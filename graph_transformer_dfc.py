#!/usr/bin/env python3
"""
Dynamic functional connectivity + graph transformer experiment.

This script keeps the leak-free subject split logic while replacing
flattened static FC features with:
  1. subject-local dynamic FC summaries
  2. a graph transformer operating on ROI nodes and connectivity edges
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from app.main import (
    DEFAULT_PREPROCESSING_CONDITION,
    DEFAULT_ROI_ATLAS,
    DEFAULT_SEED,
    compute_binary_metrics,
    compute_fisher_connectivity_matrix,
    get_data_from_abide,
    set_random_seed,
)


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clone_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def compute_dynamic_fc(subject_timeseries: np.ndarray, window_size: int = 50, stride: int = 25):
    """Compute sliding-window FC for a single subject."""
    subject_timeseries = np.asarray(subject_timeseries, dtype=float)
    n_timepoints, _ = subject_timeseries.shape

    if n_timepoints < window_size:
        static_fc = compute_fisher_connectivity_matrix(subject_timeseries)
        return static_fc, np.zeros_like(static_fc), np.expand_dims(static_fc, axis=0)

    window_fcs = []
    for start in range(0, n_timepoints - window_size + 1, stride):
        window_data = subject_timeseries[start:start + window_size, :]
        fc = compute_fisher_connectivity_matrix(window_data)
        window_fcs.append(fc)

    window_fcs = np.asarray(window_fcs, dtype=float)
    static_fc = np.mean(window_fcs, axis=0)
    fc_std = np.std(window_fcs, axis=0)
    return static_fc, fc_std, window_fcs


def build_dfc_node_features(data, window_size: int = 50, stride: int = 25):
    """
    Build node features from dynamic FC.

    Each node gets:
      - top-k strongest static connections
      - the variability of those same connections
      - a few node-level summary statistics
    """
    if not data:
      raise ValueError("data must contain at least one subject.")

    n_rois = int(np.asarray(data[0]).shape[1])
    k = max(1, min(20, n_rois - 1))

    all_node_features = []
    all_adjacency = []

    for subject_ts in data:
        static_fc, fc_std, window_fcs = compute_dynamic_fc(subject_ts, window_size, stride)

        node_strength = np.abs(static_fc).sum(axis=1, keepdims=True)
        node_pos_strength = np.maximum(static_fc, 0.0).sum(axis=1, keepdims=True)
        node_neg_strength = np.minimum(static_fc, 0.0).sum(axis=1, keepdims=True)
        node_variability = fc_std.mean(axis=1, keepdims=True)

        window_strengths = np.asarray([np.abs(wfc).sum(axis=1) for wfc in window_fcs], dtype=float)
        if window_strengths.shape[0] > 1:
            strength_std = window_strengths.std(axis=0, keepdims=True).T
            strength_range = (window_strengths.max(axis=0) - window_strengths.min(axis=0)).reshape(-1, 1)
        else:
            strength_std = np.zeros((n_rois, 1), dtype=float)
            strength_range = np.zeros((n_rois, 1), dtype=float)

        top_k_static = np.zeros((n_rois, k), dtype=float)
        top_k_variability = np.zeros((n_rois, k), dtype=float)

        for roi_index in range(n_rois):
            abs_row = np.abs(static_fc[roi_index]).copy()
            abs_row[roi_index] = 0.0
            top_indices = np.argsort(abs_row)[-k:]
            top_k_static[roi_index] = static_fc[roi_index, top_indices]
            top_k_variability[roi_index] = fc_std[roi_index, top_indices]

        node_feat = np.concatenate(
            [
                top_k_static,
                top_k_variability,
                node_strength,
                node_pos_strength,
                node_neg_strength,
                node_variability,
                strength_std,
                strength_range,
            ],
            axis=1,
        )

        all_node_features.append(node_feat.astype(np.float32))
        all_adjacency.append(static_fc.astype(np.float32))

    return np.asarray(all_node_features, dtype=np.float32), np.asarray(all_adjacency, dtype=np.float32)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.3, use_edge_features: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.d_k = self.d_model // self.n_heads
        self.use_edge_features = bool(use_edge_features)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        if self.use_edge_features:
            self.edge_proj = nn.Linear(1, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.shape

        q = self.w_q(x).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, num_nodes, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.use_edge_features and adj is not None:
            edge_bias = self.edge_proj(adj.unsqueeze(-1)).permute(0, 3, 1, 2)
            attn = attn + edge_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        out = self.w_o(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class ASDGraphTransformer(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.3,
        readout: str = "mean_max",
        num_classes: int = 2,
    ):
        super().__init__()
        self.readout = str(readout).strip().lower()

        self.input_proj = nn.Sequential(
            nn.Linear(n_node_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.transformer_layers = nn.ModuleList(
            [
                GraphTransformerLayer(d_model, n_heads, d_ff, dropout, use_edge_features=True)
                for _ in range(int(n_layers))
            ]
        )

        readout_dim = d_model * 2 if self.readout == "mean_max" else d_model
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(2, d_ff // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(2, d_ff // 2), num_classes),
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(node_features)
        for layer in self.transformer_layers:
            x = layer(x, adj)

        if self.readout == "mean_max":
            graph_repr = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        else:
            graph_repr = x.mean(dim=1)

        return self.classifier(graph_repr)


def build_graph_dataloader(node_features, adjacency, labels, batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.tensor(np.asarray(node_features), dtype=torch.float32),
        torch.tensor(np.asarray(adjacency), dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=0)


def compute_graph_average_loss(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for node_batch, adj_batch, label_batch in dataloader:
            node_batch = node_batch.to(DEVICE)
            adj_batch = adj_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            outputs = model(node_batch, adj_batch)
            loss = criterion(outputs, label_batch)
            batch_size = int(label_batch.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    return total_loss / max(1, total_samples)


def train_graph_transformer(
    train_nodes,
    train_adj,
    train_labels,
    val_nodes,
    val_adj,
    val_labels,
    config,
    verbose: bool = False,
):
    n_node_features = int(train_nodes.shape[2])
    model = ASDGraphTransformer(
        n_node_features=n_node_features,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        num_classes=2,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"])),
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = build_graph_dataloader(train_nodes, train_adj, train_labels, config["batch_size"], shuffle=True)
    val_loader = build_graph_dataloader(val_nodes, val_adj, val_labels, config["batch_size"], shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = clone_module_state(model)
    patience_counter = 0
    history = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        for node_batch, adj_batch, label_batch in train_loader:
            node_batch = node_batch.to(DEVICE)
            adj_batch = adj_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(node_batch, adj_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = int(label_batch.shape[0])
            total_train_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        average_train_loss = total_train_loss / max(1, total_samples)
        val_loss = compute_graph_average_loss(model, val_loader, criterion)
        improved = val_loss < (best_val_loss - float(config["min_delta"]))

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(average_train_loss),
                "validation_loss": float(val_loss),
                "improved": bool(improved),
            }
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = clone_module_state(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(
                f"  Epoch {epoch}/{config['epochs']} "
                f"train_loss={average_train_loss:.4f} val_loss={val_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        scheduler.step()
        if patience_counter >= int(config["patience"]):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    model.to(DEVICE)

    training_summary = {
        "epochs_trained": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "history": history,
    }
    return model, training_summary


def evaluate_graph_transformer(model, nodes, adjacency, labels):
    model.eval()
    dataloader = build_graph_dataloader(nodes, adjacency, labels, batch_size=64, shuffle=False)

    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for node_batch, adj_batch, label_batch in dataloader:
            node_batch = node_batch.to(DEVICE)
            adj_batch = adj_batch.to(DEVICE)
            outputs = model(node_batch, adj_batch)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            predicted_labels.append(predictions)
            true_labels.append(label_batch.numpy())

    predicted_labels = np.concatenate(predicted_labels)
    true_labels = np.concatenate(true_labels)
    metrics = compute_binary_metrics(true_labels, predicted_labels)
    return metrics, true_labels, predicted_labels


def run_dfc_graph_transformer(args):
    pipeline = args.pipelines[0]
    data, labels = get_data_from_abide(
        pipeline,
        preprocessing_condition=args.preprocessing_condition,
        roi_atlas=args.roi_atlas,
        return_subject_metadata=False,
    )
    labels = np.asarray(labels, dtype=int)

    if args.verbose:
        print(f"Loaded {len(labels)} subjects for pipeline '{pipeline}' and atlas '{args.roi_atlas}'")
        print(f"Computing dynamic FC (window={args.window_size}, stride={args.stride})...")

    node_features, adjacency = build_dfc_node_features(data, args.window_size, args.stride)
    if args.verbose:
        print(f"Node features shape: {node_features.shape}")
        print(f"Adjacency shape: {adjacency.shape}")

    model_config = {
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "d_ff": int(args.d_ff),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
    }

    set_random_seed(DEFAULT_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)

    fold_results = []
    for fold_id, (outer_train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n{'=' * 50}\nFold {fold_id}\n{'=' * 50}")

        stratify_labels = labels[outer_train_idx]
        train_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=0.2,
            stratify=stratify_labels,
            random_state=DEFAULT_SEED + fold_id,
        )

        train_nodes = node_features[train_idx]
        val_nodes = node_features[val_idx]
        test_nodes = node_features[test_idx]

        train_adj = adjacency[train_idx]
        val_adj = adjacency[val_idx]
        test_adj = adjacency[test_idx]

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        n_node_features = int(train_nodes.shape[2])
        scaler = StandardScaler()
        scaler.fit(train_nodes.reshape(-1, n_node_features))

        train_nodes = scaler.transform(train_nodes.reshape(-1, n_node_features)).reshape(train_nodes.shape)
        val_nodes = scaler.transform(val_nodes.reshape(-1, n_node_features)).reshape(val_nodes.shape)
        test_nodes = scaler.transform(test_nodes.reshape(-1, n_node_features)).reshape(test_nodes.shape)

        model, training_summary = train_graph_transformer(
            train_nodes,
            train_adj,
            train_labels,
            val_nodes,
            val_adj,
            val_labels,
            model_config,
            verbose=args.verbose,
        )
        metrics, true_labels, predicted_labels = evaluate_graph_transformer(model, test_nodes, test_adj, test_labels)

        fold_results.append(
            {
                "fold_id": fold_id,
                "train_indices": train_idx.tolist(),
                "validation_indices": val_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "metrics": {key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in metrics.items() if key != "confusion_matrix"},
                "training_summary": training_summary,
                "true_labels": true_labels.tolist(),
                "predicted_labels": predicted_labels.tolist(),
            }
        )
        print(
            f"Fold {fold_id}: accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}, sensitivity={metrics['sensitivity']:.4f}, "
            f"specificity={metrics['specificity']:.4f}"
        )

    accuracies = [fold["metrics"]["accuracy"] for fold in fold_results]
    f1s = [fold["metrics"]["f1"] for fold in fold_results]
    summary = {
        "pipeline": pipeline,
        "preprocessing_condition": args.preprocessing_condition,
        "roi_atlas": args.roi_atlas,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "model_config": model_config,
        "metrics": {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        },
        "fold_results": fold_results,
        "seed": DEFAULT_SEED,
    }

    artifact_root = ensure_directory(Path(args.artifact_root))
    summary_path = artifact_root / f"{pipeline}_{args.roi_atlas}_dfc_graph_transformer_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)

    print(f"\n{'=' * 60}")
    print("DYNAMIC FC + GRAPH TRANSFORMER RESULTS")
    print(f"Atlas: {args.roi_atlas}, Pipeline: {pipeline}")
    print(f"Window: {args.window_size}, Stride: {args.stride}")
    print(f"Model: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}, dropout={args.dropout}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {summary['metrics']['accuracy_mean'] * 100:.2f}% ± {summary['metrics']['accuracy_std'] * 100:.2f}%")
    print(f"F1:       {summary['metrics']['f1_mean']:.4f} ± {summary['metrics']['f1_std']:.4f}")
    print(f"Per-fold accuracies: {[f'{accuracy * 100:.1f}%' for accuracy in accuracies]}")
    print(f"Per-fold F1s:        {[f'{f1:.4f}' for f1 in f1s]}")
    print(f"Saved summary to {summary_path}")
    print(f"Seed is {DEFAULT_SEED}")

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DFC + Graph Transformer for ASD classification")
    parser.add_argument("--pipelines", nargs="*", default=["dparsf"])
    parser.add_argument("--roi_atlas", default=DEFAULT_ROI_ATLAS)
    parser.add_argument("--preprocessing_condition", default=DEFAULT_PREPROCESSING_CONDITION)
    parser.add_argument("--window_size", type=int, default=50, help="Sliding-window size for dynamic FC")
    parser.add_argument("--stride", type=int, default=25, help="Stride for sliding windows")
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of graph transformer layers")
    parser.add_argument("--d_ff", type=int, default=128, help="Feed-forward hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum validation improvement")
    parser.add_argument("--artifact_root", default="artifacts/dfc_graph_transformer", help="Directory for experiment artifacts")
    parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_dfc_graph_transformer(args)


if __name__ == "__main__":
    main()
