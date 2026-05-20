#!/usr/bin/env python3
"""
Separate connectivity-matrix CNN experiment for ABIDE ASD classification.

This runner keeps the full subject-level Fisher-z connectivity matrix intact
and trains a CNN directly on the native ROI-by-ROI matrix instead of using
flattened edge vectors or resized connectivity images.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from app.main import (
    DEFAULT_PREPROCESSING_CONDITION,
    DEFAULT_ROI_ATLAS,
    DEFAULT_SEED,
    compute_binary_metrics,
    compute_fisher_connectivity_matrix,
    ensure_directory,
    get_data_from_abide,
    normalize_site_filters,
    resolve_abide_download_dir,
    set_random_seed,
    write_json_file,
)

DEFAULT_MIN_SITE_SUBJECTS = 40
DEFAULT_VALIDATION_SIZE = 0.2
MAX_CNN_SPLITS = 5
MIN_MANUAL_SITE_SUBJECTS = 10
GRADIENT_CLIP_NORM = 1.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class ConnectivityCNNConfig:
    preprocessing_condition: str = DEFAULT_PREPROCESSING_CONDITION
    roi_atlas: str = DEFAULT_ROI_ATLAS
    min_site_subjects: int = DEFAULT_MIN_SITE_SUBJECTS
    dropout: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 16
    epochs: int = 200
    patience: int = 25
    min_delta: float = 1e-4
    artifact_root: str = "artifacts/connectivity_cnn"
    validation_size: float = DEFAULT_VALIDATION_SIZE
    random_seed: int = DEFAULT_SEED
    max_splits: int = MAX_CNN_SPLITS


class ConnectivityCNN(nn.Module):
    """Three-block CNN for native connectivity matrices."""

    def __init__(self, n_rois: int, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.n_rois = int(n_rois)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(float(dropout) * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(float(dropout) * 0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Dropout2d(float(dropout)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def dedupe_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def slugify_label(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "cohort"


def build_roi_atlas_candidates(roi_atlas: str) -> list[str]:
    normalized = str(roi_atlas or DEFAULT_ROI_ATLAS).strip() or DEFAULT_ROI_ATLAS
    candidates = [normalized]
    stripped_fd_suffix = re.sub(r"_fd\d+$", "", normalized)
    if stripped_fd_suffix != normalized:
        candidates.append(stripped_fd_suffix)
    return dedupe_preserve_order(candidates)


def resolve_roi_atlas_alias(
    pipeline: str,
    preprocessing_condition: str,
    roi_atlas: str,
    downloads_root: str | Path = "abide/downloads/Outputs",
) -> str:
    last_error = None
    for candidate in build_roi_atlas_candidates(roi_atlas):
        try:
            resolve_abide_download_dir(
                pipeline,
                preprocessing_condition=preprocessing_condition,
                roi_atlas=candidate,
                downloads_root=downloads_root,
            )
            return candidate
        except FileNotFoundError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Could not resolve roi_atlas '{roi_atlas}'.")


def load_connectivity_dataset(
    pipeline: str,
    preprocessing_condition: str,
    roi_atlas: str,
    site_filters=(),
):
    resolved_roi_atlas = resolve_roi_atlas_alias(
        pipeline,
        preprocessing_condition=preprocessing_condition,
        roi_atlas=roi_atlas,
    )
    data, labels, subject_metadata = get_data_from_abide(
        pipeline,
        preprocessing_condition=preprocessing_condition,
        roi_atlas=resolved_roi_atlas,
        site_filters=normalize_site_filters(site_filters),
        return_subject_metadata=True,
    )
    labels = np.asarray(labels, dtype=int)
    subject_metadata = subject_metadata.reset_index(drop=True)
    return list(data), labels, subject_metadata, resolved_roi_atlas


def build_connectivity_matrices(data) -> np.ndarray:
    matrices = []
    for subject_timeseries in data:
        connectivity_matrix = compute_fisher_connectivity_matrix(subject_timeseries)
        matrices.append(np.asarray(connectivity_matrix, dtype=np.float32))
    return np.stack(matrices, axis=0)


def summarize_sites(subject_metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    metadata = subject_metadata.copy()
    metadata["label"] = np.asarray(labels, dtype=int)
    summary = metadata.groupby("site_id").agg(
        total=("label", "size"),
        asd=("label", lambda series: int((series == 0).sum())),
        control=("label", lambda series: int((series == 1).sum())),
    )
    return summary.sort_values(["total", "asd", "control"], ascending=[False, False, False])


def get_eligible_sites(site_summary: pd.DataFrame, min_site_subjects: int) -> list[list[str]]:
    eligible_sites = []
    for site_name, row in site_summary.iterrows():
        if int(row["total"]) < int(min_site_subjects):
            continue
        if int(row["asd"]) < 1 or int(row["control"]) < 1:
            continue
        eligible_sites.append([str(site_name)])
    return eligible_sites


def filter_site_group(
    matrices: np.ndarray,
    labels: np.ndarray,
    subject_metadata: pd.DataFrame,
    site_filters=(),
):
    normalized_filters = normalize_site_filters(site_filters)
    if not normalized_filters:
        return matrices, labels, subject_metadata.reset_index(drop=True)

    site_mask = subject_metadata["site_id"].isin(normalized_filters)
    indices = np.flatnonzero(site_mask.to_numpy())
    return (
        matrices[indices],
        labels[indices],
        subject_metadata.iloc[indices].reset_index(drop=True),
    )


def determine_cv_splits(labels: np.ndarray, maximum_splits: int = MAX_CNN_SPLITS) -> int:
    labels = np.asarray(labels, dtype=int)
    label_counts = np.bincount(labels)
    valid_counts = label_counts[label_counts > 0]
    if len(valid_counts) < 2:
        raise ValueError("At least two classes are required for cross-validation.")
    return max(2, min(int(maximum_splits), int(valid_counts.min())))


def split_outer_train_validation(
    outer_train_indices: np.ndarray,
    labels: np.ndarray,
    validation_size: float,
    random_seed: int,
):
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    outer_labels = labels[outer_train_indices]
    unique_labels = np.unique(outer_labels)
    if len(unique_labels) < 2:
        raise ValueError("Outer training fold must contain both classes.")

    proposed_validation_size = int(round(len(outer_train_indices) * float(validation_size)))
    proposed_validation_size = max(len(unique_labels), proposed_validation_size)
    proposed_validation_size = min(proposed_validation_size, len(outer_train_indices) - 1)
    if proposed_validation_size <= 0:
        raise ValueError("Validation split would be empty.")

    bincount = np.bincount(outer_labels)
    valid_counts = bincount[bincount > 0]

    if proposed_validation_size >= len(unique_labels) and int(valid_counts.min()) >= 2:
        try:
            train_indices, validation_indices = train_test_split(
                outer_train_indices,
                test_size=proposed_validation_size,
                stratify=outer_labels,
                random_state=random_seed,
            )
            return np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int)
        except ValueError:
            pass

    train_indices, validation_indices = train_test_split(
        outer_train_indices,
        test_size=proposed_validation_size,
        shuffle=True,
        random_state=random_seed,
    )
    return np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int)


def normalize_matrices_with_train_statistics(train_matrices: np.ndarray, *other_matrices):
    train_mean = float(train_matrices.mean())
    train_std = float(train_matrices.std())
    if train_std <= 0.0:
        train_std = 1.0

    normalized_train = ((train_matrices - train_mean) / train_std).astype(np.float32)
    normalized_others = [
        ((matrices - train_mean) / train_std).astype(np.float32)
        for matrices in other_matrices
    ]
    return normalized_train, normalized_others, {"train_mean": train_mean, "train_std": train_std}


def choose_training_batch_size(num_samples: int, requested_batch_size: int) -> int:
    if num_samples <= 1:
        return 1

    batch_size = max(2, min(int(requested_batch_size), int(num_samples)))
    while batch_size > 2 and (num_samples % batch_size) == 1:
        batch_size -= 1
    return batch_size


def build_dataloader(matrices: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    matrix_tensor = torch.tensor(np.asarray(matrices, dtype=np.float32), dtype=torch.float32)
    label_tensor = torch.tensor(np.asarray(labels, dtype=int), dtype=torch.long)
    dataset = TensorDataset(matrix_tensor, label_tensor)
    if shuffle:
        batch_size = choose_training_batch_size(len(dataset), batch_size)
    else:
        batch_size = max(1, min(int(batch_size), len(dataset)))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def compute_average_loss(model: nn.Module, dataloader: DataLoader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for matrices, labels in dataloader:
            matrices = matrices.to(device)
            labels = labels.to(device)
            outputs = model(matrices)
            loss = criterion(outputs, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return float("inf")
    return total_loss / total_samples


def train_connectivity_cnn(
    train_matrices: np.ndarray,
    train_labels: np.ndarray,
    validation_matrices: np.ndarray,
    validation_labels: np.ndarray,
    config: ConnectivityCNNConfig,
    verbose: bool = False,
):
    n_rois = int(train_matrices.shape[1])
    model = ConnectivityCNN(n_rois=n_rois, dropout=config.dropout).to(device)
    train_dataloader = build_dataloader(train_matrices, train_labels, config.batch_size, shuffle=True)
    validation_dataloader = build_dataloader(validation_matrices, validation_labels, config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.epochs)))

    best_state = clone_model_state(model)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    if verbose:
        n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(f"  Model: ConnectivityCNN, Parameters: {n_parameters:,}")

    for epoch in range(int(config.epochs)):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for matrices, labels in train_dataloader:
            matrices = matrices.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(matrices)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()

            batch_size = labels.shape[0]
            total_train_loss += float(loss.item()) * batch_size
            total_train_samples += batch_size

        average_train_loss = total_train_loss / max(1, total_train_samples)
        validation_loss = compute_average_loss(model, validation_dataloader, criterion)
        improved = validation_loss < (best_validation_loss - float(config.min_delta))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(average_train_loss),
                "validation_loss": float(validation_loss),
                "improved": bool(improved),
            }
        )

        if improved:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch + 1
            best_state = clone_model_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch < 5 or (epoch + 1) % 10 == 0 or epoch + 1 == int(config.epochs)):
            print(
                f"  Epoch {epoch + 1}/{int(config.epochs)} "
                f"train_loss={average_train_loss:.6f} val_loss={validation_loss:.6f}"
            )

        scheduler.step()

        if int(config.patience) and epochs_without_improvement >= int(config.patience):
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.to(device)

    return model, {
        "model_type": "connectivity_cnn",
        "device": str(device),
        "n_rois": n_rois,
        "dropout": float(config.dropout),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "batch_size": int(config.batch_size),
        "epochs_requested": int(config.epochs),
        "epochs_trained": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "scheduler_type": "cosine",
        "gradient_clip_norm": float(GRADIENT_CLIP_NORM),
        "history": history,
    }


def evaluate_connectivity_cnn(model: nn.Module, matrices: np.ndarray, labels: np.ndarray):
    dataloader = build_dataloader(matrices, labels, batch_size=64, shuffle=False)
    true_labels = []
    predicted_labels = []

    model.eval()
    with torch.no_grad():
        for batch_matrices, batch_labels in dataloader:
            batch_matrices = batch_matrices.to(device)
            outputs = model(batch_matrices)
            predicted = outputs.argmax(dim=1).cpu().numpy()
            true_labels.extend(batch_labels.numpy().tolist())
            predicted_labels.extend(predicted.tolist())

    true_labels = np.asarray(true_labels, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    return compute_binary_metrics(true_labels, predicted_labels)


def build_fold_metrics_row(
    fold_id: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    scaling_summary: dict,
    metrics: dict,
    training_summary: dict,
):
    return {
        "fold": int(fold_id),
        "train_size": int(len(train_indices)),
        "validation_size": int(len(validation_indices)),
        "test_size": int(len(test_indices)),
        "train_mean": float(scaling_summary["train_mean"]),
        "train_std": float(scaling_summary["train_std"]),
        "accuracy": float(metrics["accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "precision": float(metrics["precision"]),
        "f1": float(metrics["f1"]),
        "epochs_trained": int(training_summary["epochs_trained"]),
        "best_epoch": int(training_summary["best_epoch"]),
        "best_validation_loss": float(training_summary["best_validation_loss"]),
    }


def summarize_metric(values: list[float]) -> dict[str, float]:
    values_array = np.asarray(values, dtype=float)
    return {
        "mean": float(values_array.mean()),
        "std": float(values_array.std()),
    }


def build_run_record(summary: dict) -> dict:
    return {
        "cohort": summary["cohort_label"],
        "pipeline": summary["pipeline"],
        "requested_roi_atlas": summary["requested_roi_atlas"],
        "resolved_roi_atlas": summary["resolved_roi_atlas"],
        "n_subjects": int(summary["n_subjects"]),
        "n_asd": int(summary["n_asd"]),
        "n_control": int(summary["n_control"]),
        "accuracy_mean": float(summary["metrics"]["accuracy"]["mean"]),
        "accuracy_std": float(summary["metrics"]["accuracy"]["std"]),
        "f1_mean": float(summary["metrics"]["f1"]["mean"]),
        "f1_std": float(summary["metrics"]["f1"]["std"]),
        "sensitivity_mean": float(summary["metrics"]["sensitivity"]["mean"]),
        "specificity_mean": float(summary["metrics"]["specificity"]["mean"]),
        "precision_mean": float(summary["metrics"]["precision"]["mean"]),
        "artifact_dir": summary["artifact_dir"],
    }


def run_cohort_evaluation(
    matrices: np.ndarray,
    labels: np.ndarray,
    subject_metadata: pd.DataFrame,
    pipeline: str,
    requested_roi_atlas: str,
    resolved_roi_atlas: str,
    cohort_label: str,
    config: ConnectivityCNNConfig,
    verbose: bool = False,
    site_filters=(),
):
    labels = np.asarray(labels, dtype=int)
    subject_count = int(len(labels))
    asd_count = int((labels == 0).sum())
    control_count = int((labels == 1).sum())
    unique_labels = np.unique(labels)

    if subject_count < MIN_MANUAL_SITE_SUBJECTS or len(unique_labels) < 2:
        raise ValueError(
            f"Cohort '{cohort_label}' is not suitable for training: "
            f"{subject_count} subjects, classes={unique_labels.tolist()}."
        )

    n_rois = int(matrices.shape[1])
    n_splits = determine_cv_splits(labels, maximum_splits=config.max_splits)
    artifact_dir = ensure_directory(Path(config.artifact_root) / pipeline / slugify_label(cohort_label))

    if verbose:
        print(f"\n{'=' * 68}")
        print(f"Cohort: {cohort_label}")
        print(f"Subjects: {subject_count} ({asd_count} ASD, {control_count} control)")
        print(f"Pipeline: {pipeline} | Atlas: {resolved_roi_atlas}")
        print(f"Matrix shape per subject: {n_rois}x{n_rois}")
        print(f"Device: {device}")
        print(f"{'=' * 68}")

    set_random_seed(config.random_seed)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )

    fold_metrics_rows = []

    for fold_id, (outer_train_indices, test_indices) in enumerate(splitter.split(matrices, labels), start=1):
        if verbose:
            print(f"\n--- Fold {fold_id}/{n_splits} ---")

        train_indices, validation_indices = split_outer_train_validation(
            outer_train_indices,
            labels,
            validation_size=config.validation_size,
            random_seed=config.random_seed + fold_id,
        )

        train_matrices = matrices[train_indices]
        validation_matrices = matrices[validation_indices]
        test_matrices = matrices[test_indices]
        train_labels = labels[train_indices]
        validation_labels = labels[validation_indices]
        test_labels = labels[test_indices]

        train_matrices, [validation_matrices, test_matrices], scaling_summary = normalize_matrices_with_train_statistics(
            train_matrices,
            validation_matrices,
            test_matrices,
        )

        model, training_summary = train_connectivity_cnn(
            train_matrices,
            train_labels,
            validation_matrices,
            validation_labels,
            config=config,
            verbose=verbose,
        )
        metrics = evaluate_connectivity_cnn(model, test_matrices, test_labels)

        fold_metrics_rows.append(
            build_fold_metrics_row(
                fold_id,
                train_indices,
                validation_indices,
                test_indices,
                scaling_summary,
                metrics,
                training_summary,
            )
        )

        if verbose:
            print(
                f"  Fold {fold_id}: accuracy={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1']:.4f}, "
                f"sensitivity={metrics['sensitivity']:.4f}, "
                f"specificity={metrics['specificity']:.4f}"
            )

    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    fold_metrics_df.to_csv(artifact_dir / "fold_metrics.csv", index=False)

    summary = {
        "cohort_label": cohort_label,
        "pipeline": pipeline,
        "preprocessing_condition": config.preprocessing_condition,
        "requested_roi_atlas": requested_roi_atlas,
        "resolved_roi_atlas": resolved_roi_atlas,
        "site_filters": list(normalize_site_filters(site_filters)),
        "artifact_dir": str(artifact_dir),
        "n_subjects": subject_count,
        "n_asd": asd_count,
        "n_control": control_count,
        "n_splits": int(n_splits),
        "native_matrix_shape": [n_rois, n_rois],
        "config": asdict(config),
        "metrics": {
            "accuracy": summarize_metric(fold_metrics_df["accuracy"].tolist()),
            "sensitivity": summarize_metric(fold_metrics_df["sensitivity"].tolist()),
            "specificity": summarize_metric(fold_metrics_df["specificity"].tolist()),
            "precision": summarize_metric(fold_metrics_df["precision"].tolist()),
            "f1": summarize_metric(fold_metrics_df["f1"].tolist()),
        },
        "folds": fold_metrics_rows,
    }
    write_json_file(artifact_dir / "summary.json", summary)
    return summary


def run_pipeline_experiments(
    pipeline: str,
    config: ConnectivityCNNConfig,
    requested_roi_atlas: str,
    site_filters=(),
    run_all_large_sites: bool = False,
    run_pooled_all_sites: bool = False,
    verbose: bool = False,
):
    if not run_all_large_sites and not run_pooled_all_sites and not site_filters:
        raise ValueError("Pass --site, --run_all_large_sites, or --run_pooled_all_sites.")

    data, labels, subject_metadata, resolved_roi_atlas = load_connectivity_dataset(
        pipeline,
        preprocessing_condition=config.preprocessing_condition,
        roi_atlas=requested_roi_atlas,
    )
    matrices = build_connectivity_matrices(data)
    site_summary = summarize_sites(subject_metadata, labels)

    output_root = ensure_directory(Path(config.artifact_root))
    payload = {
        "pipeline": pipeline,
        "preprocessing_condition": config.preprocessing_condition,
        "requested_roi_atlas": requested_roi_atlas,
        "resolved_roi_atlas": resolved_roi_atlas,
        "config": asdict(config),
        "site_summary": site_summary.reset_index().to_dict(orient="records"),
        "site_runs": [],
        "pooled_run": None,
        "requested_site_run": None,
    }

    if run_all_large_sites:
        site_records = []
        for site_group in get_eligible_sites(site_summary, config.min_site_subjects):
            cohort_matrices, cohort_labels, cohort_metadata = filter_site_group(
                matrices,
                labels,
                subject_metadata,
                site_filters=site_group,
            )
            summary = run_cohort_evaluation(
                cohort_matrices,
                cohort_labels,
                cohort_metadata,
                pipeline=pipeline,
                requested_roi_atlas=requested_roi_atlas,
                resolved_roi_atlas=resolved_roi_atlas,
                cohort_label="+".join(site_group),
                config=config,
                verbose=verbose,
                site_filters=site_group,
            )
            site_records.append(build_run_record(summary))
            payload["site_runs"].append(summary)

        if site_records:
            site_records_df = pd.DataFrame(site_records).sort_values(
                ["accuracy_mean", "f1_mean"],
                ascending=[False, False],
            ).reset_index(drop=True)
            site_csv_path = output_root / f"{pipeline}_{resolved_roi_atlas}_site_comparison.csv"
            site_json_path = output_root / f"{pipeline}_{resolved_roi_atlas}_site_comparison.json"
            site_records_df.to_csv(site_csv_path, index=False)
            write_json_file(site_json_path, {"runs": site_records_df.to_dict(orient="records"), "site_summary": payload["site_summary"]})

            if verbose:
                print("\nSite-by-site comparison:")
                print(
                    site_records_df[
                        [
                            "cohort",
                            "n_subjects",
                            "n_asd",
                            "n_control",
                            "accuracy_mean",
                            "accuracy_std",
                            "f1_mean",
                            "f1_std",
                        ]
                    ].to_string(index=False)
                )

    if site_filters and not run_all_large_sites:
        cohort_matrices, cohort_labels, cohort_metadata = filter_site_group(
            matrices,
            labels,
            subject_metadata,
            site_filters=site_filters,
        )
        manual_summary = run_cohort_evaluation(
            cohort_matrices,
            cohort_labels,
            cohort_metadata,
            pipeline=pipeline,
            requested_roi_atlas=requested_roi_atlas,
            resolved_roi_atlas=resolved_roi_atlas,
            cohort_label="+".join(normalize_site_filters(site_filters)),
            config=config,
            verbose=verbose,
            site_filters=site_filters,
        )
        payload["requested_site_run"] = manual_summary

    if run_pooled_all_sites:
        pooled_summary = run_cohort_evaluation(
            matrices,
            labels,
            subject_metadata,
            pipeline=pipeline,
            requested_roi_atlas=requested_roi_atlas,
            resolved_roi_atlas=resolved_roi_atlas,
            cohort_label="pooled_all_sites",
            config=config,
            verbose=verbose,
            site_filters=(),
        )
        payload["pooled_run"] = pooled_summary

    write_json_file(output_root / f"{pipeline}_{resolved_roi_atlas}_experiment_summary.json", payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Native connectivity-matrix CNN experiment for ABIDE.")
    parser.add_argument("--pipelines", nargs="*", default=["dparsf"])
    parser.add_argument("--preprocessing_condition", default=DEFAULT_PREPROCESSING_CONDITION)
    parser.add_argument("--roi_atlas", default=DEFAULT_ROI_ATLAS)
    parser.add_argument("--site", nargs="+", default=(), help="Optional site ID(s) combined into one manual cohort, e.g. --site NYU")
    parser.add_argument("--run_all_large_sites", action="store_true", help="Run each eligible site independently and save a comparison table.")
    parser.add_argument("--run_pooled_all_sites", action="store_true", help="Run one pooled multi-site cohort across all available subjects.")
    parser.add_argument("--min_site_subjects", type=int, default=DEFAULT_MIN_SITE_SUBJECTS)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--artifact_root", default="artifacts/connectivity_cnn")
    parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ConnectivityCNNConfig(
        preprocessing_condition=args.preprocessing_condition,
        roi_atlas=args.roi_atlas,
        min_site_subjects=args.min_site_subjects,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        artifact_root=args.artifact_root,
    )

    for pipeline in args.pipelines:
        payload = run_pipeline_experiments(
            pipeline,
            config=config,
            requested_roi_atlas=args.roi_atlas,
            site_filters=args.site,
            run_all_large_sites=bool(args.run_all_large_sites),
            run_pooled_all_sites=bool(args.run_pooled_all_sites),
            verbose=bool(args.verbose),
        )

        if payload.get("requested_site_run") is not None:
            metrics = payload["requested_site_run"]["metrics"]
            print(
                f"\nManual cohort result: accuracy={metrics['accuracy']['mean'] * 100:.2f}% "
                f"+/- {metrics['accuracy']['std'] * 100:.2f}%, "
                f"f1={metrics['f1']['mean']:.4f} +/- {metrics['f1']['std']:.4f}"
            )

        if payload.get("pooled_run") is not None:
            metrics = payload["pooled_run"]["metrics"]
            print(
                f"\nPooled result: accuracy={metrics['accuracy']['mean'] * 100:.2f}% "
                f"+/- {metrics['accuracy']['std'] * 100:.2f}%, "
                f"f1={metrics['f1']['mean']:.4f} +/- {metrics['f1']['std']:.4f}"
            )


if __name__ == "__main__":
    main()
