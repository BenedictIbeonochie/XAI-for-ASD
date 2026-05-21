#!/usr/bin/env python3
"""
Standalone 3D convolutional autoencoder for ABIDE preprocessed NIfTI volumes.

This runner lazily reads `.nii.gz` files from a manifest CSV, collapses the
4D fMRI series into a single 3D volume, trains an unsupervised autoencoder,
and exports per-subject latent vectors for downstream classification.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from app.main import DEFAULT_SEED, ensure_directory, set_random_seed, write_json_file

DEFAULT_ARTIFACT_ROOT = "artifacts/nifti_3d_autoencoder"
DEFAULT_COLLAPSE_MODE = "variance"
DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_VALIDATION_SIZE = 0.15
DEFAULT_LATENT_DIM = 1024
DEFAULT_NUM_WORKERS = 0


@dataclass
class NiftiAutoencoderConfig:
    manifest: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    nifti_root: str | None = None
    collapse_mode: str = DEFAULT_COLLAPSE_MODE
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    validation_size: float = DEFAULT_VALIDATION_SIZE
    latent_dim: int = DEFAULT_LATENT_DIM
    num_workers: int = DEFAULT_NUM_WORKERS
    random_seed: int = DEFAULT_SEED
    max_subjects: int | None = None
    verbose: bool = False


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from '{value}'.")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def ceil_divide_by_two(value: int) -> int:
    return int((int(value) + 1) // 2)


def compute_downsampled_shapes(input_shape: tuple[int, int, int], levels: int = 3) -> list[tuple[int, int, int]]:
    current_shape = tuple(int(dimension) for dimension in input_shape)
    shapes: list[tuple[int, int, int]] = []
    for _ in range(int(levels)):
        current_shape = tuple(ceil_divide_by_two(dimension) for dimension in current_shape)
        shapes.append(current_shape)
    return shapes


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    mean_value = float(volume.mean())
    std_value = float(volume.std())
    if not np.isfinite(std_value) or std_value <= 0.0:
        std_value = 1.0
    normalized = (volume - mean_value) / std_value
    return normalized.astype(np.float32, copy=False)


def collapse_fmri_time_axis(data_4d: np.ndarray, mode: str = DEFAULT_COLLAPSE_MODE) -> np.ndarray:
    data_4d = np.asarray(data_4d, dtype=np.float32)
    normalized_mode = str(mode).strip().lower()
    if data_4d.ndim != 4:
        raise ValueError(f"Expected a 4D fMRI volume, got shape {data_4d.shape}.")
    if normalized_mode == "variance":
        collapsed = np.var(data_4d, axis=3)
    elif normalized_mode == "mean":
        collapsed = np.mean(data_4d, axis=3)
    elif normalized_mode == "std":
        collapsed = np.std(data_4d, axis=3)
    else:
        raise ValueError(
            f"Unsupported collapse_mode '{mode}'. Supported options: variance, mean, std."
        )
    return collapsed.astype(np.float32, copy=False)


def resolve_nifti_path(path_value: str, nifti_root: str | None = None) -> Path:
    candidate = Path(str(path_value))
    if candidate.exists():
        return candidate.resolve()
    if nifti_root:
        fallback = Path(nifti_root) / candidate.name
        if fallback.exists():
            return fallback.resolve()
    raise FileNotFoundError(f"Could not resolve NIfTI path '{path_value}'.")


def load_manifest_dataframe(manifest_path: str | Path, nifti_root: str | None = None, max_subjects: int | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path).copy()
    required_columns = {"file_id", "site_id", "label", "nifti_path"}
    missing = sorted(required_columns.difference(manifest.columns))
    if missing:
        raise ValueError(
            f"Manifest '{manifest_path}' is missing required columns: {', '.join(missing)}."
        )

    if max_subjects is not None:
        manifest = manifest.head(int(max_subjects)).copy()

    manifest["file_id"] = manifest["file_id"].astype(str)
    manifest["site_id"] = manifest["site_id"].astype(str)
    manifest["label"] = manifest["label"].astype(int)
    manifest["resolved_nifti_path"] = [
        str(resolve_nifti_path(path_value, nifti_root=nifti_root))
        for path_value in manifest["nifti_path"].tolist()
    ]
    return manifest.reset_index(drop=True)


class NiftiManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, collapse_mode: str = DEFAULT_COLLAPSE_MODE):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.collapse_mode = str(collapse_mode).strip().lower()
        if self.manifest.empty:
            raise ValueError("Manifest is empty; at least one subject is required.")

    def __len__(self) -> int:
        return int(len(self.manifest))

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[int(index)]
        nifti_path = Path(row["resolved_nifti_path"])
        image = nib.load(str(nifti_path))
        data_4d = image.get_fdata(dtype=np.float32)
        data_3d = collapse_fmri_time_axis(data_4d, mode=self.collapse_mode)
        data_3d = normalize_volume(data_3d)
        tensor = torch.from_numpy(data_3d).unsqueeze(0)
        return {
            "volume": tensor,
            "file_id": str(row["file_id"]),
            "site_id": str(row["site_id"]),
            "label": int(row["label"]),
            "nifti_path": str(nifti_path),
        }


def build_group_norm(channels: int) -> nn.GroupNorm:
    for group_count in (8, 4, 2, 1):
        if channels % group_count == 0:
            return nn.GroupNorm(group_count, channels)
    return nn.GroupNorm(1, channels)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            build_group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            build_group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NiftiAutoencoder3D(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.input_shape = tuple(int(dimension) for dimension in input_shape)
        self.downsample_shapes = compute_downsampled_shapes(self.input_shape, levels=3)
        self.bottleneck_shape = self.downsample_shapes[-1]
        self.bottleneck_channels = 128
        flattened_size = self.bottleneck_channels * math.prod(self.bottleneck_shape)

        self.encoder_stem = ConvBlock3D(1, 16, stride=1)
        self.encoder_down1 = ConvBlock3D(16, 32, stride=2)
        self.encoder_down2 = ConvBlock3D(32, 64, stride=2)
        self.encoder_down3 = ConvBlock3D(64, self.bottleneck_channels, stride=2)
        self.latent_projection = nn.Linear(flattened_size, int(latent_dim))

        self.latent_expansion = nn.Linear(int(latent_dim), flattened_size)
        self.decoder_block3 = ConvBlock3D(self.bottleneck_channels, 64, stride=1)
        self.decoder_block2 = ConvBlock3D(64, 32, stride=1)
        self.decoder_block1 = ConvBlock3D(32, 16, stride=1)
        self.output_layer = nn.Conv3d(16, 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_stem(x)
        x = self.encoder_down1(x)
        x = self.encoder_down2(x)
        x = self.encoder_down3(x)
        x = torch.flatten(x, start_dim=1)
        return self.latent_projection(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.latent_expansion(latent)
        x = x.view(latent.shape[0], self.bottleneck_channels, *self.bottleneck_shape)
        x = F.interpolate(x, size=self.downsample_shapes[1], mode="trilinear", align_corners=False)
        x = self.decoder_block3(x)
        x = F.interpolate(x, size=self.downsample_shapes[0], mode="trilinear", align_corners=False)
        x = self.decoder_block2(x)
        x = F.interpolate(x, size=self.input_shape, mode="trilinear", align_corners=False)
        x = self.decoder_block1(x)
        return self.output_layer(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_manifest_indices(manifest: pd.DataFrame, validation_size: float, random_seed: int) -> tuple[list[int], list[int]]:
    subject_count = int(len(manifest))
    if subject_count < 2:
        raise ValueError("At least two subjects are required to create a validation split.")

    validation_count = max(1, int(round(subject_count * float(validation_size))))
    if validation_count >= subject_count:
        validation_count = subject_count - 1

    indices = np.arange(subject_count)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_count,
        shuffle=True,
        random_state=int(random_seed),
    )
    return train_indices.tolist(), validation_indices.tolist()


def compute_reconstruction_loss(model: nn.Module, dataloader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            volumes = batch["volume"].float().to(device)
            reconstruction, _ = model(volumes)
            loss = criterion(reconstruction, volumes)
            batch_size = int(volumes.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        return float("inf")
    return float(total_loss / total_samples)


def train_autoencoder(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    config: NiftiAutoencoderConfig,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(config.patience) // 3),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    model = model.to(device)

    for epoch in range(int(config.epochs)):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_dataloader:
            volumes = batch["volume"].float().to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(volumes)
            loss = criterion(reconstruction, volumes)
            loss.backward()
            optimizer.step()

            batch_size = int(volumes.shape[0])
            total_train_loss += float(loss.item()) * batch_size
            total_train_samples += batch_size

        average_train_loss = float(total_train_loss / max(1, total_train_samples))
        validation_loss = compute_reconstruction_loss(model, validation_dataloader, device, criterion)
        improved = validation_loss < (best_validation_loss - float(config.min_delta))

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": average_train_loss,
                "validation_loss": float(validation_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": bool(improved),
            }
        )

        if improved:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(validation_loss)

        if config.verbose:
            print(
                f"Epoch {epoch + 1}/{config.epochs} "
                f"train_loss={average_train_loss:.6f} val_loss={validation_loss:.6f}"
            )

        if int(config.patience) and epochs_without_improvement >= int(config.patience):
            if config.verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    summary = {
        "epochs_trained": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "history": history,
    }
    return model, summary


def export_latent_vectors(
    model: NiftiAutoencoder3D,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model = model.to(device)
    model.eval()
    latent_rows = []

    with torch.no_grad():
        for batch in dataloader:
            volumes = batch["volume"].float().to(device)
            latent = model.encode(volumes).cpu().numpy()
            file_ids = list(batch["file_id"])
            site_ids = list(batch["site_id"])
            labels = batch["label"].tolist()
            nifti_paths = list(batch["nifti_path"])

            for row_index, latent_vector in enumerate(latent):
                record = {
                    "file_id": str(file_ids[row_index]),
                    "site_id": str(site_ids[row_index]),
                    "label": int(labels[row_index]),
                    "nifti_path": str(nifti_paths[row_index]),
                }
                for latent_index, value in enumerate(latent_vector):
                    record[f"latent_{latent_index:04d}"] = float(value)
                latent_rows.append(record)

    return pd.DataFrame(latent_rows)


def write_training_history_csv(path: Path, history: list[dict]) -> None:
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(path, index=False)


def save_checkpoint(path: Path, model: NiftiAutoencoder3D, config: NiftiAutoencoderConfig, device: torch.device) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "input_shape": tuple(int(dimension) for dimension in model.input_shape),
        "latent_dim": int(config.latent_dim),
        "config": asdict(config),
        "device": str(device),
    }
    torch.save(payload, path)


def run_autoencoder_experiment(config: NiftiAutoencoderConfig) -> dict:
    set_random_seed(int(config.random_seed))
    artifact_dir = Path(config.artifact_root)
    ensure_directory(str(artifact_dir))

    manifest = load_manifest_dataframe(
        config.manifest,
        nifti_root=config.nifti_root,
        max_subjects=config.max_subjects,
    )
    dataset = NiftiManifestDataset(manifest, collapse_mode=config.collapse_mode)
    sample_batch = dataset[0]["volume"]
    input_shape = tuple(int(dimension) for dimension in sample_batch.shape[1:])

    train_indices, validation_indices = split_manifest_indices(
        manifest,
        validation_size=float(config.validation_size),
        random_seed=int(config.random_seed),
    )

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    validation_dataloader = build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    export_dataloader = build_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    device = select_device()
    model = NiftiAutoencoder3D(input_shape=input_shape, latent_dim=int(config.latent_dim))

    if config.verbose:
        print()
        print("=" * 76)
        print("3D NIfTI Autoencoder")
        print(f"Manifest: {config.manifest}")
        print(f"Subjects: {len(manifest)}")
        print(f"Validation subjects: {len(validation_indices)}")
        print(f"Collapse mode: {config.collapse_mode}")
        print(f"Input shape: {input_shape}")
        print(f"Latent dimension: {config.latent_dim}")
        print(f"Trainable parameters: {count_trainable_parameters(model):,}")
        print(f"Device: {device}")
        print("=" * 76)
        print()

    model, training_summary = train_autoencoder(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
        device=device,
    )

    checkpoint_path = artifact_dir / "best_autoencoder.pt"
    history_path = artifact_dir / "training_history.csv"
    latent_vectors_path = artifact_dir / "latent_vectors.csv"
    summary_path = artifact_dir / "summary.json"

    save_checkpoint(checkpoint_path, model, config, device)
    write_training_history_csv(history_path, training_summary["history"])

    latent_frame = export_latent_vectors(model, export_dataloader, device=device)
    latent_frame.to_csv(latent_vectors_path, index=False)

    summary = {
        "artifact_dir": str(artifact_dir.resolve()),
        "manifest": str(Path(config.manifest).resolve()),
        "nifti_root": str(Path(config.nifti_root).resolve()) if config.nifti_root else None,
        "collapse_mode": str(config.collapse_mode),
        "device": str(device),
        "subject_count": int(len(manifest)),
        "site_count": int(manifest["site_id"].nunique()),
        "label_counts": {str(key): int(value) for key, value in manifest["label"].value_counts().sort_index().items()},
        "input_shape": list(int(dimension) for dimension in input_shape),
        "latent_dimension": int(config.latent_dim),
        "trainable_parameters": count_trainable_parameters(model),
        "train_subject_count": int(len(train_indices)),
        "validation_subject_count": int(len(validation_indices)),
        "training": training_summary,
        "output_files": {
            "checkpoint": str(checkpoint_path.resolve()),
            "training_history_csv": str(history_path.resolve()),
            "latent_vectors_csv": str(latent_vectors_path.resolve()),
        },
        "config": asdict(config),
    }
    write_json_file(str(summary_path), summary)

    if config.verbose:
        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Saved latent vectors: {latent_vectors_path}")
        print(
            f"Best validation loss: {training_summary['best_validation_loss']:.6f} "
            f"(epoch {training_summary['best_epoch']})"
        )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a standalone 3D autoencoder on lazy-loaded ABIDE NIfTI volumes."
    )
    parser.add_argument("--manifest", required=True, help="Path to the manifest CSV with file_id/site_id/label/nifti_path columns.")
    parser.add_argument("--artifact_root", default=DEFAULT_ARTIFACT_ROOT, help="Directory where checkpoints and exports will be written.")
    parser.add_argument("--nifti_root", default=None, help="Optional fallback root for resolving NIfTI basenames if manifest paths are not directly readable.")
    parser.add_argument("--collapse_mode", default=DEFAULT_COLLAPSE_MODE, help="How to collapse the 4D time axis into a 3D volume: variance, mean, or std.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training and latent export.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience in epochs.")
    parser.add_argument("--min_delta", type=float, default=DEFAULT_MIN_DELTA, help="Minimum validation-loss improvement required to reset patience.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--validation_size", type=float, default=DEFAULT_VALIDATION_SIZE, help="Fraction of subjects reserved for unsupervised validation.")
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM, help="Size of the latent vector exported for each subject.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="PyTorch DataLoader workers.")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_SEED, help="Random seed for data splitting and training.")
    parser.add_argument("--max_subjects", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--verbose", type=parse_bool, default=False, help="Whether to print per-epoch training progress.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    arguments = parser.parse_args()
    config = NiftiAutoencoderConfig(
        manifest=arguments.manifest,
        artifact_root=arguments.artifact_root,
        nifti_root=arguments.nifti_root,
        collapse_mode=arguments.collapse_mode,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        patience=arguments.patience,
        min_delta=arguments.min_delta,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        validation_size=arguments.validation_size,
        latent_dim=arguments.latent_dim,
        num_workers=arguments.num_workers,
        random_seed=arguments.random_seed,
        max_subjects=arguments.max_subjects,
        verbose=arguments.verbose,
    )
    run_autoencoder_experiment(config)


if __name__ == "__main__":
    main()
