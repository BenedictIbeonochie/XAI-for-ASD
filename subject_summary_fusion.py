#!/usr/bin/env python3
"""
Standalone LOSO subject-summary fusion experiment for ABIDE ASD classification.

This runner reads each subject's raw ROI `.1D` time series directly, builds
deterministic per-subject summaries, and evaluates with a leave-one-site-out
outer loop. Each LOSO round fits every train-time artifact only on the training
sites: structured-feature scaling, ASD/control prototypes, TF-IDF vocabulary,
and the final classifier.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.main import (
    DEFAULT_PREPROCESSING_CONDITION,
    DEFAULT_ROI_ATLAS,
    DEFAULT_SEED,
    build_generic_roi_labels,
    compute_binary_metrics,
    compute_fisher_connectivity_matrix,
    ensure_directory,
    get_data_from_abide,
    load_roi_display_metadata,
    set_random_seed,
    write_json_file,
)

DEFAULT_MIN_SITE_SUBJECTS = 40
DEFAULT_VALIDATION_SIZE = 0.20
DEFAULT_ARTIFACT_ROOT = "artifacts/subject_summary_fusion"
DEFAULT_TOP_ROIS_PER_SECTION = 5
DEFAULT_TOP_CONTRIBUTION_FEATURES = 5
DEFAULT_TOP_TOKEN_COUNT = 20

CLASSIFIER_PARAMS = {
    "solver": "saga",
    "penalty": "elasticnet",
    "l1_ratio": 0.5,
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 5000,
}

ROI_METRIC_SPECS = (
    ("time_series_mean", "ts_mean"),
    ("time_series_std", "ts_std"),
    ("lag1_autocorr", "lag1_autocorr"),
    ("mean_abs_first_diff", "mean_abs_first_diff"),
    ("positive_connectivity_strength", "positive_connectivity_strength"),
    ("negative_connectivity_strength", "negative_connectivity_strength"),
    ("mean_abs_connectivity", "mean_abs_connectivity"),
)

GLOBAL_METRIC_NAMES = (
    "global_mean_connectivity",
    "global_connectivity_std",
    "global_positive_edge_fraction",
    "global_negative_edge_fraction",
    "global_mean_abs_connectivity",
    "global_top_decile_mean_abs_connectivity",
    "global_mean_roi_volatility",
    "global_mean_roi_autocorrelation",
)

ABLATION_ORDER = (
    "structured_only",
    "structured_plus_prototypes",
    "full_fused",
)

ABLATION_LABELS = {
    "structured_only": "Structured Only",
    "structured_plus_prototypes": "Structured + Prototype Similarities",
    "full_fused": "Full Fused",
}


@dataclass
class SubjectSummaryFusionConfig:
    preprocessing_condition: str = DEFAULT_PREPROCESSING_CONDITION
    roi_atlas: str = DEFAULT_ROI_ATLAS
    min_site_subjects: int = DEFAULT_MIN_SITE_SUBJECTS
    validation_size: float = DEFAULT_VALIDATION_SIZE
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    random_seed: int = DEFAULT_SEED
    top_rois_per_section: int = DEFAULT_TOP_ROIS_PER_SECTION
    top_contribution_features: int = DEFAULT_TOP_CONTRIBUTION_FEATURES
    top_token_count: int = DEFAULT_TOP_TOKEN_COUNT


@dataclass
class PreparedSubjectDataset:
    structured_features: np.ndarray
    feature_names: list[str]
    feature_text_tokens: list[str]
    roi_display_labels: list[str]
    subject_rows: pd.DataFrame


@dataclass
class PrototypeBundle:
    scaler: StandardScaler
    asd_centroid: np.ndarray
    control_centroid: np.ndarray
    midpoint: np.ndarray
    delta: np.ndarray


def slugify_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def slugify_label(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "artifact"


def label_name_from_int(label: int) -> str:
    return "ASD" if int(label) == 0 else "Control"


def summarize_metric(values: Iterable[float]) -> dict[str, float]:
    values_array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(values_array.mean()) if values_array.size else 0.0,
        "std": float(values_array.std()) if values_array.size else 0.0,
    }


def stable_text_seed(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(str(value)))


def safe_lag1_autocorrelation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    current = values[:-1]
    shifted = values[1:]
    current_centered = current - current.mean()
    shifted_centered = shifted - shifted.mean()
    numerator = float(np.dot(current_centered, shifted_centered))
    denominator = math.sqrt(float(np.dot(current_centered, current_centered)) * float(np.dot(shifted_centered, shifted_centered)))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def resolve_roi_labels(roi_atlas: str, roi_count: int) -> list[str]:
    try:
        display_metadata = load_roi_display_metadata(
            roi_atlas=roi_atlas,
            minimum_roi_count=int(roi_count),
            require_coordinates=False,
        )
        labels = np.asarray(display_metadata["labels"], dtype=object).astype(str)
        if len(labels) >= int(roi_count):
            return labels[: int(roi_count)].tolist()
    except Exception:
        pass

    return build_generic_roi_labels(int(roi_count), roi_atlas).astype(str).tolist()


def build_feature_metadata(roi_display_labels: list[str]) -> tuple[list[str], list[str]]:
    feature_names = []
    feature_text_tokens = []

    for column_name, token_suffix in ROI_METRIC_SPECS:
        for roi_index, roi_label in enumerate(roi_display_labels):
            feature_names.append(f"roi_{roi_index:03d}_{column_name}")
            feature_text_tokens.append(f"{slugify_token(roi_label)}__{token_suffix}")

    for global_metric_name in GLOBAL_METRIC_NAMES:
        feature_names.append(global_metric_name)
        feature_text_tokens.append(slugify_token(global_metric_name))

    return feature_names, feature_text_tokens


def top_k_roi_tokens(values: np.ndarray, roi_display_labels: list[str], prefix: str, top_k: int) -> list[str]:
    values = np.asarray(values, dtype=float)
    top_indices = np.argsort(-values, kind="stable")[: int(top_k)]
    return [f"{prefix}_{slugify_token(roi_display_labels[index])}" for index in top_indices]


def build_intrinsic_summary_text(
    roi_display_labels: list[str],
    volatility: np.ndarray,
    positive_strength: np.ndarray,
    negative_strength: np.ndarray,
    mean_abs_connectivity: np.ndarray,
    top_k: int,
) -> str:
    tokens = ["intrinsic_summary"]
    tokens.extend(["top_volatility", *top_k_roi_tokens(volatility, roi_display_labels, "vol", top_k)])
    tokens.extend(["top_positive_strength", *top_k_roi_tokens(positive_strength, roi_display_labels, "pos", top_k)])
    tokens.extend(["top_negative_strength", *top_k_roi_tokens(negative_strength, roi_display_labels, "neg", top_k)])
    tokens.extend(["top_abs_connectivity", *top_k_roi_tokens(mean_abs_connectivity, roi_display_labels, "abs", top_k)])
    return " ".join(tokens)


def build_subject_summary(
    subject_roi_timeseries: np.ndarray,
    roi_display_labels: list[str],
    top_rois_per_section: int,
) -> dict:
    subject_roi_timeseries = np.asarray(subject_roi_timeseries, dtype=float)
    connectivity_matrix = np.asarray(
        compute_fisher_connectivity_matrix(subject_roi_timeseries),
        dtype=float,
    )
    connectivity_matrix = np.nan_to_num(connectivity_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    n_rois = int(subject_roi_timeseries.shape[1])
    denominator = max(1, n_rois - 1)
    time_differences = np.diff(subject_roi_timeseries, axis=0)

    time_series_mean = np.nan_to_num(subject_roi_timeseries.mean(axis=0), nan=0.0)
    time_series_std = np.nan_to_num(subject_roi_timeseries.std(axis=0), nan=0.0)
    lag1_autocorr = np.asarray(
        [safe_lag1_autocorrelation(subject_roi_timeseries[:, roi_index]) for roi_index in range(n_rois)],
        dtype=float,
    )
    mean_abs_first_diff = (
        np.nan_to_num(np.abs(time_differences).mean(axis=0), nan=0.0)
        if time_differences.size
        else np.zeros(n_rois, dtype=float)
    )

    positive_connectivity_strength = np.clip(connectivity_matrix, 0.0, None).sum(axis=1) / denominator
    negative_connectivity_strength = np.clip(-connectivity_matrix, 0.0, None).sum(axis=1) / denominator
    mean_abs_connectivity = np.abs(connectivity_matrix).sum(axis=1) / denominator

    upper_triangle = connectivity_matrix[np.triu_indices(n_rois, k=1)]
    absolute_edges = np.abs(upper_triangle)
    if absolute_edges.size:
        top_decile_edge_count = max(1, int(math.ceil(absolute_edges.size * 0.10)))
        top_decile_mean_abs = float(np.partition(absolute_edges, -top_decile_edge_count)[-top_decile_edge_count:].mean())
    else:
        top_decile_mean_abs = 0.0

    structured_values = np.concatenate(
        [
            np.nan_to_num(time_series_mean, nan=0.0),
            np.nan_to_num(time_series_std, nan=0.0),
            np.nan_to_num(lag1_autocorr, nan=0.0),
            np.nan_to_num(mean_abs_first_diff, nan=0.0),
            np.nan_to_num(positive_connectivity_strength, nan=0.0),
            np.nan_to_num(negative_connectivity_strength, nan=0.0),
            np.nan_to_num(mean_abs_connectivity, nan=0.0),
            np.asarray(
                [
                    float(np.nan_to_num(upper_triangle.mean(), nan=0.0)) if upper_triangle.size else 0.0,
                    float(np.nan_to_num(upper_triangle.std(), nan=0.0)) if upper_triangle.size else 0.0,
                    float(np.mean(upper_triangle > 0.0)) if upper_triangle.size else 0.0,
                    float(np.mean(upper_triangle < 0.0)) if upper_triangle.size else 0.0,
                    float(np.nan_to_num(absolute_edges.mean(), nan=0.0)) if absolute_edges.size else 0.0,
                    float(top_decile_mean_abs),
                    float(np.nan_to_num(mean_abs_first_diff.mean(), nan=0.0)),
                    float(np.nan_to_num(lag1_autocorr.mean(), nan=0.0)),
                ],
                dtype=float,
            ),
        ]
    ).astype(np.float32)

    intrinsic_summary_text = build_intrinsic_summary_text(
        roi_display_labels=roi_display_labels,
        volatility=mean_abs_first_diff,
        positive_strength=positive_connectivity_strength,
        negative_strength=negative_connectivity_strength,
        mean_abs_connectivity=mean_abs_connectivity,
        top_k=top_rois_per_section,
    )

    return {
        "structured_values": structured_values,
        "intrinsic_summary_text": intrinsic_summary_text,
        "n_timepoints": int(subject_roi_timeseries.shape[0]),
        "n_rois": n_rois,
    }


def prepare_subject_dataset(
    data: list[np.ndarray],
    labels: np.ndarray,
    subject_metadata: pd.DataFrame,
    roi_atlas: str,
    config: SubjectSummaryFusionConfig,
) -> PreparedSubjectDataset:
    if not data:
        raise ValueError("No subject ROI time series were loaded.")

    roi_count = int(np.asarray(data[0]).shape[1])
    roi_display_labels = resolve_roi_labels(roi_atlas, roi_count)
    feature_names, feature_text_tokens = build_feature_metadata(roi_display_labels)

    structured_rows = []
    summary_rows = []

    for subject_roi_timeseries, label, metadata_row in zip(data, labels, subject_metadata.to_dict(orient="records")):
        subject_summary = build_subject_summary(
            subject_roi_timeseries,
            roi_display_labels=roi_display_labels,
            top_rois_per_section=config.top_rois_per_section,
        )
        structured_rows.append(subject_summary["structured_values"])
        summary_rows.append(
            {
                "file_id": str(metadata_row["file_id"]),
                "site_id": str(metadata_row["site_id"]),
                "label": int(label),
                "label_name": label_name_from_int(int(label)),
                "age_at_scan": metadata_row.get("age_at_scan"),
                "sex": metadata_row.get("sex"),
                "n_timepoints": int(subject_summary["n_timepoints"]),
                "n_rois": int(subject_summary["n_rois"]),
                "intrinsic_summary_text": subject_summary["intrinsic_summary_text"],
            }
        )

    return PreparedSubjectDataset(
        structured_features=np.vstack(structured_rows).astype(np.float32),
        feature_names=feature_names,
        feature_text_tokens=feature_text_tokens,
        roi_display_labels=roi_display_labels,
        subject_rows=pd.DataFrame(summary_rows).reset_index(drop=True),
    )


def summarize_sites(subject_rows: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    metadata = subject_rows[["file_id", "site_id"]].copy()
    metadata["label"] = np.asarray(labels, dtype=int)
    summary = metadata.groupby("site_id").agg(
        total=("label", "size"),
        asd=("label", lambda values: int((values == 0).sum())),
        control=("label", lambda values: int((values == 1).sum())),
    )
    return summary.sort_values(["total", "asd", "control"], ascending=[False, False, False])


def determine_loso_sites(site_summary: pd.DataFrame, min_site_subjects: int) -> tuple[list[str], list[dict]]:
    eligible_sites = []
    skipped_sites = []

    for site_name, row in site_summary.iterrows():
        total = int(row["total"])
        asd_count = int(row["asd"])
        control_count = int(row["control"])
        reason = None

        if total < int(min_site_subjects):
            reason = f"below_min_site_subjects:{min_site_subjects}"
        elif asd_count < 1 or control_count < 1:
            reason = "missing_class_balance"

        if reason is None:
            eligible_sites.append(str(site_name))
            continue

        skipped_sites.append(
            {
                "site_id": str(site_name),
                "total": total,
                "asd": asd_count,
                "control": control_count,
                "reason": reason,
            }
        )

    return eligible_sites, skipped_sites


def split_train_validation_indices(
    train_indices: np.ndarray,
    labels: np.ndarray,
    validation_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=int)
    train_labels = np.asarray(labels, dtype=int)[train_indices]
    unique_labels = np.unique(train_labels)
    if len(unique_labels) < 2:
        raise ValueError("The LOSO training pool must contain both classes.")

    requested_validation_size = int(round(len(train_indices) * float(validation_size)))
    requested_validation_size = max(len(unique_labels), requested_validation_size)
    requested_validation_size = min(requested_validation_size, len(train_indices) - 1)
    if requested_validation_size <= 0:
        raise ValueError("Validation split would be empty.")

    bincount = np.bincount(train_labels, minlength=2)
    valid_counts = bincount[bincount > 0]

    if requested_validation_size >= len(unique_labels) and int(valid_counts.min()) >= 2:
        try:
            train_subset, validation_subset = train_test_split(
                train_indices,
                test_size=requested_validation_size,
                stratify=train_labels,
                random_state=random_seed,
            )
            return np.asarray(train_subset, dtype=int), np.asarray(validation_subset, dtype=int)
        except ValueError:
            pass

    train_subset, validation_subset = train_test_split(
        train_indices,
        test_size=requested_validation_size,
        shuffle=True,
        random_state=random_seed,
    )
    return np.asarray(train_subset, dtype=int), np.asarray(validation_subset, dtype=int)


def fit_prototype_bundle(
    train_structured_features: np.ndarray,
    train_labels: np.ndarray,
) -> PrototypeBundle:
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(np.asarray(train_structured_features, dtype=float))
    train_labels = np.asarray(train_labels, dtype=int)

    asd_centroid = scaled_train_features[train_labels == 0].mean(axis=0)
    control_centroid = scaled_train_features[train_labels == 1].mean(axis=0)
    midpoint = (asd_centroid + control_centroid) / 2.0
    delta = asd_centroid - control_centroid

    return PrototypeBundle(
        scaler=scaler,
        asd_centroid=np.asarray(asd_centroid, dtype=float),
        control_centroid=np.asarray(control_centroid, dtype=float),
        midpoint=np.asarray(midpoint, dtype=float),
        delta=np.asarray(delta, dtype=float),
    )


def safe_cosine_similarity_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    numerator = matrix @ vector
    similarities = np.divide(
        numerator,
        denominator,
        out=np.zeros(matrix.shape[0], dtype=float),
        where=denominator > 0.0,
    )
    return similarities.astype(np.float32)


def build_contrast_outputs(
    scaled_structured_features: np.ndarray,
    bundle: PrototypeBundle,
    feature_text_tokens: list[str],
    top_contribution_features: int,
) -> dict:
    scaled_structured_features = np.asarray(scaled_structured_features, dtype=float)
    asd_similarity = safe_cosine_similarity_rows(scaled_structured_features, bundle.asd_centroid)
    control_similarity = safe_cosine_similarity_rows(scaled_structured_features, bundle.control_centroid)
    similarity_gap = asd_similarity - control_similarity

    signed_contributions = (scaled_structured_features - bundle.midpoint) * bundle.delta
    contrast_texts = []
    toward_asd_features = []
    toward_control_features = []

    for contribution_row in signed_contributions:
        asd_indices = np.argsort(-contribution_row, kind="stable")
        asd_indices = [int(index) for index in asd_indices if contribution_row[index] > 0.0][: int(top_contribution_features)]
        control_indices = np.argsort(contribution_row, kind="stable")
        control_indices = [int(index) for index in control_indices if contribution_row[index] < 0.0][: int(top_contribution_features)]

        asd_tokens = [feature_text_tokens[index] for index in asd_indices]
        control_tokens = [feature_text_tokens[index] for index in control_indices]

        contrast_tokens = ["contrast_summary", "toward_asd", *[f"asd_{token}" for token in asd_tokens]]
        contrast_tokens.extend(["toward_control", *[f"control_{token}" for token in control_tokens]])

        contrast_texts.append(" ".join(contrast_tokens))
        toward_asd_features.append(", ".join(asd_tokens))
        toward_control_features.append(", ".join(control_tokens))

    return {
        "prototype_features": np.column_stack([asd_similarity, control_similarity, similarity_gap]).astype(np.float32),
        "contrast_summary_text": contrast_texts,
        "toward_asd_features": toward_asd_features,
        "toward_control_features": toward_control_features,
        "asd_prototype_similarity": asd_similarity.astype(np.float32),
        "control_prototype_similarity": control_similarity.astype(np.float32),
        "prototype_similarity_gap": similarity_gap.astype(np.float32),
        "signed_contributions": signed_contributions.astype(np.float32),
    }


def build_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )


def combine_summary_texts(intrinsic_texts: Iterable[str], contrast_texts: Iterable[str]) -> list[str]:
    return [
        f"{str(intrinsic_text).strip()} {str(contrast_text).strip()}".strip()
        for intrinsic_text, contrast_text in zip(intrinsic_texts, contrast_texts)
    ]


def build_ablation_feature_blocks(
    scaled_structured_features: np.ndarray,
    prototype_features: np.ndarray,
    tfidf_matrix: sparse.csr_matrix,
) -> dict[str, sparse.csr_matrix]:
    structured_sparse = sparse.csr_matrix(np.asarray(scaled_structured_features, dtype=np.float32))
    prototype_sparse = sparse.csr_matrix(np.asarray(prototype_features, dtype=np.float32))
    return {
        "structured_only": structured_sparse,
        "structured_plus_prototypes": sparse.hstack(
            [structured_sparse, prototype_sparse],
            format="csr",
        ),
        "full_fused": sparse.hstack(
            [structured_sparse, prototype_sparse, tfidf_matrix],
            format="csr",
        ),
    }


def build_classifier(random_seed: int) -> LogisticRegression:
    return LogisticRegression(
        random_state=int(random_seed),
        **CLASSIFIER_PARAMS,
    )


def predict_asd_probability(model: LogisticRegression, feature_matrix: sparse.csr_matrix) -> np.ndarray:
    probabilities = model.predict_proba(feature_matrix)
    class_index = int(np.flatnonzero(model.classes_ == 0)[0])
    return probabilities[:, class_index].astype(np.float32)


def train_and_evaluate_ablation_models(
    train_blocks: dict[str, sparse.csr_matrix],
    train_labels: np.ndarray,
    eval_blocks: dict[str, sparse.csr_matrix],
    eval_labels: np.ndarray,
    random_seed: int,
) -> dict[str, dict]:
    results = {}
    train_labels = np.asarray(train_labels, dtype=int)
    eval_labels = np.asarray(eval_labels, dtype=int)

    for offset, ablation_name in enumerate(ABLATION_ORDER):
        model = build_classifier(random_seed + offset)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*'penalty' was deprecated.*",
                category=FutureWarning,
            )
            model.fit(train_blocks[ablation_name], train_labels)
        predicted_labels = model.predict(eval_blocks[ablation_name]).astype(int)
        predicted_probability_asd = predict_asd_probability(model, eval_blocks[ablation_name])
        metrics = compute_binary_metrics(eval_labels, predicted_labels, positive_label=0)
        results[ablation_name] = {
            "model": model,
            "predicted_labels": predicted_labels,
            "predicted_probability_asd": predicted_probability_asd,
            "metrics": {key: float(value) for key, value in metrics.items() if key not in {"confusion_matrix"}},
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }

    return results


def build_token_rankings(
    held_out_site: str,
    vectorizer: TfidfVectorizer,
    train_text_matrix: sparse.csr_matrix,
    train_labels: np.ndarray,
    top_token_count: int,
) -> list[dict]:
    train_labels = np.asarray(train_labels, dtype=int)
    if train_text_matrix.shape[1] == 0:
        return []

    feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    asd_mask = train_labels == 0
    control_mask = train_labels == 1

    if not asd_mask.any() or not control_mask.any():
        return []

    asd_mean = np.asarray(train_text_matrix[asd_mask].mean(axis=0)).ravel()
    control_mean = np.asarray(train_text_matrix[control_mask].mean(axis=0)).ravel()
    gap = asd_mean - control_mean

    records = []
    top_asd_indices = np.argsort(-gap, kind="stable")[: int(top_token_count)]
    top_control_indices = np.argsort(gap, kind="stable")[: int(top_token_count)]

    for rank, feature_index in enumerate(top_asd_indices, start=1):
        records.append(
            {
                "held_out_site": held_out_site,
                "favored_class": "ASD",
                "rank": int(rank),
                "token": str(feature_names[feature_index]),
                "asd_mean_tfidf": float(asd_mean[feature_index]),
                "control_mean_tfidf": float(control_mean[feature_index]),
                "gap": float(gap[feature_index]),
            }
        )

    for rank, feature_index in enumerate(top_control_indices, start=1):
        records.append(
            {
                "held_out_site": held_out_site,
                "favored_class": "Control",
                "rank": int(rank),
                "token": str(feature_names[feature_index]),
                "asd_mean_tfidf": float(asd_mean[feature_index]),
                "control_mean_tfidf": float(control_mean[feature_index]),
                "gap": float(gap[feature_index]),
            }
        )

    return records


def select_top_features(values: np.ndarray, feature_names: list[str], top_k: int, descending: bool = True) -> list[dict]:
    values = np.asarray(values, dtype=float)
    ordered_indices = np.argsort(-values if descending else values, kind="stable")[: int(top_k)]
    return [
        {
            "feature_name": str(feature_names[index]),
            "value": float(values[index]),
        }
        for index in ordered_indices
    ]


def build_prototype_record(
    held_out_site: str,
    train_site_names: list[str],
    bundle: PrototypeBundle,
    feature_names: list[str],
    validation_results: dict[str, dict],
    test_results: dict[str, dict],
) -> dict:
    prototype_delta = np.asarray(bundle.asd_centroid - bundle.control_centroid, dtype=float)
    return {
        "held_out_site": held_out_site,
        "train_sites": list(train_site_names),
        "fit_sites": {
            "scaler": list(train_site_names),
            "vectorizer": list(train_site_names),
            "prototypes": list(train_site_names),
        },
        "structured_scaler_mean": bundle.scaler.mean_.astype(float).tolist(),
        "structured_scaler_scale": bundle.scaler.scale_.astype(float).tolist(),
        "asd_centroid": bundle.asd_centroid.astype(float).tolist(),
        "control_centroid": bundle.control_centroid.astype(float).tolist(),
        "top_asd_centroid_features": select_top_features(bundle.asd_centroid, feature_names, top_k=10, descending=True),
        "top_control_centroid_features": select_top_features(bundle.control_centroid, feature_names, top_k=10, descending=True),
        "top_delta_toward_asd_features": select_top_features(prototype_delta, feature_names, top_k=10, descending=True),
        "top_delta_toward_control_features": select_top_features(prototype_delta, feature_names, top_k=10, descending=False),
        "validation_metrics": {
            ablation_name: ablation_result["metrics"]
            for ablation_name, ablation_result in validation_results.items()
        },
        "test_metrics": {
            ablation_name: ablation_result["metrics"]
            for ablation_name, ablation_result in test_results.items()
        },
    }


def build_site_metric_row(
    held_out_site: str,
    train_site_names: list[str],
    validation_site_names: list[str],
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    validation_results: dict[str, dict],
    test_results: dict[str, dict],
) -> dict:
    primary_metrics = test_results["full_fused"]["metrics"]
    row = {
        "held_out_site": held_out_site,
        "train_sites": "|".join(train_site_names),
        "validation_sites": "|".join(validation_site_names),
        "n_train": int(len(train_labels)),
        "n_validation": int(len(validation_labels)),
        "n_test": int(len(test_labels)),
        "train_asd": int((np.asarray(train_labels, dtype=int) == 0).sum()),
        "train_control": int((np.asarray(train_labels, dtype=int) == 1).sum()),
        "validation_asd": int((np.asarray(validation_labels, dtype=int) == 0).sum()),
        "validation_control": int((np.asarray(validation_labels, dtype=int) == 1).sum()),
        "test_asd": int((np.asarray(test_labels, dtype=int) == 0).sum()),
        "test_control": int((np.asarray(test_labels, dtype=int) == 1).sum()),
        "accuracy": float(primary_metrics["accuracy"]),
        "sensitivity": float(primary_metrics["sensitivity"]),
        "specificity": float(primary_metrics["specificity"]),
        "precision": float(primary_metrics["precision"]),
        "f1": float(primary_metrics["f1"]),
    }

    for ablation_name in ABLATION_ORDER:
        validation_metrics = validation_results[ablation_name]["metrics"]
        test_metrics = test_results[ablation_name]["metrics"]
        for metric_name in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
            row[f"{ablation_name}_validation_{metric_name}"] = float(validation_metrics[metric_name])
            row[f"{ablation_name}_test_{metric_name}"] = float(test_metrics[metric_name])

    return row


def filter_to_eligible_sites(
    prepared_dataset: PreparedSubjectDataset,
    eligible_sites: list[str],
) -> tuple[PreparedSubjectDataset, np.ndarray]:
    site_mask = prepared_dataset.subject_rows["site_id"].isin(eligible_sites).to_numpy()
    indices = np.flatnonzero(site_mask)
    filtered_subject_rows = prepared_dataset.subject_rows.iloc[indices].reset_index(drop=True)
    filtered_structured_features = prepared_dataset.structured_features[indices]
    return (
        PreparedSubjectDataset(
            structured_features=filtered_structured_features,
            feature_names=list(prepared_dataset.feature_names),
            feature_text_tokens=list(prepared_dataset.feature_text_tokens),
            roi_display_labels=list(prepared_dataset.roi_display_labels),
            subject_rows=filtered_subject_rows,
        ),
        indices,
    )


def run_single_loso_round(
    prepared_dataset: PreparedSubjectDataset,
    held_out_site: str,
    config: SubjectSummaryFusionConfig,
    verbose: bool = False,
) -> dict:
    subject_rows = prepared_dataset.subject_rows
    labels = subject_rows["label"].to_numpy(dtype=int)
    held_out_mask = subject_rows["site_id"].eq(held_out_site).to_numpy()
    test_indices = np.flatnonzero(held_out_mask)
    train_indices = np.flatnonzero(~held_out_mask)

    if test_indices.size == 0:
        raise ValueError(f"Held-out site '{held_out_site}' has no subjects after filtering.")

    inner_train_indices, validation_indices = split_train_validation_indices(
        train_indices=train_indices,
        labels=labels,
        validation_size=config.validation_size,
        random_seed=config.random_seed + stable_text_seed(held_out_site),
    )

    train_structured = prepared_dataset.structured_features[train_indices]
    inner_train_structured = prepared_dataset.structured_features[inner_train_indices]
    validation_structured = prepared_dataset.structured_features[validation_indices]
    test_structured = prepared_dataset.structured_features[test_indices]

    train_subject_rows = subject_rows.iloc[train_indices].reset_index(drop=True)
    inner_train_subject_rows = subject_rows.iloc[inner_train_indices].reset_index(drop=True)
    validation_subject_rows = subject_rows.iloc[validation_indices].reset_index(drop=True)
    test_subject_rows = subject_rows.iloc[test_indices].reset_index(drop=True)

    train_labels = labels[train_indices]
    inner_train_labels = labels[inner_train_indices]
    validation_labels = labels[validation_indices]
    test_labels = labels[test_indices]

    if verbose:
        print(f"\n--- Held-out Site: {held_out_site} ---")
        print(
            f"Train sites: {sorted(train_subject_rows['site_id'].unique().tolist())} | "
            f"Validation subjects: {len(validation_indices)} | Test subjects: {len(test_indices)}"
        )

    validation_bundle = fit_prototype_bundle(inner_train_structured, inner_train_labels)
    validation_scaled_train = validation_bundle.scaler.transform(inner_train_structured)
    validation_scaled_eval = validation_bundle.scaler.transform(validation_structured)

    validation_train_contrast = build_contrast_outputs(
        validation_scaled_train,
        validation_bundle,
        prepared_dataset.feature_text_tokens,
        top_contribution_features=config.top_contribution_features,
    )
    validation_eval_contrast = build_contrast_outputs(
        validation_scaled_eval,
        validation_bundle,
        prepared_dataset.feature_text_tokens,
        top_contribution_features=config.top_contribution_features,
    )

    validation_train_text = combine_summary_texts(
        inner_train_subject_rows["intrinsic_summary_text"].tolist(),
        validation_train_contrast["contrast_summary_text"],
    )
    validation_eval_text = combine_summary_texts(
        validation_subject_rows["intrinsic_summary_text"].tolist(),
        validation_eval_contrast["contrast_summary_text"],
    )

    validation_vectorizer = build_text_vectorizer()
    validation_train_tfidf = validation_vectorizer.fit_transform(validation_train_text)
    validation_eval_tfidf = validation_vectorizer.transform(validation_eval_text)

    validation_train_blocks = build_ablation_feature_blocks(
        validation_scaled_train,
        validation_train_contrast["prototype_features"],
        validation_train_tfidf,
    )
    validation_eval_blocks = build_ablation_feature_blocks(
        validation_scaled_eval,
        validation_eval_contrast["prototype_features"],
        validation_eval_tfidf,
    )

    validation_results = train_and_evaluate_ablation_models(
        validation_train_blocks,
        inner_train_labels,
        validation_eval_blocks,
        validation_labels,
        random_seed=config.random_seed,
    )

    full_train_bundle = fit_prototype_bundle(train_structured, train_labels)
    full_train_scaled = full_train_bundle.scaler.transform(train_structured)
    test_scaled = full_train_bundle.scaler.transform(test_structured)

    full_train_contrast = build_contrast_outputs(
        full_train_scaled,
        full_train_bundle,
        prepared_dataset.feature_text_tokens,
        top_contribution_features=config.top_contribution_features,
    )
    test_contrast = build_contrast_outputs(
        test_scaled,
        full_train_bundle,
        prepared_dataset.feature_text_tokens,
        top_contribution_features=config.top_contribution_features,
    )

    full_train_text = combine_summary_texts(
        train_subject_rows["intrinsic_summary_text"].tolist(),
        full_train_contrast["contrast_summary_text"],
    )
    test_text = combine_summary_texts(
        test_subject_rows["intrinsic_summary_text"].tolist(),
        test_contrast["contrast_summary_text"],
    )

    full_train_vectorizer = build_text_vectorizer()
    full_train_tfidf = full_train_vectorizer.fit_transform(full_train_text)
    test_tfidf = full_train_vectorizer.transform(test_text)

    full_train_blocks = build_ablation_feature_blocks(
        full_train_scaled,
        full_train_contrast["prototype_features"],
        full_train_tfidf,
    )
    test_blocks = build_ablation_feature_blocks(
        test_scaled,
        test_contrast["prototype_features"],
        test_tfidf,
    )

    test_results = train_and_evaluate_ablation_models(
        full_train_blocks,
        train_labels,
        test_blocks,
        test_labels,
        random_seed=config.random_seed + 1000,
    )

    token_rankings = build_token_rankings(
        held_out_site=held_out_site,
        vectorizer=full_train_vectorizer,
        train_text_matrix=full_train_tfidf,
        train_labels=train_labels,
        top_token_count=config.top_token_count,
    )

    subject_output_rows = test_subject_rows.copy()
    subject_output_rows["held_out_site"] = held_out_site
    subject_output_rows["contrast_summary_text"] = test_contrast["contrast_summary_text"]
    subject_output_rows["toward_asd_features"] = test_contrast["toward_asd_features"]
    subject_output_rows["toward_control_features"] = test_contrast["toward_control_features"]
    subject_output_rows["asd_prototype_similarity"] = test_contrast["asd_prototype_similarity"]
    subject_output_rows["control_prototype_similarity"] = test_contrast["control_prototype_similarity"]
    subject_output_rows["prototype_similarity_gap"] = test_contrast["prototype_similarity_gap"]
    subject_output_rows["predicted_label"] = test_results["full_fused"]["predicted_labels"].astype(int)
    subject_output_rows["predicted_label_name"] = [label_name_from_int(label) for label in subject_output_rows["predicted_label"]]
    subject_output_rows["predicted_probability_asd"] = test_results["full_fused"]["predicted_probability_asd"]
    subject_output_rows["summary_text"] = (
        subject_output_rows["intrinsic_summary_text"].astype(str)
        + " "
        + subject_output_rows["contrast_summary_text"].astype(str)
    )

    structured_output_df = pd.DataFrame(
        test_structured,
        columns=prepared_dataset.feature_names,
        index=subject_output_rows.index,
    )
    subject_output_rows = pd.concat([subject_output_rows, structured_output_df], axis=1)

    site_metric_row = build_site_metric_row(
        held_out_site=held_out_site,
        train_site_names=sorted(train_subject_rows["site_id"].unique().tolist()),
        validation_site_names=sorted(validation_subject_rows["site_id"].unique().tolist()),
        train_labels=train_labels,
        validation_labels=validation_labels,
        test_labels=test_labels,
        validation_results=validation_results,
        test_results=test_results,
    )

    prototype_record = build_prototype_record(
        held_out_site=held_out_site,
        train_site_names=sorted(train_subject_rows["site_id"].unique().tolist()),
        bundle=full_train_bundle,
        feature_names=prepared_dataset.feature_names,
        validation_results=validation_results,
        test_results=test_results,
    )

    if verbose:
        full_fused_metrics = test_results["full_fused"]["metrics"]
        print(
            f"Held-out {held_out_site}: accuracy={full_fused_metrics['accuracy']:.4f}, "
            f"f1={full_fused_metrics['f1']:.4f}, "
            f"sensitivity={full_fused_metrics['sensitivity']:.4f}, "
            f"specificity={full_fused_metrics['specificity']:.4f}"
        )

    return {
        "held_out_site": held_out_site,
        "site_metric_row": site_metric_row,
        "subject_output_rows": subject_output_rows,
        "prototype_record": prototype_record,
        "token_rankings": token_rankings,
        "validation_results": validation_results,
        "test_results": test_results,
    }


def load_subject_dataset_for_pipeline(
    pipeline: str,
    config: SubjectSummaryFusionConfig,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    data, labels, subject_metadata = get_data_from_abide(
        pipeline,
        preprocessing_condition=config.preprocessing_condition,
        roi_atlas=config.roi_atlas,
        site_filters=normalize_site_filters(()),
        return_subject_metadata=True,
    )
    return list(data), np.asarray(labels, dtype=int), subject_metadata.reset_index(drop=True)


def run_pipeline_loso_experiment(
    pipeline: str,
    config: SubjectSummaryFusionConfig,
    verbose: bool = False,
) -> dict:
    set_random_seed(config.random_seed)
    data, labels, subject_metadata = load_subject_dataset_for_pipeline(pipeline, config)
    prepared_dataset = prepare_subject_dataset(
        data=data,
        labels=labels,
        subject_metadata=subject_metadata,
        roi_atlas=config.roi_atlas,
        config=config,
    )

    site_summary = summarize_sites(prepared_dataset.subject_rows, labels)
    eligible_sites, skipped_sites = determine_loso_sites(site_summary, config.min_site_subjects)
    if len(eligible_sites) < 2:
        raise ValueError(
            f"Need at least 2 eligible LOSO sites, found {len(eligible_sites)}. "
            f"Skipped sites: {skipped_sites}"
        )

    eligible_dataset, eligible_indices = filter_to_eligible_sites(prepared_dataset, eligible_sites)
    eligible_labels = labels[eligible_indices]

    artifact_dir = ensure_directory(
        Path(config.artifact_root)
        / pipeline
        / config.preprocessing_condition
        / slugify_label(config.roi_atlas)
        / "loso_all_sites"
    )

    if verbose:
        eligible_subject_rows = eligible_dataset.subject_rows
        print(f"\n{'=' * 76}")
        print("LOSO Subject-Summary Fusion")
        print(f"Pipeline: {pipeline}")
        print(f"Preprocessing: {config.preprocessing_condition}")
        print(f"ROI Atlas: {config.roi_atlas}")
        print(
            f"Eligible subjects: {len(eligible_subject_rows)} "
            f"({int((eligible_labels == 0).sum())} ASD, {int((eligible_labels == 1).sum())} control)"
        )
        print(f"Eligible held-out sites: {eligible_sites}")
        if skipped_sites:
            print(f"Skipped sites: {[record['site_id'] for record in skipped_sites]}")
        print(f"{'=' * 76}")

    site_results = []
    subject_output_frames = []
    prototype_records = []
    token_records = []

    for held_out_site in eligible_sites:
        round_result = run_single_loso_round(
            prepared_dataset=eligible_dataset,
            held_out_site=held_out_site,
            config=config,
            verbose=verbose,
        )
        site_results.append(round_result["site_metric_row"])
        subject_output_frames.append(round_result["subject_output_rows"])
        prototype_records.append(round_result["prototype_record"])
        token_records.extend(round_result["token_rankings"])

    loso_site_metrics_df = pd.DataFrame(site_results).sort_values("held_out_site").reset_index(drop=True)
    subject_summaries_df = pd.concat(subject_output_frames, axis=0).reset_index(drop=True)
    top_tokens_df = pd.DataFrame(token_records)

    loso_site_metrics_df.to_csv(artifact_dir / "loso_site_metrics.csv", index=False)
    subject_summaries_df.to_csv(artifact_dir / "subject_summaries.csv", index=False)
    if top_tokens_df.empty:
        top_tokens_df = pd.DataFrame(
            columns=[
                "held_out_site",
                "favored_class",
                "rank",
                "token",
                "asd_mean_tfidf",
                "control_mean_tfidf",
                "gap",
            ]
        )
    top_tokens_df.to_csv(artifact_dir / "top_tokens_by_class_by_site.csv", index=False)

    class_prototypes_payload = {
        "pipeline": pipeline,
        "preprocessing_condition": config.preprocessing_condition,
        "roi_atlas": config.roi_atlas,
        "site_prototypes": prototype_records,
    }
    write_json_file(artifact_dir / "class_prototypes_by_site.json", class_prototypes_payload)

    aggregate_metrics = {
        metric_name: summarize_metric(loso_site_metrics_df[metric_name].tolist())
        for metric_name in ("accuracy", "sensitivity", "specificity", "precision", "f1")
    }

    ablation_summary = {}
    for ablation_name in ABLATION_ORDER:
        ablation_summary[ablation_name] = {
            metric_name: summarize_metric(loso_site_metrics_df[f"{ablation_name}_test_{metric_name}"].tolist())
            for metric_name in ("accuracy", "sensitivity", "specificity", "precision", "f1")
        }

    summary = {
        "pipeline": pipeline,
        "preprocessing_condition": config.preprocessing_condition,
        "roi_atlas": config.roi_atlas,
        "artifact_dir": str(artifact_dir),
        "total_subjects_loaded": int(len(labels)),
        "eligible_subjects": int(len(eligible_dataset.subject_rows)),
        "eligible_sites": list(eligible_sites),
        "skipped_sites": skipped_sites,
        "site_summary": site_summary.reset_index().to_dict(orient="records"),
        "aggregate_metrics": aggregate_metrics,
        "ablation_comparison": ablation_summary,
        "per_site_metrics": loso_site_metrics_df.to_dict(orient="records"),
        "config": asdict(config),
        "defaults": {
            "min_site_subjects": int(config.min_site_subjects),
            "validation_fraction": float(config.validation_size),
            "text_backend": "local_tfidf",
            "summary_components": [
                "structured_features",
                "intrinsic_summary_text",
                "contrast_summary_text",
            ],
            "prototype_feature_count": 3,
            "classifier": CLASSIFIER_PARAMS,
            "random_seed": int(config.random_seed),
        },
    }
    write_json_file(artifact_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone LOSO subject-summary fusion experiment for ABIDE ROI .1D files."
    )
    parser.add_argument("--pipelines", nargs="*", default=["dparsf"])
    parser.add_argument("--preprocessing_condition", default=DEFAULT_PREPROCESSING_CONDITION)
    parser.add_argument("--roi_atlas", default=DEFAULT_ROI_ATLAS)
    parser.add_argument("--artifact_root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--min_site_subjects", type=int, default=DEFAULT_MIN_SITE_SUBJECTS)
    parser.add_argument("--validation_size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", type=lambda value: str(value).lower() == "true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SubjectSummaryFusionConfig(
        preprocessing_condition=args.preprocessing_condition,
        roi_atlas=args.roi_atlas,
        artifact_root=args.artifact_root,
        min_site_subjects=args.min_site_subjects,
        validation_size=args.validation_size,
        random_seed=args.random_seed,
    )

    for pipeline in args.pipelines:
        summary = run_pipeline_loso_experiment(
            pipeline=pipeline,
            config=config,
            verbose=bool(args.verbose),
        )
        metrics = summary["aggregate_metrics"]
        print(
            f"\n{pipeline} LOSO result: accuracy={metrics['accuracy']['mean'] * 100:.2f}% "
            f"+/- {metrics['accuracy']['std'] * 100:.2f}%, "
            f"f1={metrics['f1']['mean']:.4f} +/- {metrics['f1']['std']:.4f}"
        )


if __name__ == "__main__":
    main()
