#!/usr/bin/env python3
"""
Progressive single-site to multi-site adaptation for ABIDE ROI connectivity.

This script does two things:
  1. Benchmarks several honest, leak-free single-site baselines using the
     existing corrected pipeline.
  2. Takes the best source-site setting and performs sequential adaptation
     across additional sites while preserving the learned classifier weights.

The adaptation stage uses a fixed source-site scaler/PCA basis and fine-tunes
an incremental logistic head with replay from previously seen sites to reduce
catastrophic forgetting.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from app.main import (
    DEFAULT_PREPROCESSING_CONDITION,
    DEFAULT_ROI_ATLAS,
    DEFAULT_SEED,
    ReanalysisConfig,
    build_edge_vector_features,
    compute_binary_metrics,
    ensure_directory,
    get_data_from_abide,
    set_random_seed,
    train_and_eval_model,
    write_json_file,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def determine_cv_splits(labels, maximum_splits=5):
  label_counts = np.bincount(np.asarray(labels, dtype=int))
  valid_counts = label_counts[label_counts > 0]
  if len(valid_counts) < 2:
    raise ValueError("At least two classes are required for cross-validation.")
  return max(2, min(int(maximum_splits), int(valid_counts.min())))


def safe_stratified_split(indices, labels, holdout_fraction, random_state):
  indices = np.asarray(indices, dtype=int)
  labels = np.asarray(labels, dtype=int)
  if len(indices) != len(labels):
    raise ValueError("indices and labels must have the same length.")

  unique_labels = np.unique(labels)
  if len(unique_labels) < 2:
    return None

  label_counts = np.bincount(labels)
  valid_counts = label_counts[label_counts > 0]
  if len(valid_counts) < 2 or int(valid_counts.min()) < 2:
    return None

  desired_holdout = int(math.ceil(len(indices) * float(holdout_fraction)))
  min_holdout = len(unique_labels)
  max_holdout = len(indices) - len(unique_labels)
  if max_holdout < min_holdout:
    return None

  holdout_size = min(max_holdout, max(min_holdout, desired_holdout))
  train_indices, holdout_indices = train_test_split(
    indices,
    test_size=holdout_size,
    stratify=labels,
    random_state=random_state,
  )
  return np.asarray(train_indices, dtype=int), np.asarray(holdout_indices, dtype=int)


def summarize_sites(subject_metadata, labels):
  metadata = subject_metadata.copy()
  metadata["label"] = np.asarray(labels, dtype=int)
  summary = metadata.groupby("site_id").agg(
    total=("label", "size"),
    asd=("label", lambda series: int((series == 0).sum())),
    control=("label", lambda series: int((series == 1).sum())),
  )
  return summary.sort_values(["total", "asd", "control"], ascending=[False, False, False])


def select_site_subset(data, labels, subject_metadata, site_ids):
  requested = {str(site_id).strip() for site_id in site_ids if str(site_id).strip()}
  mask = subject_metadata["site_id"].astype(str).isin(requested).to_numpy()
  filtered_data = [sample for sample, keep in zip(data, mask) if keep]
  filtered_labels = np.asarray(labels)[mask]
  filtered_metadata = subject_metadata.loc[mask].reset_index(drop=True)
  if len(filtered_data) == 0:
    raise ValueError(f"No subjects matched site_ids={sorted(requested)}.")
  return filtered_data, filtered_labels, filtered_metadata


def build_source_candidate_configs(base_config, artifact_root):
  return [
    ("logistic_l1_pca250_c003", dict(model_type="logistic_l1", feature_transform="pca", pca_components=250, logistic_c=0.03, selector_type="none")),
    ("logistic_l1_pca100_c003", dict(model_type="logistic_l1", feature_transform="pca", pca_components=100, logistic_c=0.03, selector_type="none")),
    ("linear_svm_pca250", dict(model_type="linear_svm", feature_transform="pca", pca_components=250, svm_c=1.0, selector_type="none")),
    ("logistic_l1_allfeatures_c003", dict(model_type="logistic_l1", feature_transform="none", pca_components=0, logistic_c=0.03, selector_type="none")),
  ]


def evaluate_single_site_candidates(site_name, data, labels, subject_metadata, base_config, artifact_root):
  feature_vectors, feature_indices = build_edge_vector_features(data)
  candidate_records = []
  best_summary = None
  best_name = None
  best_key = None

  site_cv_splits = determine_cv_splits(labels)

  for candidate_name, overrides in build_source_candidate_configs(base_config, artifact_root):
    config = copy.deepcopy(base_config)
    config.n_splits = site_cv_splits
    config.roi_atlas = base_config.roi_atlas
    config.site_filters = (site_name,)
    config.artifact_root = str(Path(artifact_root) / "single_site" / site_name / candidate_name)
    for key, value in overrides.items():
      setattr(config, key, value)

    summary = train_and_eval_model(
      feature_vectors,
      labels,
      pipeline=f"dparsf_{site_name}",
      feature_indices=feature_indices,
      subject_metadata=subject_metadata,
      verbose=True,
      train_model=True,
      save_model=False,
      config=config,
      raw_data=None,
    )
    accuracy = float(summary.metrics_summary["accuracy"]["mean"])
    f1 = float(summary.metrics_summary["f1"]["mean"])
    candidate_records.append({
      "candidate_name": candidate_name,
      "site_name": site_name,
      "accuracy_mean": accuracy,
      "accuracy_std": float(summary.metrics_summary["accuracy"]["std"]),
      "f1_mean": f1,
      "f1_std": float(summary.metrics_summary["f1"]["std"]),
      "artifact_dir": summary.artifact_dir,
      "config": overrides,
    })
    ranking_key = (accuracy, f1)
    if best_key is None or ranking_key > best_key:
      best_key = ranking_key
      best_name = candidate_name
      best_summary = summary

  results_df = pd.DataFrame(candidate_records).sort_values(
    ["accuracy_mean", "f1_mean"],
    ascending=[False, False],
  ).reset_index(drop=True)
  results_df.to_csv(Path(artifact_root) / "single_site" / site_name / "candidate_results.csv", index=False)
  return best_name, best_summary, results_df


def stratified_site_splits(subject_metadata, labels, site_schedule, test_size, random_seed):
  splits = {}
  skipped_sites = {}
  for offset, site_name in enumerate(site_schedule):
    site_mask = subject_metadata["site_id"].astype(str).eq(site_name).to_numpy()
    site_indices = np.where(site_mask)[0]
    site_labels = np.asarray(labels)[site_indices]
    split = safe_stratified_split(
      site_indices,
      site_labels,
      holdout_fraction=test_size,
      random_state=random_seed + offset,
    )
    if split is None:
      skipped_sites[site_name] = {
        "reason": "insufficient_samples_for_stratified_split",
        "total": int(len(site_indices)),
        "class_counts": np.bincount(site_labels, minlength=2).tolist(),
      }
      continue
    train_idx, test_idx = split
    splits[site_name] = {
      "train_indices": np.asarray(train_idx, dtype=int),
      "test_indices": np.asarray(test_idx, dtype=int),
    }
  return splits, skipped_sites


def make_class_weight_tensor(labels):
  labels = np.asarray(labels, dtype=int)
  class_counts = np.bincount(labels, minlength=2).astype(float)
  class_counts[class_counts == 0] = 1.0
  weights = class_counts.sum() / (len(class_counts) * class_counts)
  return torch.tensor(weights, dtype=torch.float32, device=device)


def make_tensor_dataset(features, labels):
  feature_tensor = torch.tensor(features, dtype=torch.float32)
  label_tensor = torch.tensor(labels, dtype=torch.long)
  return torch.utils.data.TensorDataset(feature_tensor, label_tensor)


def train_linear_head(model, train_features, train_labels, val_features, val_labels, learning_rate, weight_decay, max_epochs, patience, stage_name, verbose):
  train_loader = torch.utils.data.DataLoader(
    make_tensor_dataset(train_features, train_labels),
    batch_size=min(64, len(train_labels)),
    shuffle=True,
  )
  val_loader = torch.utils.data.DataLoader(
    make_tensor_dataset(val_features, val_labels),
    batch_size=min(128, len(val_labels)),
    shuffle=False,
  )
  criterion = nn.CrossEntropyLoss(weight=make_class_weight_tensor(train_labels))
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  best_state = copy.deepcopy(model.state_dict())
  best_val_loss = math.inf
  patience_counter = 0
  history = []

  model.to(device)
  for epoch in range(1, max_epochs + 1):
    model.train()
    running_loss = 0.0
    sample_count = 0
    for batch_features, batch_labels in train_loader:
      batch_features = batch_features.to(device)
      batch_labels = batch_labels.to(device)
      optimizer.zero_grad()
      logits = model(batch_features)
      loss = criterion(logits, batch_labels)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      running_loss += float(loss.item()) * int(batch_labels.size(0))
      sample_count += int(batch_labels.size(0))

    model.eval()
    val_loss = 0.0
    val_count = 0
    predictions = []
    with torch.no_grad():
      for batch_features, batch_labels in val_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        val_loss += float(loss.item()) * int(batch_labels.size(0))
        val_count += int(batch_labels.size(0))
        predictions.append(logits.argmax(dim=1).cpu().numpy())

    train_loss = running_loss / max(1, sample_count)
    val_loss = val_loss / max(1, val_count)
    val_predictions = np.concatenate(predictions) if predictions else np.empty(0, dtype=int)
    metrics = compute_binary_metrics(np.asarray(val_labels), val_predictions)
    history.append({
      "stage_name": stage_name,
      "epoch": epoch,
      "train_loss": train_loss,
      "val_loss": val_loss,
      "val_accuracy": float(metrics["accuracy"]),
      "val_f1": float(metrics["f1"]),
    })
    if verbose:
      print(
        f"{stage_name} epoch {epoch}/{max_epochs} "
        f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
        f"val_accuracy={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}"
      )

    if val_loss < best_val_loss - 1e-4:
      best_val_loss = val_loss
      best_state = copy.deepcopy(model.state_dict())
      patience_counter = 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        break

  model.load_state_dict(best_state)
  return history


def evaluate_model(model, features, labels):
  model.eval()
  feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
  with torch.no_grad():
    logits = model(feature_tensor)
    predictions = logits.argmax(dim=1).cpu().numpy()
  return compute_binary_metrics(np.asarray(labels), predictions), predictions


def sample_replay_features(seen_train_store, replay_samples_per_site, random_seed):
  if not seen_train_store:
    return np.empty((0, 0), dtype=float), np.empty(0, dtype=int)

  rng = np.random.default_rng(random_seed)
  feature_batches = []
  label_batches = []
  for site_name, features, labels in seen_train_store:
    sample_count = min(replay_samples_per_site, len(labels))
    if sample_count <= 0:
      continue
    indices = rng.choice(len(labels), size=sample_count, replace=False)
    feature_batches.append(features[indices])
    label_batches.append(labels[indices])

  if not feature_batches:
    return np.empty((0, 0), dtype=float), np.empty(0, dtype=int)

  return np.vstack(feature_batches), np.concatenate(label_batches)


def run_progressive_adaptation(data, labels, subject_metadata, source_site, site_schedule, artifact_root, base_config, verbose, test_size=0.2):
  features, _ = build_edge_vector_features(data)
  site_splits, skipped_sites = stratified_site_splits(
    subject_metadata,
    labels,
    site_schedule,
    test_size=test_size,
    random_seed=base_config.random_seed,
  )
  if source_site not in site_splits:
    raise ValueError(f"Source site '{source_site}' does not have a valid train/test split.")

  source_train_indices = site_splits[source_site]["train_indices"]
  source_test_indices = site_splits[source_site]["test_indices"]
  source_train_labels = labels[source_train_indices]
  source_inner_split = safe_stratified_split(
    source_train_indices,
    source_train_labels,
    holdout_fraction=0.2,
    random_state=base_config.random_seed,
  )
  if source_inner_split is None:
    raise ValueError(f"Source site '{source_site}' does not have enough training samples for an inner validation split.")
  inner_train_idx, inner_val_idx = source_inner_split

  scaler = StandardScaler()
  scaler.fit(features[inner_train_idx])
  source_train_scaled = scaler.transform(features[inner_train_idx])
  source_val_scaled = scaler.transform(features[inner_val_idx])
  all_scaled = scaler.transform(features)

  if base_config.pca_components and base_config.pca_components > 0:
    pca = PCA(n_components=min(int(base_config.pca_components), source_train_scaled.shape[0], source_train_scaled.shape[1]), random_state=base_config.random_seed)
    pca.fit(source_train_scaled)
    all_transformed = pca.transform(all_scaled)
  else:
    pca = None
    all_transformed = all_scaled

  model = nn.Linear(all_transformed.shape[1], 2)
  source_history = train_linear_head(
    model,
    all_transformed[inner_train_idx],
    labels[inner_train_idx],
    all_transformed[inner_val_idx],
    labels[inner_val_idx],
    learning_rate=0.005,
    weight_decay=1e-3,
    max_epochs=80,
    patience=10,
    stage_name=f"{source_site}_source",
    verbose=verbose,
  )

  stage_records = []
  seen_sites = []
  seen_train_store = []

  for stage_index, site_name in enumerate(site_schedule, start=1):
    split = site_splits.get(site_name)
    if split is None:
      continue

    train_indices = split["train_indices"]
    test_indices = split["test_indices"]

    if site_name == source_site:
      seen_sites.append(site_name)
      seen_train_store.append((site_name, all_transformed[train_indices], labels[train_indices]))
    else:
      current_split = safe_stratified_split(
        train_indices,
        labels[train_indices],
        holdout_fraction=0.2,
        random_state=base_config.random_seed + stage_index,
      )
      if current_split is None:
        stage_records.append({
          "stage_index": stage_index,
          "site_name": site_name,
          "seen_sites": list(seen_sites),
          "skipped": True,
          "reason": "insufficient_samples_for_inner_validation_split",
        })
        continue
      current_train_indices, current_val_indices = current_split
      replay_features, replay_labels = sample_replay_features(
        seen_train_store,
        replay_samples_per_site=64,
        random_seed=base_config.random_seed + stage_index,
      )
      train_features = all_transformed[current_train_indices]
      train_labels = labels[current_train_indices]
      if replay_features.size:
        train_features = np.vstack([train_features, replay_features])
        train_labels = np.concatenate([train_labels, replay_labels])

      train_linear_head(
        model,
        train_features,
        train_labels,
        all_transformed[current_val_indices],
        labels[current_val_indices],
        learning_rate=0.001,
        weight_decay=1e-3,
        max_epochs=40,
        patience=6,
        stage_name=f"{site_name}_adapt",
        verbose=verbose,
      )
      seen_sites.append(site_name)
      seen_train_store.append((site_name, all_transformed[train_indices], labels[train_indices]))

    cumulative_test_indices = np.concatenate([site_splits[name]["test_indices"] for name in seen_sites])
    cumulative_metrics, _ = evaluate_model(model, all_transformed[cumulative_test_indices], labels[cumulative_test_indices])
    site_metrics, _ = evaluate_model(model, all_transformed[test_indices], labels[test_indices])
    stage_records.append({
      "stage_index": stage_index,
      "site_name": site_name,
      "seen_sites": list(seen_sites),
      "site_accuracy": float(site_metrics["accuracy"]),
      "site_f1": float(site_metrics["f1"]),
      "cumulative_accuracy": float(cumulative_metrics["accuracy"]),
      "cumulative_f1": float(cumulative_metrics["f1"]),
      "site_sample_count": int(len(test_indices)),
      "cumulative_sample_count": int(len(cumulative_test_indices)),
    })
    if verbose:
      print(
        f"Stage {stage_index}: added {site_name} | "
        f"site_accuracy={site_metrics['accuracy']:.4f} site_f1={site_metrics['f1']:.4f} | "
        f"cumulative_accuracy={cumulative_metrics['accuracy']:.4f} cumulative_f1={cumulative_metrics['f1']:.4f}"
      )

  return {
    "source_history": source_history,
    "stage_records": stage_records,
    "source_site": source_site,
    "site_schedule": site_schedule,
    "skipped_sites": skipped_sites,
    "pca_components": int(base_config.pca_components),
    "model_type": "progressive_logistic_replay",
    "feature_representation": base_config.feature_representation,
    "roi_atlas": base_config.roi_atlas,
  }


def main():
  parser = argparse.ArgumentParser(description="Single-site benchmark plus progressive site adaptation for ABIDE.")
  parser.add_argument("--pipelines", nargs="*", default=["dparsf"])
  parser.add_argument("--preprocessing_condition", default=DEFAULT_PREPROCESSING_CONDITION)
  parser.add_argument("--roi_atlas", default="rois_aal_fd02")
  parser.add_argument("--artifact_root", default="artifacts/site_adaptation")
  parser.add_argument("--source_site", default="", help="Optional source site. Defaults to the largest available site.")
  parser.add_argument("--site_schedule", nargs="*", default=(), help="Optional explicit site schedule. Defaults to sites sorted by size descending.")
  parser.add_argument("--test_size", type=float, default=0.2)
  parser.add_argument("--random_seed", type=int, default=DEFAULT_SEED)
  parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=True)
  args = parser.parse_args()

  set_random_seed(args.random_seed)
  artifact_root = ensure_directory(args.artifact_root)

  pipeline = args.pipelines[0]
  data, labels, subject_metadata = get_data_from_abide(
    pipeline,
    preprocessing_condition=args.preprocessing_condition,
    roi_atlas=args.roi_atlas,
    return_subject_metadata=True,
  )
  site_summary = summarize_sites(subject_metadata, labels)
  print(site_summary.to_string())

  site_schedule = list(args.site_schedule or site_summary.index.tolist())
  source_site = str(args.source_site).strip() or site_schedule[0]
  source_site = source_site.strip()
  if source_site not in site_schedule:
    site_schedule = [source_site] + [site for site in site_schedule if site != source_site]

  print(f"\nSource site: {source_site}")
  print(f"Site schedule: {site_schedule}")

  base_config = ReanalysisConfig(
    preprocessing_condition=args.preprocessing_condition,
    roi_atlas=args.roi_atlas,
    feature_representation="edge_vector",
    selector_type="none",
    use_feature_scaling=True,
    save_artifacts=True,
    save_model_checkpoints=False,
    random_seed=args.random_seed,
  )

  source_data, source_labels, source_metadata = select_site_subset(data, labels, subject_metadata, [source_site])
  best_name, best_summary, source_results_df = evaluate_single_site_candidates(
    source_site,
    source_data,
    source_labels,
    source_metadata,
    base_config,
    artifact_root,
  )
  print(f"\nBest single-site candidate for {source_site}: {best_name}")
  print(source_results_df.to_string(index=False))

  best_config = copy.deepcopy(best_summary.config)
  progressive_summary = run_progressive_adaptation(
    data,
    labels,
    subject_metadata,
    source_site=source_site,
    site_schedule=site_schedule,
    artifact_root=artifact_root,
    base_config=best_config,
    verbose=args.verbose,
    test_size=args.test_size,
  )

  progressive_dir = ensure_directory(Path(artifact_root) / "progressive")
  pd.DataFrame(progressive_summary["stage_records"]).to_csv(progressive_dir / "stage_metrics.csv", index=False)
  write_json_file(progressive_dir / "summary.json", {
    "site_summary": site_summary.reset_index().to_dict(orient="records"),
    "source_site": source_site,
    "best_single_site_candidate": best_name,
    "best_single_site_metrics": {
      "accuracy_mean": float(best_summary.metrics_summary["accuracy"]["mean"]),
      "accuracy_std": float(best_summary.metrics_summary["accuracy"]["std"]),
      "f1_mean": float(best_summary.metrics_summary["f1"]["mean"]),
      "f1_std": float(best_summary.metrics_summary["f1"]["std"]),
    },
    "progressive_summary": progressive_summary,
  })
  print(f"\nSaved progressive adaptation summary to {progressive_dir / 'summary.json'}")


if __name__ == "__main__":
  main()
