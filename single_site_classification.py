#!/usr/bin/env python3
"""
Single-site ASD classification using the corrected leak-free pipeline.

This script reuses the main reanalysis codepath but restricts the cohort to
one ABIDE site at a time (or a user-specified group of sites) so you can
measure the single-site ceiling without multi-site heterogeneity.

Examples:
  python3 single_site_classification.py \
    --pipelines dparsf \
    --roi_atlas rois_aal_fd02 \
    --site NYU \
    --feature_representation edge_vector \
    --feature_transform pca \
    --pca_components 250 \
    --model_type logistic_l1 \
    --logistic_c 0.03 \
    --selector_type none \
    --verbose true

  python3 single_site_classification.py \
    --pipelines dparsf \
    --roi_atlas rois_cc200_fd02 \
    --run_all_large_sites \
    --min_site_subjects 40 \
    --feature_representation edge_vector \
    --feature_transform pca \
    --pca_components 250 \
    --model_type logistic_l1 \
    --logistic_c 0.03 \
    --selector_type none \
    --verbose true
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from app.main import (
  DEFAULT_PREPROCESSING_CONDITION,
  DEFAULT_ROI_ATLAS,
  DEFAULT_SEED,
  ReanalysisConfig,
  ensure_directory,
  get_data_from_abide,
  normalize_site_filters,
  run_pipeline_reanalysis,
  write_json_file,
)


def determine_cv_splits(labels, maximum_splits=5):
  labels = np.asarray(labels, dtype=int)
  label_counts = np.bincount(labels)
  valid_counts = label_counts[label_counts > 0]
  if len(valid_counts) < 2:
    raise ValueError("At least two classes are required for cross-validation.")
  return max(2, min(int(maximum_splits), int(valid_counts.min())))


def summarize_sites(subject_metadata, labels):
  metadata = subject_metadata.copy()
  metadata["label"] = np.asarray(labels, dtype=int)
  summary = metadata.groupby("site_id").agg(
    total=("label", "size"),
    asd=("label", lambda series: int((series == 0).sum())),
    control=("label", lambda series: int((series == 1).sum())),
  )
  return summary.sort_values(["total", "asd", "control"], ascending=[False, False, False])


def build_single_site_record(summary, site_label, subject_count, asd_count, control_count):
  return {
    "site": site_label,
    "n_subjects": int(subject_count),
    "n_asd": int(asd_count),
    "n_control": int(control_count),
    "pipeline": summary.pipeline,
    "preprocessing_condition": summary.config.preprocessing_condition,
    "roi_atlas": summary.config.roi_atlas,
    "site_filters": "|".join(summary.config.site_filters),
    "feature_representation": summary.config.feature_representation,
    "feature_transform": summary.config.feature_transform,
    "pca_components": int(summary.config.pca_components),
    "model_type": summary.config.model_type,
    "selector_type": summary.config.selector_type,
    "accuracy_mean": float(summary.metrics_summary["accuracy"]["mean"]),
    "accuracy_std": float(summary.metrics_summary["accuracy"]["std"]),
    "f1_mean": float(summary.metrics_summary["f1"]["mean"]),
    "f1_std": float(summary.metrics_summary["f1"]["std"]),
    "sensitivity_mean": float(summary.metrics_summary["sensitivity"]["mean"]),
    "specificity_mean": float(summary.metrics_summary["specificity"]["mean"]),
    "precision_mean": float(summary.metrics_summary["precision"]["mean"]),
    "artifact_dir": summary.artifact_dir,
  }


def run_site_evaluation(pipeline, site_ids, base_config, verbose=False):
  site_filters = normalize_site_filters(site_ids)
  site_label = "+".join(site_filters)

  data, labels, subject_metadata = get_data_from_abide(
    pipeline,
    preprocessing_condition=base_config.preprocessing_condition,
    roi_atlas=base_config.roi_atlas,
    site_filters=site_filters,
    return_subject_metadata=True,
  )
  labels = np.asarray(labels, dtype=int)
  subject_count = len(labels)
  asd_count = int((labels == 0).sum())
  control_count = int((labels == 1).sum())

  if subject_count < 10 or len(np.unique(labels)) < 2:
    print(f"Skipping {site_label}: not enough subjects/classes after filtering.")
    return None, None

  config = copy.deepcopy(base_config)
  config.site_filters = site_filters
  config.n_splits = determine_cv_splits(labels)
  config.artifact_root = str(Path(base_config.artifact_root) / pipeline / site_label.replace("+", "_"))

  print(f"\n{'=' * 64}")
  print(f"Site: {site_label}")
  print(f"Subjects: {subject_count} ({asd_count} ASD, {control_count} control)")
  print(f"Pipeline: {pipeline} | Atlas: {config.roi_atlas}")
  print(
    f"Model: {config.model_type} | Selector: {config.selector_type} | "
    f"Representation: {config.feature_representation}"
  )
  print(f"{'=' * 64}")

  summary = run_pipeline_reanalysis(pipeline, verbose=verbose, config=config)
  record = build_single_site_record(summary, site_label, subject_count, asd_count, control_count)
  print(
    f"Completed {site_label}: "
    f"accuracy={record['accuracy_mean']:.4f} ± {record['accuracy_std']:.4f}, "
    f"f1={record['f1_mean']:.4f} ± {record['f1_std']:.4f}"
  )
  return summary, record


def parse_args():
  parser = argparse.ArgumentParser(description="Leak-free single-site ABIDE classification.")
  parser.add_argument("--pipelines", nargs="*", default=["dparsf"])
  parser.add_argument("--preprocessing_condition", default=DEFAULT_PREPROCESSING_CONDITION)
  parser.add_argument("--roi_atlas", default=DEFAULT_ROI_ATLAS)
  parser.add_argument("--site", nargs="+", default=(), help="Site ID(s) to run. Example: --site NYU")
  parser.add_argument("--combine_sites", type=lambda x: str(x).lower() == "true", default=False, help="When multiple sites are passed, treat them as one combined cohort instead of running them individually.")
  parser.add_argument("--run_all_large_sites", action="store_true", help="Run each site with at least --min_site_subjects subjects.")
  parser.add_argument("--min_site_subjects", type=int, default=40)
  parser.add_argument("--artifact_root", default="artifacts/single_site")
  parser.add_argument("--feature_representation", default="edge_vector")
  parser.add_argument("--feature_transform", default="none")
  parser.add_argument("--pca_components", type=int, default=0)
  parser.add_argument("--selector_type", default="rfe")
  parser.add_argument("--num_selected_features", type=int, default=1000)
  parser.add_argument("--feature_count_candidates", nargs="*", type=int, default=None)
  parser.add_argument("--feature_count_selection_metric", default="f1")
  parser.add_argument("--rfecv_inner_splits", type=int, default=3)
  parser.add_argument("--rfe_step", type=int, default=20)
  parser.add_argument("--model_type", default="ssae")
  parser.add_argument("--svm_c", type=float, default=1.0)
  parser.add_argument("--logistic_c", type=float, default=1.0)
  parser.add_argument("--elastic_net_l1_ratio", type=float, default=0.5)
  parser.add_argument("--ae1_hidden_size", type=int, default=500)
  parser.add_argument("--ae2_hidden_size", type=int, default=100)
  parser.add_argument("--ssae_dropout_rate", type=float, default=0.0)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--ae1_epochs", type=int, default=50)
  parser.add_argument("--ae2_epochs", type=int, default=50)
  parser.add_argument("--classifier_epochs", type=int, default=300)
  parser.add_argument("--fine_tuning_epochs", type=int, default=125)
  parser.add_argument("--early_stopping_patience", type=int, default=15)
  parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
  parser.add_argument("--ae_learning_rate", type=float, default=0.001)
  parser.add_argument("--classifier_learning_rate", type=float, default=0.001)
  parser.add_argument("--fine_tuning_learning_rate", type=float, default=0.0001)
  parser.add_argument("--weight_decay", type=float, default=1e-4)
  parser.add_argument("--ae_scheduler_type", default="none")
  parser.add_argument("--supervised_scheduler_type", default="none")
  parser.add_argument("--scheduler_patience", type=int, default=5)
  parser.add_argument("--scheduler_factor", type=float, default=0.5)
  parser.add_argument("--use_feature_scaling", type=lambda x: str(x).lower() == "true", default=True)
  parser.add_argument("--harmonization_method", default="none")
  parser.add_argument("--harmonization_covariates", nargs="*", default=("age", "sex"))
  parser.add_argument("--enable_confound_regression", type=lambda x: str(x).lower() == "true", default=False)
  parser.add_argument("--confound_variables", nargs="*", default=("site", "age", "sex"))
  parser.add_argument("--random_seed", type=int, default=DEFAULT_SEED)
  parser.add_argument("--verbose", type=lambda x: str(x).lower() == "true", default=True)
  return parser.parse_args()


def main():
  args = parse_args()
  pipeline = args.pipelines[0]

  base_config = ReanalysisConfig(
    random_seed=args.random_seed,
    artifact_root=args.artifact_root,
    preprocessing_condition=args.preprocessing_condition,
    roi_atlas=args.roi_atlas,
    num_selected_features=args.num_selected_features,
    feature_representation=args.feature_representation,
    feature_transform=args.feature_transform,
    pca_components=args.pca_components,
    selector_type=args.selector_type,
    rfecv_inner_splits=args.rfecv_inner_splits,
    model_type=args.model_type,
    rfe_step=args.rfe_step,
    ae1_epochs=args.ae1_epochs,
    ae2_epochs=args.ae2_epochs,
    classifier_epochs=args.classifier_epochs,
    fine_tuning_epochs=args.fine_tuning_epochs,
    ae1_hidden_size=args.ae1_hidden_size,
    ae2_hidden_size=args.ae2_hidden_size,
    ssae_dropout_rate=args.ssae_dropout_rate,
    feature_count_candidates=tuple(args.feature_count_candidates or ()),
    feature_count_selection_metric=args.feature_count_selection_metric,
    early_stopping_patience=args.early_stopping_patience,
    early_stopping_min_delta=args.early_stopping_min_delta,
    ae_learning_rate=args.ae_learning_rate,
    classifier_learning_rate=args.classifier_learning_rate,
    fine_tuning_learning_rate=args.fine_tuning_learning_rate,
    weight_decay=args.weight_decay,
    ae_scheduler_type=args.ae_scheduler_type,
    supervised_scheduler_type=args.supervised_scheduler_type,
    scheduler_patience=args.scheduler_patience,
    scheduler_factor=args.scheduler_factor,
    svm_c=args.svm_c,
    logistic_c=args.logistic_c,
    elastic_net_l1_ratio=args.elastic_net_l1_ratio,
    use_feature_scaling=args.use_feature_scaling,
    harmonization_method=args.harmonization_method,
    harmonization_covariates=tuple(args.harmonization_covariates or ()),
    enable_confound_regression=args.enable_confound_regression,
    confound_variables=tuple(args.confound_variables or ()),
    explanation_methods=(),
    save_artifacts=True,
    save_model_checkpoints=False,
  )

  full_data, full_labels, full_metadata = get_data_from_abide(
    pipeline,
    preprocessing_condition=args.preprocessing_condition,
    roi_atlas=args.roi_atlas,
    return_subject_metadata=True,
  )
  site_summary = summarize_sites(full_metadata, full_labels)
  print(site_summary.to_string())

  if args.run_all_large_sites:
    site_groups = [
      [site_name]
      for site_name, row in site_summary.iterrows()
      if int(row["total"]) >= int(args.min_site_subjects)
    ]
  elif args.combine_sites:
    site_groups = [normalize_site_filters(args.site)]
  else:
    site_groups = [[site_name] for site_name in normalize_site_filters(args.site)]

  if not site_groups:
    raise ValueError("No site groups were selected. Pass --site or use --run_all_large_sites.")

  records = []
  summary_payload = {
    "pipeline": pipeline,
    "preprocessing_condition": args.preprocessing_condition,
    "roi_atlas": args.roi_atlas,
    "site_summary": site_summary.reset_index().to_dict(orient="records"),
    "runs": [],
  }

  for site_group in site_groups:
    summary, record = run_site_evaluation(
      pipeline,
      site_group,
      base_config,
      verbose=args.verbose,
    )
    if summary is None:
      continue
    records.append(record)
    summary_payload["runs"].append(record)

  output_root = ensure_directory(Path(args.artifact_root))
  records_df = pd.DataFrame(records)
  if not records_df.empty:
    records_df = records_df.sort_values(["accuracy_mean", "f1_mean"], ascending=[False, False]).reset_index(drop=True)
    print("\nSingle-site comparison:")
    print(
      records_df[
        ["site", "n_subjects", "n_asd", "n_control", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"]
      ].to_string(index=False)
    )
    records_df.to_csv(output_root / f"{pipeline}_{args.roi_atlas}_single_site_results.csv", index=False)

  write_json_file(output_root / f"{pipeline}_{args.roi_atlas}_single_site_results.json", summary_payload)


if __name__ == "__main__":
  main()
