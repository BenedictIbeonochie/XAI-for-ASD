import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.svm import SVC

import subject_summary_fusion


def make_synthetic_roi_timeseries_dataset():
    rng = np.random.default_rng(123)
    site_names = ["NYU", "UCLA_1", "USM"]
    labels = []
    data = []
    metadata_rows = []

    for site_index, site_name in enumerate(site_names):
        for subject_offset, label in enumerate([0, 1, 0, 1]):
            timepoints = 24
            rois = 4
            shared_signal = rng.normal(0.0, 1.0, size=(timepoints, 1))
            roi_noise = rng.normal(0.0, 0.20, size=(timepoints, rois))
            site_shift = np.linspace(0.05 * site_index, 0.20 * site_index, rois, dtype=float)
            if label == 0:
                class_pattern = np.array([0.8, 0.6, 0.3, 0.1], dtype=float)
            else:
                class_pattern = np.array([0.1, 0.3, 0.6, 0.8], dtype=float)
            subject_roi_timeseries = shared_signal * class_pattern + roi_noise + site_shift
            file_id = f"{site_name.lower()}_{subject_offset:03d}"

            labels.append(label)
            data.append(subject_roi_timeseries)
            metadata_rows.append(
                {
                    "file_id": file_id,
                    "site_id": site_name,
                    "age_at_scan": 10.0 + site_index + subject_offset,
                    "sex": 1.0 if subject_offset % 2 == 0 else 2.0,
                }
            )

    return data, np.asarray(labels, dtype=int), pd.DataFrame(metadata_rows)


class SubjectSummaryFusionTests(unittest.TestCase):
    def make_fast_config(self, artifact_root, **overrides):
        config = subject_summary_fusion.SubjectSummaryFusionConfig(
            preprocessing_condition="filt_global",
            roi_atlas="rois_aal",
            min_site_subjects=4,
            validation_size=0.25,
            artifact_root=str(artifact_root),
            random_seed=123,
            top_rois_per_section=3,
            top_contribution_features=3,
            top_token_count=5,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_build_subject_summary_returns_expected_feature_count_and_finite_values(self):
        rng = np.random.default_rng(456)
        subject_roi_timeseries = rng.normal(0.0, 1.0, size=(20, 4))
        summary = subject_summary_fusion.build_subject_summary(
            subject_roi_timeseries,
            roi_display_labels=["ROI_0", "ROI_1", "ROI_2", "ROI_3"],
            top_rois_per_section=3,
        )

        self.assertEqual(summary["structured_values"].shape[0], (4 * 7) + 8)
        self.assertTrue(np.isfinite(summary["structured_values"]).all())
        self.assertIn("intrinsic_summary", summary["intrinsic_summary_text"])

    def test_intrinsic_summary_text_is_deterministic(self):
        rng = np.random.default_rng(789)
        subject_roi_timeseries = rng.normal(0.0, 1.0, size=(18, 4))
        labels = ["ROI_0", "ROI_1", "ROI_2", "ROI_3"]

        first = subject_summary_fusion.build_subject_summary(
            subject_roi_timeseries,
            roi_display_labels=labels,
            top_rois_per_section=3,
        )["intrinsic_summary_text"]
        second = subject_summary_fusion.build_subject_summary(
            subject_roi_timeseries,
            roi_display_labels=labels,
            top_rois_per_section=3,
        )["intrinsic_summary_text"]

        self.assertEqual(first, second)

    def test_resolve_roi_labels_falls_back_to_generic_labels_when_lookup_fails(self):
        with patch("subject_summary_fusion.load_roi_display_metadata", side_effect=RuntimeError("atlas unavailable")):
            labels = subject_summary_fusion.resolve_roi_labels("rois_aal", 4)

        self.assertEqual(labels, ["rois_aal_000", "rois_aal_001", "rois_aal_002", "rois_aal_003"])

    def test_build_classifier_supports_stronger_rbf_svm_backend(self):
        model = subject_summary_fusion.build_classifier("rbf_svm", random_seed=123)
        self.assertIsInstance(model, SVC)
        self.assertTrue(model.probability)

    def test_determine_loso_sites_skips_small_or_single_class_sites(self):
        site_summary = pd.DataFrame(
            [
                {"site_id": "NYU", "total": 8, "asd": 4, "control": 4},
                {"site_id": "UCLA_1", "total": 6, "asd": 3, "control": 3},
                {"site_id": "SMALL", "total": 3, "asd": 2, "control": 1},
                {"site_id": "ONECLASS", "total": 5, "asd": 5, "control": 0},
            ]
        ).set_index("site_id")

        eligible_sites, skipped_sites = subject_summary_fusion.determine_loso_sites(site_summary, min_site_subjects=4)

        self.assertEqual(eligible_sites, ["NYU", "UCLA_1"])
        self.assertEqual({record["site_id"] for record in skipped_sites}, {"SMALL", "ONECLASS"})

    def test_load_subject_dataset_for_pipeline_uses_empty_site_filters(self):
        data, labels, metadata = make_synthetic_roi_timeseries_dataset()
        config = self.make_fast_config("artifacts/test_loader")

        with patch("subject_summary_fusion.get_data_from_abide", return_value=(data, labels, metadata)) as mocked_loader:
            loaded_data, loaded_labels, loaded_metadata = subject_summary_fusion.load_subject_dataset_for_pipeline(
                "dparsf",
                config,
            )

        self.assertEqual(len(loaded_data), len(data))
        self.assertTrue(np.array_equal(loaded_labels, labels))
        self.assertEqual(len(loaded_metadata), len(metadata))
        self.assertEqual(mocked_loader.call_args.kwargs["site_filters"], ())

    def test_single_held_out_site_mode_runs_only_requested_site(self):
        data, labels, metadata = make_synthetic_roi_timeseries_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                held_out_site="NYU",
                classifier_type="rbf_svm",
            )
            with patch("subject_summary_fusion.get_data_from_abide", return_value=(data, labels, metadata)), patch(
                "subject_summary_fusion.resolve_roi_labels",
                return_value=["ROI_0", "ROI_1", "ROI_2", "ROI_3"],
            ):
                summary = subject_summary_fusion.run_pipeline_loso_experiment(
                    pipeline="dparsf",
                    config=config,
                    verbose=False,
                )

            self.assertEqual(summary["evaluation_mode"], "single_held_out_site")
            self.assertEqual(summary["held_out_site"], "NYU")
            self.assertEqual(summary["evaluation_sites"], ["NYU"])
            self.assertEqual(summary["eligible_sites"], ["NYU", "UCLA_1", "USM"])

            artifact_dir = Path(summary["artifact_dir"])
            subject_summaries = pd.read_csv(artifact_dir / "subject_summaries.csv")
            self.assertEqual(subject_summaries["held_out_site"].unique().tolist(), ["NYU"])

    def test_run_pipeline_loso_experiment_writes_artifacts_and_keeps_held_out_site_out_of_fit_artifacts(self):
        data, labels, metadata = make_synthetic_roi_timeseries_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(temp_dir)
            with patch("subject_summary_fusion.get_data_from_abide", return_value=(data, labels, metadata)), patch(
                "subject_summary_fusion.resolve_roi_labels",
                return_value=["ROI_0", "ROI_1", "ROI_2", "ROI_3"],
            ):
                summary = subject_summary_fusion.run_pipeline_loso_experiment(
                    pipeline="dparsf",
                    config=config,
                    verbose=False,
                )

            artifact_dir = Path(summary["artifact_dir"])
            self.assertTrue((artifact_dir / "summary.json").exists())
            self.assertTrue((artifact_dir / "loso_site_metrics.csv").exists())
            self.assertTrue((artifact_dir / "subject_summaries.csv").exists())
            self.assertTrue((artifact_dir / "class_prototypes_by_site.json").exists())
            self.assertTrue((artifact_dir / "top_tokens_by_class_by_site.csv").exists())

            subject_summaries = pd.read_csv(artifact_dir / "subject_summaries.csv")
            self.assertEqual(len(subject_summaries), len(labels))
            self.assertEqual(sorted(subject_summaries["held_out_site"].unique().tolist()), ["NYU", "UCLA_1", "USM"])
            self.assertTrue(subject_summaries["predicted_probability_asd"].between(0.0, 1.0).all())

            prototype_payload = json.loads((artifact_dir / "class_prototypes_by_site.json").read_text(encoding="utf-8"))
            self.assertEqual(len(prototype_payload["site_prototypes"]), 3)
            for prototype_record in prototype_payload["site_prototypes"]:
                held_out_site = prototype_record["held_out_site"]
                self.assertNotIn(held_out_site, prototype_record["fit_sites"]["scaler"])
                self.assertNotIn(held_out_site, prototype_record["fit_sites"]["vectorizer"])
                self.assertNotIn(held_out_site, prototype_record["fit_sites"]["prototypes"])


if __name__ == "__main__":
    unittest.main()
