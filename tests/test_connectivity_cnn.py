import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import connectivity_cnn


def make_synthetic_roi_timeseries_dataset(num_samples=20, num_timepoints=24, num_rois=4):
    rng = np.random.default_rng(123)
    labels = np.resize(np.array([0, 1], dtype=int), num_samples)
    data = []

    for label in labels:
        shared_signal = rng.normal(0.0, 1.0, size=(num_timepoints, 1))
        roi_noise = rng.normal(0.0, 0.35, size=(num_timepoints, num_rois))
        class_pattern = np.linspace(0.2, 0.8, num_rois, dtype=float)
        class_signal = shared_signal * (class_pattern if label == 0 else class_pattern[::-1])
        data.append(shared_signal * 0.4 + class_signal + roi_noise)

    return data, labels


def make_synthetic_subject_metadata(labels):
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "file_id": f"subject_{index:03d}",
                "site_id": "NYU" if index < (len(labels) // 2) else "UCLA_1",
                "age_at_scan": 10.0 + index,
                "sex": 1.0 if index % 2 == 0 else 2.0,
                "label": int(label),
            }
        )
    return pd.DataFrame(rows)


def make_site_metadata():
    return pd.DataFrame(
        [
            {"file_id": "s001", "site_id": "NYU", "age_at_scan": 10.0, "sex": 1.0},
            {"file_id": "s002", "site_id": "NYU", "age_at_scan": 11.0, "sex": 2.0},
            {"file_id": "s003", "site_id": "NYU", "age_at_scan": 12.0, "sex": 1.0},
            {"file_id": "s004", "site_id": "NYU", "age_at_scan": 13.0, "sex": 2.0},
            {"file_id": "s005", "site_id": "UCLA_1", "age_at_scan": 14.0, "sex": 1.0},
            {"file_id": "s006", "site_id": "UCLA_1", "age_at_scan": 15.0, "sex": 2.0},
            {"file_id": "s007", "site_id": "UCLA_1", "age_at_scan": 16.0, "sex": 1.0},
            {"file_id": "s008", "site_id": "UCLA_1", "age_at_scan": 17.0, "sex": 2.0},
            {"file_id": "s009", "site_id": "SMALL", "age_at_scan": 18.0, "sex": 1.0},
            {"file_id": "s010", "site_id": "SMALL", "age_at_scan": 19.0, "sex": 2.0},
        ]
    )


class ConnectivityCNNTests(unittest.TestCase):
    def make_fast_config(self, artifact_root, **overrides):
        config = connectivity_cnn.ConnectivityCNNConfig(
            roi_atlas="rois_aal",
            artifact_root=str(artifact_root),
            min_site_subjects=4,
            batch_size=4,
            epochs=1,
            patience=1,
            min_delta=0.0,
            max_splits=2,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_build_connectivity_matrices_returns_symmetric_zero_diagonal_values(self):
        data, _ = make_synthetic_roi_timeseries_dataset(num_samples=3, num_timepoints=16, num_rois=4)
        matrices = connectivity_cnn.build_connectivity_matrices(data)

        self.assertEqual(matrices.shape, (3, 4, 4))
        self.assertTrue(np.isfinite(matrices).all())
        self.assertTrue(np.allclose(matrices, np.transpose(matrices, (0, 2, 1))))
        self.assertTrue(np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0.0))

    def test_connectivity_cnn_accepts_native_2d_matrix_batches(self):
        model = connectivity_cnn.ConnectivityCNN(n_rois=116, dropout=0.5)
        batch = np.zeros((3, 116, 116), dtype=np.float32)
        outputs = model(connectivity_cnn.torch.tensor(batch))

        self.assertEqual(tuple(outputs.shape), (3, 2))

    def test_resolve_roi_atlas_alias_falls_back_from_fd_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            downloads_root = Path(temp_dir) / "Outputs"
            target_dir = downloads_root / "dparsf" / "filt_global" / "rois_aal"
            target_dir.mkdir(parents=True)

            resolved_roi_atlas = connectivity_cnn.resolve_roi_atlas_alias(
                "dparsf",
                preprocessing_condition="filt_global",
                roi_atlas="rois_aal_fd02",
                downloads_root=downloads_root,
            )

        self.assertEqual(resolved_roi_atlas, "rois_aal")

    def test_run_pipeline_experiments_dispatches_manual_site_run(self):
        metadata = make_site_metadata()
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        matrices = np.zeros((10, 4, 4), dtype=np.float32)
        config = self.make_fast_config("artifacts/test_manual_dispatch")

        captured_calls = []

        def fake_run_cohort(matrices, labels, subject_metadata, **kwargs):
            captured_calls.append(
                {
                    "cohort_label": kwargs["cohort_label"],
                    "site_filters": tuple(kwargs.get("site_filters", ())),
                    "n_subjects": int(len(labels)),
                }
            )
            return {
                "cohort_label": kwargs["cohort_label"],
                "pipeline": kwargs["pipeline"],
                "requested_roi_atlas": kwargs["requested_roi_atlas"],
                "resolved_roi_atlas": kwargs["resolved_roi_atlas"],
                "n_subjects": int(len(labels)),
                "n_asd": int((labels == 0).sum()),
                "n_control": int((labels == 1).sum()),
                "artifact_dir": "artifacts/test",
                "metrics": {
                    "accuracy": {"mean": 0.6, "std": 0.0},
                    "sensitivity": {"mean": 0.6, "std": 0.0},
                    "specificity": {"mean": 0.6, "std": 0.0},
                    "precision": {"mean": 0.6, "std": 0.0},
                    "f1": {"mean": 0.6, "std": 0.0},
                },
            }

        with patch("connectivity_cnn.load_connectivity_dataset", return_value=([], labels, metadata, "rois_aal")), patch(
            "connectivity_cnn.build_connectivity_matrices",
            return_value=matrices,
        ), patch("connectivity_cnn.run_cohort_evaluation", side_effect=fake_run_cohort):
            payload = connectivity_cnn.run_pipeline_experiments(
                "dparsf",
                config=config,
                requested_roi_atlas="rois_aal",
                site_filters=("NYU",),
                run_all_large_sites=False,
                run_pooled_all_sites=False,
                verbose=False,
            )

        self.assertEqual(len(captured_calls), 1)
        self.assertEqual(captured_calls[0]["cohort_label"], "NYU")
        self.assertEqual(captured_calls[0]["site_filters"], ("NYU",))
        self.assertIsNotNone(payload["requested_site_run"])

    def test_run_pipeline_experiments_dispatches_all_large_sites_and_pooled_run(self):
        metadata = make_site_metadata()
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        matrices = np.zeros((10, 4, 4), dtype=np.float32)
        config = self.make_fast_config("artifacts/test_site_dispatch", min_site_subjects=4)

        captured_calls = []

        def fake_run_cohort(matrices, labels, subject_metadata, **kwargs):
            captured_calls.append(kwargs["cohort_label"])
            return {
                "cohort_label": kwargs["cohort_label"],
                "pipeline": kwargs["pipeline"],
                "requested_roi_atlas": kwargs["requested_roi_atlas"],
                "resolved_roi_atlas": kwargs["resolved_roi_atlas"],
                "n_subjects": int(len(labels)),
                "n_asd": int((labels == 0).sum()),
                "n_control": int((labels == 1).sum()),
                "artifact_dir": "artifacts/test",
                "metrics": {
                    "accuracy": {"mean": 0.6, "std": 0.0},
                    "sensitivity": {"mean": 0.6, "std": 0.0},
                    "specificity": {"mean": 0.6, "std": 0.0},
                    "precision": {"mean": 0.6, "std": 0.0},
                    "f1": {"mean": 0.6, "std": 0.0},
                },
            }

        with patch("connectivity_cnn.load_connectivity_dataset", return_value=([], labels, metadata, "rois_aal")), patch(
            "connectivity_cnn.build_connectivity_matrices",
            return_value=matrices,
        ), patch("connectivity_cnn.run_cohort_evaluation", side_effect=fake_run_cohort):
            payload = connectivity_cnn.run_pipeline_experiments(
                "dparsf",
                config=config,
                requested_roi_atlas="rois_aal",
                run_all_large_sites=True,
                run_pooled_all_sites=True,
                verbose=False,
            )

        self.assertEqual(captured_calls, ["NYU", "UCLA_1", "pooled_all_sites"])
        self.assertEqual(len(payload["site_runs"]), 2)
        self.assertIsNotNone(payload["pooled_run"])

    def test_run_cohort_evaluation_writes_artifacts_and_preserves_native_matrix_shape(self):
        data, labels = make_synthetic_roi_timeseries_dataset(num_samples=12, num_timepoints=18, num_rois=4)
        matrices = connectivity_cnn.build_connectivity_matrices(data)
        metadata = make_synthetic_subject_metadata(labels)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(temp_dir)
            summary = connectivity_cnn.run_cohort_evaluation(
                matrices,
                labels,
                metadata,
                pipeline="synthetic",
                requested_roi_atlas="rois_aal",
                resolved_roi_atlas="rois_aal",
                cohort_label="NYU",
                config=config,
                verbose=False,
                site_filters=("NYU",),
            )

            artifact_dir = Path(summary["artifact_dir"])
            self.assertTrue((artifact_dir / "summary.json").exists())
            self.assertTrue((artifact_dir / "fold_metrics.csv").exists())

        self.assertEqual(summary["native_matrix_shape"], [4, 4])
        self.assertIn("accuracy", summary["metrics"])


if __name__ == "__main__":
    unittest.main()
