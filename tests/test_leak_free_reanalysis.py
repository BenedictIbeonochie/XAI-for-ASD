import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.main import (
    ReanalysisConfig,
    get_top_features_from_SVM_RFE,
    load_legacy_selected_features,
    train_and_eval_model,
)


def make_synthetic_feature_matrix(num_samples=30, num_features=12):
    rng = np.random.default_rng(42)
    labels = np.array([0, 1] * (num_samples // 2), dtype=int)

    base_signal = labels[:, None] * 0.6
    noise = rng.normal(0.0, 0.15, size=(num_samples, num_features))
    feature_vectors = base_signal + noise
    feature_indices = np.array([[index, index + 1] for index in range(num_features)], dtype=int)

    return feature_vectors, labels, feature_indices


class LeakFreeReanalysisTests(unittest.TestCase):
    def make_fast_config(self, artifact_root, explanation_methods=()):
        return ReanalysisConfig(
            n_splits=5,
            num_selected_features=6,
            rfe_step=1,
            batch_size=8,
            ae1_epochs=1,
            ae2_epochs=1,
            classifier_epochs=1,
            fine_tuning_epochs=1,
            ae1_hidden_size=4,
            ae2_hidden_size=2,
            explanation_methods=tuple(explanation_methods),
            explanation_top_n=5,
            artifact_root=str(artifact_root),
            save_artifacts=True,
            save_model_checkpoints=False,
            random_seed=7,
        )

    def test_fold_selector_returns_metadata(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()
        train_indices = np.arange(20)

        selection = get_top_features_from_SVM_RFE(
            feature_vectors[train_indices],
            labels[train_indices],
            feature_indices,
            N=5,
            step=1,
            training_sample_indices=train_indices,
        )

        self.assertEqual(selection.selected_feature_indices.shape, (5,))
        self.assertEqual(selection.selected_roi_pairs.shape, (5, 2))
        np.testing.assert_array_equal(selection.training_sample_indices, train_indices)

    def test_corrected_training_uses_outer_train_only_for_feature_selection(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(temp_dir)
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                verbose=False,
                config=config,
            )

        self.assertEqual(len(summary.fold_results), 5)

        for fold_result in summary.fold_results:
            np.testing.assert_array_equal(
                np.sort(fold_result.selection.training_sample_indices),
                np.sort(fold_result.outer_train_indices),
            )
            self.assertEqual(
                len(np.intersect1d(fold_result.selection.training_sample_indices, fold_result.test_indices)),
                0,
            )
            self.assertEqual(fold_result.train_feature_shape[1], len(fold_result.selection.selected_feature_indices))
            self.assertEqual(fold_result.validation_feature_shape[1], len(fold_result.selection.selected_feature_indices))
            self.assertEqual(fold_result.test_feature_shape[1], len(fold_result.selection.selected_feature_indices))

    def test_correction_mode_blocks_legacy_feature_loading(self):
        with self.assertRaisesRegex(RuntimeError, 'Legacy globally selected feature artifacts are disabled'):
            load_legacy_selected_features(correction_mode=True)

    def test_fold_explanations_are_aggregated_across_all_folds(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(temp_dir, explanation_methods=('Integrated Gradients',))
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                verbose=False,
                config=config,
            )

            artifact_dir = Path(temp_dir) / 'synthetic'
            self.assertTrue((artifact_dir / 'summary.json').exists())
            self.assertTrue((artifact_dir / 'fold_metrics.csv').exists())

        ig_summary = summary.interpretation_summary['Integrated Gradients']
        self.assertEqual(ig_summary['folds_aggregated'], 5)
        self.assertFalse(ig_summary['connections'].empty)
        self.assertFalse(ig_summary['rois'].empty)


if __name__ == '__main__':
    unittest.main()
