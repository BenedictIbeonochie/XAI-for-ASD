import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from app.main import (
    Autoencoder,
    ExplanationRanking,
    ReanalysisConfig,
    SoftmaxClassifier,
    StackedAutoencoder,
    build_hyperparameter_sweep_configs,
    build_dataloader,
    coerce_explanation_ranking,
    compute_fold_explanations,
    get_top_features_from_SVM_RFE,
    load_legacy_selected_features,
    train_supervised_stage,
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
    def make_fast_config(self, artifact_root, explanation_methods=(), **overrides):
        config = ReanalysisConfig(
            n_splits=5,
            num_selected_features=6,
            rfe_step=1,
            batch_size=8,
            ae1_epochs=1,
            ae2_epochs=1,
            classifier_epochs=1,
            fine_tuning_epochs=1,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            ae1_hidden_size=4,
            ae2_hidden_size=2,
            explanation_methods=tuple(explanation_methods),
            explanation_top_n=5,
            artifact_root=str(artifact_root),
            save_artifacts=True,
            save_model_checkpoints=False,
            random_seed=7,
            use_feature_scaling=True,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

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

    def test_stacked_autoencoder_restores_relu_between_pretrained_encoders(self):
        ae1 = Autoencoder(2, 2)
        ae2 = Autoencoder(2, 2)
        classifier = SoftmaxClassifier(2, 2)
        model = StackedAutoencoder(ae1, ae2, classifier)

        with torch.no_grad():
            ae1.encoder.weight.copy_(torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32))
            ae1.encoder.bias.zero_()
            ae2.encoder.weight.copy_(torch.eye(2, dtype=torch.float32))
            ae2.encoder.bias.zero_()
            classifier.linear.weight.zero_()
            classifier.linear.bias.copy_(torch.tensor([0.25, -0.5], dtype=torch.float32))

        outputs = model(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        np.testing.assert_allclose(outputs.detach().numpy(), np.array([[0.25, -0.5]], dtype=np.float32))

    def test_train_supervised_stage_stops_after_validation_plateau(self):
        train_features = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        train_labels = np.array([0, 1], dtype=int)
        train_loader = build_dataloader(train_features, train_labels, batch_size=2, shuffle=False)
        val_loader = build_dataloader(train_features, train_labels, batch_size=2, shuffle=False)

        model = nn.Linear(2, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()

        summary = train_supervised_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            max_epochs=6,
            patience=1,
            min_delta=0.0,
            stage_name='UnitTest',
            verbose=False,
        )

        self.assertEqual(summary['epochs_trained'], 2)
        self.assertEqual(summary['best_epoch'], 1)

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

    def test_explanation_ranking_is_truncated_to_shortest_payload(self):
        ranking = coerce_explanation_ranking(
            roi_pairs=np.array([[0, 1], [2, 3], [4, 5]], dtype=int),
            weights=np.array([0.9, 0.5], dtype=float),
            feature_indices=np.array([7, 8, 9, 10], dtype=int),
        )

        self.assertIsInstance(ranking, ExplanationRanking)
        np.testing.assert_array_equal(ranking.roi_pairs, np.array([[0, 1], [2, 3]], dtype=int))
        np.testing.assert_array_equal(ranking.weights, np.array([0.9, 0.5], dtype=float))
        np.testing.assert_array_equal(ranking.feature_indices, np.array([7, 8], dtype=int))

    def test_compute_fold_explanations_handles_lime_style_length_mismatch(self):
        config = self.make_fast_config('artifacts/test', explanation_methods=('LIME',))
        selected_roi_pairs = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)

        def fake_lime(top_n, model, test_dataloader, train_dataloader, rois):
            return rois[:2], np.array([0.7, 0.2], dtype=float), np.array([1, 0, 2], dtype=int)

        with patch('app.main.get_interpretability_method_map', return_value={'LIME': fake_lime}):
            explanations = compute_fold_explanations(
                model=None,
                train_dataloader=None,
                test_dataloader=None,
                selected_roi_pairs=selected_roi_pairs,
                config=config,
            )

        lime_ranking = explanations['LIME']
        self.assertIsNone(lime_ranking.skipped_reason)
        np.testing.assert_array_equal(lime_ranking.roi_pairs, selected_roi_pairs[:2])
        np.testing.assert_array_equal(lime_ranking.weights, np.array([0.7, 0.2], dtype=float))
        np.testing.assert_array_equal(lime_ranking.feature_indices, np.array([1, 0], dtype=int))

    def test_scaler_uses_training_subset_only(self):
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

        for fold_result in summary.fold_results:
            raw_train_features = feature_vectors[fold_result.train_indices][:, fold_result.selection.selected_feature_indices]
            scaler = StandardScaler().fit(raw_train_features)
            np.testing.assert_allclose(fold_result.scaler_mean, scaler.mean_)
            np.testing.assert_allclose(fold_result.scaler_scale, scaler.scale_)

    def test_feature_count_candidates_are_tuned_within_each_fold(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                feature_count_candidates=(2, 4, 6),
                feature_count_selection_metric='f1',
            )
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                verbose=False,
                config=config,
            )

        for fold_result in summary.fold_results:
            candidate_counts = [record['feature_count'] for record in fold_result.feature_count_tuning_records]
            self.assertEqual(candidate_counts, [2, 4, 6])
            self.assertIn(fold_result.selected_feature_count, {2, 4, 6})

            for tuning_record in fold_result.feature_count_tuning_records:
                np.testing.assert_array_equal(
                    np.sort(np.asarray(tuning_record['selection_training_sample_indices'], dtype=int)),
                    np.sort(fold_result.train_indices),
                )

            np.testing.assert_array_equal(
                np.sort(fold_result.selection.training_sample_indices),
                np.sort(fold_result.outer_train_indices),
            )

    def test_hyperparameter_sweep_builds_fixed_configs_with_separate_artifact_roots(self):
        base_config = self.make_fast_config(
            'artifacts/reanalysis',
            explanation_methods=('Integrated Gradients',),
            feature_count_candidates=(2, 4, 6),
            num_selected_features=6,
        )

        sweep_configs = build_hyperparameter_sweep_configs(
            base_config,
            pipeline='synthetic',
            sweep_root=Path('artifacts/reanalysis/hyperparameter_sweeps/synthetic/demo'),
            feature_counts=(4, 8),
            ae1_hidden_sizes=(6,),
            ae2_hidden_sizes=(3,),
            ae_learning_rates=(0.001,),
            classifier_learning_rates=(0.0005,),
            fine_tuning_learning_rates=(0.0001,),
            weight_decays=(1e-4,),
        )

        self.assertEqual(len(sweep_configs), 2)
        self.assertEqual(
            [entry['parameters']['num_selected_features'] for entry in sweep_configs],
            [4, 8],
        )

        for sweep_entry in sweep_configs:
            config = sweep_entry['config']
            self.assertEqual(config.feature_count_candidates, ())
            self.assertEqual(config.explanation_methods, ())
            self.assertFalse(config.save_model_checkpoints)
            self.assertIn(sweep_entry['name'], config.artifact_root)


if __name__ == '__main__':
    unittest.main()
