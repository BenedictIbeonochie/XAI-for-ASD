import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from app.main import (
    Autoencoder,
    ExplanationRanking,
    ReanalysisConfig,
    PipelineRunSummary,
    SoftmaxClassifier,
    StackedAutoencoder,
    build_hyperparameter_sweep_configs,
    build_dataloader,
    build_confound_design_matrix,
    collect_reanalysis_logs,
    coerce_explanation_ranking,
    compute_fold_explanations,
    get_feature_vecs,
    get_top_features_from_selector,
    get_top_features_from_SVM_RFE,
    harmonize_feature_sets,
    load_legacy_selected_features,
    parse_reanalysis_log,
    prepare_transfer_learning_images,
    resolve_abide_download_dir,
    regress_out_confounds,
    run_repeated_evaluation,
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


def make_synthetic_subject_metadata(num_samples=30):
    rows = []
    for index in range(num_samples):
        rows.append({
            "file_id": f"subject_{index:03d}",
            "site_id": "SITE_A" if index % 2 == 0 else "SITE_B",
            "age_at_scan": 10.0 + index,
            "sex": 1.0 if index % 3 == 0 else 2.0,
        })
    return pd.DataFrame(rows)


def make_synthetic_roi_timeseries_dataset(num_samples=30, num_timepoints=24, num_rois=4):
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

    def test_none_selector_returns_all_features_without_filtering(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix(num_features=7)

        selection = get_top_features_from_selector(
            feature_vectors,
            labels,
            feature_indices,
            N=3,
            step=1,
            selector_type='none',
            training_sample_indices=np.arange(len(feature_vectors)),
        )

        np.testing.assert_array_equal(selection.selected_feature_indices, np.arange(7))
        np.testing.assert_array_equal(selection.selected_roi_pairs, feature_indices)

    def test_rfecv_selector_returns_nonempty_subset(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix(num_samples=30, num_features=10)

        selection = get_top_features_from_selector(
            feature_vectors,
            labels,
            feature_indices,
            N=6,
            step=1,
            selector_type='rfecv',
            training_sample_indices=np.arange(len(feature_vectors)),
            rfecv_inner_splits=3,
        )

        self.assertGreaterEqual(len(selection.selected_feature_indices), 1)
        self.assertLessEqual(len(selection.selected_feature_indices), feature_vectors.shape[1])

    def test_mrmr_selector_returns_requested_feature_count(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix(num_samples=30, num_features=12)

        selection = get_top_features_from_selector(
            feature_vectors,
            labels,
            feature_indices,
            N=5,
            step=1,
            selector_type='mrmr',
            training_sample_indices=np.arange(len(feature_vectors)),
        )

        self.assertEqual(len(selection.selected_feature_indices), 5)
        self.assertEqual(selection.selected_roi_pairs.shape, (5, 2))

    def test_graph_summary_representation_returns_roi_level_features(self):
        data = [
            np.array([
                [1.0, 0.5, 0.1, -0.2],
                [0.8, 0.4, 0.2, -0.1],
                [0.7, 0.3, 0.1, 0.0],
                [0.9, 0.6, 0.0, -0.3],
            ], dtype=float),
            np.array([
                [0.2, -0.1, 0.7, 0.5],
                [0.1, -0.2, 0.8, 0.6],
                [0.0, -0.3, 0.6, 0.4],
                [0.3, -0.1, 0.9, 0.7],
            ], dtype=float),
        ]

        feature_vectors, feature_indices = get_feature_vecs(data, feature_representation="graph_summary")

        self.assertEqual(feature_vectors.shape, (2, 24))
        self.assertEqual(feature_indices.shape, (2, 24, 2))
        self.assertTrue(np.isfinite(feature_vectors).all())
        np.testing.assert_array_equal(feature_indices[0, :, 0], feature_indices[0, :, 1])

    def test_tangent_representation_returns_fold_fitted_edge_features(self):
        data, _ = make_synthetic_roi_timeseries_dataset(num_samples=6, num_timepoints=20, num_rois=4)

        feature_vectors, feature_indices = get_feature_vecs(
            data[4:],
            feature_representation="tangent_vector",
            fit_data=data[:4],
        )

        self.assertEqual(feature_vectors.shape, (2, 6))
        self.assertEqual(feature_indices.shape, (6, 2))
        self.assertTrue(np.isfinite(feature_vectors).all())

    def test_partial_correlation_representation_returns_finite_edge_features(self):
        data, _ = make_synthetic_roi_timeseries_dataset(num_samples=5, num_timepoints=20, num_rois=4)

        feature_vectors, feature_indices = get_feature_vecs(
            data,
            feature_representation="partial_correlation_vector",
        )

        self.assertEqual(feature_vectors.shape, (5, 6))
        self.assertEqual(feature_indices.shape, (6, 2))
        self.assertTrue(np.isfinite(feature_vectors).all())

    def test_graph_summary_representation_skips_edge_level_explanations(self):
        config = self.make_fast_config(
            'artifacts/test',
            explanation_methods=('Integrated Gradients',),
            feature_representation='graph_summary',
        )

        explanations = compute_fold_explanations(
            model=None,
            train_dataloader=None,
            test_dataloader=None,
            selected_roi_pairs=np.array([[0, 0], [1, 1]], dtype=int),
            config=config,
        )

        self.assertIn('Integrated Gradients', explanations)
        self.assertIn('edge-based connectivity features', explanations['Integrated Gradients'].skipped_reason)

    def test_prepare_transfer_learning_images_returns_rgb_square_batch(self):
        feature_vectors, _, feature_indices = make_synthetic_feature_matrix(num_samples=4, num_features=6)
        image_tensor = prepare_transfer_learning_images(
            feature_vectors,
            feature_indices,
            roi_count=7,
            image_size=32,
        )

        self.assertEqual(tuple(image_tensor.shape), (4, 3, 32, 32))
        self.assertTrue(torch.isfinite(image_tensor).all())

    def test_prepare_transfer_learning_images_supports_single_channel_connectivity_cnn(self):
        feature_vectors, _, feature_indices = make_synthetic_feature_matrix(num_samples=4, num_features=6)
        image_tensor = prepare_transfer_learning_images(
            feature_vectors,
            feature_indices,
            roi_count=7,
            image_size=32,
            num_channels=1,
            per_subject_minmax=False,
            imagenet_normalize=False,
        )

        self.assertEqual(tuple(image_tensor.shape), (4, 1, 32, 32))
        self.assertTrue(torch.isfinite(image_tensor).all())

    def test_resolve_abide_download_dir_supports_custom_condition_and_atlas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            downloads_root = Path(temp_dir) / "Outputs"
            target_dir = downloads_root / "dparsf" / "filt_noglobal" / "rois_cc200"
            target_dir.mkdir(parents=True)

            resolved_dir = resolve_abide_download_dir(
                "dparsf",
                preprocessing_condition="filt_noglobal",
                roi_atlas="rois_cc200",
                downloads_root=downloads_root,
            )

        self.assertEqual(resolved_dir, target_dir)

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

    def test_train_and_eval_model_supports_tangent_representation_with_raw_data(self):
        data, labels = make_synthetic_roi_timeseries_dataset(num_samples=20, num_timepoints=18, num_rois=4)
        feature_indices = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                model_type='linear_svm',
                selector_type='none',
                feature_representation='tangent_vector',
            )
            summary = train_and_eval_model(
                None,
                labels,
                pipeline='synthetic_tangent',
                feature_indices=feature_indices,
                subject_metadata=make_synthetic_subject_metadata(num_samples=20),
                verbose=False,
                train_model=True,
                save_model=False,
                config=config,
                raw_data=data,
            )

        self.assertEqual(len(summary.fold_results), 5)

    def test_train_and_eval_model_supports_transfer_learning_backbone(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix(num_samples=18, num_features=6)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                n_splits=3,
                model_type='resnet18_transfer',
                selector_type='none',
                feature_transform='none',
                classifier_epochs=1,
                transfer_pretrained=False,
                transfer_freeze_backbone=True,
                transfer_image_size=32,
            )
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                verbose=False,
                config=config,
            )

        self.assertEqual(len(summary.fold_results), 3)
        self.assertEqual(summary.fold_results[0].training_summary['model_type'], 'resnet18_transfer')
        self.assertFalse(summary.fold_results[0].training_summary['transfer_learning']['pretrained_loaded'])

    def test_train_and_eval_model_supports_connectivity_cnn(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix(num_samples=18, num_features=6)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                n_splits=3,
                model_type='connectivity_cnn',
                selector_type='none',
                feature_transform='none',
                classifier_epochs=1,
                transfer_image_size=32,
            )
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                verbose=False,
                config=config,
            )

        self.assertEqual(len(summary.fold_results), 3)
        self.assertEqual(summary.fold_results[0].training_summary['model_type'], 'connectivity_cnn')
        self.assertEqual(summary.fold_results[0].training_summary['transfer_learning']['input_channels'], 1)

    def test_train_and_eval_model_applies_pca_inside_each_fold(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix(num_samples=30, num_features=8)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                model_type='linear_svm',
                selector_type='none',
                feature_transform='pca',
                pca_components=3,
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
            self.assertEqual(fold_result.train_feature_shape[1], 3)
            self.assertEqual(fold_result.validation_feature_shape[1], 3)
            self.assertEqual(fold_result.test_feature_shape[1], 3)
            self.assertEqual(fold_result.feature_transform_summary['type'], 'pca')
            self.assertEqual(fold_result.feature_transform_summary['n_components'], 3)

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

    def test_stacked_autoencoder_dropout_only_applies_during_training(self):
        ae1 = Autoencoder(2, 2)
        ae2 = Autoencoder(2, 2)
        classifier = SoftmaxClassifier(2, 2)
        model = StackedAutoencoder(ae1, ae2, classifier, dropout_rate=1.0)

        with torch.no_grad():
            ae1.encoder.weight.copy_(torch.eye(2, dtype=torch.float32))
            ae1.encoder.bias.zero_()
            ae2.encoder.weight.copy_(torch.eye(2, dtype=torch.float32))
            ae2.encoder.bias.zero_()
            classifier.linear.weight.copy_(torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32))
            classifier.linear.bias.copy_(torch.tensor([0.5, -0.25], dtype=torch.float32))

        training_input = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        model.train()
        train_outputs = model(training_input)
        np.testing.assert_allclose(train_outputs.detach().numpy(), np.array([[0.5, -0.25]], dtype=np.float32))

        model.eval()
        eval_outputs = model(training_input)
        np.testing.assert_allclose(eval_outputs.detach().numpy(), np.array([[3.5, -3.25]], dtype=np.float32))

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

    def test_confound_regression_uses_training_subset_only(self):
        train_features = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ])
        test_features = np.array([
            [5.0, 10.0],
            [6.0, 12.0],
        ])
        train_metadata = pd.DataFrame([
            {"file_id": "a", "site_id": "SITE_A", "age_at_scan": 10.0, "sex": 1.0},
            {"file_id": "b", "site_id": "SITE_A", "age_at_scan": 11.0, "sex": 2.0},
            {"file_id": "c", "site_id": "SITE_B", "age_at_scan": 12.0, "sex": 1.0},
            {"file_id": "d", "site_id": "SITE_B", "age_at_scan": 13.0, "sex": 2.0},
        ])
        test_metadata = pd.DataFrame([
            {"file_id": "e", "site_id": "SITE_A", "age_at_scan": 14.0, "sex": 1.0},
            {"file_id": "f", "site_id": "SITE_B", "age_at_scan": 15.0, "sex": 2.0},
        ])
        config = self.make_fast_config(
            "artifacts/test",
            enable_confound_regression=True,
            confound_variables=("site", "age", "sex"),
        )

        residual_train, [residual_test], summary = regress_out_confounds(
            train_features,
            train_metadata,
            config,
            (test_features, test_metadata),
        )

        train_design, _, _ = build_confound_design_matrix(train_metadata, ("site", "age", "sex"))
        expected_beta = np.linalg.pinv(train_design) @ train_features
        expected_train = train_features - train_design @ expected_beta

        test_design, _, _ = build_confound_design_matrix(
            test_metadata,
            ("site", "age", "sex"),
            site_categories=tuple(summary["site_categories"]),
            age_mean=summary["age_mean"],
            age_scale=summary["age_scale"],
            sex_mean=summary["sex_mean"],
            sex_scale=summary["sex_scale"],
        )
        expected_test = test_features - test_design @ expected_beta

        np.testing.assert_allclose(residual_train, expected_train)
        np.testing.assert_allclose(residual_test, expected_test)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["variables"], ["site", "age", "sex"])

    def test_linear_svm_baseline_runs_in_leak_free_pipeline(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                explanation_methods=(),
                model_type="linear_svm",
            )
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                verbose=False,
                config=config,
            )

        self.assertEqual(len(summary.fold_results), 5)
        self.assertIn("accuracy", summary.metrics_summary)
        self.assertEqual(summary.fold_results[0].training_summary["model_type"], "linear_svm")

    def test_combat_harmonization_uses_training_subset_only(self):
        train_features = np.array([
            [10.0, 1.0],
            [11.0, 1.5],
            [20.0, 1.2],
            [21.0, 1.7],
        ])
        test_features = np.array([
            [12.0, 1.1],
            [22.0, 1.6],
        ])
        train_metadata = pd.DataFrame([
            {"file_id": "a", "site_id": "SITE_A", "age_at_scan": 10.0, "sex": 1.0},
            {"file_id": "b", "site_id": "SITE_A", "age_at_scan": 11.0, "sex": 2.0},
            {"file_id": "c", "site_id": "SITE_B", "age_at_scan": 12.0, "sex": 1.0},
            {"file_id": "d", "site_id": "SITE_B", "age_at_scan": 13.0, "sex": 2.0},
        ])
        test_metadata = pd.DataFrame([
            {"file_id": "e", "site_id": "SITE_A", "age_at_scan": 14.0, "sex": 1.0},
            {"file_id": "f", "site_id": "SITE_B", "age_at_scan": 15.0, "sex": 2.0},
        ])
        config = self.make_fast_config(
            "artifacts/test",
            harmonization_method="combat",
            harmonization_covariates=("age", "sex"),
        )

        harmonized_train, [harmonized_test], summary = harmonize_feature_sets(
            train_features,
            train_metadata,
            config,
            (test_features, test_metadata),
        )

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["method"], "combat")
        self.assertEqual(summary["covariates"], ["age", "sex"])
        self.assertEqual(harmonized_train.shape, train_features.shape)
        self.assertEqual(harmonized_test.shape, test_features.shape)
        self.assertLess(
            abs(harmonized_train[:2, 0].mean() - harmonized_train[2:, 0].mean()),
            abs(train_features[:2, 0].mean() - train_features[2:, 0].mean()),
        )

    def test_linear_svm_baseline_runs_with_combat_harmonization(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()
        subject_metadata = make_synthetic_subject_metadata(num_samples=len(labels))

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                explanation_methods=(),
                model_type="linear_svm",
                harmonization_method="combat",
                harmonization_covariates=("age", "sex"),
            )
            summary = train_and_eval_model(
                feature_vectors,
                labels,
                pipeline='synthetic',
                feature_indices=feature_indices,
                subject_metadata=subject_metadata,
                verbose=False,
                config=config,
            )

        self.assertEqual(len(summary.fold_results), 5)
        self.assertTrue(summary.fold_results[0].confound_summary["harmonization"]["enabled"])

    def test_anova_selector_runs_in_leak_free_pipeline(self):
        feature_vectors, labels, feature_indices = make_synthetic_feature_matrix()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.make_fast_config(
                temp_dir,
                explanation_methods=(),
                selector_type="anova_f",
            )
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
            self.assertEqual(fold_result.selected_feature_count, config.num_selected_features)
            self.assertEqual(len(fold_result.selection.selected_feature_indices), config.num_selected_features)

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

    def test_parse_reanalysis_log_extracts_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "dparsf_ssae.log"
            log_path.write_text(
                "\n".join([
                    "preprocessing_condition:  filt_noglobal",
                    "roi_atlas:  rois_cc200",
                    "feature_representation:  edge_vector",
                    "feature_transform:  pca",
                    "pca_components:  1000",
                    "model_type:  ssae",
                    "selector_type:  rfe",
                    "rfecv_inner_splits:  3",
                    "ssae_dropout_rate:  0.3",
                    "harmonization_method:  combat",
                    "enable_confound_regression:  False",
                    "Accuracy: 64.70%",
                    "Specificity: 0.71",
                    "Precision: 0.63",
                    "F1_Score: 0.60",
                    "dparsf: accuracy=0.6470 ± 0.0454, f1=0.6008 ± 0.0576",
                ]),
                encoding="utf-8",
            )

            record = parse_reanalysis_log(log_path)

        self.assertEqual(record["pipeline"], "dparsf")
        self.assertEqual(record["preprocessing_condition"], "filt_noglobal")
        self.assertEqual(record["roi_atlas"], "rois_cc200")
        self.assertEqual(record["model_type"], "ssae")
        self.assertEqual(record["feature_transform"], "pca")
        self.assertEqual(record["pca_components"], "1000")
        self.assertEqual(record["ssae_dropout_rate"], "0.3")
        self.assertEqual(record["harmonization_method"], "combat")
        self.assertAlmostEqual(record["accuracy_mean"], 0.6470)
        self.assertAlmostEqual(record["f1_mean"], 0.6008)

    def test_collect_reanalysis_logs_sorts_best_accuracy_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            (logs_dir / "run_a.log").write_text(
                "model_type:  linear_svm\ndparsf: accuracy=0.6100 ± 0.0200, f1=0.5800 ± 0.0300\n",
                encoding="utf-8",
            )
            (logs_dir / "run_b.log").write_text(
                "model_type:  ssae\ndparsf: accuracy=0.6400 ± 0.0100, f1=0.6000 ± 0.0200\n",
                encoding="utf-8",
            )

            results_df = collect_reanalysis_logs(logs_dir)

        self.assertEqual(list(results_df["log_name"]), ["run_b.log", "run_a.log"])

    def test_run_repeated_evaluation_aggregates_runs(self):
        base_config = self.make_fast_config("artifacts/reanalysis", explanation_methods=())

        def make_summary(seed):
            config = ReanalysisConfig(**vars(base_config))
            config.random_seed = seed
            config.artifact_root = f"artifacts/repeated/{seed}"
            return PipelineRunSummary(
                pipeline="synthetic",
                config=config,
                fold_results=[],
                metrics_summary={
                    "accuracy": {"mean": 0.60 + (seed % 2) * 0.01, "std": 0.01, "values": []},
                    "sensitivity": {"mean": 0.60, "std": 0.01, "values": []},
                    "specificity": {"mean": 0.61, "std": 0.01, "values": []},
                    "precision": {"mean": 0.59, "std": 0.01, "values": []},
                    "f1": {"mean": 0.58 + (seed % 2) * 0.01, "std": 0.01, "values": []},
                },
                interpretation_summary={},
                artifact_dir=config.artifact_root,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.main.run_pipeline_reanalysis", side_effect=lambda pipeline, verbose, config: make_summary(config.random_seed)):
                repeat_df, aggregate_summary, repeat_records = run_repeated_evaluation(
                    "synthetic",
                    base_config,
                    Path(temp_dir),
                    num_repeats=3,
                    verbose=False,
                )

        self.assertEqual(len(repeat_df), 3)
        self.assertEqual(len(repeat_records), 3)
        self.assertEqual(aggregate_summary["num_repeats"], 3)
        self.assertIn("accuracy_across_repeats_mean", aggregate_summary)


if __name__ == '__main__':
    unittest.main()
