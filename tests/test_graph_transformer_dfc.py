import unittest

import numpy as np
import torch

from graph_transformer_dfc import (
    ASDGraphTransformer,
    build_dfc_node_features,
    compute_class_weight_tensor,
    compute_dynamic_fc,
    evaluate_graph_transformer,
    sparsify_adjacency_batch,
    sparsify_adjacency_matrix,
    train_graph_transformer,
)


def make_synthetic_roi_timeseries_dataset(num_samples=12, num_timepoints=24, num_rois=6):
    rng = np.random.default_rng(321)
    labels = np.resize(np.array([0, 1], dtype=int), num_samples)
    data = []

    for label in labels:
        shared_signal = rng.normal(0.0, 1.0, size=(num_timepoints, 1))
        roi_noise = rng.normal(0.0, 0.35, size=(num_timepoints, num_rois))
        class_pattern = np.linspace(0.2, 0.8, num_rois, dtype=float)
        class_signal = shared_signal * (class_pattern if label == 0 else class_pattern[::-1])
        data.append(shared_signal * 0.4 + class_signal + roi_noise)

    return data, labels


class GraphTransformerDFCTests(unittest.TestCase):
    def test_compute_dynamic_fc_returns_static_and_variability_matrices(self):
        data, _ = make_synthetic_roi_timeseries_dataset(num_samples=1, num_timepoints=30, num_rois=5)

        static_fc, fc_std, windows = compute_dynamic_fc(data[0], window_size=10, stride=5)

        self.assertEqual(static_fc.shape, (5, 5))
        self.assertEqual(fc_std.shape, (5, 5))
        self.assertGreaterEqual(windows.shape[0], 1)
        self.assertTrue(np.isfinite(static_fc).all())
        self.assertTrue(np.isfinite(fc_std).all())

    def test_build_dfc_node_features_returns_subject_node_feature_tensor(self):
        data, _ = make_synthetic_roi_timeseries_dataset(num_samples=4, num_timepoints=30, num_rois=5)

        node_features, adjacency = build_dfc_node_features(data, window_size=10, stride=5)

        self.assertEqual(node_features.shape[0], 4)
        self.assertEqual(node_features.shape[1], 5)
        self.assertEqual(adjacency.shape, (4, 5, 5))
        self.assertTrue(np.isfinite(node_features).all())
        self.assertTrue(np.isfinite(adjacency).all())

    def test_graph_transformer_forward_pass_has_binary_output(self):
        model = ASDGraphTransformer(n_node_features=12, d_model=16, n_heads=4, n_layers=2, d_ff=32, dropout=0.1)
        node_features = np.random.default_rng(7).normal(size=(3, 5, 12)).astype(np.float32)
        adjacency = np.random.default_rng(8).normal(size=(3, 5, 5)).astype(np.float32)

        outputs = model(
            node_features=torch.tensor(np.asarray(node_features, dtype=np.float32)),
            adj=torch.tensor(np.asarray(adjacency, dtype=np.float32)),
        )

        self.assertEqual(tuple(outputs.shape), (3, 2))

    def test_compute_class_weight_tensor_upweights_minority_class(self):
        weights = compute_class_weight_tensor(np.array([0, 0, 0, 1], dtype=int))

        self.assertEqual(tuple(weights.shape), (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_sparsify_adjacency_matrix_keeps_only_top_k_neighbors(self):
        adjacency = np.array(
            [
                [0.0, 0.9, 0.1, 0.2],
                [0.9, 0.0, 0.8, 0.3],
                [0.1, 0.8, 0.0, 0.7],
                [0.2, 0.3, 0.7, 0.0],
            ],
            dtype=np.float32,
        )

        sparse = sparsify_adjacency_matrix(adjacency, top_k=1)

        self.assertEqual(sparse.shape, adjacency.shape)
        self.assertTrue(np.allclose(sparse, sparse.T))
        self.assertLess(np.count_nonzero(np.abs(sparse) > 0), np.count_nonzero(np.abs(adjacency) > 0))

    def test_sparsify_adjacency_batch_preserves_batch_shape(self):
        adjacency = np.stack([np.eye(4, dtype=np.float32), np.ones((4, 4), dtype=np.float32)], axis=0)
        sparse_batch = sparsify_adjacency_batch(adjacency, top_k=1)
        self.assertEqual(sparse_batch.shape, adjacency.shape)

    def test_train_and_evaluate_graph_transformer_smoke_test(self):
        data, labels = make_synthetic_roi_timeseries_dataset(num_samples=12, num_timepoints=30, num_rois=6)
        node_features, adjacency = build_dfc_node_features(data, window_size=10, stride=5)

        train_nodes = node_features[:8]
        val_nodes = node_features[8:10]
        test_nodes = node_features[10:]
        train_adj = adjacency[:8]
        val_adj = adjacency[8:10]
        test_adj = adjacency[10:]
        train_labels = labels[:8]
        val_labels = labels[8:10]
        test_labels = labels[10:]

        config = {
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "batch_size": 4,
            "epochs": 1,
            "patience": 1,
            "min_delta": 0.0,
            "use_class_weights": True,
            "adjacency_top_k": 2,
        }

        model, training_summary = train_graph_transformer(
            train_nodes,
            train_adj,
            train_labels,
            val_nodes,
            val_adj,
            val_labels,
            config,
            verbose=False,
        )
        metrics, _, predictions = evaluate_graph_transformer(model, test_nodes, test_adj, test_labels)

        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertEqual(len(predictions), len(test_labels))
        self.assertEqual(training_summary["epochs_trained"], 1)
        self.assertEqual(len(training_summary["class_weights"]), 2)


if __name__ == "__main__":
    unittest.main()
