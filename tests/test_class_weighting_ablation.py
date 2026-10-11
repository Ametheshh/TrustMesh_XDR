import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import evaluate_class_weighting_ablation as ablation
from src.federated.simulated_fedavg import (
    LEARNING_RATE,
    L2,
    train_parameters,
    zero_parameters,
)


class ClassWeightingAblationTests(unittest.TestCase):
    def test_default_training_matches_existing_weighted_formula(self):
        x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
        y = np.asarray([0, 0, 1], dtype=np.int64)
        initial = zero_parameters(1)
        default = train_parameters(x, y, initial, epochs=1)
        weighted = train_parameters(x, y, initial, epochs=1, class_weighted=True)
        counts = np.bincount(y, minlength=2).astype(np.float64)
        sample_weights = (len(y) / (2.0 * counts))[y]
        probabilities = np.full(len(y), 0.5)
        residual = (probabilities - y) * sample_weights
        expected_weights = initial[0] - LEARNING_RATE * (
            (x.T @ residual) / sample_weights.sum() + L2 * initial[0]
        )
        expected_bias = initial[1] - LEARNING_RATE * residual.sum() / sample_weights.sum()
        self.assertTrue(np.array_equal(default[0], weighted[0]))
        self.assertTrue(np.allclose(default[0], expected_weights))
        self.assertTrue(np.allclose(default[1], expected_bias))

    def test_unweighted_training_uses_unit_sample_weights(self):
        x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
        y = np.asarray([0, 0, 1], dtype=np.int64)
        initial = zero_parameters(1)
        actual = train_parameters(x, y, initial, epochs=1, class_weighted=False)
        residual = 0.5 - y
        expected_weights = initial[0] - LEARNING_RATE * (
            (x.T @ residual) / len(y) + L2 * initial[0]
        )
        expected_bias = initial[1] - LEARNING_RATE * residual.sum() / len(y)
        self.assertTrue(np.allclose(actual[0], expected_weights))
        self.assertTrue(np.allclose(actual[1], expected_bias))

    def test_four_conditions_use_the_requested_weighting_policies(self):
        x = np.ones((2, len(ablation.FEATURES)), dtype=np.float64)
        y = np.asarray([0, 1], dtype=np.int64)
        partitions = {"client": (x, y, x, y)}
        central_calls = []
        fed_calls = []

        def central_stub(*args, **kwargs):
            central_calls.append(kwargs["class_weighted"])
            return [np.zeros(len(ablation.FEATURES)), np.zeros(1)]

        def fed_stub(partitions, *, class_weighted=True):
            fed_calls.append(class_weighted)
            return {}, [np.zeros(len(ablation.FEATURES)), np.zeros(1)]

        with patch.object(ablation, "train_centralized_model", side_effect=central_stub), patch.object(
            ablation, "fit_federated_global_model", side_effect=fed_stub
        ):
            models = ablation.train_conditions(partitions)
        self.assertEqual(set(models), set(ablation.CONDITIONS))
        self.assertEqual(central_calls, [True, False])
        self.assertEqual(fed_calls, [True, False])

    def test_all_conditions_use_the_same_held_out_rows_and_artifact_root(self):
        x_train = np.ones((4, len(ablation.FEATURES)), dtype=np.float64)
        y_train = np.asarray([0, 1, 0, 1], dtype=np.int64)
        x_eval = np.ones((3, len(ablation.FEATURES)), dtype=np.float64)
        y_eval = np.asarray([0, 1, 1], dtype=np.int64)
        partitions = {"client": (x_train, y_train, x_eval, y_eval)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                ablation, "predict_proba", wraps=ablation.predict_proba
            ) as predict_mock, patch.object(
                ablation, "summarize_metrics", wraps=ablation.summarize_metrics
            ) as metrics_mock:
                result = ablation.evaluate_seed(42, partitions, artifact_root=root)
            self.assertEqual(len(predict_mock.call_args_list), len(ablation.CONDITIONS))
            for call in predict_mock.call_args_list:
                self.assertTrue(np.array_equal(call.args[0], x_eval))
            self.assertEqual(len(metrics_mock.call_args_list), len(ablation.CONDITIONS))
            for call in metrics_mock.call_args_list:
                self.assertTrue(np.array_equal(call.args[0], y_eval))
            for metrics in result["client_metrics"]["client"].values():
                self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), len(y_eval))
            self.assertTrue(all(Path(path).is_relative_to(root) for path in result["artifact_paths"].values()))
            self.assertTrue(all("centralized_vs_federated" not in path for path in result["artifact_paths"].values()))

    def test_roc_auc_is_reported_only_when_both_classes_are_present(self):
        probabilities = np.asarray([0.1, 0.8, 0.6], dtype=np.float64)
        both = ablation.summarize_metrics(np.asarray([0, 1, 1]), probabilities)
        one_class = ablation.summarize_metrics(np.asarray([1, 1, 1]), probabilities)
        self.assertIsNotNone(both["roc_auc"])
        self.assertIsNotNone(both["pr_auc"])
        self.assertIsNone(one_class["roc_auc"])
        self.assertIsNone(one_class["pr_auc"])

    def test_experiment_output_root_is_separate_from_existing_benchmark(self):
        self.assertEqual(
            ablation.EXPERIMENT_ROOT,
            Path("data/models/class_weighting_ablation"),
        )
        self.assertNotEqual(
            ablation.EXPERIMENT_ROOT,
            ablation.ARTIFACT_ROOT / "centralized_vs_federated",
        )


if __name__ == "__main__":
    unittest.main()
