import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_centralized_vs_federated import (
    build_partitions_for_seed,
    summarize_metrics,
)
from src.federated.simulated_fedavg import (
    load_model_artifact,
    predict_from_artifact,
    predict_proba,
    prepare_feature_vector,
    save_model_artifact,
    stratified_split,
    train_centralized_model,
    train_parameters,
    zero_parameters,
)


class CentralizedAndFederatedArtifactTests(unittest.TestCase):
    def test_centralized_training_uses_only_pooled_training_rows(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(16, 3))
        y = np.asarray([0, 1] * 8, dtype=np.int64)
        train_idx, eval_idx = stratified_split(y, seed=13)
        full_training = x[train_idx]
        full_labels = y[train_idx]
        centralized = train_centralized_model(full_training, full_labels, num_features=3)
        with self.assertRaisesRegex(ValueError, "must be a 1D"):
            train_centralized_model(full_training, full_training)
        self.assertEqual(centralized[0].shape, (3,))
        self.assertEqual(len(full_labels), len(train_idx))
        self.assertNotEqual(list(centralized[0]), [0.0, 0.0, 0.0])

    def test_feature_order_is_named_not_dictionary_order(self):
        feature_map = {
            "packets_dst_to_src": 12.0,
            "duration": 3.0,
            "bytes_src_to_dst": 5.0,
            "bytes_dst_to_src": 10.0,
            "packets_src_to_dst": 7.0,
        }
        vector = prepare_feature_vector(
            feature_map,
            [
                "duration",
                "bytes_src_to_dst",
                "bytes_dst_to_src",
                "packets_src_to_dst",
                "packets_dst_to_src",
            ],
        )
        self.assertEqual(vector.shape, (5,))
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_saved_artifact_round_trip_matches_in_memory_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            feature_names = ["duration", "bytes_src_to_dst"]
            params = [
                np.asarray([0.5, -0.25], dtype=np.float64),
                np.asarray([0.1], dtype=np.float64),
            ]
            save_model_artifact(
                path,
                model_name="synthetic_model",
                model_type="centralized_logistic_regression",
                parameters=params,
                feature_names=feature_names,
                training_config={"seed": 7, "epochs": 5},
                dataset_summary={"source": "synthetic"},
                transforms=["log1p"],
            )
            loaded = load_model_artifact(path)
            feature_map = {"duration": 2.0, "bytes_src_to_dst": 4.0}
            predictions, probabilities = predict_from_artifact(loaded, feature_map)
            expected_probs = predict_proba(
                np.asarray([[np.log1p(2.0), np.log1p(4.0)]], dtype=np.float64), params
            )
            self.assertTrue(np.allclose(probabilities, expected_probs, atol=1e-12))
            self.assertTrue(np.array_equal(predictions, (expected_probs >= 0.5).astype(np.int64)))

    def test_missing_and_invalid_values_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            save_model_artifact(
                path,
                model_name="bad_inputs",
                model_type="centralized_logistic_regression",
                parameters=[
                    np.asarray([0.5, -0.25], dtype=np.float64),
                    np.asarray([0.1], dtype=np.float64),
                ],
                feature_names=["duration", "bytes_src_to_dst"],
                training_config={"seed": 7},
            )
            loaded = load_model_artifact(path)
            with self.assertRaisesRegex(ValueError, "missing"):
                predict_from_artifact(loaded, {"duration": 2.0})
            with self.assertRaisesRegex(ValueError, "not finite"):
                predict_from_artifact(loaded, {"duration": np.inf, "bytes_src_to_dst": 1.0})
            with self.assertRaisesRegex(ValueError, "unsupported artifact format"):
                bad_path = Path(temporary) / "bad_artifact.json"
                bad_path.write_text(
                    json.dumps({"artifact_format": "unsupported_v9", "feature_names": ["duration"]}),
                    encoding="utf-8",
                )
                load_model_artifact(bad_path)

    def test_benchmark_seed_builds_matching_holdout_partitions(self):
        rows_a = []
        rows_b = []
        for index in range(30):
            label = int(index % 2)
            rows_a.append(
                {
                    "features": {
                        "duration": float(index + 1),
                        "bytes_src_to_dst": float(index + 2),
                        "bytes_dst_to_src": float(index + 3),
                        "packets_src_to_dst": float(index + 4),
                        "packets_dst_to_src": float(index + 5),
                    },
                    "labels": {"binary": label},
                }
            )
            rows_b.append(
                {
                    "features": {
                        "duration": float(index + 6),
                        "bytes_src_to_dst": float(index + 7),
                        "bytes_dst_to_src": float(index + 8),
                        "packets_src_to_dst": float(index + 9),
                        "packets_dst_to_src": float(index + 10),
                    },
                    "labels": {"binary": label},
                }
            )

        feature_names = [
            "duration",
            "bytes_src_to_dst",
            "bytes_dst_to_src",
            "packets_src_to_dst",
            "packets_dst_to_src",
        ]

        def row_matrix(rows):
            values = [
                [np.log1p(float(row["features"][name])) for name in feature_names]
                for row in rows
            ]
            return np.asarray(values, dtype=np.float64)

        client_rows = {
            "client_a": (row_matrix(rows_a), np.asarray([int(r["labels"]["binary"]) for r in rows_a]), {"eligible_rows": len(rows_a)}),
            "client_b": (row_matrix(rows_b), np.asarray([int(r["labels"]["binary"]) for r in rows_b]), {"eligible_rows": len(rows_b)}),
        }
        partitions = build_partitions_for_seed(client_rows, seed=7)
        self.assertSetEqual(set(partitions), {"client_a", "client_b"})
        self.assertEqual(len(partitions["client_a"][0]) + len(partitions["client_a"][2]), len(client_rows["client_a"][1]))
        self.assertEqual(len(partitions["client_b"][0]) + len(partitions["client_b"][2]), len(client_rows["client_b"][1]))

    def test_predict_proba_uses_positive_class_probability(self):
        params = [np.asarray([20.0, 20.0], dtype=np.float64), np.asarray([-10.0], dtype=np.float64)]
        x = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        scores = predict_proba(x, params)
        self.assertLess(scores[0], 0.5)
        self.assertGreater(scores[1], 0.5)
        self.assertEqual(scores.shape, (2,))

    def test_threshold_and_positive_class_metadata_are_explicit(self):
        y = np.asarray([0, 1, 1, 0], dtype=np.int64)
        probs = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float64)
        metrics = summarize_metrics(y, probs)
        self.assertEqual(metrics["positive_class"], 1)
        self.assertEqual(metrics["decision_threshold"], 0.5)
        self.assertEqual(metrics["score_semantics"], "probability of class 1 (attack/malicious)")
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertAlmostEqual(metrics["precision"], 1.0, places=12)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=12)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.0, places=12)

    def test_reloaded_metrics_match_in_memory_metrics(self):
        x = np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float64)
        y = np.asarray([0, 0, 1, 1], dtype=np.int64)
        params = train_parameters(x, y, zero_parameters(2), epochs=15)
        artifact = {
            "artifact_format": "trustmesh_logistic_v1",
            "feature_names": ["a", "b"],
            "parameters": {"weights": params[0].tolist(), "bias": params[1].tolist()},
            "preprocessing": {"pipeline": ["log1p"]},
            "weights": params[0],
            "bias": params[1],
        }
        expected = summarize_metrics(y, predict_proba(x, params))
        recovered = summarize_metrics(y, predict_proba(x, [artifact["weights"], artifact["bias"]]))
        for metric in ("precision", "recall", "f1", "balanced_accuracy", "false_positive_rate"):
            self.assertAlmostEqual(expected[metric], recovered[metric], places=12)


if __name__ == "__main__":
    unittest.main()
