import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.federated.simulated_fedavg import (
    classification_metrics,
    _history_summary,
    _result_line,
    initial_parameters,
    load_client_data,
    mask_model_update,
    pairwise_mask,
    privatize_model_update,
    predict,
    run_experiment,
    run_networked_experiment,
    stratified_split,
    train_parameters,
    validate_dp_config,
)


class SimulatedFedAvgTests(unittest.TestCase):
    def test_approved_inputs_provide_only_one_numeric_feature_and_binary_target(self):
        path = Path("data/canonical/smoke_test/ctu_unsw_fl_poc/ctu_sme_total_packets.jsonl")
        x, y = load_client_data(path)
        self.assertEqual(x.shape, (9760, 1))
        self.assertEqual(set(y.tolist()), {0, 1})
        self.assertTrue(np.isfinite(x).all())

    def test_split_is_deterministic_and_stratified(self):
        y = np.asarray([0] * 20 + [1] * 80)
        train_a, test_a = stratified_split(y)
        train_b, test_b = stratified_split(y)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(test_a, test_b))
        self.assertEqual(set(y[test_a].tolist()), {0, 1})
        self.assertEqual(len(test_a), 20)

    def test_shared_parameter_order_and_metrics(self):
        parameters = initial_parameters()
        self.assertEqual([array.shape for array in parameters], [(1,), (1,)])
        x = np.asarray([[0.0], [0.5], [1.0], [2.0]])
        y = np.asarray([0, 0, 1, 1])
        trained = train_parameters(x, y, parameters, epochs=5)
        self.assertEqual([array.shape for array in trained], [(1,), (1,)])
        metrics = classification_metrics(y, predict(x, trained))
        self.assertEqual(metrics["confusion_matrix_labels"], [0, 1])
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertIn("balanced_accuracy", metrics)

    def test_two_client_experiment_uses_flower_fedavg(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for client_id, shift in (("ctu_sme", 0), ("unsw_nb15_named", 1)):
                path = Path(temporary) / f"{client_id}.jsonl"
                paths[client_id] = path
                with path.open("w", encoding="utf-8") as stream:
                    for index in range(20):
                        label = int(index >= 10)
                        value = index + shift if label else index + 1
                        stream.write(json.dumps({"features": {"total_packets": value}, "labels": {"binary": label}}) + "\n")
            report = run_experiment(paths)
        self.assertEqual(report["rounds"], 1)
        self.assertEqual(set(report["local_training"]), {"ctu_sme", "unsw_nb15_named"})
        self.assertEqual(set(report["global_model_evaluation"]), {"ctu_sme", "unsw_nb15_named"})
        self.assertEqual(report["parameter_order"], ["weight_total_packets", "bias"])

    def test_network_server_history_reads_flower_round_lists(self):
        history = SimpleNamespace(
            losses_distributed=[(1, 0.0)],
            metrics_distributed={"ctu_sme_f1": [(1, 0.75)]},
        )
        strategy = SimpleNamespace(final_parameters=[np.asarray([0.2]), np.asarray([0.1])])
        summary = _history_summary(history, strategy)
        self.assertEqual(summary["completed_rounds"], 1)
        self.assertEqual(summary["distributed_evaluation"], [{"round": 1, "loss": 0.0}])
        self.assertEqual(summary["client_evaluation_metrics"]["1"]["ctu_sme_f1"], 0.75)

    def test_result_marker_is_parsed_from_worker_output(self):
        result = _result_line("Flower log\nTRUSTMESH_RESULT={\"role\":\"server\",\"completed_rounds\":1}\n", "server")
        self.assertEqual(result["completed_rounds"], 1)

    def test_networked_mode_requires_the_two_approved_clients(self):
        with self.assertRaisesRegex(ValueError, "exactly CTU-SME"):
            run_networked_experiment({"other": Path("unused.jsonl")})

    def test_dp_clipping_caps_joint_parameter_update_norm(self):
        base = [np.zeros(1), np.zeros(1)]
        trained = [np.asarray([3.0]), np.asarray([4.0])]
        sent, diagnostics = privatize_model_update(
            base, trained, client_id="ctu_sme", clip_norm=2.0,
            noise_multiplier=0.0, random_seed=7,
        )
        self.assertTrue(diagnostics["clipped"])
        self.assertAlmostEqual(diagnostics["update_norm_before_clipping"], 5.0)
        self.assertAlmostEqual(diagnostics["update_norm_after_clipping"], 2.0)
        self.assertAlmostEqual(float(np.linalg.norm(np.concatenate(sent))), 2.0)

    def test_dp_noise_changes_the_clipped_update(self):
        base = [np.zeros(1), np.zeros(1)]
        trained = [np.asarray([0.2]), np.asarray([0.1])]
        sent, diagnostics = privatize_model_update(
            base, trained, client_id="ctu_sme", clip_norm=1.0,
            noise_multiplier=0.5, random_seed=8,
        )
        self.assertEqual(diagnostics["noise_std"], 0.5)
        self.assertGreater(diagnostics["update_norm_after_clipping_and_noise"], 0.0)
        self.assertFalse(np.array_equal(np.concatenate(sent), np.concatenate(trained)))

    def test_dp_noise_is_reproducible_for_a_fixed_client_seed(self):
        base = [np.zeros(1), np.zeros(1)]
        trained = [np.asarray([0.2]), np.asarray([0.1])]
        first, first_report = privatize_model_update(
            base, trained, client_id="unsw_nb15_named", clip_norm=1.0,
            noise_multiplier=1.0, random_seed=2026,
        )
        second, second_report = privatize_model_update(
            base, trained, client_id="unsw_nb15_named", clip_norm=1.0,
            noise_multiplier=1.0, random_seed=2026,
        )
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second, strict=True)))
        self.assertEqual(first_report, second_report)

    def test_dp_configuration_is_explicit_and_validated(self):
        self.assertEqual(
            validate_dp_config(1.0, 1.0, 2026),
            {"clip_norm": 1.0, "noise_multiplier": 1.0, "random_seed": 2026},
        )
        with self.assertRaisesRegex(ValueError, "clipping norm"):
            validate_dp_config(0.0, 1.0, 2026)
        with self.assertRaisesRegex(ValueError, "noise multiplier"):
            validate_dp_config(1.0, -0.1, 2026)
        with self.assertRaisesRegex(ValueError, "random seed"):
            validate_dp_config(1.0, 1.0, -1)

    def test_secure_pairwise_mask_generation_is_repeatable(self):
        parameters = [np.zeros(1), np.zeros(1)]
        first = pairwise_mask(parameters, 314)
        second = pairwise_mask(parameters, 314)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second, strict=True)))

    def test_secure_client_messages_are_masked_and_pair_masks_cancel(self):
        base = [np.zeros(1), np.zeros(1)]
        update_a = [np.asarray([0.3]), np.asarray([-0.2])]
        update_b = [np.asarray([-0.1]), np.asarray([0.4])]
        count_a, count_b = 8, 12
        sent_a, diag_a = mask_model_update(
            base, update_a, client_id="ctu_sme", num_examples=count_a, random_seed=9,
        )
        sent_b, diag_b = mask_model_update(
            base, update_b, client_id="unsw_nb15_named", num_examples=count_b, random_seed=9,
        )
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(sent_a, update_a, strict=True)))
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(sent_b, update_b, strict=True)))
        self.assertFalse(diag_a["unmasked_update_sent"])
        self.assertFalse(diag_b["unmasked_update_sent"])
        weighted_secure = [
            (count_a * a + count_b * b) / (count_a + count_b)
            for a, b in zip(sent_a, sent_b, strict=True)
        ]
        weighted_ordinary = [
            (count_a * a + count_b * b) / (count_a + count_b)
            for a, b in zip(update_a, update_b, strict=True)
        ]
        self.assertTrue(all(
            np.allclose(a, b, rtol=0.0, atol=1e-15)
            for a, b in zip(weighted_secure, weighted_ordinary, strict=True)
        ))


if __name__ == "__main__":
    unittest.main()
