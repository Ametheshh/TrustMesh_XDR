"""Tests for the Unified Federated Master Model."""

import unittest
from pathlib import Path
import numpy as np

from scripts.build_unified_model import aggregate_models, OUTPUT_PATH
from scripts.demo_unified_model import load_model, predict_flow


class UnifiedModelTests(unittest.TestCase):
    def setUp(self):
        if not OUTPUT_PATH.exists():
            aggregate_models()
        self.weights, self.bias, self.features, self.metadata = load_model()

    def test_artifact_format_and_metadata(self):
        """Verify unified model matches standard project artifact specifications."""
        self.assertEqual(self.metadata["artifact_format"], "trustmesh_logistic_v1")
        self.assertEqual(self.metadata["model_name"], "unified_federated_master_model")
        self.assertEqual(len(self.features), 5)
        self.assertEqual(len(self.weights), 5)
        self.assertIsInstance(self.bias, float)
        self.assertGreater(self.metadata["aggregated_checkpoint_count"], 0)

    def test_benign_traffic_prediction(self):
        """Verify normal business flow is classified as benign with high confidence."""
        normal_flow = {
            "duration": 5.0,
            "bytes_src_to_dst": 1500.0,
            "bytes_dst_to_src": 50000.0,
            "packets_src_to_dst": 20.0,
            "packets_dst_to_src": 45.0,
        }
        pred, prob = predict_flow(normal_flow, self.weights, self.bias, self.features)
        self.assertEqual(pred, 0)
        self.assertLess(prob, 0.05)

    def test_attack_probe_prediction(self):
        """Verify malicious burst flow is classified as attack with high confidence."""
        attack_flow = {
            "duration": 0.02,
            "bytes_src_to_dst": 80.0,
            "bytes_dst_to_src": 0.0,
            "packets_src_to_dst": 120.0,
            "packets_dst_to_src": 0.0,
        }
        pred, prob = predict_flow(attack_flow, self.weights, self.bias, self.features)
        self.assertEqual(pred, 1)
        self.assertGreater(prob, 0.90)

    def test_zero_flow_edge_case(self):
        """Verify zero-length connection flow does not crash and handles log1p(0) gracefully."""
        zero_flow = {"duration": 0.0, "bytes_src_to_dst": 0.0, "bytes_dst_to_src": 0.0, "packets_src_to_dst": 0.0, "packets_dst_to_src": 0.0}
        pred, prob = predict_flow(zero_flow, self.weights, self.bias, self.features)
        self.assertIn(pred, (0, 1))
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_ddos_high_volume_attack(self):
        """Verify massive flood yields high attack probability (>95%)."""
        flood_flow = {"duration": 1.0, "bytes_src_to_dst": 50000.0, "bytes_dst_to_src": 0.0, "packets_src_to_dst": 10000.0, "packets_dst_to_src": 0.0}
        pred, prob = predict_flow(flood_flow, self.weights, self.bias, self.features)
        self.assertEqual(pred, 1)
        self.assertGreater(prob, 0.95)

    def test_missing_feature_graceful_handling(self):
        """Verify dictionary missing some features defaults cleanly without exceptions."""
        partial_flow = {"packets_src_to_dst": 500.0}
        pred, prob = predict_flow(partial_flow, self.weights, self.bias, self.features)
        self.assertEqual(pred, 1)
        self.assertGreater(prob, 0.95)

    def test_cross_company_attacks(self):
        """Verify model detects attacks from Bank, Hospital, and Enterprise profiles."""
        scenarios = [
            ("Enterprise C2", {"duration": 0.05, "bytes_src_to_dst": 40.0, "bytes_dst_to_src": 0.0, "packets_src_to_dst": 80.0, "packets_dst_to_src": 0.0}),
            ("Hospital IoT Probe", {"duration": 0.12, "bytes_src_to_dst": 120.0, "bytes_dst_to_src": 0.0, "packets_src_to_dst": 150.0, "packets_dst_to_src": 2.0}),
            ("Bank DoS Burst", {"duration": 0.45, "bytes_src_to_dst": 2500.0, "bytes_dst_to_src": 120.0, "packets_src_to_dst": 450.0, "packets_dst_to_src": 4.0}),
        ]
        for name, flow in scenarios:
            pred, prob = predict_flow(flow, self.weights, self.bias, self.features)
            self.assertEqual(pred, 1, f"Failed on scenario {name}")
            self.assertGreater(prob, 0.60, f"Low confidence on {name}")

    def test_packet_monotonicity(self):
        """Verify higher outgoing packet rates increase attack probability."""
        p_low = predict_flow({"packets_src_to_dst": 5.0}, self.weights, self.bias, self.features)[1]
        p_mid = predict_flow({"packets_src_to_dst": 50.0}, self.weights, self.bias, self.features)[1]
        p_high = predict_flow({"packets_src_to_dst": 500.0}, self.weights, self.bias, self.features)[1]
        self.assertLess(p_low, p_mid)
        self.assertLess(p_mid, p_high)


if __name__ == "__main__":
    unittest.main()
