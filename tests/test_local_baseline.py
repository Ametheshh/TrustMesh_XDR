import json
import tempfile
import unittest
from pathlib import Path

from src.detection.local_baseline import load_feature_rows, run_baseline


class LocalBaselineTests(unittest.TestCase):
    def test_reads_only_feature_channel_and_binary_target(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.jsonl"
            rows = [
                {"features": {"duration": float(i)}, "labels": {"binary": i % 2, "original": "secret"},
                 "provenance": {"record_id": f"r{i}"}, "quality": {"flag": True},
                 "restricted": {"src_ip": "192.0.2.1"}}
                for i in range(4)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            x, y, names = load_feature_rows(path)
        self.assertEqual(names, ["duration"])
        self.assertEqual(x, [[0.0], [1.0], [2.0], [3.0]])
        self.assertEqual(y, [0, 1, 0, 1])

    def test_rejects_feature_schema_drift_and_nonbinary_target(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            path.write_text(
                json.dumps({"features": {"x": 1}, "labels": {"binary": 0}}) + "\n"
                + json.dumps({"features": {"y": 2}, "labels": {"binary": 1}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature schema/order changed"):
                load_feature_rows(path)

    def test_baseline_split_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "balanced.jsonl"
            rows = [
                {"features": {"signal": float(i % 2), "noise": float(i)},
                 "labels": {"binary": i % 2},
                 "provenance": {"record_id": f"r{i}"},
                 "quality": {}, "restricted": {"src_ip": "192.0.2.1"}}
                for i in range(40)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            first = run_baseline(path)
            second = run_baseline(path)
        self.assertEqual(first["split"], second["split"])
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["split"]["train_label_counts"], {"0": 16, "1": 16})
        self.assertEqual(first["split"]["validation_label_counts"], {"0": 4, "1": 4})
        self.assertEqual(first["feature_count"], 2)
        self.assertEqual(first["target"], "labels.binary")


if __name__ == "__main__":
    unittest.main()
