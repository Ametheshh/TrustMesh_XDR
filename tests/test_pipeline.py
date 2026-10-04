import csv
import tempfile
import unittest
from pathlib import Path

from src.pipeline import iter_pipeline


class PipelineTests(unittest.TestCase):
    def test_tiny_fixture_pipeline_separates_channels(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train" / "tiny.csv"
            path.parent.mkdir()
            config = __import__("json").loads(Path("configs/datasets/ciciot23.json").read_text())
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(config["columns"])
                writer.writerow(["1" if name != "label" else "Benign" for name in config["columns"]])
            record = next(iter_pipeline("ciciot23", path, dataset_config="configs/datasets/ciciot23.json",
                                        client_assigner=lambda _: "client_01"))
            self.assertEqual(record.labels.binary, 0)
            self.assertNotIn("label", record.features)
            self.assertNotIn("record_id", record.features)
            self.assertIn("duration", record.features)
            self.assertEqual(record.provenance.client_id, "client_01")
            self.assertEqual(record.restricted, {})


if __name__ == "__main__":
    unittest.main()
