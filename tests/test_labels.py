import csv
import tempfile
import unittest
from pathlib import Path

from src.ingestion.adapters.unsw_partitions import UNSWPartitionAdapter
from src.ingestion.models import LabelData
from src.normalization.labels import LabelNormalizer
from src.pipeline import iter_pipeline


class LabelTests(unittest.TestCase):
    def test_original_is_preserved_and_normalized_separately(self):
        labels = LabelData(original={"label": "DDoS-TCP_Flood"}, label_source="column:label")
        normalized = LabelNormalizer.from_config("configs/datasets/labels.json").apply(labels, "ciciot23")
        self.assertEqual(normalized.original["label"], "DDoS-TCP_Flood")
        self.assertEqual(normalized.normalized, "ciciot23:ddos-tcp_flood")
        self.assertEqual(normalized.binary, 1)

    def test_normal_fields_are_binary_benign(self):
        labels = LabelData(original={"attack_cat": "Normal", "label": "0"})
        LabelNormalizer.from_config("configs/datasets/labels.json").apply(labels, "unsw_nb15_named")
        self.assertEqual(labels.binary, 0)

    def test_ciciot_benigntraffic_is_binary_benign(self):
        labels = LabelData(original={"label": "BenignTraffic"})
        LabelNormalizer.from_config("configs/datasets/labels.json").apply(labels, "ciciot23")
        self.assertEqual(labels.normalized, "ciciot23:benigntraffic")
        self.assertEqual(labels.binary, 0)

    def test_unsw_headerless_capitalized_label_is_used_when_attack_cat_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "partition.csv"
            columns = UNSWPartitionAdapter().config["ordered_columns"]
            row = {name: "1" for name in columns}
            row.update(attack_cat="", Label="0")
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerow([row[name] for name in columns])
            record = next(iter_pipeline("unsw_nb15_partitions", path,
                                        dataset_config="configs/datasets/unsw_nb15_partitions.json"))
            self.assertEqual(record.labels.original["attack_cat"], "")
            self.assertEqual(record.labels.original["Label"], "0")
            self.assertEqual(record.labels.normalized, "unsw_nb15_partitions:0")
            self.assertEqual(record.labels.binary, 0)
            self.assertNotIn("Label", record.features)

    def test_unsw_normalization_trims_but_preserves_original_label(self):
        labels = LabelData(original={"attack_cat": " Fuzzers", "Label": "1"})
        LabelNormalizer.from_config("configs/datasets/labels.json").apply(labels, "unsw_nb15_partitions")
        self.assertEqual(labels.original["attack_cat"], " Fuzzers")
        self.assertEqual(labels.normalized, "unsw_nb15_partitions:fuzzers")
        clean = LabelData(original={"attack_cat": "Fuzzers", "Label": "1"})
        LabelNormalizer.from_config("configs/datasets/labels.json").apply(clean, "unsw_nb15_partitions")
        self.assertEqual(labels.normalized, clean.normalized)

    def test_unknown_dataset_version_is_null_and_quality_marked(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train" / "sample.csv"
            config = __import__("json").loads(Path("configs/datasets/ciciot23.json").read_text())
            path.parent.mkdir()
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(config["columns"])
                writer.writerow(["1" if name != "label" else "BenignTraffic" for name in config["columns"]])
            record = next(iter_pipeline("ciciot23", path, dataset_config="configs/datasets/ciciot23.json"))
            self.assertIsNone(record.provenance.dataset_version)
            self.assertIn("dataset_version_unknown", record.quality.flags)
            self.assertEqual(record.labels.binary, 0)

    def test_ctu_unknown_is_unresolved_and_quality_marked(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "conn.log"
            config = __import__("json").loads(Path("configs/datasets/ctu_sme_zeek.json").read_text())
            values = ["1.5", "uid1", "10.0.0.1", "123", "10.0.0.2", "80", "tcp", "http", "2.0", "10", "20", "SF", "-", "-", "0", "ShAD", "1", "50", "1", "60", "-", "Unknown", "unknown detail"]
            path.write_text("#separator \\x09\n#fields\t" + "\t".join(config["columns"]) +
                            "\n#types\t" + "\t".join(config["types"]) +
                            "\n" + "\t".join(values) + "\n")
            record = next(iter_pipeline("ctu_sme_zeek", path,
                                        dataset_config="configs/datasets/ctu_sme_zeek.json"))
            self.assertEqual(record.labels.original["label"], "Unknown")
            self.assertEqual(record.labels.binary, None)
            self.assertIn("unresolved_binary_label", record.quality.flags)
            self.assertTrue(record.quality.warnings)
            self.assertNotIn("uid", record.features)
            self.assertNotIn("label", record.features)


if __name__ == "__main__":
    unittest.main()
