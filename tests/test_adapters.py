import csv
import tempfile
import unittest
from pathlib import Path

from src.ingestion.adapter import ParseError
from src.ingestion.adapters.ciciot23 import CICIOT23Adapter
from src.ingestion.adapters.ciciomt2024_wifi_mqtt import CICIoMT2024WifiMqttAdapter
from src.ingestion.adapters.ctu_sme_zeek import CTUSMEZeekAdapter
from src.ingestion.adapters.unsw_named import UNSWNamedAdapter
from src.ingestion.adapters.unsw_partitions import UNSWPartitionAdapter
from src.normalization.labels import LabelNormalizer
from src.sanitization.policy import sanitize_record


class AdapterTests(unittest.TestCase):
    def write_csv(self, path, header, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if header:
                writer.writerow(header)
            writer.writerow(row)

    def test_ciciot23_explicit_label_and_split(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train" / "sample.csv"
            columns = CICIOT23Adapter().config["columns"]
            self.write_csv(path, columns, ["1" if c != "label" else "Attack-X" for c in columns])
            record = next(CICIOT23Adapter().iter_records(path))
            self.assertEqual(record.provenance.source_split, "train")
            self.assertEqual(record.labels.original, {"label": "Attack-X"})
            self.assertNotIn("label", record.features)

    def test_ciciomt_filename_label_and_strict_schema(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ARP_Spoofing_test.pcap.csv"
            columns = CICIoMT2024WifiMqttAdapter().config["columns"]
            self.write_csv(path, columns, ["1"] * len(columns))
            record = next(CICIoMT2024WifiMqttAdapter().iter_records(path))
            self.assertEqual(record.provenance.source_split, "test")
            self.assertEqual(record.labels.original["label_class_original"], "ARP_Spoofing")
            with self.assertRaises(ParseError):
                list(CICIoMT2024WifiMqttAdapter().iter_records(path.with_name("unexpected.csv")))

    def test_unsw_named_excludes_id_labels_and_sanitizes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "UNSW_NB15_training-set.csv"
            config = UNSWNamedAdapter().config
            row = {name: "1" for name in config["columns"]}
            row.update(id="7", attack_cat="Normal", label="0")
            self.write_csv(path, config["columns"], [row[k] for k in config["columns"]])
            record = next(UNSWNamedAdapter().iter_records(path))
            self.assertNotIn("id", record.features)
            self.assertNotIn("attack_cat", record.features)
            LabelNormalizer.from_config("configs/datasets/labels.json").apply(record.labels, "unsw_nb15_named")
            self.assertEqual(record.labels.binary, 0)
            self.assertEqual(record.labels.original["attack_cat"], "Normal")
            sanitize_record(record)
            self.assertNotIn("stcpb", record.features)

    def test_unsw_headerless_uses_verified_order_and_rejects_bad_width(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "partition.csv"
            columns = UNSWPartitionAdapter().config["ordered_columns"]
            self.write_csv(path, None, ["1"] * len(columns))
            record = next(UNSWPartitionAdapter().iter_records(path))
            self.assertIn("proto", record.features)
            self.assertIn("srcip", record.restricted)
            bad = Path(td) / "bad.csv"
            self.write_csv(bad, None, ["x"])
            with self.assertRaises(ParseError):
                list(UNSWPartitionAdapter().iter_records(bad))

    def test_ctu_zeek_metadata_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "conn.log"
            config = CTUSMEZeekAdapter().config
            fields = config["columns"]
            types = config["types"]
            values = ["1.5", "uid1", "10.0.0.1", "123", "10.0.0.2", "80", "tcp", "http", "2.0", "10", "20", "SF", "-", "-", "0", "ShAD", "1", "50", "1", "60", "-", "Benign", "benign-test"]
            path.write_text("#separator \\x09\n#empty_field\t(empty)\n#unset_field\t-\n#fields\t" +
                            "\t".join(fields) + "\n#types\t" + "\t".join(types) + "\n" +
                            "\t".join(values) + "\n")
            record = next(CTUSMEZeekAdapter().iter_records(path))
            self.assertEqual(record.labels.original["label"], "Benign")
            self.assertIn("id.orig_h", record.restricted)
            self.assertNotIn("uid", record.features)

    def test_ctu_declared_unset_and_empty_markers_remain_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "conn.log"
            config = CTUSMEZeekAdapter().config
            fields = config["columns"]
            values = ["1.5", "uid1", "10.0.0.1", "123", "10.0.0.2", "80", "tcp", "(empty)", "-", "-", "20", "SF", "-", "(empty)", "0", "ShAD", "1", "50", "1", "60", "-", "Malicious", "example"]
            path.write_text("#separator \\x09\n#empty_field\t(empty)\n#unset_field\t-\n#fields\t" +
                            "\t".join(fields) + "\n#types\t" + "\t".join(config["types"]) + "\n" +
                            "\t".join(values) + "\n")
            record = next(CTUSMEZeekAdapter().iter_records(path))
            self.assertIsNone(record.features["duration"])
            self.assertIsNone(record.features["orig_bytes"])
            self.assertEqual(record.features["service"], "")
            self.assertIsNone(record.features["local_orig"])
            self.assertIn("source_unset_fields_present", record.quality.flags)
            self.assertIn("source_empty_fields_present", record.quality.flags)


if __name__ == "__main__":
    unittest.main()
