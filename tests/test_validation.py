import unittest

from src.ingestion.models import (
    IngestRecord,
    LabelData,
    Provenance,
    QualityMetadata,
)
from src.validation.schema import validate_record, validate_records


class ValidationTests(unittest.TestCase):
    def record(self, features):
        provenance = Provenance("r1", "test", "fixture.csv")
        quality = QualityMetadata(flags=["dataset_version_unknown"])
        return IngestRecord(features, LabelData(), provenance, quality)

    def test_restricted_feature_detected(self):
        issues = validate_record(
            self.record({"src_ip": "192.0.2.1"}),
            excluded_features={"src_ip"},
        )
        self.assertTrue(any("restricted fields" in issue for issue in issues))

    def test_report_diagnostics_are_bounded(self):
        report = validate_records(
            [self.record({"src_ip": str(i)}) for i in range(20)],
            excluded_features={"src_ip"},
            max_examples=2,
        )
        self.assertEqual(report.rows_seen, 20)
        self.assertEqual(report.failures, 20)
        self.assertEqual(len(report.examples), 2)


if __name__ == "__main__":
    unittest.main()
