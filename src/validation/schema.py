"""Checks record-channel separation and declared feature constraints."""

from src.ingestion.models import IngestRecord
from .report import ValidationReport


def validate_record(record: IngestRecord, *, excluded_features: set[str] | None = None) -> list[str]:
    issues = []
    excluded = excluded_features or set()
    leaked = excluded.intersection(record.features)
    if leaked:
        issues.append(f"restricted fields present in features: {sorted(leaked)!r}")
    if not isinstance(record.provenance.record_id, str) or not record.provenance.record_id:
        issues.append("record_id is missing")
    if record.labels.binary not in (0, 1, None):
        issues.append("binary label must be 0, 1, or unresolved null")
    if (record.labels.normalized is not None and record.labels.binary is None
            and "unresolved_binary_label" not in record.quality.flags):
        issues.append("unresolved binary label lacks quality metadata")
    if (record.provenance.dataset_version is None
            and "dataset_version_unknown" not in record.quality.flags):
        issues.append("unknown dataset version lacks quality metadata")
    return issues


def validate_records(records, *, excluded_features: set[str] | None = None, max_examples: int = 5) -> ValidationReport:
    report = ValidationReport(max_examples=max_examples)
    for record in records:
        report.rows_seen += 1
        for issue in validate_record(record, excluded_features=excluded_features):
            report.add_failure(f"record {record.provenance.record_id}: {issue}")
    return report
