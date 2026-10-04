"""Apply explicit sanitization policy without hashing or mutating source data."""

import json
from pathlib import Path

from src.ingestion.models import IngestRecord


def sanitize_record(record: IngestRecord, config_path: str | Path = "configs/sanitization/policy.json") -> IngestRecord:
    policy = json.loads(Path(config_path).read_text())
    disallowed = set(policy["canonical_shared_feature_exclusions"])
    for name in list(record.features):
        if name in disallowed:
            record.restricted[name] = record.features.pop(name)
    # Labels, provenance and quality remain separate typed channels.
    return record
