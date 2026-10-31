"""Apply explicit sanitization policy without hashing or mutating source data."""

from functools import lru_cache
import json
from pathlib import Path

from src.ingestion.models import IngestRecord


@lru_cache(maxsize=16)
def _load_disallowed(config_path: str) -> frozenset:
    policy = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return frozenset(policy["canonical_shared_feature_exclusions"])


def sanitize_record(record: IngestRecord, config_path: str | Path = "configs/sanitization/policy.json") -> IngestRecord:
    disallowed = _load_disallowed(str(config_path))
    for name in list(record.features):
        if name in disallowed:
            record.restricted[name] = record.features.pop(name)
    # Labels, provenance and quality remain separate typed channels.
    return record
