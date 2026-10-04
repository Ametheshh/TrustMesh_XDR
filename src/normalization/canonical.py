"""Apply explicit source-to-canonical field mappings."""

from src.ingestion.models import IngestRecord


def normalize_features(record: IngestRecord, field_mapping: dict[str, str]) -> IngestRecord:
    unknown = set(field_mapping) - set(record.features)
    # A configured field may be absent from a dataset variant; it remains absent.
    normalized = {target: record.features[source] for source, target in field_mapping.items() if source in record.features}
    # Keep source-native values with an explicit namespace to avoid discarding semantics.
    for source, value in record.features.items():
        if source not in field_mapping:
            normalized[f"source_native.{source}"] = value
    record.features = normalized
    if unknown:
        record.quality.flags.append("configured_mapping_fields_absent")
    return record
