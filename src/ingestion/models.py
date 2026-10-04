"""Typed record containers. Labels and metadata are never part of features."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LabelData:
    original: dict[str, Any] = field(default_factory=dict)
    normalized: str | None = None
    binary: int | None = None
    label_source: str | None = None


@dataclass
class Provenance:
    record_id: str
    dataset_id: str
    source_file: str
    dataset_version: str | None = None
    source_split: str | None = None
    source_row: int | None = None
    adapter_version: str = "1.0"
    schema_version: str = "1.0"
    client_id: str | None = None


@dataclass
class QualityMetadata:
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestRecord:
    """A row with deliberately separate feature, label and metadata channels."""

    features: dict[str, Any]
    labels: LabelData
    provenance: Provenance
    quality: QualityMetadata = field(default_factory=QualityMetadata)
    # Restricted source fields (for example endpoint IPs); never model inputs.
    restricted: dict[str, Any] = field(default_factory=dict)
