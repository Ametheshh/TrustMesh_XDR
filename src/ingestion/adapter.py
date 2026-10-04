"""Base interface and bounded parsing errors for dataset adapters."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from .models import IngestRecord


class AdapterError(ValueError):
    """Input does not conform to an adapter's declared format."""


class ParseError(AdapterError):
    """A bounded, row-specific parsing failure."""


class DatasetAdapter(ABC):
    adapter_version = "1.0"
    schema_version = "1.0"

    @abstractmethod
    def iter_records(self, path: str | Path, *, chunk_size: int = 4096) -> Iterator[IngestRecord]:
        """Yield records incrementally; never load the full input into memory."""
