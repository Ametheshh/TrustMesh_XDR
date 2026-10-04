"""Shared CSV helpers with strict schema checks."""

import csv
import hashlib
from pathlib import Path
from typing import Any

from ..adapter import ParseError
from ..models import IngestRecord, LabelData, Provenance


def record_id(dataset_id: str, source: Path, row: int) -> str:
    # Hash only stable source path and ordinal, never feature or identifier values.
    raw = f"{dataset_id}\0{source.as_posix()}\0{row}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def csv_rows(path: str | Path, expected: list[str] | None, *, header: bool = True):
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if header:
            actual = next(reader, None)
            if actual is None:
                raise ParseError(f"{source}: empty CSV; expected a header")
            if expected is not None and actual != expected:
                raise ParseError(f"{source}: header mismatch; expected {expected!r}, got {actual!r}")
        row_number = 1 if header else 0
        for values in reader:
            row_number += 1
            if expected is not None and len(values) != len(expected):
                raise ParseError(f"{source}: row {row_number} has {len(values)} columns; expected {len(expected)}")
            if expected is None:
                raise ParseError("An explicit schema is required")
            yield row_number, dict(zip(expected, values, strict=True))


def make_record(dataset: str, path: Path, row: int, features: dict[str, Any], labels: dict[str, Any],
                split: str | None, label_source: str | None, *, restricted: dict[str, Any] | None = None,
                dataset_version: str | None = None) -> IngestRecord:
    return IngestRecord(features=features, labels=LabelData(original=labels, label_source=label_source),
                        provenance=Provenance(record_id=record_id(dataset, path, row), dataset_id=dataset,
                                              source_file=str(path), dataset_version=dataset_version,
                                              source_split=split, source_row=row if row else None),
                        restricted=restricted or {})


def maybe_number(value: str) -> Any:
    if value == "":
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value
