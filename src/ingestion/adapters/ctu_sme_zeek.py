"""Streaming parser for Zeek conn.log files with directive validation."""

import json
from pathlib import Path
from collections.abc import Iterator

from ..adapter import DatasetAdapter, ParseError
from ..models import IngestRecord
from ._common import make_record, maybe_number


class CTUSMEZeekAdapter(DatasetAdapter):
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path or "configs/datasets/ctu_sme_zeek.json").read_text())

    def iter_records(self, path: str | Path, *, chunk_size: int = 4096) -> Iterator[IngestRecord]:
        source = Path(path)
        separator = None
        unset_marker = None
        empty_marker = None
        fields = types = None
        data_row = 0
        with source.open("r", encoding="utf-8", newline="") as stream:
            first_data = None
            for line_no, line in enumerate(stream, 1):
                line = line.rstrip("\r\n")
                if line.startswith("#separator "):
                    encoded = line.split(" ", 1)[1]
                    try:
                        separator = bytes(encoded, "utf-8").decode("unicode_escape")
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise ParseError(f"{source}: line {line_no}: invalid #separator") from exc
                elif line.startswith("#fields\t"):
                    fields = line.split("\t")[1:]
                elif line.startswith("#types\t"):
                    types = line.split("\t")[1:]
                elif line.startswith("#unset_field\t"):
                    unset_marker = line.split("\t", 1)[1]
                elif line.startswith("#empty_field\t"):
                    empty_marker = line.split("\t", 1)[1]
                elif line.startswith("#"):
                    continue
                else:
                    first_data = (line_no, line)
                    break
            if separator is None or fields is None or types is None:
                raise ParseError(f"{source}: required Zeek #separator, #fields, or #types metadata missing")
            if len(fields) != len(types):
                raise ParseError(f"{source}: #fields has {len(fields)} entries but #types has {len(types)}")
            if fields != self.config["columns"]:
                raise ParseError(f"{source}: Zeek field schema mismatch; expected configured ordered fields")
            if types != self.config["types"]:
                raise ParseError(f"{source}: Zeek type schema mismatch; expected configured ordered types")

            def convert(line_number: int, record_line: str) -> IngestRecord:
                nonlocal data_row
                values = record_line.split(separator)
                if len(values) != len(fields):
                    raise ParseError(f"{source}: line {line_number} has {len(values)} fields; expected {len(fields)}")
                data_row += 1
                raw_row = dict(zip(fields, values, strict=True))
                unset_fields = [name for name, value in raw_row.items()
                                if unset_marker is not None and value == unset_marker]
                empty_fields = [name for name, value in raw_row.items()
                                if empty_marker is not None and value == empty_marker]
                row = {
                    name: (None if unset_marker is not None and value == unset_marker
                           else "" if empty_marker is not None and value == empty_marker
                           else value)
                    for name, value in raw_row.items()
                }
                def parse_value(value):
                    # Preserve the declared empty value as an empty string; only
                    # the declared unset marker becomes None.
                    return value if value is None or value == "" else maybe_number(value)
                labels = {name: row[name] for name in self.config["label_fields"]}
                excluded = set(self.config["label_fields"] + self.config["restricted_fields"])
                features = {name: parse_value(value) for name, value in row.items() if name not in excluded}
                restricted = {name: row[name] for name in self.config["restricted_fields"] if name in row}
                record = make_record(self.config["dataset_id"], source, data_row, features, labels, None,
                                     "columns", restricted=restricted,
                                     dataset_version=self.config.get("dataset_version"))
                if unset_fields:
                    record.quality.flags.append("source_unset_fields_present")
                    record.quality.warnings.append("Zeek unset fields: " + ", ".join(unset_fields))
                if empty_fields:
                    record.quality.flags.append("source_empty_fields_present")
                    record.quality.warnings.append("Zeek empty fields: " + ", ".join(empty_fields))
                return record

            if first_data is not None:
                yield convert(*first_data)
            for line_number, line in enumerate(stream, line_no + 1):
                line = line.rstrip("\r\n")
                if line.startswith("#"):
                    raise ParseError(f"{source}: unexpected metadata directive at line {line_number}")
                yield convert(line_number, line)
