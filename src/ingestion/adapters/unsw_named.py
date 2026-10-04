"""Adapter for UNSW-NB15 named train/test CSV files."""

import json
from pathlib import Path
from collections.abc import Iterator

from ..adapter import DatasetAdapter
from ..models import IngestRecord
from ._common import csv_rows, make_record, maybe_number


class UNSWNamedAdapter(DatasetAdapter):
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path or "configs/datasets/unsw_nb15_named.json").read_text())

    def iter_records(self, path: str | Path, *, chunk_size: int = 4096) -> Iterator[IngestRecord]:
        source = Path(path)
        split = self.config["split_by_filename"].get(source.name)
        if split is None:
            raise ValueError(f"{source}: filename is not a configured UNSW named split")
        columns = self.config["columns"]
        for row, values in csv_rows(source, columns):
            labels = {name: values[name] for name in self.config["label_fields"]}
            restricted = {name: values[name] for name in self.config["restricted_fields"]}
            excluded = set(self.config["label_fields"] + self.config["restricted_fields"] + self.config["provenance_fields"])
            features = {name: maybe_number(value) for name, value in values.items() if name not in excluded}
            yield make_record(self.config["dataset_id"], source, row, features, labels, split, "columns",
                              restricted=restricted, dataset_version=self.config.get("dataset_version"))
