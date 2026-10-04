"""Adapter for headerless UNSW partition CSVs with verified field ordering."""

import json
from pathlib import Path
from collections.abc import Iterator

from ..adapter import DatasetAdapter
from ..models import IngestRecord
from ._common import csv_rows, make_record, maybe_number


class UNSWPartitionAdapter(DatasetAdapter):
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path or "configs/datasets/unsw_nb15_partitions.json").read_text())

    def iter_records(self, path: str | Path, *, chunk_size: int = 4096) -> Iterator[IngestRecord]:
        source = Path(path)
        columns = self.config["ordered_columns"]
        for row, values in csv_rows(source, columns, header=False):
            labels = {name: values[name] for name in self.config["label_fields"]}
            restricted = {name: values[name] for name in self.config["restricted_fields"]}
            excluded = set(self.config["label_fields"] + self.config["restricted_fields"])
            features = {name: maybe_number(value) for name, value in values.items() if name not in excluded}
            yield make_record(self.config["dataset_id"], source, row, features, labels, None, "columns",
                              restricted=restricted, dataset_version=self.config.get("dataset_version"))
