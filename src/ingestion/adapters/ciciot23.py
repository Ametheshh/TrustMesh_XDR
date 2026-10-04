"""Adapter for the CICIOT23 train/validation/test CSV layout."""

import json
from pathlib import Path
from collections.abc import Iterator

from ..adapter import DatasetAdapter
from ..models import IngestRecord
from ._common import csv_rows, make_record, maybe_number


class CICIOT23Adapter(DatasetAdapter):
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path or "configs/datasets/ciciot23.json").read_text())

    def iter_records(self, path: str | Path, *, chunk_size: int = 4096) -> Iterator[IngestRecord]:
        source = Path(path)
        split = self.config["split_by_parent"].get(source.parent.name)
        if split is None:
            raise ValueError(f"{source}: expected parent directory train, validation, or test")
        columns = self.config["columns"]
        for row, values in csv_rows(source, columns):
            labels = {name: values[name] for name in self.config["label_fields"]}
            features = {name: maybe_number(value) for name, value in values.items() if name not in labels}
            yield make_record(self.config["dataset_id"], source, row, features, labels, split, "column:label",
                              dataset_version=self.config.get("dataset_version"))
