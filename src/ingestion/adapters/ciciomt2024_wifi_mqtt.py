"""Adapter for CICIoMT2024 WiFi/MQTT feature CSV files."""

import json
import re
from pathlib import Path
from collections.abc import Iterator

from ..adapter import DatasetAdapter, ParseError
from ..models import IngestRecord
from ._common import csv_rows, make_record, maybe_number


class CICIoMT2024WifiMqttAdapter(DatasetAdapter):
    def __init__(self, config_path: str | Path | None = None):
        self.config = json.loads(Path(config_path or "configs/datasets/ciciomt2024_wifi_mqtt.json").read_text())
        self.pattern = re.compile(self.config["filename_pattern"])

    def iter_records(self, path: str | Path, *, chunk_size: int = 4096) -> Iterator[IngestRecord]:
        source = Path(path)
        match = self.pattern.fullmatch(source.name)
        if not match:
            raise ParseError(f"{source}: filename must match {self.config['filename_pattern']!r}")
        split = match.group("split")
        attack = match.group("label")
        labels = {"label_class_original": "Benign" if attack.lower() == "benign" else attack}
        for row, values in csv_rows(source, self.config["columns"]):
            features = {name: maybe_number(value) for name, value in values.items()}
            yield make_record(self.config["dataset_id"], source, row, features, labels, split, "filename",
                              dataset_version=self.config.get("dataset_version"))
