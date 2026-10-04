"""Versioned label normalization preserving dataset-native labels."""

import json
from pathlib import Path

from src.ingestion.models import LabelData


class LabelNormalizer:
    def __init__(self, mappings: dict, *, benign_values: list[str] | None = None,
                 benign_values_by_dataset: dict | None = None, binary_class_mappings: dict | None = None):
        self.mappings = mappings
        self.benign_values = {value.casefold() for value in (benign_values or ["benign", "normal"])}
        self.benign_values_by_dataset = {
            dataset: {value.casefold() for value in values}
            for dataset, values in (benign_values_by_dataset or {}).items()
        }
        self.binary_class_mappings = binary_class_mappings or {}

    @classmethod
    def from_config(cls, path: str | Path) -> "LabelNormalizer":
        config = json.loads(Path(path).read_text())
        return cls(config["class_mappings"], benign_values=config["benign_values"],
                   benign_values_by_dataset=config.get("benign_values_by_dataset", {}),
                   binary_class_mappings=config.get("binary_class_mappings", {}))

    def apply(self, labels: LabelData, dataset_id: str) -> LabelData:
        raw = None
        for name in ("label_class_original", "attack_cat", "label", "Label"):
            candidate = labels.original.get(name)
            if candidate is not None and str(candidate).strip() != "":
                raw = candidate
                break
        if raw is None:
            return labels
        original = str(raw)
        normalized_source = original.strip()
        mapping = self.mappings.get(dataset_id, {})
        normalized = mapping.get(normalized_source, mapping.get(normalized_source.casefold()))
        if normalized is None and self.mappings.get("identity_datasets", {}).get(dataset_id):
            normalized = f"{dataset_id}:{normalized_source.casefold()}"
        if normalized is None:
            raise ValueError(f"No normalized label mapping for dataset {dataset_id!r}, value {original!r}")
        labels.normalized = normalized

        # Prefer an explicit source binary target, including UNSW's capitalized
        # `Label`, while keeping it in the label channel.
        explicit_binary = None
        for name in ("label", "Label"):
            candidate = labels.original.get(name)
            if candidate is None or str(candidate).strip() == "":
                continue
            try:
                numeric = float(candidate)
            except (TypeError, ValueError):
                continue
            if numeric in (0, 1):
                explicit_binary = int(numeric)
                break

        if explicit_binary is not None:
            labels.binary = explicit_binary
            return labels

        strict_map = self.binary_class_mappings.get(dataset_id)
        folded = normalized_source.casefold()
        if strict_map is not None:
            benign = {value.casefold() for value in strict_map.get("benign", [])}
            attacks = {value.casefold() for value in strict_map.get("attack", [])}
            unresolved = {value.casefold() for value in strict_map.get("unresolved", [])}
            if folded in benign:
                labels.binary = 0
            elif folded in attacks:
                labels.binary = 1
            elif folded in unresolved or strict_map.get("unmapped") == "unresolved":
                # Configured unresolved values and any unrecognized CTU label
                # stay unresolved instead of being guessed as an attack.
                labels.binary = None
            else:
                raise ValueError(f"No binary label mapping for dataset {dataset_id!r}, value {original!r}")
            return labels

        dataset_benign = self.benign_values_by_dataset.get(dataset_id, set())
        labels.binary = 0 if folded in (self.benign_values | dataset_benign) or normalized == "benign" else 1
        return labels
