"""Lazy pipeline orchestration; caller controls input and output destinations."""

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from src.ingestion.registry import get_adapter
from src.normalization.canonical import normalize_features
from src.normalization.labels import LabelNormalizer
from src.sanitization.policy import sanitize_record
from src.validation.schema import validate_record


def iter_staging(adapter_id: str, source_path: str | Path) -> Iterator:
    """Yield parsed source-shaped staging records lazily."""
    yield from get_adapter(adapter_id).iter_records(source_path)


def iter_sanitized(records: Iterable, *, policy_config: str | Path) -> Iterator:
    """Yield records after policy-based separation of restricted fields."""
    for record in records:
        yield sanitize_record(record, policy_config)


def iter_canonical(
    records: Iterable,
    *,
    dataset_config: str | Path,
    label_config: str | Path = "configs/datasets/labels.json",
    policy_config: str | Path = "configs/sanitization/policy.json",
) -> Iterator:
    """Normalize labels/features and yield canonical records without restricted values."""
    config = json.loads(Path(dataset_config).read_text())
    policy = json.loads(Path(policy_config).read_text())
    label_normalizer = LabelNormalizer.from_config(label_config)

    for record in records:
        label_normalizer.apply(
            record.labels,
            record.provenance.dataset_id,
        )

        if record.provenance.dataset_version is None:
            record.quality.flags.append("dataset_version_unknown")
            record.quality.warnings.append(
                "Dataset release/version is not configured; dataset_version is null"
            )

        if record.labels.original and record.labels.binary is None:
            record.quality.flags.append("unresolved_binary_label")
            record.quality.warnings.append(
                "Binary label unresolved by configured dataset label mapping"
            )

        normalize_features(
            record,
            config.get("canonical_mapping", {}),
        )

        issues = validate_record(
            record,
            excluded_features=set(
                policy["canonical_shared_feature_exclusions"]
            ),
        )

        if issues:
            raise ValueError(
                f"{record.provenance.source_file}: "
                f"row {record.provenance.source_row}: "
                f"{'; '.join(issues)}"
            )

        # Restricted values remain staging-only and are dropped from canonical output.
        record.restricted.clear()
        yield record


def iter_pipeline(
    adapter_id: str,
    source_path: str | Path,
    *,
    dataset_config: str | Path,
    label_config: str | Path = "configs/datasets/labels.json",
    policy_config: str | Path = "configs/sanitization/policy.json",
    client_assigner: Callable | None = None,
) -> Iterator:
    """Stream raw/source through staging, sanitized and canonical to client routing.

    No files are written. A caller may assign client metadata; no federated learning
    operation is performed here.
    """
    staged = iter_staging(adapter_id, source_path)
    sanitized = iter_sanitized(
        staged,
        policy_config=policy_config,
    )
    canonical = iter_canonical(
        sanitized,
        dataset_config=dataset_config,
        label_config=label_config,
        policy_config=policy_config,
    )

    for record in canonical:
        if client_assigner is not None:
            record.provenance.client_id = client_assigner(record)
        yield record
