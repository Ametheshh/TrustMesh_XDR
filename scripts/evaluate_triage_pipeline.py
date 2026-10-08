"""Run a reproducible held-out detector-to-Top-50 triage simulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]

from src.correlation.engine import EpisodeCorrelator
from src.correlation.models import CanonicalAlert
from src.detection.local_baseline import LocalDetectorBaseline
from src.triage.metrics import calculate_incident_recall_at_k, calculate_precision_at_k
from src.triage.ranker import EpisodeRanker


DEFAULT_ARTIFACT = Path("data/canonical/smoke_test/ctu_sme_conn_labeled.jsonl")
FEATURES = (
    "duration",
    "bytes_src_to_dst",
    "bytes_dst_to_src",
    "packets_src_to_dst",
    "packets_dst_to_src",
)
MAX_ROWS = 10_000
INCIDENT_WINDOW_ROWS = 100
TOP_K = 50
ALERT_THRESHOLD = 0.5


def load_benchmark_rows(path: Path, max_rows: int = MAX_ROWS) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Load labeled rows with complete, finite, nonnegative shared CTU features."""
    records: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(records) >= max_rows:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            label = record.get("labels", {}).get("binary")
            features = record.get("features", {})
            provenance = record.get("provenance", {})
            if label not in (0, 1) or isinstance(label, bool):
                continue
            values = [features.get(name) for name in FEATURES]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                for value in values
            ):
                continue
            source_row = provenance.get("source_row")
            if not isinstance(source_row, int) or isinstance(source_row, bool):
                raise ValueError(f"provenance.source_row is required at {path}:{line_number}")
            records.append(record)
            vectors.append([math.log1p(float(value)) for value in values])
            labels.append(int(label))

    y = np.asarray(labels, dtype=np.int64)
    if not records or set(y.tolist()) != {0, 1}:
        raise ValueError(f"{path} must provide eligible rows from both binary classes")
    return records, np.asarray(vectors, dtype=np.float64), y


def build_incident_manifest(
    evaluation_records: list[dict[str, Any]],
    evaluation_labels: np.ndarray,
    *,
    window_rows: int = INCIDENT_WINDOW_ROWS,
) -> dict[int, str]:
    """Map attack-bearing fixed source-row windows to controlled incidents.

    The window assignment uses source provenance only; it does not depend on model
    predictions. The resulting IDs are benchmark ground truth, not public-dataset
    incident identifiers.
    """
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")
    attack_windows = sorted({
        (record["provenance"]["source_row"] - 1) // window_rows
        for record, label in zip(evaluation_records, evaluation_labels, strict=True)
        if int(label) == 1
    })
    return {
        window: f"SIM-CTU-SME-ROW-WINDOW-{window + 1:04d}"
        for window in attack_windows
    }


def run_evaluation(
    path: Path = DEFAULT_ARTIFACT,
    *,
    seed: int = 42,
    max_rows: int = MAX_ROWS,
    incident_window_rows: int = INCIDENT_WINDOW_ROWS,
) -> dict[str, Any]:
    records, x, y = load_benchmark_rows(path, max_rows=max_rows)
    all_indices = np.arange(len(y))
    train_indices, evaluation_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    train_records = [records[int(i)] for i in train_indices]
    evaluation_records = [records[int(i)] for i in evaluation_indices]
    y_train, y_evaluation = y[train_indices], y[evaluation_indices]
    train_ids = {record["provenance"]["record_id"] for record in train_records}
    evaluation_ids = {record["provenance"]["record_id"] for record in evaluation_records}
    if train_ids.intersection(evaluation_ids):
        raise AssertionError("training and evaluation records overlap")

    detector = LocalDetectorBaseline(random_state=seed)
    detector.fit(x[train_indices].tolist(), y_train.tolist())
    probabilities = detector.predict_proba(x[evaluation_indices].tolist())
    predictions = np.asarray([p >= ALERT_THRESHOLD for p in probabilities], dtype=bool)

    incident_manifest = build_incident_manifest(
        evaluation_records,
        y_evaluation,
        window_rows=incident_window_rows,
    )
    alerts: list[CanonicalAlert] = []
    for record, probability, is_alert, label in zip(
        evaluation_records, probabilities, predictions, y_evaluation, strict=True
    ):
        if not is_alert:
            continue
        source_row = record["provenance"]["source_row"]
        window = (source_row - 1) // incident_window_rows
        alerts.append(CanonicalAlert(
            alert_id=str(record["provenance"]["record_id"]),
            timestamp=None,
            src_ip="",
            dst_ip="",
            src_port=0,
            dst_port=0,
            protocol="",
            detection_confidence=float(probability),
            is_attack=bool(label),
            incident_id=incident_manifest.get(window) if int(label) == 1 else None,
            raw_metadata={"source_row": source_row},
            correlation_key=f"SIM-CTU-SME-ROW-WINDOW-{window + 1:04d}",
        ))

    episodes = EpisodeCorrelator().correlate_alerts(alerts)
    ranked_queue = EpisodeRanker().rank_episodes(episodes, top_k=None)
    top_k = ranked_queue[:TOP_K]
    total_incidents = len(incident_manifest)
    surfaced_incidents = {
        incident_id
        for episode, _ in top_k
        for incident_id in episode.ground_truth_incident_ids
    }

    return {
        "dataset": "CTU-SME Zeek canonical smoke artifact",
        "artifact": str(path),
        "features": list(FEATURES),
        "seed": seed,
        "incident_window_rows": incident_window_rows,
        "rows": len(records),
        "train_rows": len(train_indices),
        "evaluation_rows": len(evaluation_indices),
        "train_positive_rows": int(np.sum(y_train == 1)),
        "train_negative_rows": int(np.sum(y_train == 0)),
        "evaluation_positive_rows": int(np.sum(y_evaluation == 1)),
        "evaluation_negative_rows": int(np.sum(y_evaluation == 0)),
        "alerts": len(alerts),
        "episodes": len(episodes),
        "true_incidents": total_incidents,
        "top_k": TOP_K,
        "episodes_surfaced": len(top_k),
        "incidents_surfaced": len(surfaced_incidents),
        "incident_recall_at_50": calculate_incident_recall_at_k(
            ranked_queue,
            episodes,
            k=TOP_K,
            ground_truth_incident_ids=set(incident_manifest.values()),
        ),
        "precision_at_50": calculate_precision_at_k(ranked_queue, k=TOP_K),
        "alert_threshold": ALERT_THRESHOLD,
        "train_record_ids": sorted(train_ids),
        "evaluation_record_ids": sorted(evaluation_ids),
        "ground_truth_note": (
            "Controlled simulation ground truth: each fixed 100-source-row window "
            "containing at least one held-out labeled attack is one simulated incident."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--incident-window-rows", type=int, default=INCIDENT_WINDOW_ROWS)
    args = parser.parse_args()
    result = run_evaluation(
        args.input,
        seed=args.seed,
        max_rows=args.max_rows,
        incident_window_rows=args.incident_window_rows,
    )
    # Keep the report compact; record IDs remain available from run_evaluation for tests.
    result.pop("train_record_ids")
    result.pop("evaluation_record_ids")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
