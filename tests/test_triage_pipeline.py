"""Scientific-sanity tests for the controlled held-out triage experiment."""

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_triage_pipeline import build_incident_manifest, run_evaluation
from src.correlation.models import AttackEpisode, CanonicalAlert
from src.triage.metrics import calculate_incident_recall_at_k, calculate_precision_at_k


def test_incident_manifest_is_deterministic_and_uses_source_windows():
    records = [
        {"provenance": {"source_row": row}}
        for row in (1, 25, 101, 199, 201)
    ]
    labels = np.asarray([1, 0, 1, 1, 0])

    first = build_incident_manifest(records, labels, window_rows=100)
    second = build_incident_manifest(records, labels, window_rows=100)

    assert first == second
    assert list(first) == [0, 1]
    assert len(set(first.values())) == 2


def test_top_50_incident_metrics_use_manifest_denominator():
    episodes = []
    for index in range(51):
        alert = CanonicalAlert(
            alert_id=f"alert-{index}", timestamp=None, src_ip="", dst_ip="",
            src_port=0, dst_port=0, protocol="", detection_confidence=0.9,
            is_attack=True, incident_id=f"incident-{index}",
        )
        episode = AttackEpisode(episode_id=f"episode-{index}")
        episode.add_alert(alert)
        episodes.append((episode, 0.9))

    manifest_ids = {f"incident-{index}" for index in range(51)}
    assert calculate_incident_recall_at_k(
        episodes, [episode for episode, _ in episodes], k=50,
        ground_truth_incident_ids=manifest_ids,
    ) == 50 / 51
    assert calculate_precision_at_k(episodes, k=50) == 1.0


def test_pipeline_runs_on_heldout_rows_without_record_overlap():
    artifact = Path("data/canonical/smoke_test/ctu_sme_conn_labeled.jsonl")
    if not artifact.exists():
        pytest.skip("local canonical CTU-SME smoke artifact is not present")

    result = run_evaluation(artifact)

    assert result["train_rows"] + result["evaluation_rows"] == result["rows"]
    assert not set(result["train_record_ids"]).intersection(result["evaluation_record_ids"])
    assert result["episodes_surfaced"] <= 50
    assert 0.0 <= result["incident_recall_at_50"] <= 1.0
