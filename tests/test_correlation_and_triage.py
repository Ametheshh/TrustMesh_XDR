"""Unit tests for episode correlation, triage ranking, and Incident Recall metrics."""

import pytest
from src.correlation.engine import EpisodeCorrelator
from src.correlation.models import AttackEpisode, CanonicalAlert
from src.triage.metrics import (
    calculate_deduplication_ratio,
    calculate_incident_recall_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    evaluate_triage_summary,
)
from src.triage.ranker import EpisodeRanker, RankingWeights


def test_canonical_alert_and_episode_properties():
    """Verify AttackEpisode aggregation and calculated properties."""
    alert1 = CanonicalAlert(
        alert_id="A1",
        timestamp=100.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=44321,
        dst_port=80,
        protocol="tcp",
        detection_confidence=0.85,
        is_attack=True,
        tactic="Reconnaissance",
        technique="T1046",
        incident_id="INC-01",
    )

    alert2 = CanonicalAlert(
        alert_id="A2",
        timestamp=150.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=44322,
        dst_port=443,
        protocol="tcp",
        detection_confidence=0.95,
        is_attack=True,
        tactic="Execution",
        technique="T1059",
        incident_id="INC-01",
    )

    episode = AttackEpisode(episode_id="EP-01", primary_entity="10.0.0.5")
    episode.add_alert(alert1)
    episode.add_alert(alert2)

    assert episode.alert_count == 2
    assert episode.start_time == 100.0
    assert episode.end_time == 150.0
    assert episode.duration == 50.0
    assert episode.max_confidence == 0.95
    assert episode.mean_confidence == pytest.approx(0.90)
    assert episode.stage_coverage == 2  # Reconnaissance, Execution
    assert episode.is_true_incident is True
    assert episode.ground_truth_incident_ids == {"INC-01"}
    # Evidence diversity: 2 IPs (192.168.1.10, 10.0.0.5) + 3 ports (44321, 44322, 80, 443 -> wait 4 ports)
    assert episode.evidence_diversity == 2 + 4  # 6 total distinct entities & ports


def test_episode_correlator_time_window_split():
    """Verify episode correlator splits alerts when time window is exceeded."""
    correlator = EpisodeCorrelator(time_window_seconds=100.0)

    alerts = [
        CanonicalAlert("A1", 100.0, "192.168.1.1", "10.0.0.1", 1000, 80, "tcp", 0.9, True),
        CanonicalAlert("A2", 150.0, "192.168.1.2", "10.0.0.1", 1001, 80, "tcp", 0.8, True),
        # A3 is 300s later -> should trigger new episode for 10.0.0.1
        CanonicalAlert("A3", 450.0, "192.168.1.3", "10.0.0.1", 1002, 80, "tcp", 0.95, True),
    ]

    episodes = correlator.correlate_alerts(alerts)

    assert len(episodes) == 2
    assert episodes[0].alert_count == 2
    assert episodes[1].alert_count == 1


def test_episode_ranker_ordering_and_abstention():
    """Verify EpisodeRanker scores episodes and filters noise below threshold."""
    ranker = EpisodeRanker(abstention_threshold=0.20)

    ep_high = AttackEpisode("EP-HIGH", max_confidence=0.95, asset_criticality=3.0)
    ep_high.tactics_covered = {"Reconnaissance", "Initial Access", "Execution"}
    ep_high.involved_entities = {"10.0.0.1", "192.168.1.1"}
    ep_high.involved_ports = {80, 443, 22}

    ep_low = AttackEpisode("EP-LOW", max_confidence=0.10, asset_criticality=1.0)
    ep_low.tactics_covered = set()
    ep_low.involved_entities = {"10.0.0.2"}

    episodes = [ep_low, ep_high]
    ranked = ranker.rank_episodes(episodes, top_k=50)

    # Low confidence episode should be filtered out by abstention threshold
    assert len(ranked) == 1
    assert ranked[0][0].episode_id == "EP-HIGH"
    assert ranked[0][1] > 0.50


def test_triage_metrics_calculation():
    """Verify Incident Recall@K, Precision@K, NDCG@K, and deduplication ratio."""
    ep_true1 = AttackEpisode("EP-1")
    ep_true1.add_alert(CanonicalAlert("A1", 10.0, "1.1.1.1", "2.2.2.2", 1, 2, "tcp", 0.9, True, incident_id="INC-A"))

    ep_true2 = AttackEpisode("EP-2")
    ep_true2.add_alert(CanonicalAlert("A2", 20.0, "1.1.1.1", "2.2.2.2", 1, 3, "tcp", 0.8, True, incident_id="INC-B"))

    ep_false = AttackEpisode("EP-3")
    ep_false.add_alert(CanonicalAlert("A3", 30.0, "1.1.1.1", "2.2.2.2", 1, 4, "tcp", 0.1, False))

    all_episodes = [ep_true1, ep_true2, ep_false]
    ranked_queue = [(ep_true1, 0.9), (ep_false, 0.5), (ep_true2, 0.2)]

    # Recall at 2 includes ep_true1 (INC-A) but misses ep_true2 (INC-B)
    recall_at_2 = calculate_incident_recall_at_k(ranked_queue, all_episodes, k=2)
    assert recall_at_2 == 0.5  # 1 out of 2 incidents captured

    precision_at_2 = calculate_precision_at_k(ranked_queue, k=2)
    assert precision_at_2 == 0.5  # 1 true out of 2 in queue

    ndcg_at_3 = calculate_ndcg_at_k(ranked_queue, k=3)
    assert 0.0 < ndcg_at_3 <= 1.0

    dedup = calculate_deduplication_ratio(100, 10)
    assert dedup == 10.0

    summary = evaluate_triage_summary(ranked_queue, all_episodes, raw_alerts_count=30, k_list=[2, 3])
    assert summary["incident_recall_at_2"] == 0.5
    assert summary["deduplication_ratio"] == 10.0
