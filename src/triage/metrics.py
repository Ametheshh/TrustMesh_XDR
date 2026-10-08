"""Triage and ranking evaluation metrics for TrustMesh XDR."""

import math
from typing import Dict, List, Set, Tuple
from src.correlation.models import AttackEpisode


def calculate_incident_recall_at_k(
    ranked_queue: List[Tuple[AttackEpisode, float]],
    all_episodes: List[AttackEpisode],
    k: int = 50,
    ground_truth_incident_ids: Set[str] | None = None,
) -> float:
    """Calculate Incident Recall@K: proportion of true-attack episodes surfaced in top-K queue.

    If ground-truth incident IDs exist on alerts, measures proportion of unique incident IDs captured.
    Otherwise, measures proportion of true attack episodes captured in top-K.
    """
    if not all_episodes:
        return 0.0

    # Collect ground-truth incident IDs across all episodes if available
    total_incident_ids: Set[str] = set(ground_truth_incident_ids or ())
    if ground_truth_incident_ids is None:
        for ep in all_episodes:
            total_incident_ids.update(ep.ground_truth_incident_ids)

    top_k_queue = ranked_queue[:k]

    if total_incident_ids:
        # Measure ID-based incident recall
        surfaced_incident_ids: Set[str] = set()
        for ep, _ in top_k_queue:
            surfaced_incident_ids.update(ep.ground_truth_incident_ids)
        return len(surfaced_incident_ids) / len(total_incident_ids)
    else:
        # Measure episode-based incident recall
        total_true_incidents = sum(1 for ep in all_episodes if ep.is_true_incident)
        if total_true_incidents == 0:
            return 1.0  # No true attacks exist
        surfaced_true_incidents = sum(1 for ep, _ in top_k_queue if ep.is_true_incident)
        return surfaced_true_incidents / total_true_incidents


def calculate_precision_at_k(
    ranked_queue: List[Tuple[AttackEpisode, float]],
    k: int = 50,
) -> float:
    """Calculate Precision@K: proportion of top-K queue items that are true attacks."""
    if not ranked_queue or k <= 0:
        return 0.0

    eval_k = min(k, len(ranked_queue))
    top_k_queue = ranked_queue[:eval_k]
    true_positives = sum(1 for ep, _ in top_k_queue if ep.is_true_incident)

    return true_positives / eval_k


def calculate_ndcg_at_k(
    ranked_queue: List[Tuple[AttackEpisode, float]],
    k: int = 50,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at K."""
    if not ranked_queue or k <= 0:
        return 0.0

    eval_k = min(k, len(ranked_queue))
    top_k_queue = ranked_queue[:eval_k]

    # DCG calculation
    dcg = 0.0
    for i, (ep, _) in enumerate(top_k_queue):
        rel = 1.0 if ep.is_true_incident else 0.0
        dcg += rel / math.log2(i + 2)

    # Ideal DCG calculation (all true incidents placed first)
    true_incidents_count = sum(1 for ep, _ in ranked_queue if ep.is_true_incident)
    idcg = 0.0
    for i in range(min(eval_k, true_incidents_count)):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_deduplication_ratio(
    raw_alerts_count: int,
    episodes_count: int,
) -> float:
    """Calculate alert-to-episode deduplication ratio (reduction factor)."""
    if episodes_count == 0:
        return 0.0
    return raw_alerts_count / episodes_count


def evaluate_triage_summary(
    ranked_queue: List[Tuple[AttackEpisode, float]],
    all_episodes: List[AttackEpisode],
    raw_alerts_count: int,
    k_list: List[int] = [25, 50, 100],
    ground_truth_incident_ids: Set[str] | None = None,
) -> Dict[str, float]:
    """Generate a comprehensive triage evaluation report."""
    total_episodes = len(all_episodes)
    dedup_ratio = calculate_deduplication_ratio(raw_alerts_count, total_episodes)

    results: Dict[str, float] = {
        "raw_alerts_count": float(raw_alerts_count),
        "total_episodes_count": float(total_episodes),
        "deduplication_ratio": dedup_ratio,
    }

    for k in k_list:
        results[f"incident_recall_at_{k}"] = calculate_incident_recall_at_k(
            ranked_queue, all_episodes, k=k,
            ground_truth_incident_ids=ground_truth_incident_ids,
        )
        results[f"precision_at_{k}"] = calculate_precision_at_k(ranked_queue, k=k)
        results[f"ndcg_at_{k}"] = calculate_ndcg_at_k(ranked_queue, k=k)

    return results
