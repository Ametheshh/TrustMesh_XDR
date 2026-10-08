"""Analyst triage ranker for risk-prioritized episode ordering."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.correlation.models import AttackEpisode


@dataclass
class RankingWeights:
    """Weights for composite risk scoring."""

    confidence: float = 0.40
    evidence_diversity: float = 0.20
    stage_coverage: float = 0.25
    asset_criticality: float = 0.15


class EpisodeRanker:
    """Ranks correlated attack episodes for analyst investigation queues."""

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        abstention_threshold: float = 0.10,
    ):
        """Initialize the ranker with custom scoring weights and abstention threshold."""
        self.weights = weights or RankingWeights()
        self.abstention_threshold = abstention_threshold

    def calculate_risk_score(self, episode: AttackEpisode) -> float:
        """Calculate a composite risk score for an attack episode in range [0, 1]."""
        # 1. Detection Confidence [0, 1]
        conf_score = max(0.0, min(1.0, episode.max_confidence))

        # 2. Evidence Diversity Score [0, 1] (normalized, max at 10 distinct entities/ports)
        div_score = min(1.0, episode.evidence_diversity / 10.0)

        # 3. Stage Coverage Score [0, 1] (normalized, max at 5 ATT&CK stages)
        stage_score = min(1.0, episode.stage_coverage / 5.0)

        # 4. Asset Criticality Score [0, 1] (normalized, assuming max asset criticality 3.0)
        criticality_score = min(1.0, episode.asset_criticality / 3.0)

        composite_score = (
            self.weights.confidence * conf_score
            + self.weights.evidence_diversity * div_score
            + self.weights.stage_coverage * stage_score
            + self.weights.asset_criticality * criticality_score
        )

        return max(0.0, min(1.0, composite_score))

    def rank_episodes(
        self,
        episodes: List[AttackEpisode],
        top_k: Optional[int] = 50,
    ) -> List[Tuple[AttackEpisode, float]]:
        """Rank episodes in descending order of risk score.

        Args:
            episodes: List of correlated attack episodes.
            top_k: Optional maximum number of episodes to return in top queue.

        Returns:
            List of (episode, risk_score) tuples ordered from highest to lowest risk.
        """
        scored_episodes = [
            (ep, self.calculate_risk_score(ep))
            for ep in episodes
        ]

        # Filter out episodes below abstention threshold if required
        filtered_episodes = [
            item for item in scored_episodes
            if item[1] >= self.abstention_threshold
        ]

        # Sort descending by risk score, breaking ties by max_confidence and alert count
        sorted_episodes = sorted(
            filtered_episodes,
            key=lambda item: (item[1], item[0].max_confidence, item[0].alert_count),
            reverse=True,
        )

        if top_k is not None:
            return sorted_episodes[:top_k]
        return sorted_episodes
