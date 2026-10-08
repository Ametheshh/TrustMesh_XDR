"""Episode correlation engine for streaming and batch alert aggregation."""

from typing import Dict, List, Optional
from src.correlation.models import AttackEpisode, CanonicalAlert


class EpisodeCorrelator:
    """Correlates canonical alerts into multi-stage attack episodes."""

    def __init__(
        self,
        time_window_seconds: float = 300.0,
        asset_criticality_map: Optional[Dict[str, float]] = None,
    ):
        """Initialize the correlator with a time window and optional asset criticality map.

        Args:
            time_window_seconds: Max time delta (in seconds) between alerts to be correlated.
            asset_criticality_map: Map of entity IP/name to asset criticality score (default 1.0).
        """
        self.time_window_seconds = time_window_seconds
        self.asset_criticality_map = asset_criticality_map or {}

    def correlate_alerts(self, alerts: List[CanonicalAlert]) -> List[AttackEpisode]:
        """Correlate a batch of canonical alerts into attack episodes.

        Args:
            alerts: Uncorrelated canonical alerts.

        Returns:
            List of aggregated AttackEpisode objects.
        """
        if not alerts:
            return []

        # Sort alerts chronologically
        sorted_alerts = sorted(alerts, key=lambda a: (a.timestamp is None, a.timestamp or 0.0))
        episodes: List[AttackEpisode] = []
        active_episodes: Dict[str, AttackEpisode] = {}
        episode_counter = 0

        for alert in sorted_alerts:
            # Determine primary entity key for entity-based correlation
            primary_entity = alert.correlation_key or (alert.dst_ip if alert.dst_ip else alert.src_ip)
            if not primary_entity:
                primary_entity = "global"

            criticality = self.asset_criticality_map.get(primary_entity, 1.0)
            existing_ep = active_episodes.get(primary_entity)

            if existing_ep is None:
                # Start new episode for entity
                episode_counter += 1
                new_ep = AttackEpisode(
                    episode_id=f"EP-{episode_counter:04d}",
                    primary_entity=primary_entity,
                    asset_criticality=criticality,
                )
                new_ep.add_alert(alert)
                active_episodes[primary_entity] = new_ep
            else:
                # Check if alert fits within the active episode time window
                time_delta = (
                    alert.timestamp - existing_ep.end_time
                    if alert.timestamp is not None and existing_ep.end_time is not None
                    else None
                )
                # An explicit correlation key is authoritative; it represents
                # grouping metadata, not a timestamp or an invented network ID.
                if alert.correlation_key or (time_delta is not None and time_delta <= self.time_window_seconds):
                    existing_ep.add_alert(alert)
                else:
                    # Close previous episode and start a new one
                    episodes.append(existing_ep)
                    episode_counter += 1
                    new_ep = AttackEpisode(
                        episode_id=f"EP-{episode_counter:04d}",
                        primary_entity=primary_entity,
                        asset_criticality=criticality,
                    )
                    new_ep.add_alert(alert)
                    active_episodes[primary_entity] = new_ep

        # Flush remaining active episodes
        for ep in active_episodes.values():
            episodes.append(ep)

        return episodes
