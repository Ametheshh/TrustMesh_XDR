"""Data models for alert correlation and attack episode aggregation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class CanonicalAlert:
    """Represents a normalized security alert or detection event."""

    alert_id: str
    timestamp: Optional[float]
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    detection_confidence: float
    is_attack: bool
    tactic: str = "Unknown"
    technique: str = "Unknown"
    incident_id: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional non-network grouping key for controlled benchmarks and sources
    # that already provide an explicit correlation identifier.
    correlation_key: Optional[str] = None


@dataclass
class AttackEpisode:
    """Aggregated attack episode combining related alerts over time and entities."""

    episode_id: str
    alerts: List[CanonicalAlert] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    primary_entity: str = ""
    involved_entities: Set[str] = field(default_factory=set)
    involved_ports: Set[int] = field(default_factory=set)
    tactics_covered: Set[str] = field(default_factory=set)
    techniques_covered: Set[str] = field(default_factory=set)
    max_confidence: float = 0.0
    asset_criticality: float = 1.0

    def add_alert(self, alert: CanonicalAlert) -> None:
        """Add a canonical alert to the episode and update aggregate metrics."""
        if not self.alerts:
            self.start_time = alert.timestamp
            self.end_time = alert.timestamp
            if not self.primary_entity:
                self.primary_entity = alert.dst_ip or alert.src_ip
        elif alert.timestamp is not None:
            if self.start_time is None or self.end_time is None:
                self.start_time = alert.timestamp
                self.end_time = alert.timestamp
            else:
                self.start_time = min(self.start_time, alert.timestamp)
                self.end_time = max(self.end_time, alert.timestamp)

        self.alerts.append(alert)
        if alert.src_ip:
            self.involved_entities.add(alert.src_ip)
        if alert.dst_ip:
            self.involved_entities.add(alert.dst_ip)
        if alert.src_port:
            self.involved_ports.add(alert.src_port)
        if alert.dst_port:
            self.involved_ports.add(alert.dst_port)

        if alert.tactic and alert.tactic != "Unknown":
            self.tactics_covered.add(alert.tactic)
        if alert.technique and alert.technique != "Unknown":
            self.techniques_covered.add(alert.technique)

        self.max_confidence = max(self.max_confidence, alert.detection_confidence)

    @property
    def duration(self) -> float:
        """Return episode duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return max(0.0, self.end_time - self.start_time)

    @property
    def alert_count(self) -> int:
        """Return the total number of alerts in this episode."""
        return len(self.alerts)

    @property
    def evidence_diversity(self) -> int:
        """Return the number of distinct entities and ports involved."""
        return len(self.involved_entities) + len(self.involved_ports)

    @property
    def stage_coverage(self) -> int:
        """Return the number of unique MITRE ATT&CK stages covered."""
        return len(self.tactics_covered)

    @property
    def mean_confidence(self) -> float:
        """Return mean detection confidence across constituent alerts."""
        if not self.alerts:
            return 0.0
        return sum(a.detection_confidence for a in self.alerts) / len(self.alerts)

    @property
    def is_true_incident(self) -> bool:
        """Return True if any alert in the episode represents a genuine ground-truth attack."""
        return any(a.is_attack for a in self.alerts)

    @property
    def ground_truth_incident_ids(self) -> Set[str]:
        """Return the set of ground-truth incident IDs present in constituent alerts."""
        return {a.incident_id for a in self.alerts if a.incident_id is not None}
