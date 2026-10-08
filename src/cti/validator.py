"""STIX 2.1 Cyber Threat Intelligence (CTI) validation and local relevance scoring engine."""

from typing import Any, Dict, List, Set
from src.correlation.models import AttackEpisode


class CTIValidator:
    """Validates STIX 2.1 threat intelligence bundles and computes local relevance scores."""

    def __init__(self, min_confidence: int = 50):
        """Initialize CTIValidator.

        Args:
            min_confidence: Minimum confidence threshold (0-100) to accept CTI indicators.
        """
        self.min_confidence = min_confidence

    def validate_stix_bundle(self, bundle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a STIX 2.1 bundle structure and compute quality metrics."""
        if not isinstance(bundle_data, dict):
            return {"is_valid": False, "reason": "Bundle must be a JSON object", "quality_score": 0.0}

        bundle_type = bundle_data.get("type")
        if bundle_type != "bundle":
            return {"is_valid": False, "reason": f"Invalid bundle type: {bundle_type}", "quality_score": 0.0}

        objects = bundle_data.get("objects", [])
        if not isinstance(objects, list) or len(objects) == 0:
            return {"is_valid": False, "reason": "Bundle contains no STIX objects", "quality_score": 0.0}

        indicator_count = 0
        attack_pattern_count = 0
        malware_count = 0
        tactics_found: Set[str] = set()

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("type")
            confidence = obj.get("confidence", 75)

            if confidence < self.min_confidence:
                continue

            if obj_type == "indicator":
                indicator_count += 1
            elif obj_type == "attack-pattern":
                attack_pattern_count += 1
                name = obj.get("name")
                if name:
                    tactics_found.add(name)
            elif obj_type == "malware":
                malware_count += 1

        total_valid_objects = indicator_count + attack_pattern_count + malware_count
        quality_score = min(1.0, (total_valid_objects / max(1, len(objects))) * 0.8 + (indicator_count > 0) * 0.2)

        return {
            "is_valid": True,
            "total_objects": len(objects),
            "indicator_count": indicator_count,
            "attack_pattern_count": attack_pattern_count,
            "malware_count": malware_count,
            "tactics_found": sorted(list(tactics_found)),
            "quality_score": round(quality_score, 4),
        }

    def compute_local_relevance(self, stix_report: Dict[str, Any], episodes: List[AttackEpisode]) -> float:
        """Compute local relevance score matching CTI tactics against active local episodes."""
        if not stix_report.get("is_valid", False) or not episodes:
            return 0.0

        cti_tactics = set(stix_report.get("tactics_found", []))
        if not cti_tactics:
            return 0.0

        matching_episodes = 0
        for ep in episodes:
            if ep.tactics_covered.intersection(cti_tactics):
                matching_episodes += 1

        return round(matching_episodes / len(episodes), 4)
