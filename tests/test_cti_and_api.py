"""Unit tests for CTI validation engine and FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api import main as api_main
from src.api.main import app
from src.correlation.models import AttackEpisode, CanonicalAlert
from src.cti.validator import CTIValidator

client = TestClient(app)


def test_cti_validator_stix_bundle():
    """Verify STIX 2.1 bundle validation and quality scoring."""
    validator = CTIValidator(min_confidence=50)

    sample_stix_bundle = {
        "type": "bundle",
        "id": "bundle--12345",
        "objects": [
            {
                "type": "indicator",
                "id": "indicator--001",
                "pattern": "[file:hashes.MD5 = 'd41d8cd98f00b204e9800998ecf8427e']",
                "confidence": 80,
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--002",
                "name": "Execution",
                "confidence": 90,
            },
        ],
    }

    report = validator.validate_stix_bundle(sample_stix_bundle)
    assert report["is_valid"] is True
    assert report["indicator_count"] == 1
    assert report["attack_pattern_count"] == 1
    assert "Execution" in report["tactics_found"]
    assert report["quality_score"] > 0.5


def test_cti_validator_local_relevance():
    """Verify local relevance score calculation."""
    validator = CTIValidator()
    stix_report = {
        "is_valid": True,
        "tactics_found": ["Execution", "Exfiltration"],
    }

    ep1 = AttackEpisode("EP-1")
    ep1.tactics_covered = {"Execution"}

    ep2 = AttackEpisode("EP-2")
    ep2.tactics_covered = {"Reconnaissance"}

    relevance = validator.compute_local_relevance(stix_report, [ep1, ep2])
    assert relevance == 0.5  # 1 out of 2 episodes matched


def test_api_health_endpoint():
    """Test /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_triage_queue_endpoint():
    """Test /api/triage/queue endpoint."""
    response = client.get("/api/triage/queue")
    assert response.status_code == 200
    data = response.json()
    assert "queue" in data
    assert data["top_k_capacity"] == 50


def test_dashboard_restores_demo_queue_and_inspection_controls():
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "SIMULATED CONTROLLED DEMO — NOT LIVE TELEMETRY" in response.text
    assert 'id="triage-body"' in response.text
    assert 'id="refresh-queue"' in response.text
    assert 'id="episode-modal"' in response.text
    assert "/api/triage/queue" in response.text
    assert "/api/triage/episode/" in response.text
    assert "cdn.tailwindcss.com" not in response.text
    assert "prefers-reduced-motion" in response.text


def test_api_episode_inspection_returns_demo_alert_details(monkeypatch):
    episode = AttackEpisode("EP-0001")
    episode.add_alert(
        CanonicalAlert(
            alert_id="ALT-0001",
            timestamp=3.0,
            src_ip="DEMO-SRC-1",
            dst_ip="DEMO-DST-1",
            src_port=1001,
            dst_port=80,
            protocol="tcp",
            detection_confidence=0.85,
            is_attack=True,
            tactic="Execution",
        )
    )
    monkeypatch.setattr(
        api_main,
        "_get_demo_triage_state",
        lambda: ([episode], [(episode, 0.75)], 1),
    )

    queue = client.get("/api/triage/queue").json()
    details = client.get("/api/triage/episode/EP-0001").json()

    assert queue["data_status"] == "controlled_demo"
    assert queue["queue"][0]["episode_id"] == "EP-0001"
    assert queue["queue"][0]["risk_score"] == 0.75
    assert details["alerts"][0]["alert_id"] == "ALT-0001"
    assert details["alerts"][0]["timestamp"] == 3.0
    assert details["alerts"][0]["tactic"] == "Execution"


def test_api_cti_validate_endpoint():
    """Test POST /api/cti/validate endpoint."""
    sample_stix_bundle = {
        "type": "bundle",
        "id": "bundle--test",
        "objects": [{"type": "indicator", "confidence": 85}],
    }
    response = client.post("/api/cti/validate", json={"bundle": sample_stix_bundle})
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True



def test_api_metrics_summary_endpoint():
    """Test /api/metrics/summary endpoint."""
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()

    assert "triage_evaluation" in data
    assert "privacy_status" in data

    assert "no formal epsilon-DP guarantee" in data["privacy_status"]["dp"]
    assert "not production secure aggregation" in data["privacy_status"]["secure_aggregation"]
    assert "privacy_rdp_accounting" not in data
