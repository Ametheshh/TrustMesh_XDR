"""FastAPI REST service for TrustMesh XDR triage queue, CTI validation, and system metrics."""

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.correlation.engine import EpisodeCorrelator
from src.correlation.models import CanonicalAlert
from src.cti.validator import CTIValidator
from src.detection.local_baseline import LocalDetectorBaseline, load_feature_rows
from src.triage.ranker import EpisodeRanker

app = FastAPI(
    title="TrustMesh XDR Triage & Intelligence API",
    description="Privacy-Bounded Federated Incident Correlation & Analyst Triage Service",
    version="2.0.0",
)

base_dir = Path(__file__).resolve().parent.parent.parent
validator = CTIValidator(min_confidence=50)


class STIXValidationRequest(BaseModel):
    bundle: Dict[str, Any]


_CACHED_DEMO_STATE = None

def _get_demo_triage_state():
    """Generate active correlated episodes and ranked triage queue for demo endpoints."""
    global _CACHED_DEMO_STATE
    if _CACHED_DEMO_STATE is not None:
        return _CACHED_DEMO_STATE
    ciciot_path = base_dir / "data" / "canonical" / "smoke_test" / "ciciot23_train.jsonl"
    cats = []
    if ciciot_path.exists():
        X, y, _ = load_feature_rows(ciciot_path, max_rows=1000)
        detector = LocalDetectorBaseline()
        detector.fit(X, y)
        probs = detector.predict_proba(X)
    else:
        unsw_csv = base_dir / "data" / "UNSW-NB15" / "UNSW_NB15_testing-set.csv"
        if not unsw_csv.exists():
            return [], [], 0
        import csv
        X, y = [], []
        c0, c1 = 0, 0
        with open(unsw_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = int(row["label"])
                if lbl == 0 and c0 < 250:
                    c0 += 1
                elif lbl == 1 and c1 < 250:
                    c1 += 1
                else:
                    continue
                dur = float(row["dur"])
                sbytes = float(row["sbytes"])
                dbytes = float(row["dbytes"])
                spkts = float(row["spkts"])
                dpkts = float(row["dpkts"])
                X.append([dur, sbytes, dbytes, spkts, dpkts])
                y.append(lbl)
                cats.append(row.get("attack_cat", "Attack"))
                if c0 >= 250 and c1 >= 250:
                    break
        detector = LocalDetectorBaseline()
        detector.fit(X, y)
        probs = detector.predict_proba(X)

    alerts = []
    for idx, (prob, label) in enumerate(zip(probs, y)):
        tactic = cats[idx] if (cats and label == 1) else ("Execution" if label == 1 else "Benign")
        alert = CanonicalAlert(
            alert_id=f"ALT-{idx:04d}",
            timestamp=float(idx * 3.0),
            src_ip=f"DEMO-SRC-{idx % 10}",
            dst_ip=f"DEMO-DST-{(idx % 3) + 1}",
            src_port=1000 + idx,
            dst_port=80,
            protocol="tcp",
            detection_confidence=float(prob),
            is_attack=bool(label == 1),
            tactic=tactic,
            incident_id=f"INC-{label}",
        )
        alerts.append(alert)

    correlator = EpisodeCorrelator(time_window_seconds=120.0)
    episodes = correlator.correlate_alerts(alerts)

    ranker = EpisodeRanker(abstention_threshold=0.05)
    ranked_queue = ranker.rank_episodes(episodes, top_k=50)

    _CACHED_DEMO_STATE = (episodes, ranked_queue, len(alerts))
    return _CACHED_DEMO_STATE


DASHBOARD_FILE = base_dir / "docs" / "dashboard.html"

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def root_dashboard() -> str:
    """Render the research dashboard and controlled demo triage console."""
    if DASHBOARD_FILE.exists():
        return DASHBOARD_FILE.read_text(encoding="utf-8")
    return "<h1>TrustMesh XDR Dashboard</h1>"




@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "TrustMesh XDR API", "version": "2.0.0"}


@app.get("/api/triage/queue")
def get_triage_queue() -> Dict[str, Any]:
    """Get the Top-50 ranked analyst triage investigation queue."""
    episodes, ranked_queue, raw_count = _get_demo_triage_state()
    queue_list = []
    for ep, risk_score in ranked_queue:
        queue_list.append(
            {
                "episode_id": ep.episode_id,
                "risk_score": round(risk_score, 4),
                "primary_entity": ep.primary_entity,
                "alert_count": ep.alert_count,
                "max_confidence": round(ep.max_confidence, 4),
                "stage_coverage": ep.stage_coverage,
                "tactics_covered": sorted(list(ep.tactics_covered)),
                "is_true_incident": ep.is_true_incident,
            }
        )

    return {
        "top_k_capacity": 50,
        "returned_episodes": len(queue_list),
        "data_status": "controlled_demo",
        "research_metric_source": "not_this_queue",
        "queue": queue_list,
    }


@app.get("/api/triage/episode/{episode_id}")
def get_episode_detail(episode_id: str) -> Dict[str, Any]:
    """Get detailed evidence breakdown for a specific attack episode."""
    episodes, ranked_queue, _ = _get_demo_triage_state()
    for ep, risk_score in ranked_queue:
        if ep.episode_id == episode_id:
            return {
                "episode_id": ep.episode_id,
                "risk_score": round(risk_score, 4),
                "primary_entity": ep.primary_entity,
                "start_time": ep.start_time,
                "end_time": ep.end_time,
                "duration": ep.duration,
                "evidence_diversity": ep.evidence_diversity,
                "tactics_covered": sorted(list(ep.tactics_covered)),
                "alerts": [
                    {
                        "alert_id": a.alert_id,
                        "timestamp": a.timestamp,
                        "src_ip": a.src_ip,
                        "dst_ip": a.dst_ip,
                        "src_port": a.src_port,
                        "dst_port": a.dst_port,
                        "detection_confidence": round(a.detection_confidence, 4),
                        "tactic": a.tactic,
                    }
                    for a in ep.alerts
                ],
            }
    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found in active triage queue")


@app.post("/api/cti/validate")
def validate_stix_threat_intel(request: STIXValidationRequest) -> Dict[str, Any]:
    """Validate a STIX 2.1 threat intelligence bundle and evaluate local relevance."""
    report = validator.validate_stix_bundle(request.bundle)
    episodes, _, _ = _get_demo_triage_state()
    relevance = validator.compute_local_relevance(report, episodes)
    report["local_relevance_score"] = relevance
    return report


@app.get("/api/model/unified")
def get_unified_model() -> Dict[str, Any]:
    """Return the master multi-domain federated model combining all client models."""
    path = base_dir / "data" / "models" / "unified_federated_master_model.json"
    if not path.exists():
        from scripts.build_unified_model import aggregate_models
        return aggregate_models()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/metrics/summary")
def get_metrics_summary() -> Dict[str, Any]:
    """Return verified Phase 2 held-out Local-vs-Federated triage results.

    These values come from the completed three-seed experiment.
    They are benchmark results, not real-world SOC performance.
    """

    return {
        "data_status": "verified_phase2_experiment",

        "primary_endpoint": {
            "metric": "Incident Recall@50",
            "dataset": "CTU-SME",
            "condition": "Federated",
            "value": 0.693228,
            "percentage": 69.32,
            "seeds": [42, 43, 44],
            "interpretation": (
                "Controlled held-out benchmark result. Incident ground truth "
                "is derived from public dataset labels and deterministic "
                "source-row windows; these are not real SOC incidents."
            ),
        },

        "triage_evaluation": {
            "ctu_sme": {
                "local_recall_at_50": 0.683838,
                "federated_recall_at_50": 0.693228,
                "random_recall_at_50": 0.521191,
            },
            "unsw_nb15": {
                "local_recall_at_50": 0.508486,
                "federated_recall_at_50": 0.501718,
                "random_recall_at_50": 0.501684,
            },
        },

        "federated_delta_at_50": {
            "ctu_sme_absolute": 0.009390,
            "unsw_nb15_absolute": -0.006768,
            "unit": "absolute recall",
        },

        "milestone_recovery": {
            "unsw_nb15": {
                "m5_global_f1": 0.0,
                "m7_global_f1": 0.6946,
                "interpretation": (
                    "Shared normalization removed the M5 all-benign "
                    "failure pattern but did not close the gap to the "
                    "UNSW local model."
                ),
            },
        },

        "privacy_status": {
            "dp": (
                "Experimental clipping and Gaussian perturbation POC; "
                "no formal epsilon-DP guarantee."
            ),
            "secure_aggregation": (
                "Two-client pairwise masking proof-of-concept; "
                "not production secure aggregation."
            ),
        },

        "limitations": [
            "Dataset-derived simulated clients.",
            "Controlled benchmark incident ground truth.",
            "UNSW evaluated split is highly attack-heavy.",
            "Results do not establish real-world SOC performance.",
        ],
    }
