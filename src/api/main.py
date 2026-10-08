"""FastAPI REST service for TrustMesh XDR triage queue, CTI validation, and system metrics."""

from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
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


def _get_demo_triage_state():
    """Generate active correlated episodes and ranked triage queue for demo endpoints."""
    ciciot_path = base_dir / "data" / "canonical" / "smoke_test" / "ciciot23_train.jsonl"
    if not ciciot_path.exists():
        return [], [], 0

    X, y, _ = load_feature_rows(ciciot_path, max_rows=1000)
    detector = LocalDetectorBaseline()
    detector.fit(X, y)
    probs = detector.predict_proba(X)

    alerts = []
    for idx, (prob, label) in enumerate(zip(probs, y)):
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
            tactic="Execution" if label == 1 else "Benign",
            incident_id=f"INC-{label}",
        )
        alerts.append(alert)

    correlator = EpisodeCorrelator(time_window_seconds=120.0)
    episodes = correlator.correlate_alerts(alerts)

    ranker = EpisodeRanker(abstention_threshold=0.05)
    ranked_queue = ranker.rank_episodes(episodes, top_k=50)

    return episodes, ranked_queue, len(alerts)


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def root_dashboard() -> str:
    """Render interactive SOC Triage Dashboard UI."""
    return """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustMesh XDR | SOC Triage Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 500: '#00f0ff', 600: '#00c3d0', 900: '#061325' },
                        dark: { 800: '#0d192b', 900: '#070f1e', 950: '#030712' }
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background-color: #030712; color: #f3f4f6; }
        .glass { background: rgba(13, 25, 43, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    </style>
</head>
<body class="min-h-screen bg-dark-950 text-gray-100 flex flex-col">
    <!-- Header -->
    <header class="border-b border-gray-800 bg-dark-900/80 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-9 w-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
                    XDR
                </div>
                <div>
                    <h1 class="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">TrustMesh XDR</h1>
                    <p class="text-xs text-gray-400">Privacy-Bounded Federated Incident Correlation Platform</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-2 h-2 mr-2 rounded-full bg-emerald-400 animate-pulse"></span> Service Active (127.0.0.1:8000)
                </span>
                <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-gray-800 hover:bg-gray-700 text-gray-200 transition">
                    Swagger API Docs
                </a>
            </div>
        </div>
    </header>

    <!-- Main Content Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-6 py-8 space-y-8">

        <!-- KPI Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div class="glass p-6 rounded-xl border border-gray-800 shadow-lg">
                <p class="text-xs font-semibold uppercase tracking-wider text-cyan-400">Primary Endpoint</p>
                <h3 id="kpi-recall" class="text-3xl font-extrabold text-white mt-2">69.32%</h3>
                <p class="text-xs text-gray-400 mt-1">CTU-SME Federated Recall@50</p>
            </div>
            <div class="glass p-6 rounded-xl border border-gray-800 shadow-lg">
                <p class="text-xs font-semibold uppercase tracking-wider text-emerald-400">Federated Effect</p>
                <h3 id="kpi-dedup" class="text-3xl font-extrabold text-white mt-2">+0.94 pp</h3>
                <p class="text-xs text-gray-400 mt-1">CTU-SME vs Local at K=50</p>
            </div>
            <div class="glass p-6 rounded-xl border border-gray-800 shadow-lg">
                <p class="text-xs font-semibold uppercase tracking-wider text-purple-400">Detection Metric</p>
                <h3 class="text-3xl font-extrabold text-white mt-2">N/A</h3>
                <p class="text-xs text-gray-400 mt-1">Not reported in Phase 2</p>
            </div>
            <div class="glass p-6 rounded-xl border border-gray-800 shadow-lg">
                <p class="text-xs font-semibold uppercase tracking-wider text-amber-400">Privacy Status</p>
                <h3 id="kpi-dp" class="text-3xl font-extrabold text-white mt-2">POC</h3>
                <p class="text-xs text-gray-400 mt-1">Experimental clipping + Gaussian perturbation; no formal ε-DP guarantee</p>
            </div>
        </div>

        <!-- Triage Queue Table Section -->
        <div class="glass rounded-xl border border-gray-800 shadow-xl overflow-hidden">
            <div class="px-6 py-5 border-b border-gray-800 flex items-center justify-between bg-dark-900/50">
                <div>
                    <h2 class="text-lg font-bold text-white flex items-center">
                        <svg class="w-5 h-5 mr-2 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                        Controlled Demonstration Triage Queue
                    </h2>
                    <p class="text-xs text-gray-400 mt-0.5">UI demonstration only; not the source of the validated Phase 2 held-out metrics</p>
                </div>
                <button onclick="fetchTriageQueue()" class="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-dark-950 font-bold text-xs rounded-lg transition shadow-md shadow-cyan-500/10 flex items-center">
                    <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                    Refresh Queue
                </button>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="bg-gray-900/60 text-xs uppercase text-gray-400 border-b border-gray-800">
                        <tr>
                            <th class="px-6 py-3.5 font-semibold">Rank</th>
                            <th class="px-6 py-3.5 font-semibold">Episode ID</th>
                            <th class="px-6 py-3.5 font-semibold">Demo Entity</th>
                            <th class="px-6 py-3.5 font-semibold">Risk Score</th>
                            <th class="px-6 py-3.5 font-semibold">Alert Count</th>
                            <th class="px-6 py-3.5 font-semibold">Detection Stage</th>
                            <th class="px-6 py-3.5 font-semibold text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody id="triage-body" class="divide-y divide-gray-800/60 text-gray-300">
                        <tr>
                            <td colspan="7" class="px-6 py-12 text-center text-gray-500">
                                Loading live triage queue from API...
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- CTI Validator & Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">

            <!-- STIX CTI Validator -->
            <div class="glass p-6 rounded-xl border border-gray-800 shadow-xl space-y-4">
                <h3 class="text-base font-bold text-white flex items-center">
                    <svg class="w-5 h-5 mr-2 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011 1h2a1 1 0 011 1v5m-4 0h4"></path></svg>
                    Federated Evaluation Snapshot
                </h3>

                <p class="text-xs text-gray-400">
                    Three-seed held-out triage benchmark. No unsupported PFL performance claims.
                </p>

                <div class="space-y-3">
                    <div class="p-3.5 rounded-lg bg-dark-900/80 border border-gray-800/80 flex justify-between items-center">
                        <div>
                            <p class="text-xs font-bold text-cyan-400">CTU-SME</p>
                            <p class="text-[11px] text-gray-400">Federated vs Local Incident Recall@50</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-cyan-500/10 text-cyan-400 text-xs font-bold border border-cyan-500/20">+0.94 pp</span>
                    </div>

                    <div class="p-3.5 rounded-lg bg-dark-900/80 border border-gray-800/80 flex justify-between items-center">
                        <div>
                            <p class="text-xs font-bold text-emerald-400">UNSW-NB15</p>
                            <p class="text-[11px] text-gray-400">Federated vs Local Incident Recall@50</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 text-xs font-bold border border-emerald-500/20">−0.68 pp</span>
                    </div>

                    <div class="p-3.5 rounded-lg bg-dark-900/80 border border-gray-800/80 flex justify-between items-center">
                        <div>
                            <p class="text-xs font-bold text-purple-400">UNSW M5 → M7</p>
                            <p class="text-[11px] text-gray-400">Shared normalization recovery from all-benign collapse</p>
                        </div>
                        <span class="px-2.5 py-1 rounded bg-purple-500/10 text-purple-400 text-xs font-bold border border-purple-500/20">Recovery observed</span>
                    </div>
                </div>

        </div>

    </main>

    <!-- Episode Detail Modal -->
    <div id="episode-modal" class="fixed inset-0 bg-black/80 backdrop-blur-sm hidden flex items-center justify-center z-50 p-4">
        <div class="glass max-w-2xl w-full rounded-2xl border border-gray-800 p-6 space-y-6 max-h-[85vh] overflow-y-auto">
            <div class="flex justify-between items-center border-b border-gray-800 pb-4">
                <h3 id="modal-title" class="text-lg font-bold text-white">Episode Detail</h3>
                <button onclick="closeModal()" class="text-gray-400 hover:text-white font-bold text-xl">&times;</button>
            </div>
            <div id="modal-body" class="space-y-4 text-xs text-gray-300">
                Loading details...
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script>
        async function fetchMetricsSummary() {
            try {
                const res = await fetch('/api/metrics/summary');
                const data = await res.json();
                if (data.primary_endpoint) {
                    document.getElementById('kpi-recall').innerText =
                        `${data.primary_endpoint.percentage.toFixed(2)}%`;
                }
            } catch (err) {
                console.error("Error loading metrics:", err);
            }
        }

        async function fetchTriageQueue() {
            try {
                const res = await fetch('/api/triage/queue');
                const data = await res.json();
                const tbody = document.getElementById('triage-body');
                tbody.innerHTML = '';

                if (!data.queue || data.queue.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-8 text-center text-gray-500">No active triage episodes found.</td></tr>';
                    return;
                }

                data.queue.forEach((item, index) => {
                    const badgeClass = item.risk_score >= 0.7 ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                                      (item.risk_score >= 0.4 ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20');

                    const filteredTactics = item.tactics_covered.filter(t => t !== 'Benign' && t !== 'Unknown');
                    const tacticsBadges = filteredTactics.length > 0
                        ? filteredTactics.map(t => `<span class="inline-block bg-purple-500/10 text-purple-300 border border-purple-500/20 px-2 py-0.5 rounded text-[10px] mr-1.5 font-semibold">${t}</span>`).join('')
                        : '<span class="text-gray-500 text-xs">Standard Telemetry</span>';

                    const tr = document.createElement('tr');
                    tr.className = 'hover:bg-gray-800/30 transition';
                    tr.innerHTML = `
                        <td class="px-6 py-4 font-bold text-gray-400">#${index + 1}</td>
                        <td class="px-6 py-4 font-mono text-cyan-400 font-semibold">${item.episode_id}</td>
                        <td class="px-6 py-4 font-mono text-gray-200">${item.primary_entity}</td>
                        <td class="px-6 py-4">
                            <span class="px-2.5 py-1 rounded-full border text-xs font-bold ${badgeClass}">
                                ${item.risk_score} Risk
                            </span>
                        </td>
                        <td class="px-6 py-4 text-gray-300">${item.alert_count} alerts</td>
                        <td class="px-6 py-4">${tacticsBadges}</td>
                        <td class="px-6 py-4 text-right">
                            <button onclick="inspectEpisode('${item.episode_id}')" class="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-cyan-400 rounded-lg text-xs font-semibold transition">
                                Inspect
                            </button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                console.error("Error fetching queue:", err);
            }
        }

        async function inspectEpisode(episodeId) {
            const modal = document.getElementById('episode-modal');
            const body = document.getElementById('modal-body');
            document.getElementById('modal-title').innerText = `Episode Inspection: ${episodeId}`;
            modal.classList.remove('hidden');

            try {
                const res = await fetch(`/api/triage/episode/${episodeId}`);
                const ep = await res.json();
                body.innerHTML = `
                    <div class="grid grid-cols-2 gap-4 bg-dark-900 p-4 rounded-lg border border-gray-800">
                        <div><span class="text-gray-500">Primary Entity:</span> <strong class="text-cyan-400">${ep.primary_entity}</strong></div>
                        <div><span class="text-gray-500">Risk Score:</span> <strong class="text-amber-400">${ep.risk_score}</strong></div>
                        <div><span class="text-gray-500">Duration:</span> ${ep.duration}s</div>
                        <div><span class="text-gray-500">Evidence Diversity:</span> ${ep.evidence_diversity} items</div>
                    </div>
                    <div>
                        <h4 class="font-bold text-white mb-2">Constituent Alerts (${ep.alerts.length})</h4>
                        <div class="space-y-2 max-h-48 overflow-y-auto pr-2">
                            ${ep.alerts.map(a => `
                                <div class="p-2.5 rounded bg-dark-900 border border-gray-800/80 flex justify-between items-center">
                                    <div>
                                        <p class="font-mono text-cyan-400 text-[11px]">${a.alert_id} &bull; ${a.src_ip}:${a.src_port} &rarr; ${a.dst_ip}:${a.dst_port}</p>
                                        <p class="text-[10px] text-gray-500">Tactic: ${a.tactic}</p>
                                    </div>
                                    <span class="font-bold text-emerald-400 text-[11px]">Conf: ${(a.detection_confidence * 100).toFixed(1)}%</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            } catch (err) {
                body.innerText = "Error loading episode details.";
            }
        }

        function closeModal() {
            document.getElementById('episode-modal').classList.add('hidden');
        }

        async function validateCTI() {
            const input = document.getElementById('stix-input').value;
            const resDiv = document.getElementById('cti-result');
            resDiv.classList.remove('hidden');
            resDiv.innerHTML = 'Validating intelligence bundle...';

            try {
                const bundle = JSON.parse(input);
                const res = await fetch('/api/cti/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ bundle })
                });
                const data = await res.json();
                resDiv.innerHTML = `
                    <div class="flex justify-between font-bold text-emerald-400 mb-1">
                        <span>Validation Status: ${data.is_valid ? 'VALID STIX 2.1' : 'INVALID'}</span>
                        <span>Quality Score: ${data.quality_score}</span>
                    </div>
                    <p class="text-gray-400">Indicators: ${data.indicator_count} | Tactics Found: ${data.tactics_found ? data.tactics_found.join(', ') : 'None'}</p>
                    <p class="text-cyan-400 font-semibold mt-1">Local Observability Relevance: ${(data.local_relevance_score * 100).toFixed(1)}%</p>
                `;
            } catch (err) {
                resDiv.innerHTML = `<span class="text-red-400">Invalid JSON format: ${err.message}</span>`;
            }
        }

        // Auto-fetch on load
        fetchMetricsSummary();
        fetchTriageQueue();
    </script>
</body>
</html>"""


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
