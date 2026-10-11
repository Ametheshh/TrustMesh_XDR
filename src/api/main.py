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
    """Render the research dashboard and controlled demo triage console."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#11120f">
    <title>TrustMesh XDR | Experimental Operations Console</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #11120f; --panel: #191a16; --panel-raised: #20211c; --line: #44443b;
            --green: #b6f36a; --amber: #f2b65d; --red: #f16e62; --ink: #f0efe6;
            --muted: #aaa99c; --dim: #75756b; --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
            --display: "Arial Narrow", "Roboto Condensed", Impact, sans-serif;
        }
        * { box-sizing: border-box; }
        body { margin: 0; min-width: 320px; background-color: var(--bg); color: var(--ink); font-family: system-ui, sans-serif; }
        body::before { position: fixed; inset: 0; z-index: -1; content: ""; pointer-events: none; opacity: .13;
            background: repeating-linear-gradient(0deg, transparent 0 3px, #000 4px), linear-gradient(#b6f36a0a 1px, transparent 1px), linear-gradient(90deg, #b6f36a0a 1px, transparent 1px); background-size: auto, 32px 32px, 32px 32px; }
        button, textarea { font: inherit; }
        button { color: inherit; }
        a { color: var(--green); }
        .shell { width: min(1440px, 100% - 40px); margin: 0 auto; }
        .topbar { border-bottom: 1px solid var(--line); background: #11120fee; }
        .topbar-inner { min-height: 76px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }
        .identity { display: flex; align-items: center; gap: 14px; }
        .mark { width: 46px; height: 46px; display: grid; place-items: center; border: 2px solid var(--green); color: var(--green); font: 800 13px var(--mono); transform: rotate(-3deg); box-shadow: 4px 4px 0 #b6f36a33; }
        .eyebrow, .micro { color: var(--muted); font: 700 10px var(--mono); letter-spacing: .12em; text-transform: uppercase; }
        h1, h2, h3, p { margin-top: 0; }
        h1 { margin-bottom: 3px; font: 800 clamp(22px, 3vw, 30px)/1 var(--display); letter-spacing: .035em; text-transform: uppercase; }
        .identity p { margin: 0; color: var(--muted); font: 11px var(--mono); }
        .top-actions { display: flex; align-items: center; gap: 16px; }
        .status-chip { display: flex; align-items: center; gap: 8px; color: var(--muted); font: 10px var(--mono); text-transform: uppercase; }
        .status-light { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px #b6f36a66; }
        .button { min-height: 38px; padding: 8px 13px; border: 1px solid var(--green); background: var(--green); color: #171910; font: 800 11px var(--mono); letter-spacing: .02em; text-transform: uppercase; cursor: pointer; box-shadow: 3px 3px 0 #000; transition: transform .15s, box-shadow .15s, background .15s; }
        .button:hover { transform: translate(-1px, -1px); box-shadow: 5px 5px 0 #000; }
        .button:active { transform: translate(2px, 2px); box-shadow: 1px 1px 0 #000; }
        .button:disabled { cursor: wait; opacity: .65; }
        .button.secondary { border-color: var(--line); background: var(--panel-raised); color: var(--ink); }
        .button.ghost { border-color: var(--line); background: transparent; color: var(--muted); }
        .console { padding: 30px 0 48px; }
        .masthead { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 260px); align-items: end; gap: 24px; margin: 4px 0 24px; }
        .masthead h2 { max-width: 760px; margin: 7px 0 0; font: 800 clamp(32px, 6vw, 64px)/.92 var(--display); letter-spacing: -.025em; text-transform: uppercase; }
        .masthead h2 span { color: var(--green); }
        .mast-meta { padding: 12px 14px; border-left: 2px solid var(--amber); color: var(--muted); font: 11px/1.8 var(--mono); }
        .mast-meta strong { color: var(--amber); font-weight: 700; }
        .layout { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 16px; align-items: start; }
        .panel { min-width: 0; border: 1px solid var(--line); background: var(--panel); box-shadow: 5px 5px 0 #0008; }
        .panel-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; padding: 17px 18px; border-bottom: 1px solid var(--line); }
        .panel-head h2, .panel-head h3 { margin: 4px 0 0; font: 800 20px/1 var(--display); letter-spacing: .04em; text-transform: uppercase; }
        .panel-index { color: var(--green); font: 11px var(--mono); }
        .benchmark { grid-column: span 7; }
        .benchmark-inner { padding: 19px; }
        .metric-primary { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 16px; padding: 17px; border: 1px solid var(--green); background: #b6f36a0a; }
        .metric-primary .value { color: var(--green); font: 800 clamp(44px, 7vw, 76px)/.9 var(--mono); letter-spacing: -.08em; }
        .metric-primary .metric-label { margin-bottom: 9px; color: var(--ink); font: 700 11px var(--mono); text-transform: uppercase; }
        .metric-note { max-width: 230px; color: var(--muted); font: 10px/1.5 var(--mono); text-align: right; }
        .metric-deltas { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 11px; }
        .delta { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 12px; border-left: 2px solid var(--line); background: var(--panel-raised); }
        .delta-label { color: var(--muted); font: 10px var(--mono); text-transform: uppercase; }
        .delta-value { color: var(--ink); font: 700 13px var(--mono); }
        .source-note { margin: 14px 0 0; color: var(--dim); font: 10px/1.5 var(--mono); }
        .research-note { grid-column: span 5; padding: 19px; border-color: #8e6633; }
        .research-note h2 { margin: 6px 0 12px; font: 800 23px/1 var(--display); text-transform: uppercase; }
        .research-note p { margin: 0; color: var(--muted); font: 12px/1.7 var(--mono); }
        .privacy-line { margin-top: 14px !important; padding-top: 12px; border-top: 1px solid var(--line); color: var(--amber) !important; }
        .queue { grid-column: span 8; }
        .queue-tools { display: flex; align-items: center; gap: 14px; }
        .queue-body { padding: 0 18px 14px; }
        .disclaimer { display: flex; align-items: center; gap: 9px; margin: 16px 0 11px; padding: 10px 12px; border: 1px solid #8a332d; background: #f16e620c; color: #ff9186; font: 700 10px var(--mono); letter-spacing: .04em; }
        .disclaimer::before { content: "!"; display: grid; flex: 0 0 17px; height: 17px; place-items: center; border: 1px solid currentColor; }
        .queue-status { min-height: 20px; margin: 0 0 8px; color: var(--muted); font: 10px var(--mono); }
        .table-wrap { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th { padding: 10px 9px; border-bottom: 1px solid var(--line); color: var(--dim); font: 700 9px var(--mono); letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; }
        td { padding: 12px 9px; border-bottom: 1px solid #383830; color: var(--muted); font: 11px var(--mono); vertical-align: middle; }
        tbody tr:hover { background: #ffffff06; }
        .rank { color: var(--dim); }
        .episode-id { color: var(--green); font-weight: 700; }
        .risk { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; }
        .risk::before { width: 7px; height: 7px; content: ""; background: var(--green); }
        .risk.medium::before { background: var(--amber); }
        .risk.high::before { background: var(--red); }
        .inspect { padding: 6px 8px; border: 1px solid var(--line); background: transparent; color: var(--green); font: 700 10px var(--mono); text-transform: uppercase; cursor: pointer; }
        .inspect:hover { border-color: var(--green); background: #b6f36a0c; }
        .empty-row { padding: 28px 10px; color: var(--muted); text-align: center; }
        .cti { grid-column: span 4; }
        .cti-body { padding: 17px; }
        .cti-body p { color: var(--muted); font: 11px/1.6 var(--mono); }
        textarea { width: 100%; min-height: 150px; resize: vertical; padding: 11px; border: 1px solid var(--line); outline: none; background: #10110e; color: var(--green); font: 11px/1.5 var(--mono); }
        textarea:focus { border-color: var(--green); }
        .cti-actions { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 10px; }
        .cti-result { min-height: 20px; margin-top: 12px; color: var(--muted); font: 11px/1.6 var(--mono); overflow-wrap: anywhere; }
        dialog { width: min(760px, calc(100% - 28px)); max-height: min(84vh, 780px); padding: 0; border: 1px solid var(--green); background: var(--panel); color: var(--ink); box-shadow: 10px 10px 0 #000b; }
        dialog::backdrop { background: #050604dd; }
        .inspect-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 17px 19px; border-bottom: 1px solid var(--line); }
        .inspect-head h2 { margin: 5px 0 0; color: var(--green); font: 800 25px var(--mono); }
        .inspect-body { max-height: calc(84vh - 80px); overflow-y: auto; padding: 17px 19px; }
        .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 0 0 18px; }
        .detail-cell { min-width: 0; padding: 10px; border: 1px solid var(--line); background: #10110e; }
        .detail-cell dt { color: var(--dim); font: 9px var(--mono); letter-spacing: .08em; text-transform: uppercase; }
        .detail-cell dd { margin: 6px 0 0; color: var(--ink); font: 12px var(--mono); overflow-wrap: anywhere; }
        .section-label { margin: 0 0 8px; color: var(--amber); font: 700 10px var(--mono); letter-spacing: .08em; text-transform: uppercase; }
        .alert-list { display: grid; gap: 7px; }
        .alert-row { display: grid; grid-template-columns: minmax(95px, .7fr) minmax(0, 2fr) auto; gap: 10px; align-items: start; padding: 10px; border-left: 2px solid var(--line); background: #10110e; font: 10px/1.5 var(--mono); }
        .alert-row strong { color: var(--green); overflow-wrap: anywhere; }
        .alert-row span { color: var(--muted); overflow-wrap: anywhere; }
        .footer { padding: 18px 0 0; color: var(--dim); font: 10px var(--mono); }
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
        :focus-visible { outline: 2px solid var(--amber); outline-offset: 3px; }
        @media (max-width: 900px) { .benchmark, .research-note { grid-column: span 12; } .queue { grid-column: span 12; } .cti { grid-column: span 12; } .cti-body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; } .cti-body > p { grid-column: 1 / -1; margin: 0; } }
        @media (max-width: 600px) { .shell { width: min(100% - 24px, 1440px); } .topbar-inner { min-height: 66px; } .top-actions .status-chip { display: none; } .console { padding-top: 23px; } .masthead { grid-template-columns: 1fr; gap: 14px; } .mast-meta { max-width: 300px; } .panel-head { padding: 14px; } .queue-tools { gap: 6px; } .button { min-height: 35px; padding: 7px 9px; font-size: 9px; } .queue-body { padding: 0 12px 12px; } table { min-width: 570px; } .metric-primary { grid-template-columns: 1fr; } .metric-note { text-align: left; max-width: none; } .metric-deltas { grid-template-columns: 1fr; } .cti-body { display: block; } .cti-body > p { margin: 0 0 12px; } .alert-row { grid-template-columns: 1fr auto; } .alert-row span:nth-child(2) { grid-column: 1 / -1; grid-row: 2; } }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; } }
    </style>
</head>
<body>
<header class="topbar">
    <div class="shell topbar-inner">
        <div class="identity">
            <div class="mark" aria-hidden="true">TXR</div>
            <div><h1>TrustMesh XDR</h1><p>Experimental cyber-defense console / research build</p></div>
        </div>
        <div class="top-actions">
            <span class="status-chip"><span class="status-light" aria-hidden="true"></span>API-backed view</span>
            <a class="button secondary" href="/docs">API index ↗</a>
        </div>
    </header>
<main class="shell console">
    <section class="masthead" aria-labelledby="page-title">
        <div><div class="eyebrow">Research console / 02—XDR</div><h2 id="page-title">Signals in the <span>dark.</span><br>Evidence in view.</h2></div>
        <div class="mast-meta"><div>MODE / <strong>OFFLINE RESEARCH</strong></div><div>FEED / <strong>CONTROLLED DEMO</strong></div><div>DATA / REQUEST-DRIVEN</div></div>
    </section>
    <div class="layout">
        <section class="panel benchmark" aria-labelledby="benchmark-title">
            <div class="panel-head"><div><div class="micro">01 / measured output</div><h2 id="benchmark-title">Federated evaluation</h2></div><span class="panel-index">BENCHMARK</span></div>
            <div class="benchmark-inner">
                <div class="metric-primary">
                    <div><div class="metric-label">CTU-SME Federated Recall@50</div><div class="value" id="kpi-recall" aria-live="polite">Loading…</div></div>
                    <div class="metric-note">HELD-OUT TRIAGE BENCHMARK<br>NOT LIVE TELEMETRY</div>
                </div>
                <div class="metric-deltas">
                    <div class="delta"><span class="delta-label">Federated Effect / CTU-SME vs Local @50</span><span id="kpi-dedup" class="delta-value" aria-live="polite">Loading…</span></div>
                    <div class="delta"><span class="delta-label">CTU-SME / Recall Δ @50</span><span id="ctu-delta" class="delta-value">Loading…</span></div>
                    <div class="delta"><span class="delta-label">UNSW-NB15 / Recall Δ @50</span><span id="unsw-delta" class="delta-value">Loading…</span></div>
                </div>
                <p id="metrics-status" class="source-note" role="status">Values load from GET /api/metrics/summary.</p>
            </div>
        </section>
        <aside class="panel research-note" aria-labelledby="scope-title">
            <div class="micro">02 / scope markers</div><h2 id="scope-title">Keep the signal honest.</h2>
            <p>Benchmark metrics are fixed results from the documented held-out experiment. The queue below is generated from a local dataset artifact each time the API is requested; it is not saved operational data.</p>
            <p class="privacy-line">PRIVACY / Experimental clipping + Gaussian perturbation. No formal differential-privacy guarantee.</p>
        </aside>
        <section class="panel queue" aria-labelledby="queue-title">
            <div class="panel-head">
                <div><div class="micro">03 / generated records</div><h2 id="queue-title">Controlled demonstration triage queue</h2></div>
                <div class="queue-tools"><span id="queue-count" class="micro" aria-live="polite"></span><button id="refresh-queue" class="button" type="button">↻ Refresh</button></div>
            </div>
            <div class="queue-body">
                <div class="disclaimer">SIMULATED CONTROLLED DEMO — NOT LIVE TELEMETRY</div>
                <p id="queue-status" class="queue-status" role="status">Loading queue from GET /api/triage/queue…</p>
                <div class="table-wrap">
                    <table>
                        <caption class="sr-only">Ranked episodes generated by the controlled CICIoT23 demonstration</caption>
                        <thead><tr><th scope="col">Rank</th><th scope="col">Episode ID</th><th scope="col">Demo entity</th><th scope="col">Calculated demo episode risk</th><th scope="col">Alerts</th><th scope="col">Signals</th><th scope="col" aria-label="Inspection"></th></tr></thead>
                        <tbody id="triage-body"><tr><td colspan="7" class="empty-row">Requesting demo queue…</td></tr></tbody>
                    </table>
                </div>
            </div>
        </section>
        <section class="panel cti" aria-labelledby="cti-title">
            <div class="panel-head"><div><div class="micro">04 / local utility</div><h2 id="cti-title">STIX validation</h2></div><span class="panel-index">CTI</span></div>
            <div class="cti-body">
                <p>Validate a supplied STIX 2.1 bundle and compare its tactics with the controlled demo queue. No external feed is contacted.</p>
                <label class="sr-only" for="stix-input">STIX 2.1 bundle JSON</label>
                <textarea id="stix-input" spellcheck="false" placeholder='{"type":"bundle","id":"bundle--…","objects":[]}'></textarea>
                <div><div class="cti-actions"><button id="validate-cti" class="button secondary" type="button">Validate bundle</button><span class="micro">LOCAL API</span></div><div id="cti-result" class="cti-result" role="status" aria-live="polite"></div></div>
            </div>
        </section>
    </div>
    <footer class="footer">TrustMesh XDR / research proof of concept / no live telemetry</footer>
</main>
<dialog id="episode-modal" aria-labelledby="modal-title">
    <div class="inspect-head"><div><div class="micro">Episode detail / generated demo record</div><h2 id="modal-title">Inspection</h2></div><button id="close-modal" class="button secondary" type="button" aria-label="Close episode inspection" autofocus>Close ×</button></div>
    <div id="modal-body" class="inspect-body"></div>
</dialog>
<script>
    const setText = (id, value) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    };

        async function fetchMetricsSummary() {
            try {
                const res = await fetch('/api/metrics/summary');
                if (!res.ok) throw new Error(`Metrics API returned ${res.status}`);
                const data = await res.json();
                const ctu = data.triage_evaluation?.ctu_sme;
                const unsw = data.triage_evaluation?.unsw_nb15;
                if (!data.primary_endpoint || !ctu || !unsw) {
                    throw new Error("Metrics response is incomplete");
                }
                const formatDelta = (federated, local) => {
                    const delta = (federated - local) * 100;
                    return `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)} pp`;
                };
                setText('kpi-recall', `${(data.primary_endpoint.value * 100).toFixed(2)}%`);
                setText('kpi-dedup', formatDelta(ctu.federated_recall_at_50, ctu.local_recall_at_50));
                setText('ctu-delta', formatDelta(ctu.federated_recall_at_50, ctu.local_recall_at_50));
                setText('unsw-delta', formatDelta(unsw.federated_recall_at_50, unsw.local_recall_at_50));
                setText('metrics-status', `Source: GET /api/metrics/summary · ${data.primary_endpoint.dataset} / ${data.primary_endpoint.condition} · ${data.primary_endpoint.seeds.join(', ')} · fixed held-out benchmark`);
            } catch (err) {
                console.error("Error loading metrics:", err);
                for (const id of ['kpi-recall', 'kpi-dedup', 'ctu-delta', 'unsw-delta']) {
                    setText(id, 'Unavailable');
                }
                setText('metrics-status', 'Benchmark metrics unavailable from the API.');
            }
        }

        async function fetchTriageQueue() {
            const tbody = document.getElementById('triage-body');
            const status = document.getElementById('queue-status');
            const button = document.getElementById('refresh-queue');
            if (!tbody || !status) return;
            button.disabled = true;
            status.textContent = 'Refreshing controlled demo queue…';
            tbody.replaceChildren();
            try {
                const res = await fetch('/api/triage/queue');
                if (!res.ok) throw new Error(`Triage API returned ${res.status}`);
                const data = await res.json();
                if (!Array.isArray(data.queue)) throw new Error('Triage response is missing its queue');
                setText('queue-count', `${data.queue.length} / ${data.top_k_capacity} EPISODES`);
                if (data.queue.length === 0) {
                    showQueueMessage('No demo episodes available. Check that the local source artifact is present.');
                    status.textContent = `No episodes · ${data.data_status || 'unknown source'}`;
                    return;
                }
                data.queue.forEach((item, index) => {
                    const risk = Number(item.risk_score);
                    const tr = document.createElement('tr');
                    const cell = (className, text) => {
                        const td = document.createElement('td');
                        td.className = className;
                        td.textContent = text;
                        return td;
                    };
                    tr.append(cell('rank', String(index + 1).padStart(2, '0')));
                    tr.append(cell('episode-id', item.episode_id));
                    tr.append(cell('', item.primary_entity));
                    const riskCell = document.createElement('td');
                    const marker = document.createElement('span');
                    marker.className = `risk${risk >= 0.7 ? ' high' : risk >= 0.4 ? ' medium' : ''}`;
                    marker.textContent = `${Number.isFinite(risk) ? risk.toFixed(4) : '—'} / calculated demo risk`;
                    riskCell.append(marker);
                    tr.append(riskCell, cell('', `${item.alert_count} alerts`));
                    const tactics = Array.isArray(item.tactics_covered)
                        ? item.tactics_covered.filter(tactic => tactic !== 'Benign' && tactic !== 'Unknown').join(', ')
                        : '';
                    tr.append(cell('', tactics || 'No attack tactic reported'));
                    const actionCell = document.createElement('td');
                    const inspect = document.createElement('button');
                    inspect.className = 'inspect';
                    inspect.type = 'button';
                    inspect.textContent = 'Inspect';
                    inspect.setAttribute('aria-label', `Inspect episode ${item.episode_id}`);
                    inspect.addEventListener('click', () => inspectEpisode(item.episode_id));
                    actionCell.append(inspect);
                    tr.append(actionCell);
                    tbody.appendChild(tr);
                });
                status.textContent = `Source: local CICIoT23 artifact · ${data.data_status || 'controlled demo'} · generated on request; not persisted`;
            } catch (err) {
                console.error("Error fetching queue:", err);
                setText('queue-count', '');
                showQueueMessage('Queue unavailable. Check the API and local demo artifact, then refresh.');
                status.textContent = 'Could not load the controlled demo queue.';
            } finally {
                button.disabled = false;
            }
        }

        function showQueueMessage(message) {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.className = 'empty-row';
            cell.colSpan = 7;
            cell.textContent = message;
            row.append(cell);
            document.getElementById('triage-body').replaceChildren(row);
        }

        async function inspectEpisode(episodeId) {
            const modal = document.getElementById('episode-modal');
            const body = document.getElementById('modal-body');
            setText('modal-title', episodeId);
            body.textContent = 'Loading episode details…';
            if (!modal.open) modal.showModal();

            try {
                const res = await fetch(`/api/triage/episode/${encodeURIComponent(episodeId)}`);
                if (!res.ok) throw new Error(`Episode API returned ${res.status}`);
                const ep = await res.json();
                body.replaceChildren();
                const grid = document.createElement('dl');
                grid.className = 'detail-grid';
                const fields = [
                    ['Primary entity', ep.primary_entity],
                    ['Calculated demo episode risk', ep.risk_score],
                    ['Generated start offset', ep.start_time == null ? null : `T+${Number(ep.start_time).toFixed(3)}s`],
                    ['Generated end offset', ep.end_time == null ? null : `T+${Number(ep.end_time).toFixed(3)}s`],
                    ['Generated duration', ep.duration == null ? null : `${ep.duration}s`],
                    ['Evidence diversity', ep.evidence_diversity],
                    ['Tactics covered', Array.isArray(ep.tactics_covered) ? ep.tactics_covered.join(', ') : null],
                ];
                fields.forEach(([label, value]) => {
                    if (value == null) return;
                    const item = document.createElement('div');
                    item.className = 'detail-cell';
                    const term = document.createElement('dt');
                    term.textContent = label;
                    const detail = document.createElement('dd');
                    detail.textContent = String(value);
                    item.append(term, detail);
                    grid.append(item);
                });
                body.append(grid);
                const sectionLabel = document.createElement('h3');
                sectionLabel.className = 'section-label';
                sectionLabel.textContent = `Constituent alerts · ${Array.isArray(ep.alerts) ? ep.alerts.length : 0}`;
                body.append(sectionLabel);
                const list = document.createElement('div');
                list.className = 'alert-list';
                (Array.isArray(ep.alerts) ? ep.alerts : []).forEach(alert => {
                    const row = document.createElement('div');
                    row.className = 'alert-row';
                    const id = document.createElement('strong');
                    id.textContent = alert.alert_id;
                    const endpoints = document.createElement('span');
                    endpoints.textContent = `${alert.src_ip}:${alert.src_port} → ${alert.dst_ip}:${alert.dst_port}`;
                    const metadata = document.createElement('span');
                    const offset = alert.timestamp == null ? 'time unavailable' : `T+${Number(alert.timestamp).toFixed(3)}s`;
                    metadata.textContent = `${offset} · ${alert.tactic} · ${(Number(alert.detection_confidence) * 100).toFixed(1)}% confidence`;
                    row.append(id, endpoints, metadata);
                    list.append(row);
                });
                body.append(list);
            } catch (err) {
                console.error("Error fetching episode:", err);
                body.textContent = 'Episode details unavailable. Close this inspection and refresh the queue.';
            }
        }

        async function validateCTI() {
            const input = document.getElementById('stix-input').value;
            const resDiv = document.getElementById('cti-result');
            resDiv.textContent = 'Validating bundle…';

            try {
                const bundle = JSON.parse(input);
                const res = await fetch('/api/cti/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ bundle })
                });
                if (!res.ok) throw new Error(`CTI API returned ${res.status}`);
                const data = await res.json();
                resDiv.textContent = `Validation: ${data.is_valid ? 'VALID STIX 2.1' : 'INVALID'} · Quality ${data.quality_score} · Indicators ${data.indicator_count} · Tactics ${(data.tactics_found || []).join(', ') || 'None'} · Demo relevance ${((data.local_relevance_score || 0) * 100).toFixed(1)}%`;
            } catch (err) {
                console.error("Error validating STIX bundle:", err);
                resDiv.textContent = err instanceof SyntaxError ? `Invalid JSON: ${err.message}` : 'Validation unavailable. Check the bundle and CTI API.';
            }
        }

        document.getElementById('refresh-queue').addEventListener('click', fetchTriageQueue);
        document.getElementById('close-modal').addEventListener('click', () => document.getElementById('episode-modal').close());
        const modal = document.getElementById('episode-modal');
        modal.addEventListener('click', event => {
            if (event.target === event.currentTarget) modal.close();
        });
        document.addEventListener('keydown', event => {
            if (event.key === 'Escape' && modal.open) modal.close();
        });
        document.getElementById('validate-cti').addEventListener('click', validateCTI);
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
