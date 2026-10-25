# TrustMesh XDR Phase 2 Triage Experiment

## Research question

Can the system rank genuine simulated attack incidents into a fixed Top-50 analyst queue?

## Dataset

The experiment uses `data/canonical/smoke_test/ctu_sme_conn_labeled.jsonl`, a local, ignored canonical artifact derived from CTU-SME Zeek connection data. The script keeps records with binary labels and complete, finite, nonnegative values for `duration`, `bytes_src_to_dst`, `bytes_dst_to_src`, `packets_src_to_dst`, and `packets_dst_to_src`. Each selected feature receives `log1p` preprocessing. The bounded input is the first 10,000 eligible records; 9,690 qualified for this run.

## Experimental design

The random seed is 42. A stratified 80/20 split is applied before model fitting. The existing `LocalDetectorBaseline` (StandardScaler + class-weighted LogisticRegression) is fitted on 7,752 rows and predicts probabilities only for the separate 1,938-row held-out partition. Alerts are created from held-out predictions at probability `>= 0.5`. Training and evaluation record IDs are checked for overlap by the script.

## Ground truth

The public dataset does not provide genuine enterprise incident IDs. This experiment therefore uses **controlled simulation ground truth**, not real-world incident ground truth. The canonical `provenance.source_row` is divided into fixed, non-overlapping windows of 100 source rows. Every window containing one or more held-out records labeled as attacks is defined as one simulated incident. This deterministic grouping is independent of predictions and does not use random IDs.

Ground-truth IDs and labels are attached only as evaluation metadata. The ranker uses the detector confidence and the existing fixed risk score. Correlation uses a separate `correlation_key` derived from the same fixed source-row window. It does not use source IPs, destination IPs, ports, protocol, timestamp, ATT&CK technique or the ground-truth incident ID. Those fields are absent from the selected canonical records. Alert timestamp is represented as unavailable (`None`); grouping is window-key based, not time based.

## Pipeline

Held-out record → local detector probability → alert if probability is at least 0.5 → correlate alerts sharing a controlled 100-source-row window → existing episode risk ranker → first 50 episodes.

The incident Recall@50 formula is:

```text
unique manifest incident IDs represented in the Top-50
------------------------------------------------------
all unique incident IDs in held-out ground-truth manifest
```

Precision@50 is the fraction of returned Top-50 episodes containing at least one held-out attack. The queue contained 50 episodes, so its denominator is 50.

## Results

Run from the repository root with:

```sh
.venv/bin/python -m scripts.evaluate_triage_pipeline
```

| Metric | Result |
|---|---:|
| Seed | 42 |
| Eligible dataset rows | 9,690 |
| Training rows (attack / benign) | 7,752 (5,438 / 2,314) |
| Evaluation rows (attack / benign) | 1,938 (1,359 / 579) |
| Held-out alerts | 1,360 |
| Correlated episodes | 72 |
| Controlled true incidents in manifest | 71 |
| Top-50 episodes returned | 50 |
| Incidents surfaced in Top-50 | 49 |
| Incident Recall@50 | 0.6901408451 (49 / 71) |
| Precision@50 | 0.98 (49 / 50) |

These are the observed outputs of the command above in the audited workspace; the result depends on the local ignored canonical artifact and installed dependency versions.

## Limitations

- This is a controlled simulation, not real SOC incidents.
- The input represents simulated organization data; it does not involve independently operated organizations.
- Public dataset labels are used as controlled ground truth where applicable.
- The 100-source-row grouping is a benchmark convention, not evidence that records in a window belong to one real incident.
- Correlation uses that controlled window key, not temporal or network-entity evidence.
- The experiment makes no claim of production deployment, production detection quality, privacy guarantees or operational analyst savings.
