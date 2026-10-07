# TrustMesh XDR

TrustMesh XDR is a research proof of concept for collaborative network-threat detection. It demonstrates streaming dataset ingestion, a local classifier, controlled Flower federated-learning experiments, a qualified threat-intelligence artifact, and a small SHAP explanation report. It is not a production XDR platform and does not establish production privacy guarantees.

## Architecture and data flow

```text
local source data → dataset adapter → staging record → sanitization → canonical record → caller-controlled routing
```

The pipeline is lazy and in-memory. Adapters preserve feature, label, provenance, quality, and restricted-data channels. Sanitization and dataset mappings produce canonical records; `iter_pipeline` does not write files or train a model. Federated experiments consume separately prepared canonical JSONL artifacts. See [Data pipeline](docs/data-pipeline.md) for schema and adapter details.

## Current components

- Streaming adapters for CICIOT23, CICIoMT2024 WiFi/MQTT CSV, UNSW-NB15 named and headerless CSV, and CTU-SME Zeek connection logs.
- Canonicalization, label normalization, validation, and policy-based exclusion of identifiers and label fields from shared model features.
- A local binary Logistic Regression baseline using canonical JSONL records.
- Controlled local and federated Logistic Regression experiments using dataset-derived CTU-SME and UNSW-NB15 clients.
- A client-update clipping/noise experiment and a two-client masking experiment, both with the limitations below.
- A qualified ATT&CK association and static STIX 2.1 bundle for a CICIoMT2024 dataset label.
- A small SHAP report for one CTU-SME local model.

## Data pipeline

The source adapters validate their configured schemas and stream records. The pipeline keeps original labels separate from normalized labels and features, and removes restricted values before yielding canonical records. Source data is read-only. The pipeline does not write canonical files, partition clients, or execute full-data workflows. Dataset files and generated artifacts are local and ignored by Git; they are not included in a fresh checkout.

## Local detection

`src/detection/local_baseline.py` provides a bounded binary Logistic Regression baseline over canonical numeric features, with a stratified holdout and majority-class baseline. Its default input is `data/canonical/smoke_test/ciciot23_train.jsonl`; this artifact must exist locally. The five-feature CTU-SME SHAP example uses a separate class-weighted Logistic Regression implementation from the federated evaluation code.

## Federated learning

`src/federated/simulated_fedavg.py` demonstrates one-round Flower FedAvg using the designated one-feature `total_packets` client artifacts. The broader Milestone 5 and Milestone 7 evaluators use five mapped connection features and separate held-out sets. The default federated clients are simulated from datasets on one machine; they are not independent organizations. Results are controlled experiments, not a claim of general federation benefit or deployment performance.

## Milestones 4–8: results and research progression

Results below are from the existing reports and are not new evaluations. Mean ± standard deviation refers to the documented three seeds where applicable.

- **Milestone 4 — one-feature comparison:** With `total_packets`, CTU-SME local and global F1 were both `0.8794 ± 0.0032` and balanced accuracy both `0.6851 ± 0.0083`. On UNSW-NB15, local F1/balanced accuracy were `0.7009 ± 0.0097` / `0.6322 ± 0.0411`; global results were `0.6992 ± 0.0092` / `0.6480 ± 0.0255`, a modest metric trade-off. See [Milestone 4](docs/federated_local_global_milestone4.md).
- **Milestone 5 — five shared features:** CTU-SME local/global F1 were `0.9966 ± 0.0015` / `0.9967 ± 0.0013`. UNSW-NB15 local F1 was `0.9542 ± 0.0033`; the global model predicted every UNSW evaluation row as benign, with F1 `0` and balanced accuracy `0.5000`. See [Milestone 5](docs/federated_shared_features_milestone5.md).
- **Milestone 6 — diagnosis:** Feature distributions, seed-42 coefficients, cross-client scores, and near-even FedAvg client weights supported feature-distribution and feature-to-label heterogeneity as the diagnosis. This was not proof of a sole cause. See [Milestone 6](docs/federated_heterogeneity_milestone6.md).
- **Milestone 7 — shared-normalization ablation:** A scaler fitted on combined training partitions only changed UNSW global F1/balanced accuracy to `0.6946 ± 0.0058` / `0.7358 ± 0.0299`. This removed the all-benign prediction pattern but did not close the gap to the UNSW local model (`0.9524 ± 0.0040` F1). See [Milestone 7](docs/federated_shared_normalization_milestone7.md).
- **Milestone 8 — SHAP:** A small report explains selected held-out predictions and global feature importance for the CTU-SME local five-feature model. SHAP did not change model accuracy and is not a causal explanation. See [SHAP examples](docs/shap_local_model_examples.md).

## Privacy proof-of-concept limits

- **DP-inspired client-update clipping and noise:** the optional networked experiment clips client updates and adds seeded Gaussian noise before transmission. There is no privacy accountant or formal epsilon-DP guarantee; this must not be described as formally differentially private.
- **Secure-aggregation masking:** the two-client experiment uses deterministic pairwise masks designed to cancel under example-count-weighted FedAvg. It is not cryptographically secure secure aggregation and has no production security guarantee.
- **Local data:** the controlled localhost mode sends model arrays, example counts, and scalar evaluation summaries rather than raw records. This does not demonstrate independent organization deployment, secure transport between organizations, or a complete privacy threat model.

## Threat intelligence

The static mapping in `docs/threat-intelligence/ciciomt2024_arp_spoofing.json` associates the CICIoMT2024 `ARP_Spoofing` dataset label with MITRE ATT&CK T1557.002 (ARP Cache Poisoning). Its evidence basis is the dataset label, not independent packet-level verification. A corresponding static STIX 2.1 bundle is in `docs/threat-intelligence/ciciomt2024_arp_spoofing.stix.json`.

No Sigma rule is generated. The available dataset features do not establish a defensible mapping to an actual event/log field and condition. In particular, a dataset feature name alone is not a SIEM telemetry field. The project has no TAXII service or operational threat-intelligence exchange.

## Research findings and limitations

- The clients are dataset-derived simulations, not real organizations.
- CTU-SME and UNSW-NB15 have materially different feature distributions and feature-to-label behavior. Milestone 6 treats heterogeneity as the supported diagnosis, not a mathematically proven sole cause.
- In the five-feature Milestone 5 setup, the global model failed on UNSW-NB15 by predicting every evaluation row benign. Shared normalization improved that failure in Milestone 7, but the UNSW global model still had substantially lower recall and F1 than its local model.
- UNSW-NB15 is severely class-imbalanced: the Milestone 5 eligible set contains 243 benign and 9,757 attack rows; its held-out set contains 49 benign and 1,951 attack rows. CTU-SME also has exclusions for unknown labels or missing selected features.
- The repository does not include a dashboard, live network-event detector, alert-delivery workflow, TAXII service, formal DP accounting, or cryptographic secure aggregation.
- Dataset and canonical smoke artifacts are ignored by Git. The current repository has no artifact-preparation script or file writer; the pipeline yields records in memory. Reproduction therefore requires local source data and already prepared canonical artifacts at the documented paths.

## Reproducibility and demo

Use the ordered commands and local-data checklist in [Reproduction guide](docs/reproduction.md). The guide distinguishes the local Logistic Regression baseline, one-feature in-memory FedAvg demo, five-feature Milestone 5/7 evaluators, static threat-intelligence artifacts, SHAP example, and test suite. A clean checkout does not contain dataset inputs or generated canonical smoke artifacts.

## Testing

With the project dependencies and `pytest` installed, run:

```sh
.venv/bin/python -m pytest -q
```

The tests cover adapter and pipeline fixtures, labels, validation, local baseline behavior, federated helpers, and the threat-intelligence artifact. Dataset-backed evaluation scripts require their ignored local inputs.

## Current project status

The implemented scope is a tested research proof of concept with documented local/federated experiments, a static STIX artifact, and one limited SHAP demonstration. It is suitable for a carefully qualified final-year project/demo of that scope. It is not a complete or production-ready privacy-preserving XDR platform. Source datasets and generated artifacts are not included in Git; the demonstration commands do not modify source datasets or the permanent canonical schema.
