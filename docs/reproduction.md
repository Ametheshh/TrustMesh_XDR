# Reproducing the TrustMesh XDR demonstrations

Run commands from the repository root. This guide documents the current tools and their local inputs; it does not add an orchestrator or create dataset artifacts.

## 1. Environment setup

The repository has `requirements.txt` for runtime dependencies. `pytest` is not listed there and must be installed separately for the test command.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install pytest
```

These commands install the dependencies declared by the checked-in `requirements.txt`; exact versions are not fully locked.

## 2. Local data and controlled smoke artifacts

Dataset inputs and canonical artifacts are excluded from Git by `.gitignore`. A fresh checkout does not contain them. Obtain the dataset files separately and place them under the source roots configured in `configs/datasets/` (including `data/CICIOT23/`, `data/CTU-SME/conn/`, and `data/UNSW-NB15/`). CICIoMT2024 WiFi/MQTT sources use the root in `configs/datasets/ciciomt2024_wifi_mqtt.json`.

The current repository **does not have an artifact-preparation script or file writer**. `src/pipeline.py` exposes `iter_pipeline`, which streams canonical records in memory and does not write JSONL. The dataset-backed demo scripts consume already prepared artifacts; they do not generate those artifacts from source data. Therefore a fresh checkout cannot reproduce those demonstrations from source datasets using a documented repository command alone. Provide the local artifacts below before continuing:

| Demonstration | Required local canonical artifact(s) |
|---|---|
| Local baseline | `data/canonical/smoke_test/ciciot23_train.jsonl` |
| One-feature in-memory federated POC | `data/canonical/smoke_test/ctu_unsw_fl_poc/ctu_sme_total_packets.jsonl`; `data/canonical/smoke_test/ctu_unsw_fl_poc/unsw_nb15_named_total_packets.jsonl` |
| Milestones 5 and 7, plus SHAP | `data/canonical/smoke_test/ctu_sme_conn_labeled.jsonl`; `data/canonical/smoke_test/unsw_nb15_named_train.jsonl` |

Milestone 4's report documents a three-seed one-feature evaluation, but no dedicated Milestone 4 evaluation script is present. The one-feature POC command below is runnable, but should not be presented as a reproduction of every Milestone 4 table. The five-feature Milestone 5 and 7 scripts are present.

You can check the three artifact groups before running the corresponding demos:

```sh
test -f data/canonical/smoke_test/ciciot23_train.jsonl
test -f data/canonical/smoke_test/ctu_unsw_fl_poc/ctu_sme_total_packets.jsonl
test -f data/canonical/smoke_test/ctu_unsw_fl_poc/unsw_nb15_named_total_packets.jsonl
test -f data/canonical/smoke_test/ctu_sme_conn_labeled.jsonl
test -f data/canonical/smoke_test/unsw_nb15_named_train.jsonl
```

## 3. Local baseline

After the CICIOT23 canonical artifact is available, run the existing baseline module. It uses the artifact named above by default and prints its holdout metrics and majority-class baseline as JSON.

```sh
.venv/bin/python -m src.detection.local_baseline
```

## 4. Federated demonstrations

Run the existing one-feature, one-round in-memory Flower FedAvg proof of concept. It reads the two `total_packets` artifacts listed above.

```sh
.venv/bin/python -m src.federated.simulated_fedavg --mode in-memory
```

The default clients are dataset-derived simulations, not separate organizations. The default command does not enable the clipping/noise or masking experiments.

## 5. Milestone evaluation scripts

Milestone 4 has a report but no retained dedicated three-seed evaluation script. Its documented one-feature setup should not be conflated with the five-feature evaluators below.

Milestone 5 evaluates the five shared features on separate held-out CTU-SME and UNSW-NB15 sets over seeds 42, 43, and 44:

```sh
.venv/bin/python -m scripts.evaluate_shared_feature_federation
```

Milestone 7 runs the documented shared-normalization ablation with the same local inputs:

```sh
.venv/bin/python -m scripts.evaluate_shared_normalization_ablation
```

Both commands require `ctu_sme_conn_labeled.jsonl` and `unsw_nb15_named_train.jsonl`. They print JSON results; the corresponding checked-in reports are [Milestone 5](federated_shared_features_milestone5.md) and [Milestone 7](federated_shared_normalization_milestone7.md). Milestone 6's read-only diagnosis is documented in [Milestone 6](federated_heterogeneity_milestone6.md); no new evaluation command is associated with it.

## 6. Threat-intelligence artifact inspection

The mapping and STIX bundle are static files; no server is required. Inspect the qualified ATT&CK mapping and validate/display the STIX JSON:

```sh
cat docs/threat-intelligence/ciciomt2024_arp_spoofing.json
.venv/bin/python -m json.tool docs/threat-intelligence/ciciomt2024_arp_spoofing.stix.json
```

The ATT&CK association is dataset-label-derived. No Sigma rule is generated because the available data does not establish a defensible event/log field condition. The repository contains no TAXII server.

## 7. SHAP demonstration

With the five-feature CTU-SME and UNSW-NB15 artifacts in place, run the existing CTU-SME local-model explainer:

```sh
.venv/bin/python -m scripts.explain_local_model_shap
```

It prints a compact JSON explanation for selected held-out examples and global importance. It explains one fitted model's behavior; it does not establish causation or improve accuracy. The report is [SHAP examples](shap_local_model_examples.md).

## 8. Test suite

Run the repository tests:

```sh
.venv/bin/python -m pytest -q
```

The tests use temporary fixtures for adapters and pipeline behavior and a deterministic temporary JSONL fixture for the one-feature data-loader check. They do not require the ignored smoke artifacts used by the dataset-backed evaluation commands.
