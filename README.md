# TrustMesh XDR

TrustMesh XDR is a research proof of concept for privacy-aware collaborative
cybersecurity detection. The current implementation contains dataset ingestion
and normalization code, a local classifier, and controlled federated-learning
experiments. It is not a production security platform.

## Implemented

- Streaming ingestion adapters for CICIOT23, CICIoMT2024 WiFi/MQTT CSV,
  UNSW-NB15 named and headerless partition CSVs, and CTU-SME Zeek connection
  logs. The adapters validate source schemas and retain label provenance.
- Sanitization and canonicalization steps that exclude configured identifiers
  and restricted fields from canonical shared features. Dataset source files
  are read only; the pipeline yields records and does not write datasets.
- A local binary Logistic Regression baseline using canonical JSONL records.
- A Flower FedAvg proof of concept with dataset-derived CTU-SME and UNSW-NB15
  clients, both in-memory and in separate processes communicating over
  localhost. These are controlled experiments, not independent organizations.
- An experimental client-update clipping and Gaussian-noise mode. It has no
  privacy accountant or formal epsilon-DP guarantee.
- A controlled two-client secure-aggregation proof of concept using
  deterministic pairwise masks. It is not production-grade cryptographic secure
  aggregation.
- Initial MITRE ATT&CK and STIX threat-intelligence work: a qualified,
  dataset-label-derived association of CICIoMT2024 `ARP_Spoofing` with
  ATT&CK T1557.002 and a STIX 2.1 bundle. The association is not independently
  verified from packet behavior. No Sigma rule has been generated.

## Data and tests

Dataset inputs and generated canonical, sanitized, and staging artifacts are
local and ignored by Git. The default federated examples expect prepared
smoke-test JSONL files under `data/canonical/smoke_test/`; those generated
inputs are not committed.

Install dependencies from `requirements.txt` in a Python environment that also
has `pytest`, then run the test suite with:

```sh
python -m pytest -q
```
