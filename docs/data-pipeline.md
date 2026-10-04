# Data ingestion, sanitization, and normalization

## Scope

Phase 1 supports streaming readers for CICIOT23 CSV, CICIoMT2024 WiFi/MQTT CSV, UNSW-NB15 named train/test CSV, UNSW-NB15 headerless partition CSV, and CTU-SME Zeek `conn.log`. CICIoMT PCAP/Bluetooth ingestion is out of scope. Separate from this pipeline, `src/federated/simulated_fedavg.py` implements a controlled Flower FedAvg proof of concept. Its experimental clipping/noise has no formal epsilon-DP guarantee, and its two-client masking experiment is not production-grade cryptographic secure aggregation.

Existing `data/` source files are inputs only. The implementation does not move, edit, or delete source data and `iter_pipeline` does not write generated datasets.

## Data flow

`raw/source → staging → sanitized → canonical → optional client routing`

Adapters yield staging records with distinct `features`, `labels`, `provenance`, `quality`, and optional `restricted` channels. Sanitization moves configured restricted feature fields out of features. Normalization applies explicit field mappings and keeps unmapped values under `source_native.*`; restricted values are removed before canonical records are yielded. The last pipeline hook can attach caller-assigned `client_id` provenance for routing. This pipeline has no file writer or built-in partition strategy; the separate federated proof of concept consumes prepared client JSONL inputs.

## Schema and policy

Dataset configuration files declare ordered schemas, split rules, labels, and canonical mappings. Header mismatches, wrong field counts, and malformed Zeek metadata raise `ParseError`; adapters never silently align columns or drop rows. Headerless UNSW files use the ordered dictionary in `unsw_nb15_partitions.json`. Zeek parsing validates `#separator`, `#fields`, and `#types` against configuration.

Original labels remain in `labels.original`; normalized labels and `label_binary` are separate. Current normalized labels use a dataset-prefixed, case-folded source label because cross-dataset class equivalence has not been established. This preserves source distinctions and is not an ATT&CK mapping. Labels are not features. `record_id` is a deterministic internal identifier derived from dataset ID, source path, and row ordinal; it is provenance-only and never part of `features`.

Endpoint IPs from the UNSW headerless and Zeek inputs are held in `restricted` and are not canonical model features. There is no hashing. Existing source files remain untouched. The sanitization policy also excludes IDs, labels, and TCP base sequence numbers from shared features.

## Streaming and diagnostics

CSV adapters yield one row at a time using Python's standard-library CSV reader. Zeek logs are read line by line. `chunk_size` is an interface hint reserved for later vectorized readers; current adapters are row-streaming. Validation reports keep aggregate counts and at most a configured number of short examples. Parser exceptions include source path and row/line where available. No diagnostic emits full rows.

## Running fixture tests

From the repository root, run `python -m pytest -q` in an environment with the dependencies in `requirements.txt` and pytest installed. Adapter and pipeline tests use small temporary examples; federated proof-of-concept tests also need NumPy and Flower. Do not point `iter_pipeline` at real datasets until the label and schema caveats are reviewed.

## Phase 1 limitations

- Label mappings intentionally preserve dataset-local class names; a reviewed taxonomy is still needed for cross-dataset attack classes.
- Numeric parsing is conservative and unit conversions/scaling are not performed.
- CTU field/type definitions are checked against the observed conn log schema; other CTU archive members have not been inventoried.
- The named UNSW schema has no IP fields. Headerless UNSW endpoints remain restricted; ports remain candidate behavior features.
- The pipeline returns canonical records in memory and has no dataset writer, client partitioner, or full-data execution path yet.
