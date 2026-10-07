# Shared-feature local-versus-federated evaluation (Milestone 5)

## Compatibility decision

All five requested features are supported by explicit source-to-canonical mappings in `configs/datasets/ctu_sme_zeek.json` and `configs/datasets/unsw_nb15_named.json`:

| Canonical feature | CTU-SME source | UNSW-NB15 source |
|---|---|---|
| `duration` | `duration` | `dur` |
| `bytes_src_to_dst` | `orig_bytes` | `sbytes` |
| `bytes_dst_to_src` | `resp_bytes` | `dbytes` |
| `packets_src_to_dst` | `orig_pkts` | `spkts` |
| `packets_dst_to_src` | `resp_pkts` | `dpkts` |

Both mappings explicitly encode the same connection duration and direction-specific byte/packet counts. The canonical artifacts represent these as numeric values. An audit found no nonnumeric, nonfinite, or negative values for these features. CTU has 72 records with missing values in `duration` and both directional byte fields; 70 have binary labels and are excluded as incomplete. The other 2 also have unknown labels and are excluded as nonbinary. CTU has 240 rows with no binary label in total. UNSW has no missing feature values or missing binary labels. Missing values are not imputed as zero.

## Experiment configuration

- Inputs: `data/canonical/smoke_test/ctu_sme_conn_labeled.jsonl` and `data/canonical/smoke_test/unsw_nb15_named_train.jsonl`.
- Features: exactly the five listed above, in that order.
- Target: existing `labels.binary` (`0=benign/normal`, `1=attack/malicious`). Rows with unknown labels or any missing selected feature are excluded before splitting.
- Preprocessing: `log1p` applied separately to each validated nonnegative feature; no fitted statistics or evaluation-set information is used.
- Model: the existing class-weighted binary logistic regression with full-batch gradient descent, 500 epochs, learning rate `0.1`, L2 `1e-4`, and prediction at logit `>= 0`.
- Split: stratified 80/20 independently for each client, seeds `42`, `43`, `44`.
- Federation: one in-memory Flower FedAvg round, weighted by each client's training row count. Local models and the global model use the same input representation and model code. Evaluation sets remain separate.
- Confusion matrices use `[[TN, FP], [FN, TP]]`. Mean ± standard deviation below uses sample standard deviation across the three seeds.

The reusable evaluator is `scripts/evaluate_shared_feature_federation.py`, runnable from the repository root with:

```sh
.venv/bin/python -m scripts.evaluate_shared_feature_federation
```

## Sample counts and class distribution

| Client | Source rows | Excluded | Eligible (0 / 1) | Train (0 / 1) | Evaluation (0 / 1) |
|---|---:|---:|---:|---:|---:|
| CTU-SME | 10,000 | 310 | 9,690 (2,893 / 6,797) | 7,752 (2,314 / 5,438) | 1,938 (579 / 1,359) |
| UNSW-NB15 | 10,000 | 0 | 10,000 (243 / 9,757) | 8,000 (194 / 7,806) | 2,000 (49 / 1,951) |

Counts are identical across the three seeds because the split is stratified. CTU's 310 excluded rows comprise 240 rows with unknown binary labels and 70 labeled rows with missing selected features.

## Per-seed held-out results

| Seed | Model → client evaluation set | Precision | Recall | F1 | Balanced accuracy | Confusion matrix |
|---:|---|---:|---:|---:|---:|---|
| 42 | CTU local → CTU | 0.9963 | 0.9978 | 0.9971 | 0.9946 | `[[574,5],[3,1356]]` |
| 42 | Global → CTU | 0.9963 | 0.9978 | 0.9971 | 0.9946 | `[[574,5],[3,1356]]` |
| 42 | UNSW local → UNSW | 0.9891 | 0.9288 | 0.9580 | 0.7603 | `[[29,20],[139,1812]]` |
| 42 | Global → UNSW | 0.0000 | 0.0000 | 0.0000 | 0.5000 | `[[49,0],[1951,0]]` |
| 43 | CTU local → CTU | 0.9934 | 0.9963 | 0.9949 | 0.9904 | `[[570,9],[5,1354]]` |
| 43 | Global → CTU | 0.9941 | 0.9963 | 0.9952 | 0.9913 | `[[571,8],[5,1354]]` |
| 43 | UNSW local → UNSW | 0.9890 | 0.9190 | 0.9527 | 0.7554 | `[[29,20],[158,1793]]` |
| 43 | Global → UNSW | 0.0000 | 0.0000 | 0.0000 | 0.5000 | `[[49,0],[1951,0]]` |
| 44 | CTU local → CTU | 0.9978 | 0.9978 | 0.9978 | 0.9963 | `[[576,3],[3,1356]]` |
| 44 | Global → CTU | 0.9978 | 0.9978 | 0.9978 | 0.9963 | `[[576,3],[3,1356]]` |
| 44 | UNSW local → UNSW | 0.9900 | 0.9165 | 0.9518 | 0.7746 | `[[31,18],[163,1788]]` |
| 44 | Global → UNSW | 0.0000 | 0.0000 | 0.0000 | 0.5000 | `[[49,0],[1951,0]]` |

## Mean ± sample standard deviation

| Model → client evaluation set | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| CTU local → CTU | 0.9958 ± 0.0022 | 0.9973 ± 0.0008 | 0.9966 ± 0.0015 | 0.9938 ± 0.0030 |
| Global → CTU | 0.9961 ± 0.0018 | 0.9973 ± 0.0008 | 0.9967 ± 0.0013 | 0.9940 ± 0.0026 |
| UNSW local → UNSW | 0.9894 ± 0.0006 | 0.9214 ± 0.0065 | 0.9542 ± 0.0033 | 0.7634 ± 0.0099 |
| Global → UNSW | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |

## Interpretation and limitations

- On CTU, FedAvg is effectively unchanged from the CTU-local model across these splits: mean F1 and balanced accuracy differ by less than `0.0002`.
- On UNSW, FedAvg fails severely: it predicts every held-out row as benign in all three seeds, yielding zero attack recall and F1. The UNSW-local model performs substantially better. The global result therefore represents a strong client-specific regression, not a useful cross-client improvement.
- This result is consistent with incompatible client distributions and local updates under a single shared decision boundary, but the experiment does not isolate the cause. No extra rounds, tuning, or alternative aggregation were tried.
- UNSW is highly imbalanced: the evaluation set has only 49 benign rows. CTU also excludes rows without labels or complete selected features, so reported metrics apply only to eligible records.
- These are dataset-derived simulated clients, not real organizations. Results cover two fixed smoke artifacts, one model setup, and three stratified partitions; they do not establish production performance or privacy properties.
- Source datasets and permanent canonical schemas were not modified.
