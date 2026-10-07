# Shared-normalization ablation (Milestone 7)

## Configuration

This is one controlled ablation on the same five-feature canonical inputs and eligible rows as Milestone 5. Features are `duration`, `bytes_src_to_dst`, `bytes_dst_to_src`, `packets_src_to_dst`, and `packets_dst_to_src`. Rows without binary labels or any required feature are excluded as before.

- Seeds: 42, 43, 44; deterministic stratified 80/20 splits independently per client.
- Preprocessing: `log1p`, then one `StandardScaler`-equivalent fit per seed using only the concatenated CTU and UNSW training partitions. Population standard deviation (`ddof=0`) is used; one is substituted for any zero scale. That same mean and scale transform both clients' training and held-out rows.
- Model: existing class-weighted logistic regression; 500 epochs, learning rate `0.1`, L2 `1e-4`, threshold at logit `>= 0`.
- Federation: one in-memory Flower FedAvg round, weighted by each client's training-example count. No other training or aggregation settings changed.

Eligible samples and split counts are unchanged from Milestone 5:

| Client | Eligible (benign / attack) | Train (benign / attack) | Evaluation (benign / attack) |
|---|---:|---:|---:|
| CTU-SME | 9,690 (2,893 / 6,797) | 7,752 (2,314 / 5,438) | 1,938 (579 / 1,359) |
| UNSW-NB15 | 10,000 (243 / 9,757) | 8,000 (194 / 7,806) | 2,000 (49 / 1,951) |

## Per-seed held-out results

Confusion matrices use `[[TN, FP], [FN, TP]]`; class `1` is positive.

| Seed | Model → evaluation client | Precision | Recall | F1 | Balanced accuracy | Confusion matrix | UNSW attack predictions |
|---:|---|---:|---:|---:|---:|---|---:|
| 42 | CTU local → CTU | 0.9883 | 0.9978 | 0.9930 | 0.9851 | `[[563,16],[3,1356]]` | — |
| 42 | CTU global → CTU | 0.9862 | 0.9978 | 0.9920 | 0.9825 | `[[560,19],[3,1356]]` | — |
| 42 | UNSW local → UNSW | 0.9891 | 0.9267 | 0.9569 | 0.7593 | `[[29,20],[143,1808]]` | 1,808 |
| 42 | UNSW global → UNSW | 0.9942 | 0.5259 | 0.6879 | 0.7017 | `[[43,6],[925,1026]]` | 1,026 |
| 43 | CTU local → CTU | 0.9890 | 0.9963 | 0.9927 | 0.9852 | `[[564,15],[5,1354]]` | — |
| 43 | CTU global → CTU | 0.9869 | 0.9963 | 0.9916 | 0.9826 | `[[561,18],[5,1354]]` | — |
| 43 | UNSW local → UNSW | 0.9895 | 0.9154 | 0.9510 | 0.7638 | `[[30,19],[165,1786]]` | 1,786 |
| 43 | UNSW global → UNSW | 0.9990 | 0.5356 | 0.6974 | 0.7576 | `[[48,1],[906,1045]]` | 1,045 |
| 44 | CTU local → CTU | 0.9941 | 0.9978 | 0.9960 | 0.9920 | `[[571,8],[3,1356]]` | — |
| 44 | CTU global → CTU | 0.9905 | 0.9978 | 0.9941 | 0.9877 | `[[566,13],[3,1356]]` | — |
| 44 | UNSW local → UNSW | 0.9900 | 0.9118 | 0.9493 | 0.7722 | `[[31,18],[172,1779]]` | 1,779 |
| 44 | UNSW global → UNSW | 0.9981 | 0.5372 | 0.6984 | 0.7482 | `[[47,2],[903,1048]]` | 1,048 |

## Mean ± sample standard deviation

| Model → evaluation client | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| CTU local → CTU | 0.9905 ± 0.0032 | 0.9973 ± 0.0008 | 0.9939 ± 0.0018 | 0.9874 ± 0.0040 |
| CTU global → CTU | 0.9879 ± 0.0023 | 0.9973 ± 0.0008 | 0.9926 ± 0.0014 | 0.9843 ± 0.0030 |
| UNSW local → UNSW | 0.9895 ± 0.0005 | 0.9180 ± 0.0078 | 0.9524 ± 0.0040 | 0.7651 ± 0.0066 |
| UNSW global → UNSW | 0.9971 ± 0.0026 | 0.5329 ± 0.0061 | 0.6946 ± 0.0058 | 0.7358 ± 0.0299 |

The UNSW global model predicted **1,026, 1,045, and 1,048 attacks out of 2,000** for seeds 42, 43, and 44 respectively. This contrasts with Milestone 5, where it predicted zero attacks in every seed.

## Comparison with Milestone 5 and conclusion

| UNSW global metric | Milestone 5 (`log1p` only) | Milestone 7 (shared scaler) |
|---|---:|---:|
| Precision | 0.0000 ± 0.0000 | 0.9971 ± 0.0026 |
| Recall | 0.0000 ± 0.0000 | 0.5329 ± 0.0061 |
| F1 | 0.0000 ± 0.0000 | 0.6946 ± 0.0058 |
| Balanced accuracy | 0.5000 ± 0.0000 | 0.7358 ± 0.0299 |

Shared normalization **materially improves the literal all-benign collapse**: the global model detects attacks and its balanced accuracy rises by about 0.236. It does **not** close the client gap. UNSW-local F1 is 0.9524 and balanced accuracy is 0.7651; global recall remains about 0.385 lower than local recall. Shared normalization also slightly reduces CTU global performance relative to Milestone 5.

The ablation shows that scale differences contributed to the collapse, but Milestone 6's client distribution and feature-to-label heterogeneity remains. A single common scale improves the shared decision boundary without making it fit the UNSW attack relationship well. Class imbalance remains material: UNSW's evaluation set contains only 49 benign examples, and its training partition has 194.

The results support documenting heterogeneity as a limitation. Do not add rounds or continue broad FL tuning based on this ablation alone. These are dataset-derived simulated clients and fixed smoke artifacts, not independent organizations or deployment evidence.

## Reproduction

Run `.venv/bin/python -m scripts.evaluate_shared_normalization_ablation` from the repository root. The evaluator is separate from the Milestone 5 baseline and uses only training partitions to fit each seed's shared scaler.
