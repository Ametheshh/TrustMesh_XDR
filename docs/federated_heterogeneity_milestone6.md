# Federated heterogeneity diagnosis (Milestone 6)

## Purpose

Milestone 6 was a read-only diagnosis of the Milestone 5 result. It examined the five shared features, the seed-42 local and global model parameters, the one-round FedAvg weights, and client evaluation scores to understand why the global model predicted every UNSW-NB15 evaluation row as benign. It did not change the training algorithm or datasets.

## Milestone 5 failure

Milestone 5 used five features, `log1p` preprocessing, a class-weighted logistic regression model, and one in-memory Flower FedAvg round. The means and sample standard deviations below are across seeds 42, 43, and 44; evaluation sets remained separate.

| Model → evaluation client | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| CTU-SME local → CTU-SME | 0.9958 ± 0.0022 | 0.9973 ± 0.0008 | 0.9966 ± 0.0015 | 0.9938 ± 0.0030 |
| Global → CTU-SME | 0.9961 ± 0.0018 | 0.9973 ± 0.0008 | 0.9967 ± 0.0013 | 0.9940 ± 0.0026 |
| UNSW-NB15 local → UNSW-NB15 | 0.9894 ± 0.0006 | 0.9214 ± 0.0065 | 0.9542 ± 0.0033 | 0.7634 ± 0.0099 |
| Global → UNSW-NB15 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |

On CTU-SME, the global model was effectively unchanged from the local model. On UNSW-NB15, it predicted all 2,000 evaluation rows as benign in every seed. The seed-42 confusion matrix was `[[49, 0], [1951, 0]]` (`[[TN, FP], [FN, TP]]`).

## Client feature distributions

The read-only comparison used the eligible Milestone 5 records: 9,690 CTU-SME rows and 10,000 UNSW-NB15 rows. The table shows established raw-value statistics. Differences in medians and upper tails are visible across several fields; the feature shifts do not all point in the same direction.

| Feature | CTU-SME median / mean / p99 / max | UNSW-NB15 median / mean / p99 / max |
|---|---:|---:|
| `duration` | 1.016392 / 6.839713 / 171.033785 / 530.219788 | 0.000009 / 1.262719 / 28.213135 / 59.995678 |
| `bytes_src_to_dst` | 0 / 499.440970 / 6,888.08 / 433,620 | 200 / 16,100.0773 / 172,856.22 / 14,355,774 |
| `bytes_dst_to_src` | 0 / 3,380.8451 / 21,148.78 / 4,442,566 | 0 / 9,194.158 / 162,505.26 / 14,657,531 |
| `packets_src_to_dst` | 4 / 7.716821 / 62 / 2,082 | 2 / 20.2007 / 164.04 / 10,646 |
| `packets_dst_to_src` | 0 / 5.899897 / 58 / 4,160 | 0 / 12.7742 / 160.02 / 11,018 |

The existing `log1p` transformation compresses these ranges but does not make the client distributions identical. For example, the `bytes_src_to_dst` median after `log1p` is 0 for CTU-SME and 5.3033049 for UNSW-NB15; the corresponding means are 1.704452 and 6.302843. For `duration`, the `log1p` medians are 0.701310 and 0.000009, respectively. Thus feature-scale and distribution differences remained after Milestone 5 preprocessing.

## Seed-42 parameters and decision behavior

Weights are listed in feature order: `duration`, `bytes_src_to_dst`, `bytes_dst_to_src`, `packets_src_to_dst`, `packets_dst_to_src`. These are the coefficients from the existing `log1p` model; the bias is shown separately.

| Model | Coefficients | Bias |
|---|---|---:|
| CTU-SME local | `[0.730386, -1.252741, -0.681422, 1.580161, -0.258562]` | 1.047614 |
| UNSW-NB15 local | `[-1.774414, -0.052342, -0.193103, 0.780570, 0.408602]` | 0.295691 |
| FedAvg global | `[-0.541732, -0.643092, -0.433418, 1.174071, 0.080272]` | 0.665734 |

The local coefficient signs differ for `duration` and `packets_dst_to_src`; magnitudes also differ substantially for other fields. The global coefficients combine those client-specific fits. In the seed-42 cross-client score check, the CTU-SME local model predicted all 2,000 UNSW-NB15 evaluation rows as benign; its mean logit was `-3.373` for the 49 benign rows and `-6.095` for the 1,951 attack rows. Conversely, the UNSW-NB15 local model predicted 1,835 of 1,938 CTU-SME evaluation rows as attacks. These results are consistent with client-specific feature-to-label relationships that do not transfer cleanly.

## FedAvg weighting and diagnosis

FedAvg weighted each local model by its training-example count. The counts were 7,752 for CTU-SME and 8,000 for UNSW-NB15, corresponding to 49.21% and 50.79% of the aggregate weight. These counts and weights were the same for all three seeds.

The count weighting was not identified as the main cause: the contributions were close to even, with UNSW-NB15 contributing slightly more, while its global evaluation still collapsed to all-benign predictions. This does not prove that weighting interactions had no effect; it shows that a large client-count imbalance was not present to explain the result.

The supported diagnosis was feature-distribution and feature-to-label heterogeneity. The raw and `log1p` distributions differ between clients, local coefficients differ (including sign changes), and cross-client local score behavior is markedly different. These observations provide a coherent explanation for why one shared decision boundary did not serve both clients. They do not mathematically prove heterogeneity as the sole cause.

## Why shared normalization was the next diagnostic

The next controlled experiment was chosen to isolate whether residual scale differences contributed to the collapse: keep the features, model, seeds, split method, one-round FedAvg, and example-count weighting fixed, then fit one scaler on the combined client training partitions only. Applying that same scaler to both clients' training and held-out data tests shared feature scaling without fitting on evaluation data or introducing client-specific scalers. This was a diagnostic ablation, not a model-tuning program.

More federated rounds were not justified at this point. The one-round result already established a severe, repeatable client-specific failure, while extra rounds would change the training exposure without identifying whether feature scaling contributed. The scale ablation was the narrower next experiment.

## Limitations

- CTU-SME and UNSW-NB15 are dataset-derived simulated clients, not independent organizations.
- UNSW-NB15 is severely imbalanced: its eligible data contains 243 benign and 9,757 attack rows; the held-out set contains only 49 benign rows. CTU-SME's eligible data contains 2,893 benign and 6,797 attack rows.
- Feature distributions are descriptive, and coefficient differences describe fitted model behavior. Neither establishes a causal mechanism.
- This is a diagnosis supported by the available datasets, seeds, model, and one-round setup. It is not proof that heterogeneity is the sole cause, nor evidence about production clients or privacy properties.

## Related results

The baseline evaluation and sample counts are documented in [Milestone 5](federated_shared_features_milestone5.md). The subsequent shared-normalization ablation is documented in [Milestone 7](federated_shared_normalization_milestone7.md).
