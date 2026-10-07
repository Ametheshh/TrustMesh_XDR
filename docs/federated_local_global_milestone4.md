# Federated local-versus-global evaluation (Milestone 4)

## Feature decision

The existing controlled federation inputs are:

- `data/canonical/smoke_test/ctu_unsw_fl_poc/ctu_sme_total_packets.jsonl`
- `data/canonical/smoke_test/ctu_unsw_fl_poc/unsw_nb15_named_total_packets.jsonl`

Both contain exactly one numeric model feature, `total_packets`, plus the binary target. Therefore the broadest feature set available in these *designated POC inputs* is `{total_packets}`. This is supported by the source adapter mappings: CTU `orig_pkts`/`resp_pkts` and UNSW `spkts`/`dpkts` map to the same directional packet-count concepts, whose sum is the POC feature. No other source fields are present in these POC files.

The separate canonical smoke-test inputs contain additional mapped common fields (`duration`, directional bytes, and directional packet counts). They are not the designated POC inputs used here, so this run does not silently substitute them. They can support a separately scoped, broader-input experiment after the feature set is explicitly approved.

## Configuration

- Clients: dataset-derived CTU-SME and UNSW-NB15; simulated clients, not real organizations.
- Input: the two POC JSONL files above; binary labels `0=benign/normal`, `1=attack/malicious`.
- Preprocessing: fixed `log1p(total_packets)` transform, identical at both clients; no fitted cross-client statistics.
- Model: the existing weighted binary logistic regression, full-batch gradient descent, 500 epochs, learning rate `0.1`, L2 `1e-4`, prediction at logit `>= 0`.
- Local models: trained independently from zero parameters on each client's training partition.
- Global model: one in-memory Flower FedAvg round, weighted by local training rows; both clients start from zero parameters.
- Splits: stratified 80/20 per client, seeds `42`, `43`, `44`. Each local and global model was evaluated separately on each client's held-out partition. Cross-client rows evaluate each local model on the other client's holdout.
- Metrics use class `1` as positive. Confusion matrix layout is `[[TN, FP], [FN, TP]]`. Mean and standard deviation below use the sample standard deviation across the three seeds.

## Counts and class balance

The stratification preserves these class counts in each seed:

| Client | Total (0 / 1) | Train (0 / 1) | Evaluation (0 / 1) |
|---|---:|---:|---:|
| CTU-SME | 9,760 (2,939 / 6,821) | 7,808 (2,351 / 5,457) | 1,952 (588 / 1,364) |
| UNSW-NB15 | 10,000 (243 / 9,757) | 8,000 (194 / 7,806) | 2,000 (49 / 1,951) |

## Per-seed results

FedAvg had the same metric values and confusion matrices as the CTU-local model on both held-out clients for all three seeds. The table lists each unique result once; `global → CTU` equals `CTU local → CTU`, and `global → UNSW` equals `CTU local → UNSW` in every seed.

| Seed | Model → evaluation client | Precision | Recall | F1 | Balanced accuracy | Confusion matrix |
|---:|---|---:|---:|---:|---:|---|
| 42 | CTU local → CTU | 0.7852 | 0.9971 | 0.8786 | 0.6822 | `[[216,372],[4,1360]]` |
| 42 | UNSW local → UNSW | 0.9839 | 0.5336 | 0.6919 | 0.5933 | `[[32,17],[910,1041]]` |
| 42 | CTU local → UNSW (cross) | 0.9867 | 0.5315 | 0.6909 | 0.6229 | `[[35,14],[914,1037]]` |
| 42 | UNSW local → CTU (cross) | 0.7803 | 0.9971 | 0.8754 | 0.6729 | `[[205,383],[4,1360]]` |
| 43 | CTU local → CTU | 0.7835 | 0.9949 | 0.8766 | 0.6786 | `[[213,375],[7,1357]]` |
| 43 | UNSW local → UNSW | 0.9908 | 0.5546 | 0.7111 | 0.6753 | `[[39,10],[869,1082]]` |
| 43 | CTU local → UNSW (cross) | 0.9908 | 0.5520 | 0.7090 | 0.6740 | `[[39,10],[874,1077]]` |
| 43 | UNSW local → CTU (cross) | 0.7794 | 0.9949 | 0.8741 | 0.6709 | `[[204,384],[7,1357]]` |
| 44 | CTU local → CTU | 0.7917 | 0.9978 | 0.8829 | 0.6945 | `[[230,358],[3,1361]]` |
| 44 | UNSW local → UNSW | 0.9869 | 0.5418 | 0.6995 | 0.6280 | `[[35,14],[894,1057]]` |
| 44 | CTU local → UNSW (cross) | 0.9887 | 0.5392 | 0.6978 | 0.6472 | `[[37,12],[899,1052]]` |
| 44 | UNSW local → CTU (cross) | 0.7863 | 0.9978 | 0.8795 | 0.6843 | `[[218,370],[3,1361]]` |

## Mean ± sample standard deviation

Global CTU results equal the CTU-local row; global UNSW results equal the CTU-local cross-client row, as noted above.

| Model → evaluation client | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| CTU local → CTU | 0.7868 ± 0.0044 | 0.9966 ± 0.0015 | 0.8794 ± 0.0032 | 0.6851 ± 0.0083 |
| Global → CTU | 0.7868 ± 0.0044 | 0.9966 ± 0.0015 | 0.8794 ± 0.0032 | 0.6851 ± 0.0083 |
| UNSW local → UNSW | 0.9872 ± 0.0035 | 0.5433 ± 0.0106 | 0.7009 ± 0.0097 | 0.6322 ± 0.0411 |
| Global → UNSW | 0.9887 ± 0.0021 | 0.5409 ± 0.0104 | 0.6992 ± 0.0092 | 0.6480 ± 0.0255 |
| CTU local → UNSW (cross-client) | 0.9887 ± 0.0021 | 0.5409 ± 0.0104 | 0.6992 ± 0.0092 | 0.6480 ± 0.0255 |
| UNSW local → CTU (cross-client) | 0.7820 ± 0.0037 | 0.9966 ± 0.0015 | 0.8763 ± 0.0028 | 0.6760 ± 0.0072 |

## Interpretation and limits

- On CTU-SME, the global model's held-out metrics match the CTU-local metrics in all seeds.
- On UNSW-NB15, global balanced accuracy is higher than UNSW-local by about `0.0158`, while F1 is lower by about `0.0016` and recall lower by about `0.0024`. This is a modest trade-off, not a consistent improvement.
- Cross-client evaluation is possible because the POC uses the same numeric feature and transform at both clients. CTU-local and global give the same reported UNSW confusion matrices; UNSW-local on CTU has slightly lower mean balanced accuracy than CTU-local on CTU.
- UNSW is highly imbalanced: only 49 benign evaluation examples per seed. This makes specificity-related estimates and their seed variation sensitive to a small number of rows. High precision/F1 should be read alongside balanced accuracy and recall.
- Three seeded stratified partitions measure split sensitivity within these fixed artifacts, not independent dataset replications. The experiment remains a single-feature, one-round, in-memory simulation; it does not represent independently operated organizations or prove a general federation benefit.
- No source datasets, canonical schemas, or project architecture were changed.
