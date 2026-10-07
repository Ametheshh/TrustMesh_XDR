# Local flow-classification explanations with SHAP

## Model and method

This report explains the existing **CTU-SME local** five-feature class-weighted logistic-regression model, trained with seed 42 on the Milestone 5 preprocessing (`log1p` applied feature-wise; no shared scaler). It uses the held-out CTU partition: 1,938 rows (579 benign, 1,359 attack); training uses 7,752 eligible rows. The existing held-out metrics for this model were precision `0.9963`, recall `0.9978`, F1 `0.9971`, and balanced accuracy `0.9946`. SHAP does not change or improve those metrics.

The explainer is SHAP `LinearExplainer` with an `Independent` masker built from 100 deterministic training rows. Attributions are in **logit/log-odds units**: a positive contribution pushes the model toward attack, and a negative contribution pushes it toward benign. The SHAP base value is `0.8889` logit. For each example, base value plus all feature attributions reconstructs the model logit (maximum observed numerical additivity error was below `2e-15`). Applying the logistic function to that logit gives the reported attack probability.

## Global importance on a small held-out sample

Mean absolute SHAP contribution across a deterministic, class-balanced sample of 200 held-out flows (100 from each true class):

```text
bytes_src_to_dst    3.675  ████████████████████
bytes_dst_to_src    2.217  ████████████
packets_src_to_dst  0.825  ████
duration            0.426  ██
packets_dst_to_src  0.264  █
```

These values summarize this model's behavior on this sampled CTU holdout and use the transformed model inputs. They are not a ranking of causal attack mechanisms.

## Individual held-out predictions

Raw feature values are shown; SHAP contributions are for the corresponding `log1p` model input. Each list is ordered by absolute contribution. `0=benign`, `1=attack`.

| Case | Held-out row | Actual | Predicted | Attack probability | Top logit contributions (feature: raw value → SHAP value) |
|---|---:|---:|---:|---:|---|
| Correct benign | 0 | 0 | 0 | 0.0015 | `bytes_src_to_dst: 88 → -3.596`; `bytes_dst_to_src: 144 → -2.178`; `packets_src_to_dst: 2 → -0.898` |
| Correct attack | 1 | 1 | 1 | 0.9838 | `bytes_src_to_dst: 0 → +2.027`; `bytes_dst_to_src: 0 → +1.213`; `packets_dst_to_src: 0 → +0.129` |
| False positive | 765 | 0 | 1 | 0.9959 | `bytes_src_to_dst: 0 → +2.027`; `bytes_dst_to_src: 0 → +1.213`; `packets_src_to_dst: 8 → +0.838` |
| False negative | 632 | 1 | 0 | 0.0495 | `bytes_src_to_dst: 423 → -5.552`; `bytes_dst_to_src: 888 → -3.414`; `packets_src_to_dst: 34 → +2.984` |

For the correct benign example, the largest contributions lower the attack logit; for the correct attack example, the largest contributions raise it. The false-positive and false-negative rows show how those same learned associations can produce incorrect predictions.

## Limits

SHAP describes how this fitted model combines the input values relative to its training background. It does **not** establish that a feature caused an attack or that changing the feature would change the real-world event. The independent masker treats features independently, while bytes, packets, and duration can be correlated; therefore some masked combinations may not represent realistic flows. This explains one local CTU model only, not the UNSW model or the federated global model. The 200-row importance summary is intentionally small and sample-dependent.

Reproduce with `.venv/bin/python -m scripts.explain_local_model_shap`. The script checks prediction consistency and SHAP additivity before emitting its compact JSON result.
