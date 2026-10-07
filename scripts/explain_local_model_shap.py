"""Explain a few held-out CTU-SME local logistic-regression predictions with SHAP."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import shap

from scripts.evaluate_shared_feature_federation import CLIENTS, FEATURES, load_rows
from src.federated.simulated_fedavg import predict, stratified_split, train_parameters


SEED = 42
CLIENT = "ctu_sme"
BACKGROUND_SIZE = 100
GLOBAL_SAMPLE_PER_CLASS = 100


def _logit(x: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
    return np.clip((x @ parameters[0]).reshape(-1) + float(parameters[1][0]), -30.0, 30.0)


def explain() -> dict[str, Any]:
    x, y, audit = load_rows(CLIENTS[CLIENT])
    train_idx, eval_idx = stratified_split(y, seed=SEED)
    x_train, y_train = x[train_idx], y[train_idx]
    x_eval, y_eval = x[eval_idx], y[eval_idx]

    parameters = train_parameters(
        x_train,
        y_train,
        [np.zeros(len(FEATURES), dtype=np.float64), np.zeros(1, dtype=np.float64)],
    )

    rng = np.random.default_rng(SEED)
    background_idx = rng.choice(len(x_train), size=min(BACKGROUND_SIZE, len(x_train)), replace=False)
    background = x_train[background_idx]
    explainer = shap.LinearExplainer(
        (parameters[0], float(parameters[1][0])),
        shap.maskers.Independent(background),
    )

    eval_predictions = predict(x_eval, parameters)
    selected: list[tuple[str, int]] = []
    cases = (
        ("correct_benign", 0, 0),
        ("correct_attack", 1, 1),
        ("false_positive", 0, 1),
        ("false_negative", 1, 0),
    )
    for name, actual, predicted in cases:
        matches = np.flatnonzero((y_eval == actual) & (eval_predictions == predicted))
        if len(matches):
            selected.append((name, int(matches[0])))

    global_indices = np.concatenate([
        np.flatnonzero(y_eval == label)[:GLOBAL_SAMPLE_PER_CLASS] for label in (0, 1)
    ])
    global_shap = np.asarray(explainer(x_eval[global_indices]).values, dtype=np.float64)
    mean_abs = np.mean(np.abs(global_shap), axis=0)

    examples = []
    for case, index in selected:
        values = np.asarray(explainer(x_eval[index:index + 1]).values[0], dtype=np.float64)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        score_logit = float(_logit(x_eval[index:index + 1], parameters)[0])
        reconstructed = base_value + float(values.sum())
        error = abs(score_logit - reconstructed)
        if error > 1e-8:
            raise AssertionError(f"SHAP additivity check failed: error={error}")
        probability = 1.0 / (1.0 + math.exp(-score_logit))
        predicted = int(eval_predictions[index])
        if predicted != int(probability >= 0.5):
            raise AssertionError("reported probability and classifier threshold disagree")
        impacts = [
            {"feature": FEATURES[j], "raw_value": float(np.expm1(x_eval[index, j])),
             "model_value_log1p": float(x_eval[index, j]), "shap_logit_contribution": float(values[j])}
            for j in range(len(FEATURES))
        ]
        examples.append({
            "case": case,
            "evaluation_row_index": index,
            "actual_class": int(y_eval[index]),
            "predicted_class": predicted,
            "attack_probability": probability,
            "logit": score_logit,
            "base_value_logit": base_value,
            "shap_sum_logit": float(values.sum()),
            "additivity_error": error,
            "feature_contributions": impacts,
            "top_contributors": sorted(impacts, key=lambda item: abs(item["shap_logit_contribution"]), reverse=True),
        })

    return {
        "model": "CTU-SME local class-weighted logistic regression",
        "dataset": str(CLIENTS[CLIENT]),
        "seed": SEED,
        "features": list(FEATURES),
        "preprocessing": "log1p per feature; no shared scaler (Milestone 5 local baseline)",
        "training_rows": int(len(train_idx)),
        "heldout_rows": int(len(eval_idx)),
        "heldout_class_counts": {str(k): int(np.sum(y_eval == k)) for k in (0, 1)},
        "data_audit": audit,
        "shap": {
            "method": "SHAP LinearExplainer with an Independent training-background masker",
            "output_space": "model logit (log-odds); positive contributions increase attack logit",
            "background_rows": int(len(background)),
            "global_importance_rows": int(len(global_indices)),
            "global_importance_sampling": "up to 100 held-out rows per actual class",
        },
        "global_importance_mean_abs_shap_logit": [
            {"feature": FEATURES[j], "mean_abs_shap": float(mean_abs[j])}
            for j in np.argsort(-mean_abs)
        ],
        "examples": examples,
    }


if __name__ == "__main__":
    print(json.dumps(explain(), indent=2))
