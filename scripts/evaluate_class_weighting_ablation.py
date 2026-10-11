"""Compare class-weighting policies in centralized and simulated federated training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.evaluate_centralized_vs_federated import (
    ARTIFACT_ROOT,
    build_partitions_for_seed,
    fit_federated_global_model,
)
from scripts.evaluate_shared_feature_federation import CLIENTS, FEATURES, SEEDS, load_rows
from src.federated.simulated_fedavg import (
    L2,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    classification_metrics,
    predict_proba,
    save_model_artifact,
    train_centralized_model,
    zero_parameters,
)

EXPERIMENT_ROOT = ARTIFACT_ROOT / "class_weighting_ablation"
CONDITIONS = {
    "A_centralized_global_weighted": ("centralized", True, "global pooled class weights"),
    "B_federated_local_weighted": ("federated", True, "per-client local class weights"),
    "C_centralized_unweighted": ("centralized", False, "unit sample weights"),
    "D_federated_unweighted": ("federated", False, "unit sample weights"),
}


def summarize_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Report threshold-0.5 classification and ranking metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = classification_metrics(y_true, predictions)
    tn, fp = metrics["confusion_matrix"][0]
    metrics["false_positive_rate"] = fp / (tn + fp) if tn + fp else 0.0
    if np.unique(y_true).size == 2:
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None
    return metrics


def train_conditions(
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    """Fit all four models using the supplied shared train/evaluation partitions."""
    pooled_x = np.concatenate([part[0] for part in partitions.values()], axis=0)
    pooled_y = np.concatenate([part[1] for part in partitions.values()], axis=0)
    return {
        "A_centralized_global_weighted": train_centralized_model(
            pooled_x, pooled_y, num_features=len(FEATURES), class_weighted=True
        ),
        "B_federated_local_weighted": fit_federated_global_model(
            partitions, class_weighted=True
        )[1],
        "C_centralized_unweighted": train_centralized_model(
            pooled_x, pooled_y, num_features=len(FEATURES), class_weighted=False
        ),
        "D_federated_unweighted": fit_federated_global_model(
            partitions, class_weighted=False
        )[1],
    }


def evaluate_seed(
    seed: int,
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    artifact_root: Path = EXPERIMENT_ROOT,
) -> dict[str, Any]:
    """Train, save, and evaluate the four conditions against shared client holdouts."""
    models = train_conditions(partitions)
    client_metrics: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, str] = {}
    for condition, parameters in models.items():
        regime, class_weighted, weighting = CONDITIONS[condition]
        model_path = artifact_root / "models" / condition / f"seed_{seed}.json"
        save_model_artifact(
            model_path,
            model_name=f"{condition}_seed_{seed}",
            model_type=f"{regime}_logistic_regression",
            parameters=parameters,
            feature_names=list(FEATURES),
            training_config={
                "seed": seed,
                "epochs": LOCAL_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "l2": L2,
                "class_weighted": class_weighted,
                "class_weighting_policy": weighting,
                "training_protocol": (
                    "pooled client training partitions only"
                    if regime == "centralized"
                    else "local training then one in-memory Flower FedAvg round weighted by training rows"
                ),
            },
            dataset_summary={
                "client_names": list(CLIENTS),
                "benchmark": "class_weighting_ablation",
            },
            transforms=["log1p"],
            preprocessing={
                "pipeline": ["log1p"],
                "feature_order": list(FEATURES),
                "notes": "Reuses the canonical shared-feature benchmark representation.",
            },
        )
        artifact_paths[condition] = str(model_path)

    for client, (_, _, x_eval, y_eval) in partitions.items():
        client_metrics[client] = {
            condition: summarize_metrics(y_eval, predict_proba(x_eval, parameters))
            for condition, parameters in models.items()
        }
    return {
        "seed": seed,
        "artifact_paths": artifact_paths,
        "training_rows": {name: int(len(part[1])) for name, part in partitions.items()},
        "evaluation_rows": {name: int(len(part[3])) for name, part in partitions.items()},
        "client_metrics": client_metrics,
    }


def evaluate_class_weighting_ablation() -> dict[str, Any]:
    """Run the four weighting conditions on the established seeds and shared partitions."""
    client_rows = {name: load_rows(path) for name, path in CLIENTS.items()}
    result: dict[str, Any] = {
        "experiment": "class_weighting_ablation",
        "features": list(FEATURES),
        "preprocessing": "log1p applied to each feature; no fitted scaler",
        "decision_threshold": 0.5,
        "split": {"method": "stratified per client", "seeds": list(SEEDS)},
        "conditions": {
            condition: {"training_regime": regime, "class_weighting": weighting}
            for condition, (regime, _, weighting) in CONDITIONS.items()
        },
        "artifacts": {"root": str(EXPERIMENT_ROOT)},
        "per_seed": [],
    }
    for seed in SEEDS:
        partitions = build_partitions_for_seed(client_rows, seed)
        result["per_seed"].append(
            evaluate_seed(seed, partitions, artifact_root=EXPERIMENT_ROOT)
        )

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = EXPERIMENT_ROOT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["summary_path"] = str(summary_path)
    return result


if __name__ == "__main__":
    output = evaluate_class_weighting_ablation()
    print(json.dumps({"summary_path": output["summary_path"], "seeds": list(SEEDS)}, indent=2))
