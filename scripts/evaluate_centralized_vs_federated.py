"""Train and evaluate a centralized and federated detection model on the fixed shared-feature protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from sklearn.metrics import average_precision_score

from scripts.evaluate_shared_feature_federation import CLIENTS, FEATURES, SEEDS, load_rows
from src.federated.simulated_fedavg import (
    L2,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    classification_metrics,
    predict,
    predict_proba,
    save_model_artifact,
    stratified_split,
    train_centralized_model,
    train_parameters,
    zero_parameters,
)

ARTIFACT_ROOT = Path("data/models")


def build_partitions_for_seed(client_rows: dict[str, tuple[np.ndarray, np.ndarray, dict[str, int]],], seed: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create train/eval partitions for each client using the benchmark's deterministic split protocol."""
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (x, y, _) in client_rows.items():
        train_idx, eval_idx = stratified_split(y, seed=seed)
        x_train, y_train = x[train_idx], y[train_idx]
        x_eval, y_eval = x[eval_idx], y[eval_idx]
        partitions[name] = (x_train, y_train, x_eval, y_eval)
    return partitions


def fit_federated_global_model(
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    class_weighted: bool = True,
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    """Train local client models and aggregate them with the project’s in-memory Flower FedAvg strategy."""
    local_params: dict[str, list[np.ndarray]] = {}
    fit_results: list[tuple[None, FitRes]] = []
    for name, (x_train, y_train, _, _) in partitions.items():
        params = train_parameters(
            x_train,
            y_train,
            zero_parameters(len(FEATURES)),
            class_weighted=class_weighted,
        )
        local_params[name] = params
        fit_results.append((
            None,
            FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(params),
                num_examples=len(y_train),
                metrics={},
            ),
        ))

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        accept_failures=False,
        fit_metrics_aggregation_fn=lambda _: {},
    )
    aggregate, _ = strategy.aggregate_fit(1, fit_results, failures=[])
    if aggregate is None:
        raise RuntimeError("Flower FedAvg did not return global parameters")
    global_params = parameters_to_ndarrays(aggregate)
    return local_params, global_params


def summarize_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Return classification metrics and false-positive rate for a binary logistic score vector.

    The score is the probability of the positive class (label 1). The benchmark
    applies the default 0.5 decision threshold before computing confusion-matrix
    metrics, while PR-AUC remains a ranking metric that can remain high even when
    the 0.5-threshold recall is zero.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_pred = (probabilities >= 0.5).astype(np.int64)
    metrics = classification_metrics(y_true, y_pred)
    tn, fp = metrics["confusion_matrix"][0]
    metrics["false_positive_rate"] = fp / (tn + fp) if (tn + fp) else 0.0
    if np.unique(y_true).size == 2 and not np.all(y_true == y_true[0]):
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["pr_auc"] = None
    return metrics


def save_benchmark_artifacts(seed: int, centralized_params: list[np.ndarray], federated_params: list[np.ndarray]) -> tuple[Path, Path]:
    """Persist the centralized and federated models under data/models using the project artifact format."""
    artifact_root = ARTIFACT_ROOT / "centralized_vs_federated"
    central_path = artifact_root / "centralized" / f"seed_{seed}.json"
    federated_path = artifact_root / "federated" / f"seed_{seed}.json"
    central_training = {
        "seed": seed,
        "model_family": "class_weighted_logistic_regression",
        "positive_class": 1,
        "decision_threshold": 0.5,
        "score_semantics": "probability of class 1 (attack/malicious)",
        "training_protocol": "pooled client training partitions only",
        "partition_mode": "stratified train/eval split per client",
        "epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2": L2,
        "class_weighting": "count-based weights: len(y)/(2*counts) with sample_weights applied across pooled training rows",
    }
    federated_training = {
        "seed": seed,
        "model_family": "class_weighted_logistic_regression",
        "positive_class": 1,
        "decision_threshold": 0.5,
        "score_semantics": "probability of class 1 (attack/malicious)",
        "training_protocol": "local training then one in-memory Flower FedAvg round weighted by training rows",
        "partition_mode": "stratified train/eval split per client",
        "epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2": L2,
        "class_weighting": "count-based weights: len(y)/(2*counts) with sample_weights applied within each client local training run",
        "federation": "Flower FedAvg",
    }
    save_model_artifact(
        central_path,
        model_name=f"centralized_seed_{seed}",
        model_type="centralized_logistic_regression",
        parameters=centralized_params,
        feature_names=list(FEATURES),
        training_config=central_training,
        dataset_summary={"client_names": list(CLIENTS), "benchmark": "centralized_vs_federated"},
        transforms=["log1p"],
        preprocessing={
            "pipeline": ["log1p"],
            "fit_on": "pooled client training partitions only",
            "feature_order": list(FEATURES),
            "notes": "The same canonical five-feature representation and log1p transform are used for centralized and federated models.",
        },
    )
    save_model_artifact(
        federated_path,
        model_name=f"federated_seed_{seed}",
        model_type="federated_logistic_regression",
        parameters=federated_params,
        feature_names=list(FEATURES),
        training_config=federated_training,
        dataset_summary={"client_names": list(CLIENTS), "benchmark": "centralized_vs_federated"},
        transforms=["log1p"],
        preprocessing={
            "pipeline": ["log1p"],
            "fit_on": "local client training partitions only; aggregation via FedAvg",
            "feature_order": list(FEATURES),
            "notes": "The federated model is the existing Flower FedAvg workflow from the controlled experiment.",
        },
    )
    return central_path, federated_path


def evaluate_centralized_vs_federated() -> dict[str, Any]:
    """Train, save, and compare centralized and federated models across the benchmark seeds."""
    client_rows = {name: load_rows(path) for name, path in CLIENTS.items()}
    output: dict[str, Any] = {
        "features": list(FEATURES),
        "preprocessing": "log1p applied to each feature; no fitted scaler in the primary comparison",
        "model_family": "class_weighted binary logistic regression",
        "positive_class": 1,
        "decision_threshold": 0.5,
        "score_semantics": "probability of class 1 (attack/malicious)",
        "split": {"method": "stratified per client", "seeds": list(SEEDS)},
        "artifacts": {"root": str(ARTIFACT_ROOT / "centralized_vs_federated")},
        "per_seed": [],
    }

    for seed in SEEDS:
        partitions = build_partitions_for_seed(client_rows, seed)
        pooled_x = np.concatenate([part[0] for part in partitions.values()], axis=0)
        pooled_y = np.concatenate([part[1] for part in partitions.values()], axis=0)
        centralized_params = train_centralized_model(pooled_x, pooled_y, num_features=len(FEATURES))
        _, federated_params = fit_federated_global_model(partitions)
        central_path, federated_path = save_benchmark_artifacts(seed, centralized_params, federated_params)

        per_client: dict[str, Any] = {}
        for client_name, (_, _, x_eval, y_eval) in partitions.items():
            central_scores = predict_proba(x_eval, centralized_params)
            federated_scores = predict_proba(x_eval, federated_params)
            per_client[client_name] = {
                "centralized": summarize_metrics(y_eval, central_scores),
                "federated": summarize_metrics(y_eval, federated_scores),
            }
        output["per_seed"].append({
            "seed": seed,
            "artifacts": {"centralized": str(central_path), "federated": str(federated_path)},
            "local_train_rows": {name: int(len(partitions[name][0])) for name in partitions},
            "evaluation_rows": {name: int(len(partitions[name][2])) for name in partitions},
            "client_metrics": per_client,
        })

    return output


if __name__ == "__main__":
    result = evaluate_centralized_vs_federated()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = ARTIFACT_ROOT / "centralized_vs_federated_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary_path": str(summary_path), "seeds": result["split"]["seeds"]}, indent=2))
