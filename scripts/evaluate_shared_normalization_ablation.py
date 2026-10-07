"""Run the shared-training-partition scaler ablation for the five-feature POC."""

from __future__ import annotations

import json
import statistics
from typing import Any

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from scripts.evaluate_shared_feature_federation import CLIENTS, FEATURES, SEEDS, load_rows
from src.federated.simulated_fedavg import (
    L2,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    classification_metrics,
    predict,
    stratified_split,
    train_parameters,
)


def evaluate() -> dict[str, Any]:
    clients = {name: load_rows(path) for name, path in CLIENTS.items()}
    output: dict[str, Any] = {
        "features": list(FEATURES),
        "preprocessing": "log1p, then one StandardScaler fit on concatenated client training partitions only",
        "scaler": "population mean and standard deviation (ddof=0); zero scales replaced with one",
        "model": {"epochs": LOCAL_EPOCHS, "learning_rate": LEARNING_RATE, "l2": L2},
        "federation": "one in-memory Flower FedAvg round weighted by client training row count",
        "per_seed": [],
    }

    for seed in SEEDS:
        raw_partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, (x, y, _) in clients.items():
            train_idx, eval_idx = stratified_split(y, seed=seed)
            raw_partitions[name] = (x[train_idx], y[train_idx], x[eval_idx], y[eval_idx])

        # Fit shared statistics on training rows only; evaluation rows are untouched.
        pooled_train = np.concatenate([part[0] for part in raw_partitions.values()], axis=0)
        scaler_mean = pooled_train.mean(axis=0)
        scaler_scale = pooled_train.std(axis=0, ddof=0)
        scaler_scale[scaler_scale == 0.0] = 1.0
        partitions = {
            name: ((x_train - scaler_mean) / scaler_scale, y_train,
                   (x_eval - scaler_mean) / scaler_scale, y_eval)
            for name, (x_train, y_train, x_eval, y_eval) in raw_partitions.items()
        }

        local_params: dict[str, list[np.ndarray]] = {}
        counts: dict[str, Any] = {}
        for name, (x_train, y_train, x_eval, y_eval) in partitions.items():
            init = [np.zeros(len(FEATURES), dtype=np.float64), np.zeros(1, dtype=np.float64)]
            local_params[name] = train_parameters(x_train, y_train, init)
            counts[name] = {
                "training_rows": int(len(y_train)),
                "training_classes": {str(k): int(np.sum(y_train == k)) for k in (0, 1)},
                "evaluation_rows": int(len(y_eval)),
                "evaluation_classes": {str(k): int(np.sum(y_eval == k)) for k in (0, 1)},
            }

        fit_results = []
        for name, params in local_params.items():
            y_train = partitions[name][1]
            fit_results.append((None, FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(params), num_examples=len(y_train), metrics={},
            )))
        strategy = FedAvg(fraction_fit=1.0, min_fit_clients=2, min_available_clients=2,
                          accept_failures=False, fit_metrics_aggregation_fn=lambda _: {})
        aggregate, _ = strategy.aggregate_fit(1, fit_results, failures=[])
        if aggregate is None:
            raise RuntimeError("Flower FedAvg did not return global parameters")
        global_params = parameters_to_ndarrays(aggregate)

        metrics: dict[str, Any] = {}
        for name, params in local_params.items():
            metrics[f"local_{name}"] = {
                client: classification_metrics(y_eval, predict(x_eval, params))
                for client, (_, _, x_eval, y_eval) in partitions.items()
            }
        metrics["global"] = {
            client: classification_metrics(y_eval, predict(x_eval, global_params))
            for client, (_, _, x_eval, y_eval) in partitions.items()
        }
        output["per_seed"].append({
            "seed": seed,
            "counts": counts,
            "scaler_mean": scaler_mean.tolist(),
            "scaler_scale": scaler_scale.tolist(),
            "fedavg_training_rows": {name: counts[name]["training_rows"] for name in partitions},
            "global_parameters": {"weights": global_params[0].tolist(), "bias": float(global_params[1][0])},
            "metrics": metrics,
        })

    metric_names = ("precision", "recall", "f1", "balanced_accuracy")
    summary: dict[str, Any] = {}
    for model in ("local_ctu_sme", "local_unsw_nb15_named", "global"):
        summary[model] = {}
        for client in CLIENTS:
            summary[model][client] = {}
            for metric in metric_names:
                vals = [row["metrics"][model][client][metric] for row in output["per_seed"]]
                summary[model][client][metric] = {
                    "mean": statistics.mean(vals),
                    "sample_std": statistics.stdev(vals),
                }
    output["mean_and_sample_std"] = summary
    return output


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
