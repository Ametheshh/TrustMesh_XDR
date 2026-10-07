"""Evaluate the five mapped CTU-SME/UNSW shared fields with one-round FedAvg."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.federated.simulated_fedavg import (
    classification_metrics,
    predict,
    stratified_split,
    train_parameters,
    LOCAL_EPOCHS,
    LEARNING_RATE,
    L2,
    TEST_FRACTION,
)


ROOT = Path("data/canonical/smoke_test")
CLIENTS = {
    "ctu_sme": ROOT / "ctu_sme_conn_labeled.jsonl",
    "unsw_nb15_named": ROOT / "unsw_nb15_named_train.jsonl",
}
FEATURES = (
    "duration",
    "bytes_src_to_dst",
    "bytes_dst_to_src",
    "packets_src_to_dst",
    "packets_dst_to_src",
)
SEEDS = (42, 43, 44)


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    vectors: list[list[float]] = []
    labels: list[int] = []
    audit = {
        "rows": 0,
        "excluded_missing_features_with_binary_label": 0,
        "excluded_nonbinary_labels": 0,
    }
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            record: dict[str, Any] = json.loads(line)
            audit["rows"] += 1
            feature_map = record.get("features", {})
            label = record.get("labels", {}).get("binary")
            if label not in (0, 1) or isinstance(label, bool):
                audit["excluded_nonbinary_labels"] += 1
                continue
            values = [feature_map.get(name) for name in FEATURES]
            if any(value is None for value in values):
                audit["excluded_missing_features_with_binary_label"] += 1
                continue
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"invalid nonnegative numeric feature at {path}:{line_no}")
            vectors.append([math.log1p(float(value)) for value in values])
            labels.append(int(label))
    x = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if not len(y) or set(y.tolist()) != {0, 1}:
        raise ValueError(f"{path} must have valid rows from both classes")
    audit["eligible_rows"] = int(len(y))
    audit["eligible_class_counts"] = {str(k): int(np.sum(y == k)) for k in (0, 1)}
    return x, y, audit


def evaluate() -> dict[str, Any]:
    clients = {name: load_rows(path) for name, path in CLIENTS.items()}
    output: dict[str, Any] = {
        "features": list(FEATURES),
        "preprocessing": "log1p applied independently to each nonnegative numeric feature",
        "model": "existing class-weighted logistic regression via full-batch gradient descent",
        "model_parameters": {"epochs": LOCAL_EPOCHS, "learning_rate": LEARNING_RATE, "l2": L2},
        "federation": "one in-memory Flower FedAvg round weighted by training row count",
        "split": {"method": "stratified per client", "train_fraction": 1 - TEST_FRACTION,
                  "evaluation_fraction": TEST_FRACTION, "seeds": list(SEEDS)},
        "clients": {name: {"input": str(CLIENTS[name]), "audit": data[2]}
                    for name, data in clients.items()},
        "per_seed": [],
    }
    for seed in SEEDS:
        partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        local_params: dict[str, list[np.ndarray]] = {}
        split_counts: dict[str, Any] = {}
        for name, (x, y, _) in clients.items():
            train_idx, eval_idx = stratified_split(y, seed=seed)
            x_train, y_train = x[train_idx], y[train_idx]
            x_eval, y_eval = x[eval_idx], y[eval_idx]
            partitions[name] = (x_train, y_train, x_eval, y_eval)
            params = [np.zeros(len(FEATURES), dtype=np.float64), np.zeros(1, dtype=np.float64)]
            local_params[name] = train_parameters(x_train, y_train, params)
            split_counts[name] = {
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
            metrics[f"local_{name}"] = {}
            for eval_name, (_, _, x_eval, y_eval) in partitions.items():
                metrics[f"local_{name}"][eval_name] = classification_metrics(
                    y_eval, predict(x_eval, params)
                )
        metrics["global"] = {
            eval_name: classification_metrics(y_eval, predict(x_eval, global_params))
            for eval_name, (_, _, x_eval, y_eval) in partitions.items()
        }
        output["per_seed"].append({"seed": seed, "counts": split_counts, "metrics": metrics})

    metric_names = ("precision", "recall", "f1", "balanced_accuracy")
    summary: dict[str, Any] = {}
    for model in ("local_ctu_sme", "local_unsw_nb15_named", "global"):
        summary[model] = {}
        for eval_name in CLIENTS:
            summary[model][eval_name] = {}
            for metric in metric_names:
                values = [row["metrics"][model][eval_name][metric] for row in output["per_seed"]]
                summary[model][eval_name][metric] = {
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                }
    output["mean_and_sample_std"] = summary
    return output


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
