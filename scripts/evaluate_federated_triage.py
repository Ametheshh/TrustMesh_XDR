"""
Local vs Federated analyst-budget triage experiment.

Research experiment:
- Clients: CTU-SME and UNSW-NB15
- Seeds: 42, 43, 44
- Model conditions:
    1. Local detector + full EpisodeRanker
    2. Federated global detector + full EpisodeRanker
- Baselines:
    3. Confidence-only
    4. Random
- Analyst budgets: K=25, 50, 100, 200

The experiment uses:
- Existing five shared features
- Existing log1p preprocessing
- Shared scaler fitted only on training partitions
- Existing local logistic training
- Existing Flower FedAvg
- Existing correlation representation
- Existing EpisodeRanker
- Existing triage metrics

This is a controlled benchmark experiment, not a production SOC evaluation.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

from scripts.evaluate_shared_feature_federation import (
    CLIENTS,
    FEATURES,
)

from src.correlation.engine import EpisodeCorrelator
from src.correlation.models import CanonicalAlert

from src.federated.simulated_fedavg import (
    classification_metrics,
    predict,
    stratified_split,
    train_parameters,
)

from src.triage.metrics import (
    calculate_incident_recall_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
)

from src.triage.ranker import EpisodeRanker


SEEDS = (42, 43, 44)

TOP_K_VALUES = (25, 50, 100, 200)

INCIDENT_WINDOW_ROWS = 100


def load_records(
    path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Load records using the same eligibility behavior as the existing
    shared-feature federation experiment.

    Rows with missing shared features are skipped.
    Invalid non-numeric, negative, NaN, or infinite values raise an error.
    """

    records: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    labels: list[int] = []

    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            record: dict[str, Any] = json.loads(line)

            label = record.get("labels", {}).get("binary")

            if label not in (0, 1) or isinstance(label, bool):
                continue

            features = record.get("features", {})

            values = [
                features.get(name)
                for name in FEATURES
            ]

            # Rows with missing shared features are excluded.
            if any(value is None for value in values):
                continue

            # Other invalid values remain data errors.
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in values
            ):
                raise ValueError(
                    f"Invalid shared feature values at "
                    f"{path}:{line_no}"
                )

            provenance = record.get("provenance", {})

            source_row = provenance.get("source_row")
            record_id = provenance.get("record_id")

            if (
                not isinstance(source_row, int)
                or isinstance(source_row, bool)
            ):
                raise ValueError(
                    f"provenance.source_row missing/invalid at "
                    f"{path}:{line_no}"
                )

            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"provenance.record_id missing/invalid at "
                    f"{path}:{line_no}"
                )

            records.append(record)

            vectors.append(
                [
                    math.log1p(float(value))
                    for value in values
                ]
            )

            labels.append(int(label))

    if not records:
        raise ValueError(
            f"No eligible records found in {path}"
        )

    if set(labels) != {0, 1}:
        raise ValueError(
            f"{path} must contain both binary classes"
        )

    x = np.asarray(
        vectors,
        dtype=np.float64,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    return records, x, y


def build_incident_manifest(
    records: list[dict[str, Any]],
    labels: np.ndarray,
) -> dict[int, str]:
    """
    Define controlled benchmark incidents.

    Each fixed source-row window containing at least one held-out
    attack row becomes one benchmark incident.
    """

    attack_windows = set()

    for record, label in zip(
        records,
        labels,
        strict=True,
    ):
        if int(label) != 1:
            continue

        source_row = int(
            record["provenance"]["source_row"]
        )

        window = (
            source_row - 1
        ) // INCIDENT_WINDOW_ROWS

        attack_windows.add(window)

    dataset_id = str(
        records[0]["provenance"]["dataset_id"]
    ).upper()

    return {
        window: (
            f"SIM-{dataset_id}-"
            f"ROW-WINDOW-{window + 1:04d}"
        )
        for window in sorted(attack_windows)
    }


def build_episodes(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    incident_manifest: dict[int, str],
) -> list[Any]:
    """
    Convert every held-out row into a candidate alert.

    There is deliberately no probability >= 0.5 threshold.

    The experiment evaluates analyst-budget prioritization, so every
    held-out record is eligible for the ranking queue.
    """

    alerts: list[CanonicalAlert] = []

    for record, label, probability in zip(
        records,
        labels,
        probabilities,
        strict=True,
    ):
        source_row = int(
            record["provenance"]["source_row"]
        )

        window = (
            source_row - 1
        ) // INCIDENT_WINDOW_ROWS

        dataset_id = str(
            record["provenance"]["dataset_id"]
        ).upper()

        correlation_key = (
            f"SIM-{dataset_id}-"
            f"ROW-WINDOW-{window + 1:04d}"
        )

        incident_id = None

        if int(label) == 1:
            incident_id = incident_manifest.get(window)

        alerts.append(
            CanonicalAlert(
                alert_id=str(
                    record["provenance"]["record_id"]
                ),
                timestamp=None,
                src_ip="",
                dst_ip="",
                src_port=0,
                dst_port=0,
                protocol="",
                detection_confidence=float(
                    probability
                ),
                is_attack=bool(label),
                incident_id=incident_id,
                raw_metadata={
                    "source_row": source_row,
                },
                correlation_key=correlation_key,
            )
        )

    return EpisodeCorrelator().correlate_alerts(
        alerts
    )


def rank_by_confidence(
    episodes: list[Any],
) -> list[tuple[Any, float]]:
    """
    Confidence-only ranking baseline.
    """

    scored = [
        (
            episode,
            float(episode.max_confidence),
        )
        for episode in episodes
    ]

    return sorted(
        scored,
        key=lambda item: (
            item[1],
            item[0].max_confidence,
            item[0].alert_count,
        ),
        reverse=True,
    )


def random_rank(
    episodes: list[Any],
    seed: int,
) -> list[tuple[Any, float]]:
    """
    Deterministic random ranking baseline.
    """

    rng = random.Random(seed)

    shuffled = list(episodes)

    rng.shuffle(shuffled)

    return [
        (
            episode,
            0.0,
        )
        for episode in shuffled
    ]


def rank_with_episode_ranker(
    episodes: list[Any],
) -> list[tuple[Any, float]]:
    """
    Use the existing TrustMesh EpisodeRanker.

    The ranker returns the analyst-prioritized episode queue.
    """

    ranker = EpisodeRanker()

    return ranker.rank_episodes(
        episodes
    )


def evaluate_queue(
    ranked_queue: list[tuple[Any, float]],
    episodes: list[Any],
    incident_manifest: dict[int, str],
) -> dict[str, float]:
    """
    Calculate Incident Recall@K, Precision@K and NDCG@K.
    """

    ground_truth_ids = set(
        incident_manifest.values()
    )

    result: dict[str, float] = {}

    for k in TOP_K_VALUES:

        result[
            f"incident_recall_at_{k}"
        ] = calculate_incident_recall_at_k(
            ranked_queue,
            episodes,
            k=k,
            ground_truth_incident_ids=ground_truth_ids,
        )

        result[
            f"precision_at_{k}"
        ] = calculate_precision_at_k(
            ranked_queue,
            k=k,
        )

        result[
            f"ndcg_at_{k}"
        ] = calculate_ndcg_at_k(
            ranked_queue,
            k=k,
        )

    return result


def probabilities_from_params(
    x_values: np.ndarray,
    params: list[np.ndarray],
) -> np.ndarray:
    """
    Convert logistic model parameters into probabilities.

    p = 1 / (1 + exp(-(x.w + b)))
    """

    weights = np.asarray(
        params[0],
        dtype=np.float64,
    )

    bias = float(
        np.asarray(
            params[1]
        ).reshape(-1)[0]
    )

    logits = (
        x_values @ weights
    ) + bias

    logits = np.clip(
        logits,
        -700.0,
        700.0,
    )

    return 1.0 / (
        1.0 + np.exp(-logits)
    )


def federated_global_parameters(
    local_params: dict[str, list[np.ndarray]],
    training_counts: dict[str, int],
) -> list[np.ndarray]:
    """
    Perform one Flower FedAvg round using the existing project model.
    """

    fit_results = []

    for client_name, params in local_params.items():

        fit_results.append(
            (
                None,
                FitRes(
                    status=Status(
                        code=Code.OK,
                        message="",
                    ),
                    parameters=ndarrays_to_parameters(
                        params
                    ),
                    num_examples=training_counts[
                        client_name
                    ],
                    metrics={},
                ),
            )
        )

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=len(local_params),
        min_available_clients=len(local_params),
        accept_failures=False,
        fit_metrics_aggregation_fn=lambda _: {},
    )

    aggregate, _ = strategy.aggregate_fit(
        server_round=1,
        results=fit_results,
        failures=[],
    )

    if aggregate is None:
        raise RuntimeError(
            "Flower FedAvg returned no global parameters"
        )

    return parameters_to_ndarrays(
        aggregate
    )


def evaluate_seed(
    seed: int,
) -> dict[str, Any]:
    """
    Run one complete seed of the experiment.
    """

    clients = {
        name: load_records(path)
        for name, path in CLIENTS.items()
    }

    raw_partitions: dict[
        str,
        tuple[
            list[dict[str, Any]],
            np.ndarray,
            np.ndarray,
            list[int],
            list[int],
        ],
    ] = {}

    # ---------------------------------------------------------
    # 1. Deterministic train/evaluation partitions.
    # ---------------------------------------------------------

    for name, (
        records,
        x,
        y,
    ) in clients.items():

        train_idx, eval_idx = stratified_split(
            y,
            seed=seed,
        )

        raw_partitions[name] = (
            records,
            x,
            y,
            train_idx.tolist(),
            eval_idx.tolist(),
        )

    # ---------------------------------------------------------
    # 2. Shared scaler fitted ONLY on training partitions.
    # ---------------------------------------------------------

    pooled_train = np.concatenate(
        [
            x[
                np.asarray(
                    train_idx,
                    dtype=np.int64,
                )
            ]
            for (
                _,
                x,
                _,
                train_idx,
                _,
            ) in raw_partitions.values()
        ],
        axis=0,
    )

    scaler_mean = pooled_train.mean(
        axis=0
    )

    scaler_scale = pooled_train.std(
        axis=0,
        ddof=0,
    )

    scaler_scale[
        scaler_scale == 0.0
    ] = 1.0

    # ---------------------------------------------------------
    # 3. Train local models.
    # ---------------------------------------------------------

    local_params: dict[
        str,
        list[np.ndarray],
    ] = {}

    training_counts: dict[
        str,
        int,
    ] = {}

    evaluation_data: dict[
        str,
        tuple[
            list[dict[str, Any]],
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    local_classification: dict[
        str,
        dict[str, float],
    ] = {}

    for name, (
        records,
        x,
        y,
        train_idx,
        eval_idx,
    ) in raw_partitions.items():

        train_idx_array = np.asarray(
            train_idx,
            dtype=np.int64,
        )

        eval_idx_array = np.asarray(
            eval_idx,
            dtype=np.int64,
        )

        x_train = (
            x[train_idx_array]
            - scaler_mean
        ) / scaler_scale

        x_eval = (
            x[eval_idx_array]
            - scaler_mean
        ) / scaler_scale

        y_train = y[
            train_idx_array
        ]

        y_eval = y[
            eval_idx_array
        ]

        initial_params = [
            np.zeros(
                len(FEATURES),
                dtype=np.float64,
            ),
            np.zeros(
                1,
                dtype=np.float64,
            ),
        ]

        local_params[name] = train_parameters(
            x_train,
            y_train,
            initial_params,
        )

        training_counts[name] = len(
            y_train
        )

        eval_records = [
            records[int(index)]
            for index in eval_idx_array
        ]

        evaluation_data[name] = (
            eval_records,
            x_eval,
            y_eval,
        )

        local_predictions = predict(
            x_eval,
            local_params[name],
        )

        local_classification[name] = (
            classification_metrics(
                y_eval,
                local_predictions,
            )
        )

    # ---------------------------------------------------------
    # 4. One-round Flower FedAvg.
    # ---------------------------------------------------------

    global_params = federated_global_parameters(
        local_params,
        training_counts,
    )

    # ---------------------------------------------------------
    # 5. Evaluate analyst-budget triage.
    # ---------------------------------------------------------

    client_results: dict[str, Any] = {}

    for name, (
        eval_records,
        x_eval,
        y_eval,
    ) in evaluation_data.items():

        local_probabilities = (
            probabilities_from_params(
                x_eval,
                local_params[name],
            )
        )

        global_probabilities = (
            probabilities_from_params(
                x_eval,
                global_params,
            )
        )

        # Ground truth is constructed ONLY from held-out rows.
        incident_manifest = (
            build_incident_manifest(
                eval_records,
                y_eval,
            )
        )

        local_episodes = build_episodes(
            eval_records,
            y_eval,
            local_probabilities,
            incident_manifest,
        )

        federated_episodes = build_episodes(
            eval_records,
            y_eval,
            global_probabilities,
            incident_manifest,
        )

        # -----------------------------------------------------
        # Four ranking conditions.
        # -----------------------------------------------------

        # 1. Local detector + full TrustMesh EpisodeRanker.
        local_ranked = rank_with_episode_ranker(
            local_episodes
        )

        # 2. Federated detector + full TrustMesh EpisodeRanker.
        federated_ranked = rank_with_episode_ranker(
            federated_episodes
        )

        # 3. Detector confidence only.
        confidence_ranked = rank_by_confidence(
            local_episodes
        )

        # 4. Random baseline.
        random_ranked = random_rank(
            local_episodes,
            seed=seed,
        )

        client_results[name] = {
            "evaluation_rows": len(y_eval),
            "evaluation_positive_rows": int(
                np.sum(y_eval == 1)
            ),
            "evaluation_negative_rows": int(
                np.sum(y_eval == 0)
            ),
            "benchmark_incidents": len(
                incident_manifest
            ),
            "episodes": len(
                local_episodes
            ),
            "local_classification": (
                local_classification[name]
            ),
            "methods": {
                "local": evaluate_queue(
                    local_ranked,
                    local_episodes,
                    incident_manifest,
                ),
                "federated_global": evaluate_queue(
                    federated_ranked,
                    federated_episodes,
                    incident_manifest,
                ),
                "confidence_only": evaluate_queue(
                    confidence_ranked,
                    local_episodes,
                    incident_manifest,
                ),
                "random": evaluate_queue(
                    random_ranked,
                    local_episodes,
                    incident_manifest,
                ),
            },
        }

    return {
        "seed": seed,
        "features": list(FEATURES),
        "preprocessing": (
            "log1p followed by shared scaling fitted "
            "only on concatenated training partitions"
        ),
        "federation": (
            "one in-memory Flower FedAvg round weighted "
            "by client training row count"
        ),
        "triage": (
            "Existing TrustMesh EpisodeRanker using "
            "confidence, evidence diversity, stage coverage "
            "and asset criticality."
        ),
        "incident_definition": (
            "A fixed 100-source-row window containing at "
            "least one held-out attack row is one controlled "
            "benchmark incident."
        ),
        "top_k_values": list(
            TOP_K_VALUES
        ),
        "clients": client_results,
    }


def summarize(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate mean and sample standard deviation across seeds.
    """

    summary: dict[str, Any] = {}

    methods = (
        "local",
        "federated_global",
        "confidence_only",
        "random",
    )

    for client_name in CLIENTS:

        summary[client_name] = {}

        for method in methods:

            summary[client_name][method] = {}

            for k in TOP_K_VALUES:

                metric_names = (
                    f"incident_recall_at_{k}",
                    f"precision_at_{k}",
                    f"ndcg_at_{k}",
                )

                for metric in metric_names:

                    values = [
                        result[
                            "clients"
                        ][
                            client_name
                        ][
                            "methods"
                        ][
                            method
                        ][metric]
                        for result in results
                    ]

                    summary[
                        client_name
                    ][
                        method
                    ][
                        metric
                    ] = {
                        "mean": statistics.mean(
                            values
                        ),
                        "sample_std": (
                            statistics.stdev(
                                values
                            )
                            if len(values) > 1
                            else 0.0
                        ),
                    }

    return summary


def main() -> None:
    results = []

    for seed in SEEDS:
        print(
            f"Running seed {seed}...",
            flush=True,
        )

        results.append(
            evaluate_seed(seed)
        )

    output = {
        "experiment": (
            "Local vs Federated analyst-budget "
            "triage on heterogeneous simulated clients"
        ),
        "seeds": list(SEEDS),
        "top_k_values": list(
            TOP_K_VALUES
        ),
        "per_seed": results,
        "mean_and_sample_std": summarize(
            results
        ),
    }

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
