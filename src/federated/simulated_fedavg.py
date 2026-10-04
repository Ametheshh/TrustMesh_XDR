"""Controlled simulation using dataset-derived CTU-SME and UNSW clients.

The default mode uses Flower FedAvg aggregation in memory. ``--mode networked``
starts one localhost Flower server process and one process per dataset-derived
client. ``--mode networked-dp`` adds experimental client-update clipping and
Gaussian noise, without a formal privacy accountant or epsilon guarantee. None
of these modes represents independent organizations. ``networked-secure`` is a
controlled two-client secure-aggregation proof of concept using deterministic
pairwise masks; it is not production-grade or cryptographically secure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CLIENTS = {
    "ctu_sme": Path("data/canonical/smoke_test/ctu_unsw_fl_poc/ctu_sme_total_packets.jsonl"),
    "unsw_nb15_named": Path("data/canonical/smoke_test/ctu_unsw_fl_poc/unsw_nb15_named_total_packets.jsonl"),
}
SPLIT_SEED = 42
TEST_FRACTION = 0.2
LOCAL_EPOCHS = 500
LEARNING_RATE = 0.1
L2 = 1e-4
PARAMETER_ORDER = ("weight_total_packets", "bias")
NETWORK_ROUNDS = 1
NETWORK_HOST = "127.0.0.1"
_RESULT_PREFIX = "TRUSTMESH_RESULT="
DP_DEFAULTS = {"clip_norm": 1.0, "noise_multiplier": 1.0, "random_seed": 2026}


def load_client_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the approved one-feature and binary-label channels."""
    values: list[float] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            features = record.get("features")
            label = record.get("labels", {}).get("binary")
            value = features.get("total_packets") if isinstance(features, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"features.total_packets must be finite numeric at {path}:{line_number}")
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(f"labels.binary must be 0 or 1 at {path}:{line_number}")
            values.append(math.log1p(float(value)))
            labels.append(int(label))
    if not values or set(labels) != {0, 1}:
        raise ValueError(f"{path} must contain records from both binary classes")
    # Fixed, identical feature transform at both clients. No client-specific statistics.
    return np.asarray(values, dtype=np.float64).reshape(-1, 1), np.asarray(labels, dtype=np.int64)


def stratified_split(y: np.ndarray, *, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/test indices while retaining both classes."""
    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label).tolist()
        if len(indices) < 2:
            raise ValueError("each client needs at least two rows per binary class")
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * TEST_FRACTION))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def initial_parameters() -> list[np.ndarray]:
    """Return the shared parameter ordering: [weight_total_packets, bias]."""
    return [np.zeros((1,), dtype=np.float64), np.zeros((1,), dtype=np.float64)]


def validate_dp_config(clip_norm: float, noise_multiplier: float, random_seed: int) -> dict[str, float | int]:
    """Validate the experimental update-clipping/noise settings."""
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("DP clipping norm must be finite and greater than zero")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("DP noise multiplier must be finite and non-negative")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("DP random seed must be a non-negative integer")
    return {
        "clip_norm": float(clip_norm),
        "noise_multiplier": float(noise_multiplier),
        "random_seed": random_seed,
    }


def _client_noise_seed(random_seed: int, client_id: str) -> int:
    """Derive a stable, client-specific RNG seed from the configured base seed."""
    digest = hashlib.sha256(f"{random_seed}:{client_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def privatize_model_update(
    base_parameters: list[np.ndarray],
    trained_parameters: list[np.ndarray],
    *,
    client_id: str,
    clip_norm: float,
    noise_multiplier: float,
    random_seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Clip one client delta as a vector and add isotropic Gaussian noise locally.

    Returns model parameters reconstructed as base + noisy_delta for Flower's
    FedAvg strategy. No raw records or unnoised update arrays are returned.
    """
    config = validate_dp_config(clip_norm, noise_multiplier, random_seed)
    if len(base_parameters) != len(trained_parameters):
        raise ValueError("base and trained model parameter lists must have equal length")
    deltas = []
    for base, trained in zip(base_parameters, trained_parameters, strict=True):
        base_array = np.asarray(base, dtype=np.float64)
        trained_array = np.asarray(trained, dtype=np.float64)
        if base_array.shape != trained_array.shape:
            raise ValueError("base and trained parameter shapes must match")
        deltas.append(trained_array - base_array)
    before_norm = math.sqrt(sum(float(np.sum(delta * delta)) for delta in deltas))
    scale = min(1.0, config["clip_norm"] / before_norm) if before_norm > 0 else 1.0
    clipped_deltas = [delta * scale for delta in deltas]
    clipped_norm = math.sqrt(sum(float(np.sum(delta * delta)) for delta in clipped_deltas))
    client_seed = _client_noise_seed(config["random_seed"], client_id)
    rng = np.random.default_rng(client_seed)
    noise_std = config["noise_multiplier"] * config["clip_norm"]
    noisy_deltas = [
        delta + rng.normal(loc=0.0, scale=noise_std, size=delta.shape)
        for delta in clipped_deltas
    ]
    after_norm = math.sqrt(sum(float(np.sum(delta * delta)) for delta in noisy_deltas))
    transmitted = [np.asarray(base, dtype=np.float64) + delta
                   for base, delta in zip(base_parameters, noisy_deltas, strict=True)]
    return transmitted, {
        "clipped": before_norm > config["clip_norm"],
        "clip_norm": config["clip_norm"],
        "noise_multiplier": config["noise_multiplier"],
        "noise_std": noise_std,
        "update_norm_before_clipping": before_norm,
        "update_norm_after_clipping": clipped_norm,
        "update_norm_after_clipping_and_noise": after_norm,
        "random_seed": config["random_seed"],
    }


def pairwise_mask(parameters: list[np.ndarray], random_seed: int) -> list[np.ndarray]:
    """Create a repeatable shared mask matching the model parameter shapes."""
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("secure aggregation random seed must be a non-negative integer")
    rng = np.random.default_rng(random_seed)
    return [rng.standard_normal(np.asarray(parameter).shape) for parameter in parameters]


def mask_model_update(
    base_parameters: list[np.ndarray],
    trained_parameters: list[np.ndarray],
    *,
    client_id: str,
    num_examples: int,
    random_seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Mask a model delta so weighted Flower FedAvg cancels the pairwise mask.

    Each client transmits base + delta + sign*mask/num_examples. Since Flower
    weights returned parameters by ``num_examples``, signed mask contributions
    cancel in the aggregate while preserving ordinary example-weighted FedAvg.
    """
    if client_id not in {"ctu_sme", "unsw_nb15_named"}:
        raise ValueError("secure aggregation requires one of the two approved clients")
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if len(base_parameters) != len(trained_parameters):
        raise ValueError("base and trained model parameter lists must have equal length")
    deltas = []
    for base, trained in zip(base_parameters, trained_parameters, strict=True):
        base_array = np.asarray(base, dtype=np.float64)
        trained_array = np.asarray(trained, dtype=np.float64)
        if base_array.shape != trained_array.shape:
            raise ValueError("base and trained parameter shapes must match")
        deltas.append(trained_array - base_array)
    masks = pairwise_mask(deltas, random_seed)
    sign = 1.0 if client_id == "ctu_sme" else -1.0
    masked_deltas = [delta + sign * mask / num_examples
                     for delta, mask in zip(deltas, masks, strict=True)]
    transmitted = [base + delta for base, delta in zip(base_parameters, masked_deltas, strict=True)]
    return transmitted, {
        "client_id": client_id,
        "num_examples": num_examples,
        "mask_sign": int(sign),
        "random_seed": random_seed,
        "unmasked_update_sent": False,
    }


def train_parameters(
    x: np.ndarray,
    y: np.ndarray,
    parameters: list[np.ndarray],
    *,
    epochs: int = LOCAL_EPOCHS,
) -> list[np.ndarray]:
    """Train one weighted binary logistic-regression model locally."""
    weights = np.asarray(parameters[0], dtype=np.float64).copy()
    bias = float(np.asarray(parameters[1], dtype=np.float64)[0])
    counts = np.bincount(y, minlength=2).astype(np.float64)
    class_weights = len(y) / (2.0 * counts)
    sample_weights = class_weights[y]
    scale = float(sample_weights.sum())
    for _ in range(epochs):
        logits = np.clip((x @ weights).reshape(-1) + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = (probabilities - y) * sample_weights
        gradient_w = (x.T @ residual) / scale + L2 * weights
        gradient_b = float(residual.sum() / scale)
        weights -= LEARNING_RATE * gradient_w
        bias -= LEARNING_RATE * gradient_b
    return [weights, np.asarray([bias], dtype=np.float64)]


def predict(x: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
    weights, bias = parameters
    logits = np.clip((x @ weights).reshape(-1) + float(bias[0]), -30.0, 30.0)
    return (logits >= 0.0).astype(np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def run_experiment(client_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    """Run client-local training and one in-memory Flower FedAvg aggregation round."""
    from flwr.server.strategy import FedAvg

    paths = client_paths or DEFAULT_CLIENTS
    if len(paths) != 2:
        raise ValueError("this controlled proof of concept requires exactly two clients")

    client_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    local_results: dict[str, dict[str, Any]] = {}
    for client_id, path in paths.items():
        x, y = load_client_data(path)
        train_idx, test_idx = stratified_split(y)
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        client_data[client_id] = (x_train, y_train, x_test, y_test)
        local_params = train_parameters(x_train, y_train, initial_parameters())
        local_results[client_id] = {
            "training_rows": int(len(y_train)),
            "evaluation_rows": int(len(y_test)),
            "evaluation_label_counts": {str(label): int(np.sum(y_test == label)) for label in (0, 1)},
            "metrics": classification_metrics(y_test, predict(x_test, local_params)),
        }

    # Each simulated client trains from the same initial vector locally. The only
    # values handed to Flower's server strategy are these model arrays and counts.
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        accept_failures=False,
        fit_metrics_aggregation_fn=lambda _: {},
    )
    fit_results = []
    for client_id in paths:
        x_train, y_train, _, _ = client_data[client_id]
        client_parameters = train_parameters(x_train, y_train, initial_parameters())
        fit_results.append(
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(client_parameters),
                    num_examples=len(y_train),
                    metrics={},
                ),
            )
        )
    aggregated_parameters, _ = strategy.aggregate_fit(1, fit_results, failures=[])
    if aggregated_parameters is None:
        raise RuntimeError("Flower FedAvg did not produce global parameters")
    global_parameters = parameters_to_ndarrays(aggregated_parameters)

    global_metrics: dict[str, dict[str, Any]] = {}
    for client_id in paths:
        _, _, x_test, y_test = client_data[client_id]
        global_metrics[client_id] = classification_metrics(y_test, predict(x_test, global_parameters))

    return {
        "experiment": "controlled simulated federation using dataset-derived clients; not real organizations",
        "framework": "Flower FedAvg",
        "rounds": 1,
        "clients": {client_id: str(path) for client_id, path in paths.items()},
        "model": "binary logistic regression; one coefficient and intercept; class-weighted full-batch gradient descent",
        "parameter_order": list(PARAMETER_ORDER),
        "feature_input": "[features.total_packets] transformed identically with log1p",
        "target": "labels.binary (0=benign/normal, 1=attack/malicious)",
        "local_epochs": LOCAL_EPOCHS,
        "local_training": local_results,
        "global_model_evaluation": global_metrics,
        "communication": "in-memory simulated client-to-server model arrays and training example counts only; no raw records sent",
        "fedavg_weighting": "number of local training examples",
        "execution": "in-memory Flower FedAvg strategy aggregation; no network transport or real organizations",
    }


def _history_summary(history: Any, strategy: Any) -> dict[str, Any]:
    """Convert Flower's round-indexed history lists into bounded JSON values."""
    losses = [
        {"round": int(server_round), "loss": float(loss)}
        for server_round, loss in history.losses_distributed
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for name, values in history.metrics_distributed.items():
        for server_round, value in values:
            metrics.setdefault(str(server_round), {})[name] = value
    return {
        "completed_rounds": len(losses),
        "distributed_evaluation": losses,
        "client_evaluation_metrics": metrics,
        "global_parameters": [array.tolist() for array in (strategy.final_parameters or [])],
    }


def _run_flower_server(address: str) -> None:
    """Worker entrypoint: start one Flower server and emit its final summary."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg

    def aggregate_client_metrics(results):
        aggregated: dict[str, float | int | str] = {}
        for num_examples, client_metrics in results:
            client_id = str(client_metrics["client_id"])
            aggregated[f"{client_id}_evaluation_rows"] = int(num_examples)
            for name, value in client_metrics.items():
                if name != "client_id":
                    aggregated[f"{client_id}_{name}"] = value
        return aggregated

    class CapturingFedAvg(FedAvg):
        final_parameters: list[np.ndarray] | None = None

        def aggregate_fit(self, server_round, results, failures):
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            if parameters is not None:
                self.final_parameters = parameters_to_ndarrays(parameters)
            return parameters, metrics

    strategy = CapturingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        accept_failures=False,
        initial_parameters=ndarrays_to_parameters(initial_parameters()),
        fit_metrics_aggregation_fn=lambda _: {},
        evaluate_metrics_aggregation_fn=aggregate_client_metrics,
    )
    history = start_server(
        server_address=address,
        config=ServerConfig(num_rounds=NETWORK_ROUNDS, round_timeout=120),
        strategy=strategy,
    )
    summary = _history_summary(history, strategy)
    print(_RESULT_PREFIX + json.dumps({"role": "server", **summary}), flush=True)


def _run_flower_client(
    client_id: str,
    input_path: Path,
    address: str,
    dp_config: dict[str, float | int] | None = None,
    secure_seed: int | None = None,
) -> None:
    """Worker entrypoint: load one client's input and participate in Flower rounds."""
    from flwr.client import NumPyClient, start_client

    x, y = load_client_data(input_path)
    train_idx, test_idx = stratified_split(y)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    local_parameters = train_parameters(x_train, y_train, initial_parameters())
    local_metrics = classification_metrics(y_test, predict(x_test, local_parameters))

    class Client(NumPyClient):
        global_metrics: dict[str, Any] | None = None
        update_diagnostics: dict[str, Any] | None = None
        secure_diagnostics: dict[str, Any] | None = None

        def get_parameters(self, config):
            del config
            return initial_parameters()

        def fit(self, parameters, config):
            del config
            updated = train_parameters(x_train, y_train, parameters)
            if secure_seed is not None:
                updated, self.secure_diagnostics = mask_model_update(
                    parameters, updated, client_id=client_id,
                    num_examples=len(y_train), random_seed=secure_seed,
                )
            if dp_config is not None:
                updated, self.update_diagnostics = privatize_model_update(
                    parameters,
                    updated,
                    client_id=client_id,
                    clip_norm=float(dp_config["clip_norm"]),
                    noise_multiplier=float(dp_config["noise_multiplier"]),
                    random_seed=int(dp_config["random_seed"]),
                )
            return updated, len(y_train), {"client_id": client_id}

        def evaluate(self, parameters, config):
            del config
            self.global_metrics = classification_metrics(y_test, predict(x_test, parameters))
            tn, fp = self.global_metrics["confusion_matrix"][0]
            fn, tp = self.global_metrics["confusion_matrix"][1]
            metrics = {
                "client_id": client_id,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": float(self.global_metrics["precision"]),
                "recall": float(self.global_metrics["recall"]),
                "f1": float(self.global_metrics["f1"]),
                "balanced_accuracy": float(self.global_metrics["balanced_accuracy"]),
            }
            return 0.0, len(y_test), metrics

    client = Client()
    start_client(
        server_address=address,
        client=client.to_client(),
        max_retries=50,
        max_wait_time=90,
        insecure=True,
    )
    if client.global_metrics is None:
        raise RuntimeError(f"Flower client {client_id} did not receive global evaluation")
    result = {
        "role": "client",
        "client_id": client_id,
        "training_rows": int(len(y_train)),
        "evaluation_rows": int(len(y_test)),
        "local_metrics": local_metrics,
        "global_metrics": client.global_metrics,
    }
    if dp_config is not None:
        if client.update_diagnostics is None:
            raise RuntimeError(f"Flower client {client_id} did not produce DP update diagnostics")
        result["dp_update"] = client.update_diagnostics
    if secure_seed is not None:
        if client.secure_diagnostics is None:
            raise RuntimeError(f"Flower client {client_id} did not produce secure-mask diagnostics")
        result["secure_update"] = client.secure_diagnostics
    print(_RESULT_PREFIX + json.dumps(result), flush=True)


def _result_line(output: str, role: str) -> dict[str, Any]:
    for line in output.splitlines():
        if line.startswith(_RESULT_PREFIX):
            result = json.loads(line[len(_RESULT_PREFIX):])
            if result.get("role") == role:
                return result
    raise RuntimeError(f"Flower {role} process did not emit a result summary; output tail: {output[-1200:]}")


def run_networked_experiment(
    client_paths: dict[str, Path] | None = None,
    *,
    dp_config: dict[str, float | int] | None = None,
    secure_seed: int | None = None,
) -> dict[str, Any]:
    """Run one actual localhost Flower round with separate server and client processes."""
    paths = client_paths or DEFAULT_CLIENTS
    if set(paths) != {"ctu_sme", "unsw_nb15_named"}:
        raise ValueError("networked mode requires exactly CTU-SME and UNSW-NB15 named clients")
    if dp_config is not None:
        dp_config = validate_dp_config(
            float(dp_config["clip_norm"]),
            float(dp_config["noise_multiplier"]),
            dp_config["random_seed"],
        )
    if secure_seed is not None:
        if isinstance(secure_seed, bool) or not isinstance(secure_seed, int) or secure_seed < 0:
            raise ValueError("secure aggregation random seed must be a non-negative integer")
        if dp_config is not None:
            raise ValueError("DP and secure aggregation modes must be run separately")

    try:
        with socket.socket() as sock:
            sock.bind((NETWORK_HOST, 0))
            port = sock.getsockname()[1]
    except OSError as exc:
        raise RuntimeError("localhost sockets are unavailable; run networked mode where loopback is permitted") from exc
    address = f"{NETWORK_HOST}:{port}"
    module_command = [sys.executable, "-m", "src.federated.simulated_fedavg"]
    server = subprocess.Popen(
        [*module_command, "--worker-role", "server", "--address", address],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    clients: dict[str, subprocess.Popen] = {}
    try:
        for client_id, input_path in paths.items():
            client_command = [
                *module_command,
                "--worker-role", "client",
                "--client-id", client_id,
                "--input", str(input_path),
                "--address", address,
            ]
            if dp_config is not None:
                client_command.extend([
                    "--dp-enabled",
                    "--dp-clip-norm", str(dp_config["clip_norm"]),
                    "--dp-noise-multiplier", str(dp_config["noise_multiplier"]),
                    "--dp-seed", str(dp_config["random_seed"]),
                ])
            if secure_seed is not None:
                client_command.extend(["--secure-enabled", "--secure-seed", str(secure_seed)])
            clients[client_id] = subprocess.Popen(
                client_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        client_results: dict[str, dict[str, Any]] = {}
        for client_id, process in clients.items():
            output, _ = process.communicate(timeout=150)
            if process.returncode != 0:
                raise RuntimeError(f"Flower client {client_id} failed: {output[-1200:]}")
            client_results[client_id] = _result_line(output, "client")
        server_output, _ = server.communicate(timeout=30)
        if server.returncode != 0:
            raise RuntimeError(f"Flower server failed: {server_output[-1600:]}")
        server_result = _result_line(server_output, "server")
    except Exception:
        for process in [*clients.values(), server]:
            if process.poll() is None:
                process.terminate()
        for process in [*clients.values(), server]:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise

    if server_result["completed_rounds"] != NETWORK_ROUNDS:
        raise RuntimeError(f"expected exactly {NETWORK_ROUNDS} completed Flower round")
    result = {
        "experiment": "controlled simulated federation using dataset-derived clients; not real organizations",
        "execution": "separate Flower server and client processes communicating over localhost",
        "framework": "Flower FedAvg",
        "rounds": server_result["completed_rounds"],
        "clients": client_results,
        "server_summary": server_result,
        "model": "binary logistic regression; one coefficient and intercept; class-weighted full-batch gradient descent",
        "parameter_order": list(PARAMETER_ORDER),
        "feature_input": "[features.total_packets] transformed identically with log1p",
        "target": "labels.binary (0=benign/normal, 1=attack/malicious)",
        "local_epochs": LOCAL_EPOCHS,
        "fedavg_weighting": "number of local training examples",
        "communication": "Flower gRPC localhost; model arrays, training example counts, and scalar evaluation summaries only",
    }
    if dp_config is not None:
        result["privacy_experiment"] = {
            "mechanism": "experimental client-update L2 clipping followed by Gaussian noise before Flower transmission",
            **dp_config,
            "epsilon_guarantee": None,
            "accounting_note": "No privacy accountant or formal epsilon guarantee is implemented; the seeded NumPy noise is for reproducible experimentation, not deployment privacy.",
        }
    if secure_seed is not None:
        result["secure_aggregation_experiment"] = {
            "description": "controlled two-client secure-aggregation proof of concept",
            "mechanism": "deterministic pairwise Gaussian mask on local model deltas with opposite client signs",
            "random_seed": secure_seed,
            "server_receives": "model parameters containing masked client deltas and training example counts; no unmasked client update is transmitted",
            "fedavg_weighting": "each signed mask is divided by that client's training-row count so example-weighted FedAvg cancels the masks",
            "production_grade": False,
            "cryptographic_security_claim": False,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("in-memory", "networked", "networked-dp", "networked-secure"), default="in-memory")
    parser.add_argument("--ctu", type=Path, default=DEFAULT_CLIENTS["ctu_sme"])
    parser.add_argument("--unsw", type=Path, default=DEFAULT_CLIENTS["unsw_nb15_named"])
    parser.add_argument("--worker-role", choices=("server", "client"), help=argparse.SUPPRESS)
    parser.add_argument("--client-id", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--address", help=argparse.SUPPRESS)
    parser.add_argument("--dp-enabled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--secure-enabled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dp-clip-norm", type=float, default=DP_DEFAULTS["clip_norm"])
    parser.add_argument("--dp-noise-multiplier", type=float, default=DP_DEFAULTS["noise_multiplier"])
    parser.add_argument("--dp-seed", type=int, default=DP_DEFAULTS["random_seed"])
    parser.add_argument("--secure-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.worker_role == "server":
        _run_flower_server(args.address)
    elif args.worker_role == "client":
        if not args.client_id or args.input is None or not args.address:
            parser.error("client worker requires --client-id, --input, and --address")
        dp_config = None
        if args.dp_enabled:
            dp_config = validate_dp_config(args.dp_clip_norm, args.dp_noise_multiplier, args.dp_seed)
        if args.dp_enabled and args.secure_enabled:
            parser.error("DP and secure aggregation worker modes cannot be combined")
        secure_seed = args.secure_seed if args.secure_enabled else None
        _run_flower_client(args.client_id, args.input, args.address, dp_config, secure_seed)
    elif args.mode == "networked":
        print(json.dumps(run_networked_experiment({"ctu_sme": args.ctu, "unsw_nb15_named": args.unsw}), indent=2))
    elif args.mode == "networked-dp":
        dp_config = validate_dp_config(args.dp_clip_norm, args.dp_noise_multiplier, args.dp_seed)
        print(json.dumps(run_networked_experiment(
            {"ctu_sme": args.ctu, "unsw_nb15_named": args.unsw}, dp_config=dp_config
        ), indent=2))
    elif args.mode == "networked-secure":
        print(json.dumps(run_networked_experiment(
            {"ctu_sme": args.ctu, "unsw_nb15_named": args.unsw}, secure_seed=args.secure_seed
        ), indent=2))
    else:
        print(json.dumps(run_experiment({"ctu_sme": args.ctu, "unsw_nb15_named": args.unsw}), indent=2))


if __name__ == "__main__":
    main()
