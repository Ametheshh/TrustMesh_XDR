"""Robust federated aggregators resilient against Byzantine and poisoning attacks."""

from typing import List, Tuple
import numpy as np


def aggregate_trimmed_mean(
    client_updates: List[Tuple[np.ndarray, float, int]],
    beta: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Trimmed Mean aggregation across client parameter updates.

    Trims top and bottom beta fraction of values per parameter coordinate.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates")

    n_clients = len(client_updates)
    weights_matrix = np.array([w for w, _, _ in client_updates])
    biases = np.array([b for _, b, _ in client_updates])

    k_trim = int(np.floor(n_clients * beta))

    if k_trim > 0 and (n_clients - 2 * k_trim) > 0:
        sorted_weights = np.sort(weights_matrix, axis=0)
        trimmed_weights = sorted_weights[k_trim : n_clients - k_trim]
        w_global = np.mean(trimmed_weights, axis=0)

        sorted_biases = np.sort(biases)
        trimmed_biases = sorted_biases[k_trim : n_clients - k_trim]
        b_global = float(np.mean(trimmed_biases))
    else:
        w_global = np.mean(weights_matrix, axis=0)
        b_global = float(np.mean(biases))

    return w_global, b_global


def aggregate_median(
    client_updates: List[Tuple[np.ndarray, float, int]],
) -> Tuple[np.ndarray, float]:
    """Coordinate-wise Median aggregation across client parameter updates."""
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates")

    weights_matrix = np.array([w for w, _, _ in client_updates])
    biases = np.array([b for _, b, _ in client_updates])

    w_global = np.median(weights_matrix, axis=0)
    b_global = float(np.median(biases))

    return w_global, b_global


def aggregate_krum(
    client_updates: List[Tuple[np.ndarray, float, int]],
    num_byzantine: int = 1,
) -> Tuple[np.ndarray, float]:
    """Krum aggregation algorithm selecting update closest to its nearest neighbors.

    Args:
        client_updates: List of (weights, bias, n_samples).
        num_byzantine: Assumed maximum number of malicious/poisoned clients.
    """
    n_clients = len(client_updates)
    if n_clients <= 2 * num_byzantine + 2:
        # Fall back to median if client count is too small for Krum bound
        return aggregate_median(client_updates)

    # Flatten weights and bias into one parameter vector per client
    vectors = [
        np.concatenate([w.flatten(), np.array([b])])
        for w, b, _ in client_updates
    ]

    scores = []
    # k closest neighbors to sum
    k_neighbors = n_clients - num_byzantine - 2

    for i in range(n_clients):
        distances = []
        for j in range(n_clients):
            if i != j:
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances.append(dist)
        distances.sort()
        score = sum(distances[:k_neighbors])
        scores.append(score)

    best_idx = int(np.argmin(scores))
    w_best, b_best, _ = client_updates[best_idx]
    return w_best, b_best
