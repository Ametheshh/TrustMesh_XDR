"""FedProx regularized federated learning solver and server aggregator."""

from typing import Dict, List, Tuple
import numpy as np


class FedProxTrainer:
    """Solves local client optimization with FedProx proximal regularization penalty."""

    def __init__(self, mu: float = 0.01, learning_rate: float = 0.05, epochs: int = 20):
        """Initialize FedProx local trainer.

        Args:
            mu: Proximal regularization parameter (mu=0 reduces to standard FedAvg).
            learning_rate: Gradient descent learning rate.
            epochs: Local optimization epochs per round.
        """
        self.mu = mu
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_global: np.ndarray,
        b_global: float,
    ) -> Tuple[np.ndarray, float]:
        """Perform local gradient descent with proximal penalty towards global weights.

        Objective: Loss(w, b) + (mu / 2) * ||w - w_global||^2 + (mu / 2) * (b - b_global)^2
        """
        w = np.copy(w_global)
        b = float(b_global)
        n_samples = len(y)

        if n_samples == 0:
            return w, b

        for _ in range(self.epochs):
            # Compute linear predictions and sigmoid probabilities
            z = np.dot(X, w) + b
            z_clipped = np.clip(z, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-z_clipped))

            # Standard binary cross-entropy gradient
            grad_w = np.dot(X.T, (probs - y)) / n_samples
            grad_b = np.sum(probs - y) / n_samples

            # Add FedProx proximal penalty gradients
            if self.mu > 0.0:
                grad_w += self.mu * (w - w_global)
                grad_b += self.mu * (b - b_global)

            # Gradient descent update step
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        return w, b


def aggregate_fedprox_weights(
    client_updates: List[Tuple[np.ndarray, float, int]],
) -> Tuple[np.ndarray, float]:
    """Perform sample-weighted parameter aggregation (FedAvg/FedProx server step).

    Args:
        client_updates: List of tuples (client_weights, client_bias, n_samples).

    Returns:
        Aggregated (w_global, b_global) tuple.
    """
    total_samples = sum(n for _, _, n in client_updates)
    if total_samples == 0:
        raise ValueError("Cannot aggregate zero client samples")

    w_dim = client_updates[0][0].shape[0]
    w_global = np.zeros(w_dim, dtype=np.float32)
    b_global = 0.0

    for w_k, b_k, n_k in client_updates:
        weight_factor = n_k / total_samples
        w_global += weight_factor * w_k
        b_global += weight_factor * b_k

    return w_global, float(b_global)
