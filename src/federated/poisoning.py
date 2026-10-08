"""Poisoning and backdoor attack simulators for federated robustness testing."""

from typing import Tuple
import numpy as np


def apply_label_flipping_attack(y: np.ndarray, flip_ratio: float = 1.0) -> np.ndarray:
    """Flip labels (1 -> 0 and 0 -> 1) for a proportion of training samples."""
    y_poisoned = np.copy(y)
    n_flip = int(len(y) * flip_ratio)
    if n_flip > 0:
        flip_indices = np.random.choice(len(y), size=n_flip, replace=False)
        y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
    return y_poisoned


def apply_weight_poisoning_attack(
    w: np.ndarray,
    b: float,
    scale_factor: float = -2.0,
) -> Tuple[np.ndarray, float]:
    """Invert or scale client weights to attempt to disrupt global model aggregation."""
    w_poisoned = w * scale_factor
    b_poisoned = b * scale_factor
    return w_poisoned, b_poisoned
