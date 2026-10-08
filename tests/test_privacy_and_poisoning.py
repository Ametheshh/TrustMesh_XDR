"""Unit tests for robust aggregators, poisoning attacks, and DP accountant."""

import numpy as np
import pytest
from src.federated.dp_accountant import RDPAccountant
from src.federated.poisoning import apply_label_flipping_attack, apply_weight_poisoning_attack
from src.federated.robust_aggregators import (
    aggregate_krum,
    aggregate_median,
    aggregate_trimmed_mean,
)


def test_robust_aggregators_poisoning_resilience():
    """Verify Trimmed Mean, Median, and Krum neutralize malicious updates."""
    clean_w1 = np.array([1.0, 1.1], dtype=np.float32)
    clean_w2 = np.array([0.9, 1.0], dtype=np.float32)
    clean_w3 = np.array([1.0, 0.9], dtype=np.float32)

    # Poisoned client sends extreme outlier (+100.0)
    poisoned_w = np.array([100.0, 100.0], dtype=np.float32)

    updates = [
        (clean_w1, 0.0, 100),
        (clean_w2, 0.0, 100),
        (clean_w3, 0.0, 100),
        (clean_w1, 0.0, 100),
        (poisoned_w, 0.0, 100),
    ]

    # Standard mean would be skewed by +100.0 (~20.0)
    mean_w = np.mean([w for w, _, _ in updates], axis=0)
    assert mean_w[0] > 15.0

    # Trimmed mean (beta=0.2) removes 1 highest and 1 lowest
    tm_w, _ = aggregate_trimmed_mean(updates, beta=0.2)
    assert tm_w[0] < 2.0

    # Median selects center value
    med_w, _ = aggregate_median(updates)
    assert med_w[0] < 2.0

    # Krum selects closest neighbor vector
    krum_w, _ = aggregate_krum(updates, num_byzantine=2)
    assert krum_w[0] < 2.0


def test_label_flipping_attack():
    """Verify label flipping attack flips target binary labels."""
    np.random.seed(42)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    y_flip = apply_label_flipping_attack(y, flip_ratio=1.0)

    # All labels flipped
    assert np.all(y_flip == (1 - y))


def test_weight_poisoning_attack():
    """Verify weight poisoning attack scaling."""
    w = np.array([1.0, 2.0], dtype=np.float32)
    b = 0.5
    w_p, b_p = apply_weight_poisoning_attack(w, b, scale_factor=-3.0)

    assert np.array_equal(w_p, np.array([-3.0, -6.0], dtype=np.float32))
    assert b_p == -1.5


def test_rdp_accountant_epsilon_bounds():
    """Verify Rényi Differential Privacy accountant calculations."""
    accountant = RDPAccountant(noise_multiplier=1.0, target_delta=1e-5)

    eps_1 = accountant.compute_epsilon(num_rounds=1)
    eps_5 = accountant.compute_epsilon(num_rounds=5)
    eps_10 = accountant.compute_epsilon(num_rounds=10)

    # Epsilon must increase monotonically with number of rounds
    assert 0 < eps_1 < eps_5 < eps_10

    # Higher noise multiplier reduces epsilon (stronger privacy guarantee)
    accountant_noisy = RDPAccountant(noise_multiplier=2.0, target_delta=1e-5)
    eps_10_noisy = accountant_noisy.compute_epsilon(num_rounds=10)
    assert eps_10_noisy < eps_10

    report = accountant.generate_privacy_report(rounds_list=[1, 5, 10])
    assert report["noise_multiplier"] == 1.0
    assert "eps_at_round_10" in report
