"""Unit tests for FedProx, Personalized FL, and non-IID organization profiles."""

from pathlib import Path
import numpy as np
import pytest
from src.federated.fedprox import FedProxTrainer, aggregate_fedprox_weights
from src.federated.pfl import PersonalizedFLManager
from src.federated.profiles import OrganizationProfile, load_organization_profiles


def test_organization_profile_properties():
    """Verify OrganizationProfile properties and metrics."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = np.array([0, 1, 1], dtype=np.int32)
    profile = OrganizationProfile(
        name="BankTest",
        description="Bank Profile Test",
        dataset_name="UNSW",
        feature_names=["f1", "f2"],
        X=X,
        y=y,
    )

    assert profile.sample_count == 3
    assert profile.attack_ratio == pytest.approx(2.0 / 3.0)


def test_fedprox_trainer_proximal_penalty():
    """Verify FedProx proximal term constrains local weight drift."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    w_global = np.zeros(5, dtype=np.float32)
    b_global = 0.0

    # Train with mu = 0 (FedAvg) vs mu = 5.0 (Strong FedProx penalty)
    trainer_fedavg = FedProxTrainer(mu=0.0, learning_rate=0.1, epochs=20)
    w_fedavg, _ = trainer_fedavg.fit_local(X, y, w_global, b_global)

    trainer_fedprox = FedProxTrainer(mu=5.0, learning_rate=0.1, epochs=20)
    w_fedprox, _ = trainer_fedprox.fit_local(X, y, w_global, b_global)

    # FedProx weight drift from global should be significantly smaller than FedAvg drift
    drift_fedavg = np.linalg.norm(w_fedavg - w_global)
    drift_fedprox = np.linalg.norm(w_fedprox - w_global)

    assert drift_fedprox < drift_fedavg


def test_aggregate_fedprox_weights():
    """Verify sample-weighted parameter aggregation."""
    w1 = np.array([1.0, 2.0], dtype=np.float32)
    w2 = np.array([3.0, 4.0], dtype=np.float32)
    updates = [(w1, 1.0, 100), (w2, 2.0, 300)]

    w_agg, b_agg = aggregate_fedprox_weights(updates)

    # 100/400 * 1.0 + 300/400 * 3.0 = 0.25 + 2.25 = 2.5
    assert w_agg[0] == pytest.approx(2.5)
    # 100/400 * 2.0 + 300/400 * 4.0 = 0.5 + 3.0 = 3.5
    assert w_agg[1] == pytest.approx(3.5)
    # 100/400 * 1.0 + 300/400 * 2.0 = 0.25 + 1.5 = 1.75
    assert b_agg == pytest.approx(1.75)


def test_personalized_fl_manager_training():
    """Verify PersonalizedFLManager runs rounds and outputs client-specific models."""
    np.random.seed(42)
    X1 = np.random.randn(50, 4).astype(np.float32)
    y1 = np.ones(50, dtype=np.int32)

    X2 = np.random.randn(50, 4).astype(np.float32)
    y2 = np.zeros(50, dtype=np.int32)

    profiles = {
        "Client1": OrganizationProfile("Client1", "Desc1", "DS1", ["f1", "f2", "f3", "f4"], X1, y1),
        "Client2": OrganizationProfile("Client2", "Desc2", "DS2", ["f1", "f2", "f3", "f4"], X2, y2),
    }

    pfl_manager = PersonalizedFLManager(mu=0.01, learning_rate=0.05, fine_tune_epochs=5)
    w_global, b_global, personalized_models = pfl_manager.train_personalized(profiles, num_rounds=3)

    assert len(w_global) == 4
    assert "Client1" in personalized_models
    assert "Client2" in personalized_models

    w1, b1 = personalized_models["Client1"]
    w2, b2 = personalized_models["Client2"]

    # Personalized weights should differ between client 1 and client 2
    assert not np.allclose(w1, w2)
