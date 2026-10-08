"""Unit tests for NeuralDetectorMLP and GradientBoostedDetector."""

import pytest
import numpy as np
from src.detection.neural_detector import NeuralDetectorMLP
from src.detection.tabular_detector import GradientBoostedDetector
from src.correlation.engine import EpisodeCorrelator
from src.correlation.models import CanonicalAlert
from src.triage.ranker import EpisodeRanker
from src.triage.metrics import calculate_incident_recall_at_k


@pytest.fixture
def synthetic_data():
    """Generate reproducible 40-feature synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 40

    # Class 0 (Benign): mean -1.0
    X_benign = np.random.randn(n_samples // 2, n_features) - 1.0
    y_benign = np.zeros(n_samples // 2, dtype=int)

    # Class 1 (Attack): mean +1.0
    X_attack = np.random.randn(n_samples // 2, n_features) + 1.0
    y_attack = np.ones(n_samples // 2, dtype=int)

    X = np.vstack([X_benign, X_attack])
    y = np.hstack([y_benign, y_attack])

    return X, y


def test_neural_detector_mlp_fit_predict(synthetic_data):
    """Verify NeuralDetectorMLP fits, predicts, and outputs calibrated probabilities."""
    X, y = synthetic_data
    detector = NeuralDetectorMLP(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)

    detector.fit(X, y)
    preds = detector.predict(X)
    probs = detector.predict_proba(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

    assert len(probs) == len(y)
    assert all(0.0 <= p <= 1.0 for p in probs)

    # High accuracy on easily separable synthetic data
    accuracy = sum(p == label for p, label in zip(preds, y)) / len(y)
    assert accuracy >= 0.85


def test_gradient_boosted_detector_fit_predict(synthetic_data):
    """Verify GradientBoostedDetector fits, predicts, and outputs valid probabilities."""
    X, y = synthetic_data
    detector = GradientBoostedDetector(max_iter=50, random_state=42)

    detector.fit(X, y)
    preds = detector.predict(X)
    probs = detector.predict_proba(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

    assert len(probs) == len(y)
    assert all(0.0 <= p <= 1.0 for p in probs)

    # High accuracy on easily separable synthetic data
    accuracy = sum(p == label for p, label in zip(preds, y)) / len(y)
    assert accuracy >= 0.85


def test_integration_detectors_with_triage_engine(synthetic_data):
    """Verify integration of neural predictions with EpisodeCorrelator and EpisodeRanker."""
    X, y = synthetic_data
    detector = NeuralDetectorMLP(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    detector.fit(X, y)
    probs = detector.predict_proba(X)

    alerts = []
    for idx, (prob, label) in enumerate(zip(probs, y)):
        alert = CanonicalAlert(
            alert_id=f"A-{idx}",
            timestamp=float(idx * 5.0),
            src_ip=f"192.168.1.{10 + (idx % 5)}",
            dst_ip=f"10.0.0.{(idx % 3) + 1}",
            src_port=1000 + idx,
            dst_port=80,
            protocol="tcp",
            detection_confidence=prob,
            is_attack=bool(label == 1),
            tactic="Execution" if label == 1 else "Benign",
            incident_id=f"INC-{label}",
        )
        alerts.append(alert)

    correlator = EpisodeCorrelator(time_window_seconds=100.0)
    episodes = correlator.correlate_alerts(alerts)

    ranker = EpisodeRanker(abstention_threshold=0.0)
    ranked_queue = ranker.rank_episodes(episodes, top_k=10)

    recall = calculate_incident_recall_at_k(ranked_queue, episodes, k=10)
    assert recall > 0.0
