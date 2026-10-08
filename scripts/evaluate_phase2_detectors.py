"""Benchmark script comparing Phase 2 detectors across full canonical feature schemas."""

import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import numpy as np
from sklearn.metrics import (  # type: ignore[import-not-found]
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]

from src.detection.local_baseline import LocalDetectorBaseline, load_feature_rows
from src.detection.neural_detector import NeuralDetectorMLP
from src.detection.tabular_detector import GradientBoostedDetector


def evaluate_detector(model_name: str, detector, X_train, y_train, X_valid, y_valid) -> dict:
    """Fit a detector model and evaluate performance metrics on held-out validation set."""
    detector.fit(X_train, y_train)
    preds = detector.predict(X_valid)
    probs = detector.predict_proba(X_valid)

    prec = float(precision_score(y_valid, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_valid, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_valid, preds, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_valid, preds))
    acc = float(accuracy_score(y_valid, preds))

    try:
        auc = float(roc_auc_score(y_valid, probs))
    except Exception:
        auc = 0.5

    return {
        "model_name": model_name,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "roc_auc": auc,
    }


def run_phase2_benchmarks():
    """Run comparative benchmark across Logistic Regression, Neural MLP, and Gradient Boosting."""
    print("=" * 75)
    print("      TrustMesh XDR - Phase 2 Detector Benchmark Evaluation")
    print("=" * 75)

    ciciot_path = base_dir / "data" / "canonical" / "smoke_test" / "ciciot23_train.jsonl"
    if not ciciot_path.exists():
        print(f"[!] Artifact not found: {ciciot_path}")
        return

    X, y, feature_names = load_feature_rows(ciciot_path, max_rows=10000)
    print(f"[*] Loaded dataset: {ciciot_path.name}")
    print(f"[*] Total rows: {len(y)} | Feature count: {len(feature_names)}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Train set: {len(y_train)} rows | Validation set: {len(y_valid)} rows\n")

    # 1. 5-Feature Baseline (First 5 features)
    X_train_5 = [row[:5] for row in X_train]
    X_valid_5 = [row[:5] for row in X_valid]
    log_reg_5 = LocalDetectorBaseline(random_state=42)
    res_5 = evaluate_detector("LogisticReg (5 Features)", log_reg_5, X_train_5, y_train, X_valid_5, y_valid)

    # 2. 46-Feature Logistic Regression
    log_reg_full = LocalDetectorBaseline(random_state=42)
    res_full_log = evaluate_detector("LogisticReg (46 Features)", log_reg_full, X_train, y_train, X_valid, y_valid)

    # 3. Neural Network MLP (46 Features)
    mlp_detector = NeuralDetectorMLP(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    res_mlp = evaluate_detector("Neural MLP (128x64)", mlp_detector, X_train, y_train, X_valid, y_valid)

    # 4. Gradient Boosted Decision Trees (46 Features)
    gb_detector = GradientBoostedDetector(max_iter=100, learning_rate=0.1, random_state=42)
    res_gb = evaluate_detector("Gradient Boosting (Hist)", gb_detector, X_train, y_train, X_valid, y_valid)

    results = [res_5, res_full_log, res_mlp, res_gb]

    print("=" * 75)
    print("                       BENCHMARK RESULTS REPORT")
    print("=" * 75)
    print(f" {'Model Architecture':<28} |  Precision  |  Recall   |    F1     | Bal Acc  | ROC AUC")
    print("-" * 75)
    for r in results:
        print(
            f" {r['model_name']:<28} |   {r['precision']*100:6.2f}%  |  {r['recall']*100:6.2f}% |  {r['f1']*100:6.2f}% |  {r['balanced_accuracy']*100:6.2f}% |  {r['roc_auc']:.4f}"
        )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_phase2_benchmarks()
