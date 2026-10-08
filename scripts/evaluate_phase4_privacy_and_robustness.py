"""Benchmark evaluation of poisoning attack resilience and formal DP (epsilon, delta) accounting."""

import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import numpy as np
from sklearn.metrics import (  # type: ignore[import-not-found]
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.federated.dp_accountant import RDPAccountant
from src.federated.fedprox import FedProxTrainer, aggregate_fedprox_weights
from src.federated.poisoning import apply_label_flipping_attack, apply_weight_poisoning_attack
from src.federated.profiles import OrganizationProfile, load_organization_profiles
from src.federated.robust_aggregators import aggregate_krum, aggregate_trimmed_mean


def evaluate_predictions(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """Evaluate classification metrics given ground truth labels and model logits."""
    z_clipped = np.clip(logits, -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-z_clipped))
    preds = (probs >= 0.5).astype(int)

    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))

    return {"precision": prec, "recall": rec, "f1": f1, "balanced_accuracy": bal_acc}


def run_phase4_evaluation():
    """Run comprehensive poisoning resilience and differential privacy benchmark."""
    print("=" * 80)
    print("      TrustMesh XDR - Phase 4 Poisoning Resilience & DP Accounting")
    print("=" * 80)

    # 1. Load Organization Profiles
    profiles = load_organization_profiles(base_dir, max_rows_per_client=3000)
    min_dim = min(p.X.shape[1] for p in profiles.values())
    for p in profiles.values():
        p.X = p.X[:, :min_dim]

    train_data = {}
    test_data = {}
    for name, p in profiles.items():
        n_train = int(len(p.y) * 0.8)
        train_data[name] = (p.X[:n_train], p.y[:n_train])
        test_data[name] = (p.X[n_train:], p.y[n_train:])

    trainer = FedProxTrainer(mu=0.01, learning_rate=0.05, epochs=15)

    # Setting A: Clean FedAvg
    clean_updates = []
    for name, (X_tr, y_tr) in train_data.items():
        w0 = np.zeros(min_dim, dtype=np.float32)
        w_k, b_k = trainer.fit_local(X_tr, y_tr, w0, 0.0)
        clean_updates.append((w_k, b_k, len(y_tr)))
    w_clean, b_clean = aggregate_fedprox_weights(clean_updates)

    # Setting B: Poisoned FedAvg (1 client poisoned with label flipping + weight inversion)
    poisoned_updates = list(clean_updates)
    # Poison Hospital client (index 1)
    X_hosp, y_hosp = train_data["Hospital"]
    y_hosp_flip = apply_label_flipping_attack(y_hosp, flip_ratio=1.0)
    w_hosp_pois, b_hosp_pois = trainer.fit_local(X_hosp, y_hosp_flip, np.zeros(min_dim, dtype=np.float32), 0.0)
    w_hosp_pois, b_hosp_pois = apply_weight_poisoning_attack(w_hosp_pois, b_hosp_pois, scale_factor=-5.0)
    poisoned_updates[1] = (w_hosp_pois, b_hosp_pois, len(y_hosp))

    w_pois_fedavg, b_pois_fedavg = aggregate_fedprox_weights(poisoned_updates)

    # Setting C: Robust Aggregation under Attack (Trimmed Mean)
    w_robust, b_robust = aggregate_trimmed_mean(poisoned_updates, beta=0.2)

    # Setting D: Robust Aggregation under Attack (Krum)
    w_krum, b_krum = aggregate_krum(poisoned_updates, num_byzantine=1)

    print("=" * 80)
    print("                POISONING ATTACK RESILIENCE REPORT")
    print("=" * 80)
    print(f" {'Aggregator Setting':<25} | {'Attack Active':<13} | Precision |  Recall   |    F1     | Bal Acc")
    print("-" * 80)

    # Evaluate on Hospital Test Partition
    X_hosp_te, y_hosp_te = test_data["Hospital"]
    for setting_name, w_m, b_m, attacked in [
        ("Clean FedAvg", w_clean, b_clean, "No"),
        ("Poisoned FedAvg", w_pois_fedavg, b_pois_fedavg, "Yes (1/3)"),
        ("Trimmed Mean Robust", w_robust, b_robust, "Yes (1/3)"),
        ("Krum Robust", w_krum, b_krum, "Yes (1/3)"),
    ]:
        res = evaluate_predictions(y_hosp_te, np.dot(X_hosp_te, w_m) + b_m)
        prec = res["precision"] * 100
        rec = res["recall"] * 100
        f1 = res["f1"] * 100
        bal = res["balanced_accuracy"] * 100
        print(f" {setting_name:<25} | {attacked:<13} |   {prec:6.2f}% |  {rec:6.2f}% |  {f1:6.2f}% |  {bal:6.2f}%")
    print("=" * 80 + "\n")

    # 2. Formal Differential Privacy Accounting
    print("=" * 80)
    print("             FORMAL RDP DIFFERENTIAL PRIVACY ACCOUNTING")
    print("=" * 80)
    print(" Target Delta (failure prob): delta = 1e-5\n")
    print(f" {'Noise Multiplier (sigma)':<25} | {'Eps @ 5 Rounds':<16} | {'Eps @ 10 Rounds':<16} | {'Eps @ 20 Rounds':<16}")
    print("-" * 80)

    for sigma in [0.5, 1.0, 1.5, 2.0]:
        acc = RDPAccountant(noise_multiplier=sigma, target_delta=1e-5)
        e5 = acc.compute_epsilon(5)
        e10 = acc.compute_epsilon(10)
        e20 = acc.compute_epsilon(20)
        print(f" sigma = {sigma:<17} | eps = {e5:<10.2f}    | eps = {e10:<10.2f}    | eps = {e20:<10.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_phase4_evaluation()
