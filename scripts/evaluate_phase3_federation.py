"""Benchmark evaluation of Local-Only, FedAvg, FedProx, and Personalized FL (PFL) across non-IID profiles."""

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
from src.federated.fedprox import FedProxTrainer
from src.federated.pfl import PersonalizedFLManager
from src.federated.profiles import OrganizationProfile, load_organization_profiles


def evaluate_predictions(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """Evaluate classification metrics given ground truth labels and model logits."""
    z_clipped = np.clip(logits, -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-z_clipped))
    preds = (probs >= 0.5).astype(int)

    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, preds))

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
    }


def run_phase3_federation_evaluation():
    """Run comprehensive non-IID federated experiment across Bank, Hospital, and Enterprise."""
    print("=" * 80)
    print("      TrustMesh XDR - Phase 3 Heterogeneous Federated Benchmark")
    print("=" * 80)

    # 1. Load Organization Profiles
    profiles = load_organization_profiles(base_dir, max_rows_per_client=3000)
    print(f"[*] Loaded {len(profiles)} non-IID organization profiles:")
    for name, p in profiles.items():
        print(f"    - {name:<12} ({p.dataset_name:<10}): {p.sample_count} samples | Attack Ratio: {p.attack_ratio*100:.2f}%")
    print()

    # Determine common feature dimension (using 5 shared features or truncating/padding to minimum common dimension)
    min_dim = min(p.X.shape[1] for p in profiles.values())
    print(f"[*] Harmonizing feature space to {min_dim} shared canonical dimensions across clients.\n")

    # Truncate profiles X to min_dim
    for p in profiles.values():
        p.X = p.X[:, :min_dim]

    # Split local train/test splits (80% train, 20% test per client)
    train_data = {}
    test_data = {}
    for name, p in profiles.items():
        n_train = int(len(p.y) * 0.8)
        train_data[name] = (p.X[:n_train], p.y[:n_train])
        test_data[name] = (p.X[n_train:], p.y[n_train:])

    # 2. Local-Only Models (No collaboration)
    print("[1/4] Training Local-Only Models...")
    local_models = {}
    local_evals = {}
    for name, (X_tr, y_tr) in train_data.items():
        trainer = FedProxTrainer(mu=0.0, learning_rate=0.05, epochs=30)
        w_init = np.zeros(min_dim, dtype=np.float32)
        w_loc, b_loc = trainer.fit_local(X_tr, y_tr, w_init, 0.0)
        local_models[name] = (w_loc, b_loc)

        X_te, y_te = test_data[name]
        logits = np.dot(X_te, w_loc) + b_loc
        local_evals[name] = evaluate_predictions(y_te, logits)

    # 3. Standard FedAvg (mu = 0.0)
    print("[2/4] Executing Standard FedAvg (5 Rounds, mu=0.0)...")
    fedavg_pfl = PersonalizedFLManager(mu=0.0, learning_rate=0.05, fine_tune_epochs=0)
    # Create profile objects with train data
    train_profiles = {
        name: OrganizationProfile(name, p.description, p.dataset_name, p.feature_names[:min_dim], train_data[name][0], train_data[name][1])
        for name, p in profiles.items()
    }
    w_fedavg, b_fedavg, _ = fedavg_pfl.train_personalized(train_profiles, num_rounds=5)

    fedavg_evals = {}
    for name, (X_te, y_te) in test_data.items():
        logits = np.dot(X_te, w_fedavg) + b_fedavg
        fedavg_evals[name] = evaluate_predictions(y_te, logits)

    # 4. FedProx (mu = 0.1)
    print("[3/4] Executing FedProx (5 Rounds, mu=0.10)...")
    fedprox_pfl = PersonalizedFLManager(mu=0.10, learning_rate=0.05, fine_tune_epochs=0)
    w_fedprox, b_fedprox, _ = fedprox_pfl.train_personalized(train_profiles, num_rounds=5)

    fedprox_evals = {}
    for name, (X_te, y_te) in test_data.items():
        logits = np.dot(X_te, w_fedprox) + b_fedprox
        fedprox_evals[name] = evaluate_predictions(y_te, logits)

    # 5. Personalized FL (PFL with local adaptation)
    print("[4/4] Executing Personalized FL (FedProx + Local Adaptation)...")
    pfl_manager = PersonalizedFLManager(mu=0.10, learning_rate=0.05, fine_tune_epochs=10)
    _, _, pfl_models = pfl_manager.train_personalized(train_profiles, num_rounds=5)

    pfl_evals = {}
    for name, (X_te, y_te) in test_data.items():
        w_p, b_p = pfl_models[name]
        logits = np.dot(X_te, w_p) + b_p
        pfl_evals[name] = evaluate_predictions(y_te, logits)

    # Print Comparative Benchmark Results Table
    print("\n" + "=" * 80)
    print("                FEDERATED HETEROGENEITY BENCHMARK REPORT")
    print("=" * 80)
    print(f" {'Client Profile':<12} | {'Setting':<16} | Precision |  Recall   |    F1     | Bal Acc")
    print("-" * 80)

    for client in ["Bank", "Hospital", "Enterprise"]:
        for setting_name, eval_dict in [
            ("Local-Only", local_evals[client]),
            ("FedAvg", fedavg_evals[client]),
            ("FedProx", fedprox_evals[client]),
            ("Personalized FL", pfl_evals[client]),
        ]:
            prec = eval_dict["precision"] * 100
            rec = eval_dict["recall"] * 100
            f1 = eval_dict["f1"] * 100
            bal = eval_dict["balanced_accuracy"] * 100
            print(f" {client:<12} | {setting_name:<16} |   {prec:6.2f}% |  {rec:6.2f}% |  {f1:6.2f}% |  {bal:6.2f}%")
        print("-" * 80)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_phase3_federation_evaluation()
