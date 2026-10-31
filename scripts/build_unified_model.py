"""Build and persist the unified federated master model combining all client models."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
OUTPUT_PATH = MODELS_DIR / "unified_federated_master_model.json"

FEATURES = [
    "duration",
    "bytes_src_to_dst",
    "bytes_dst_to_src",
    "packets_src_to_dst",
    "packets_dst_to_src",
]


def aggregate_models() -> dict:
    """Aggregate weights and biases from the trained federated checkpoints."""
    model_paths = sorted(
        list(MODELS_DIR.glob("centralized_vs_federated/federated/seed_*.json"))
        + list(MODELS_DIR.glob("class_weighting_ablation/models/B_federated_local_weighted/seed_*.json"))
    )

    if not model_paths:
        raise FileNotFoundError(f"No federated model checkpoints found under {MODELS_DIR}")

    weights_list = []
    biases_list = []

    for path in model_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data["parameters"]
        weights_list.append(np.asarray(params["weights"], dtype=np.float64))
        biases_list.append(float(params["bias"][0]))

    master_weights = np.mean(weights_list, axis=0)
    master_bias = float(np.mean(biases_list))

    artifact = {
        "artifact_format": "trustmesh_logistic_v1",
        "model_name": "unified_federated_master_model",
        "model_type": "unified_multi_domain_federated_logistic_regression",
        "version": "2.0.0",
        "decision_threshold": 0.5,
        "feature_names": FEATURES,
        "label_mapping": {
            "0": "benign",
            "1": "attack",
        },
        "parameters": {
            "bias": [round(master_bias, 6)],
            "weights": [round(float(w), 6) for w in master_weights],
        },
        "participating_domains": [
            "Bank (UNSW-NB15 Telemetry)",
            "Enterprise Campus (CTU-SME Zeek Telemetry)",
            "Hospital & Medical IoT (CICIoMT Telemetry)",
        ],
        "aggregated_checkpoint_count": len(model_paths),
        "source_checkpoints": [str(p.relative_to(BASE_DIR)) for p in model_paths],
        "preprocessing": {
            "feature_order": FEATURES,
            "pipeline": ["log1p"],
            "notes": "Unified canonical 5-feature representation with log1p scaling.",
        },
        "score_semantics": "probability of class 1 (attack/malicious)",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return artifact


if __name__ == "__main__":
    result = aggregate_models()
    print(f"[+] Successfully built unified federated model: {OUTPUT_PATH}")
    print(f"[*] Aggregated {result['aggregated_checkpoint_count']} checkpoints across {len(result['participating_domains'])} domains.")
    print(f"[*] Weights: {result['parameters']['weights']}")
    print(f"[*] Bias:    {result['parameters']['bias'][0]}")
