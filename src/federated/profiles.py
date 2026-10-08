"""Organization profile loaders for non-IID federated client simulation."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class OrganizationProfile:
    """Represents a simulated participant organization with distinct network/attack characteristics."""

    name: str
    description: str
    dataset_name: str
    feature_names: List[str]
    X: np.ndarray
    y: np.ndarray

    @property
    def sample_count(self) -> int:
        """Return total samples for this organization profile."""
        return len(self.y)

    @property
    def attack_ratio(self) -> float:
        """Return proportion of attack samples."""
        if len(self.y) == 0:
            return 0.0
        return float(np.mean(self.y))


def _load_canonical_jsonl(path: Path, max_rows: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load feature matrix X and label vector y from a canonical JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Canonical smoke file not found: {path}")

    vectors: List[List[float]] = []
    labels: List[int] = []
    feature_names: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(vectors) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            features = record.get("features", {})
            binary = record.get("labels", {}).get("binary")

            if not feature_names:
                feature_names = [
                    name for name, val in features.items()
                    if isinstance(val, (int, float)) and not isinstance(val, bool)
                ]

            vector = []
            for name in feature_names:
                val = features.get(name)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    vector.append(float(val))

            if len(vector) == len(feature_names) and binary in (0, 1):
                vectors.append(vector)
                labels.append(int(binary))

    return np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.int32), feature_names


def load_organization_profiles(base_dir: Path, max_rows_per_client: int = 3000) -> Dict[str, OrganizationProfile]:
    """Load 3 heterogeneous organization profiles (Bank, Hospital, Enterprise).

    Args:
        base_dir: Root repository directory.
        max_rows_per_client: Maximum rows to load per organization.

    Returns:
        Dict mapping profile names ('Bank', 'Hospital', 'Enterprise') to OrganizationProfile instances.
    """
    smoke_dir = base_dir / "data" / "canonical" / "smoke_test"

    # 1. Bank Profile (UNSW-NB15 partition)
    unsw_path = smoke_dir / "unsw_nb15_named_train.jsonl"
    X_bank, y_bank, feat_bank = _load_canonical_jsonl(unsw_path, max_rows=max_rows_per_client)
    bank_profile = OrganizationProfile(
        name="Bank",
        description="Transaction & Enterprise Services (UNSW Telemetry)",
        dataset_name="UNSW-NB15",
        feature_names=feat_bank,
        X=X_bank,
        y=y_bank,
    )

    # 2. Hospital Profile (CICIoMT2024 ARP + Benign)
    ciciomt_path = smoke_dir / "ciciomt_arp_spoofing_train.jsonl"
    X_hosp, y_hosp, feat_hosp = _load_canonical_jsonl(ciciomt_path, max_rows=max_rows_per_client)
    hospital_profile = OrganizationProfile(
        name="Hospital",
        description="Medical IoT & Sensor Infrastructure (CICIoMT Telemetry)",
        dataset_name="CICIoMT2024",
        feature_names=feat_hosp,
        X=X_hosp,
        y=y_hosp,
    )

    # 3. Enterprise Profile (CTU-SME Zeek conn logs)
    ctu_path = smoke_dir / "ctu_sme_conn_labeled.jsonl"
    X_ent, y_ent, feat_ent = _load_canonical_jsonl(ctu_path, max_rows=max_rows_per_client)
    enterprise_profile = OrganizationProfile(
        name="Enterprise",
        description="Large-Scale Campus Network (CTU Zeek Telemetry)",
        dataset_name="CTU-SME",
        feature_names=feat_ent,
        X=X_ent,
        y=y_ent,
    )

    return {
        "Bank": bank_profile,
        "Hospital": hospital_profile,
        "Enterprise": enterprise_profile,
    }
