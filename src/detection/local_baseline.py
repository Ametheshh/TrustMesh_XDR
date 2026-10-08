"""Small local binary-classification baseline for canonical JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT = Path("data/canonical/smoke_test/ciciot23_train.jsonl")
MAX_ROWS = 10_000
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_feature_rows(path: Path, max_rows: int = MAX_ROWS) -> tuple[list[list[float]], list[int], list[str]]:
    """Read at most max_rows, taking model inputs only from features."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    vectors: list[list[float]] = []
    labels: list[int] = []
    feature_names: list[str] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(vectors) >= max_rows:
                break
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            features = record.get("features")
            binary = record.get("labels", {}).get("binary")
            if not isinstance(features, dict) or not features:
                raise ValueError(f"missing/non-object features at {path}:{line_number}")
            if binary not in (0, 1) or isinstance(binary, bool):
                raise ValueError(f"labels.binary must be 0 or 1 at {path}:{line_number}")
            names = list(features)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError(f"feature schema/order changed at {path}:{line_number}")
            vector: list[float] = []
            for name in feature_names:
                value = features[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"feature {name!r} is not numeric at {path}:{line_number}")
                vector.append(float(value))
            vectors.append(vector)
            labels.append(int(binary))

    if not vectors:
        raise ValueError(f"no records found in {path}")
    if set(labels) != {0, 1}:
        raise ValueError("binary classification requires both classes in the bounded sample")
    return vectors, labels, feature_names or []


def _scores(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    from sklearn.metrics import (  # type: ignore[import-not-found]
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.tolist(),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def run_baseline(path: Path = DEFAULT_ARTIFACT, max_rows: int = MAX_ROWS) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
        from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required; install the declared project dependency") from exc

    x, y, feature_names = load_feature_rows(path, max_rows)
    indices = list(range(len(y)))
    train_idx, valid_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    x_train = [x[i] for i in train_idx]
    x_valid = [x[i] for i in valid_idx]
    y_train = [y[i] for i in train_idx]
    y_valid = [y[i] for i in valid_idx]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid).tolist()
    majority_class = max((0, 1), key=lambda label: (y_train.count(label), -label))
    baseline_predictions = [majority_class] * len(y_valid)

    return {
        "artifact": str(path),
        "rows_read": len(y),
        "feature_count": len(feature_names),
        "model_features": feature_names,
        "target": "labels.binary",
        "model": "StandardScaler + LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)",
        "split": {
            "method": "stratified train/validation holdout",
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx),
            "train_label_counts": {str(label): y_train.count(label) for label in (0, 1)},
            "validation_label_counts": {str(label): y_valid.count(label) for label in (0, 1)},
        },
        "metrics": _scores(y_valid, [int(value) for value in predictions]),
        "majority_class_baseline": {
            "predicted_class": majority_class,
            "metrics": _scores(y_valid, baseline_predictions),
        },
    }


class LocalDetectorBaseline:
    """StandardScaler + LogisticRegression binary classifier wrapper."""

    def __init__(self, random_state: int = RANDOM_STATE, max_iter: int = 1000):
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
            from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]
            from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required; install the declared project dependency") from exc

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=max_iter),
        )

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        """Fit the scaler and logistic regression on training data."""
        self.model.fit(X, y)

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict binary class labels (0 for benign, 1 for attack)."""
        preds = self.model.predict(X)
        return [int(p) for p in preds]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Predict positive class (malicious) probabilities."""
        probs = self.model.predict_proba(X)
        return [float(p[1]) for p in probs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.input, args.max_rows), indent=2))


if __name__ == "__main__":
    main()
