"""Gradient Boosted decision tree ensemble detector for tabular network telemetry."""

from typing import List, Union
import numpy as np


class GradientBoostedDetector:
    """Histogram-based Gradient Boosting decision tree ensemble for canonical tabular telemetry."""

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        """Initialize GradientBoostedDetector.

        Args:
            max_iter: Max boosting iterations (number of trees).
            learning_rate: Shrinkage rate for updates.
            max_depth: Maximum tree depth.
            min_samples_leaf: Minimum samples required per leaf node.
            random_state: Random seed for reproducibility.
            class_weight: Class weight mode ('balanced' or None).
        """
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-not-found]
            from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]
            from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required; install the declared project dependency") from exc

        self.random_state = random_state

        gb = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
        )

        self.model = make_pipeline(StandardScaler(), gb)

    def fit(self, X: Union[List[List[float]], np.ndarray], y: Union[List[int], np.ndarray]) -> "GradientBoostedDetector":
        """Fit scaler and gradient boosted ensemble on feature matrix X and labels y."""
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[List[List[float]], np.ndarray]) -> List[int]:
        """Predict binary class labels (0 for benign, 1 for attack)."""
        preds = self.model.predict(X)
        return [int(p) for p in preds]

    def predict_proba(self, X: Union[List[List[float]], np.ndarray]) -> List[float]:
        """Predict probability of positive class (attack)."""
        probs = self.model.predict_proba(X)
        return [float(p[1]) for p in probs]
