"""Multi-Layer Perceptron (MLP) Neural Network detector for high-dimensional canonical telemetry."""

from typing import List, Tuple, Union
import numpy as np


class NeuralDetectorMLP:
    """Multi-Layer Perceptron neural network detector with StandardScaler preprocessing."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        random_state: int = 42,
        early_stopping: bool = True,
    ):
        """Initialize NeuralDetectorMLP.

        Args:
            hidden_layer_sizes: Tuple of layer sizes (e.g. (128, 64)).
            activation: Activation function ('relu', 'tanh').
            alpha: L2 penalty parameter.
            learning_rate_init: Initial learning rate.
            max_iter: Max optimization iterations.
            random_state: Random seed for reproducibility.
            early_stopping: Whether to use early stopping validation.
        """
        try:
            from sklearn.neural_network import MLPClassifier  # type: ignore[import-not-found]
            from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]
            from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required; install the declared project dependency") from exc

        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            n_iter_no_change=10,
        )

        self.model = make_pipeline(StandardScaler(), mlp)

    def fit(self, X: Union[List[List[float]], np.ndarray], y: Union[List[int], np.ndarray]) -> "NeuralDetectorMLP":
        """Fit scaler and neural network on training feature matrix X and labels y."""
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
