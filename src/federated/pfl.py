"""Personalized Federated Learning (PFL) manager with local adaptation."""

from typing import Dict, Tuple
import numpy as np
from src.federated.fedprox import FedProxTrainer, aggregate_fedprox_weights
from src.federated.profiles import OrganizationProfile


class PersonalizedFLManager:
    """Manages Personalized Federated Learning with global backbone + local client fine-tuning."""

    def __init__(self, mu: float = 0.01, learning_rate: float = 0.05, fine_tune_epochs: int = 10):
        """Initialize PFL Manager.

        Args:
            mu: Proximal penalty for FedProx server rounds.
            learning_rate: Gradient descent learning rate.
            fine_tune_epochs: Number of local adaptation epochs post global aggregation.
        """
        self.mu = mu
        self.learning_rate = learning_rate
        self.fine_tune_epochs = fine_tune_epochs
        self.trainer = FedProxTrainer(mu=mu, learning_rate=learning_rate, epochs=15)

    def train_personalized(
        self,
        profiles: Dict[str, OrganizationProfile],
        num_rounds: int = 5,
    ) -> Tuple[np.ndarray, float, Dict[str, Tuple[np.ndarray, float]]]:
        """Execute Personalized FL rounds and fine-tuning.

        Args:
            profiles: Dict of client OrganizationProfile objects.
            num_rounds: Number of global aggregation rounds.

        Returns:
            Tuple of (w_global, b_global, personalized_client_models).
            personalized_client_models is a dict mapping client name to (w_local, b_local).
        """
        if not profiles:
            raise ValueError("No profiles provided for PFL training")

        # Determine feature dimension from first profile
        first_profile = list(profiles.values())[0]
        n_features = first_profile.X.shape[1]

        # Initialize global parameters
        w_global = np.zeros(n_features, dtype=np.float32)
        b_global = 0.0

        # Global Aggregation Rounds
        for round_num in range(num_rounds):
            client_updates = []
            for name, profile in profiles.items():
                if len(profile.y) == 0:
                    continue
                w_k, b_k = self.trainer.fit_local(profile.X, profile.y, w_global, b_global)
                client_updates.append((w_k, b_k, profile.sample_count))

            if client_updates:
                w_global, b_global = aggregate_fedprox_weights(client_updates)

        # Local Fine-Tuning Step for Personalization
        fine_tuner = FedProxTrainer(mu=0.0, learning_rate=self.learning_rate, epochs=self.fine_tune_epochs)
        personalized_models: Dict[str, Tuple[np.ndarray, float]] = {}

        for name, profile in profiles.items():
            if len(profile.y) == 0:
                personalized_models[name] = (w_global, b_global)
            else:
                w_p, b_p = fine_tuner.fit_local(profile.X, profile.y, w_global, b_global)
                personalized_models[name] = (w_p, b_p)

        return w_global, b_global, personalized_models
