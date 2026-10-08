"""Rényi Differential Privacy (RDP) accountant for formal (epsilon, delta) privacy bounds."""

import math
from typing import Dict, List


class RDPAccountant:
    """Computes exact (epsilon, delta) differential privacy guarantees via Rényi DP composition."""

    def __init__(
        self,
        noise_multiplier: float,
        target_delta: float = 1e-5,
        orders: List[float] = None,
    ):
        """Initialize RDP Accountant.

        Args:
            noise_multiplier: Noise multiplier sigma (std_dev / clipping_bound).
            target_delta: Target failure probability delta (e.g. 1e-5).
            orders: List of Rényi orders alpha to search over.
        """
        if noise_multiplier <= 0:
            raise ValueError("Noise multiplier sigma must be strictly positive for DP guarantees")
        if not (0 < target_delta < 1):
            raise ValueError("Target delta must be in range (0, 1)")

        self.noise_multiplier = float(noise_multiplier)
        self.target_delta = float(target_delta)
        self.orders = orders or [1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0]

    def compute_rdp_per_round(self, alpha: float) -> float:
        """Compute Rényi divergence at order alpha for Gaussian mechanism: alpha / (2 * sigma^2)."""
        return alpha / (2.0 * (self.noise_multiplier ** 2))

    def compute_epsilon(self, num_rounds: int) -> float:
        """Compute minimum epsilon at target_delta after num_rounds of federated aggregation.

        Formula: eps(delta) = min_{alpha > 1} [ num_rounds * (alpha / (2 * sigma^2)) + ln(1/delta) / (alpha - 1) ]
        """
        best_eps = float("inf")

        for alpha in self.orders:
            if alpha <= 1.0:
                continue
            rdp_total = num_rounds * self.compute_rdp_per_round(alpha)
            eps_alpha = rdp_total + (math.log(1.0 / self.target_delta) / (alpha - 1.0))
            if eps_alpha < best_eps:
                best_eps = eps_alpha

        return best_eps

    def generate_privacy_report(self, rounds_list: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """Generate a privacy budget report mapping rounds to (epsilon, delta) bounds."""
        report = {
            "noise_multiplier": self.noise_multiplier,
            "target_delta": self.target_delta,
        }
        for r in rounds_list:
            eps = self.compute_epsilon(num_rounds=r)
            report[f"eps_at_round_{r}"] = round(eps, 4)
        return report
