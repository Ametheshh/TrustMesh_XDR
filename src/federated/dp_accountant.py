"""Illustrative Gaussian Rényi Differential Privacy (RDP) calculations."""

import math
from typing import Dict, List


class RDPAccountant:
    """Compute illustrative epsilon values from hypothetical Gaussian RDP parameters.

    This simplified calculation is not connected to the experiment's actual
    updates and does not establish a differential privacy guarantee.
    """

    def __init__(
        self,
        noise_multiplier: float,
        target_delta: float = 1e-5,
        orders: List[float] = None,
    ):
        """Initialize the illustrative calculator with hypothetical parameters.

        Args:
            noise_multiplier: Assumed Gaussian sigma (std_dev / clipping_bound).
            target_delta: Hypothetical delta value for the calculation.
            orders: Finite list of Rényi orders alpha to search over.
        """
        if noise_multiplier <= 0:
            raise ValueError("Noise multiplier sigma must be strictly positive for this calculation")
        if not (0 < target_delta < 1):
            raise ValueError("Target delta must be in range (0, 1)")

        self.noise_multiplier = float(noise_multiplier)
        self.target_delta = float(target_delta)
        self.orders = orders or [1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0]

    def compute_rdp_per_round(self, alpha: float) -> float:
        """Apply the illustrative Gaussian RDP formula alpha / (2 * sigma^2)."""
        return alpha / (2.0 * (self.noise_multiplier ** 2))

    def compute_epsilon(self, num_rounds: int) -> float:
        """Return a hypothetical epsilon from the simplified Gaussian RDP formula.

        The minimum searches only the configured finite orders. This value is
        not an accounting of actual experiment updates or a privacy guarantee.

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

    def generate_privacy_report(self, rounds_list: List[int] | None = None) -> Dict[str, float]:
        """Return illustrative epsilon values for hypothetical round counts and delta."""
        rounds = list(rounds_list) if rounds_list is not None else [1, 5, 10, 20]
        report = {
            "noise_multiplier": self.noise_multiplier,
            "target_delta": self.target_delta,
        }
        for r in rounds:
            eps = self.compute_epsilon(num_rounds=r)
            report[f"eps_at_round_{r}"] = round(eps, 4)
        return report
