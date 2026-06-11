"""t-conditioned lambda prior pi_lambda(. | t) for decoupled training.

Implements the exponential prior from Eq.~(25) of the design document, with
a time-dependent mean that mimics the magnitude of lambda_t observed during
inference: near zero at large t (noisy) and growing at small t (clean).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import math
import torch


class LambdaPrior(ABC):
    """Abstract interface for sampling lambda conditional on the diffusion timestep."""

    @abstractmethod
    def sample(
        self,
        t: torch.Tensor,  # [B] long
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a sampled lambda tensor of shape [B, N]."""


class ExponentialLambdaPrior(LambdaPrior):
    """
    Per-node independent Exponential(mean=mu(t)) with schedule

        mu(t) = mu_min + (mu_max - mu_min) * (1 - sqrt(alpha_bar_t)),

    so that at t=T (noisy, sqrt_ab ~ 0) mu(t) -> mu_max and at t=0
    (clean, sqrt_ab ~ 1) mu(t) -> mu_min.

    This direction matches the empirically observed inference pattern on the
    WRA medium best-config run: lambda_t is initialised at ``dual_lambda_init``
    (large) at t=T and refined *downward* by dual ascent as t decreases,
    ending at a small-mean, heavy-tailed distribution at t=0.
    Calibration (400 samples, dual_lambda_init=80, 2026-03-16 best run):
    mean at t=0 ~ 11, mean across all t ~ 32, q95 at t=0 ~ 49.
    """

    def __init__(
        self,
        sqrt_alphas_cumprod: torch.Tensor,  # [T_diffusion]
        *,
        mu_min: float = 5.0,
        mu_max: float = 80.0,
    ):
        if mu_min <= 0.0 or mu_max <= 0.0:
            raise ValueError(f"mu_min and mu_max must be > 0, got ({mu_min}, {mu_max})")
        if mu_min > mu_max:
            raise ValueError(f"mu_min must be <= mu_max, got ({mu_min}, {mu_max})")
        self._sqrt_ab = sqrt_alphas_cumprod.detach().clone()
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)

    def mean_at_t(self, t: torch.Tensor) -> torch.Tensor:
        """Return mean lambda magnitude at each timestep, shape [B]."""
        sqrt_ab = self._sqrt_ab.to(device=t.device)[t.clamp_min(0).clamp_max(self._sqrt_ab.numel() - 1)]
        return self.mu_min + (self.mu_max - self.mu_min) * (1.0 - sqrt_ab)

    def sample(
        self,
        t: torch.Tensor,
        num_nodes: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mu = self.mean_at_t(t).to(device=device, dtype=dtype).view(-1, 1)  # [B, 1]
        # Exponential with mean mu <=> -mu * log(U) for U ~ Uniform(0, 1)
        u = torch.rand((t.shape[0], num_nodes), device=device, dtype=dtype).clamp_min(1e-12)
        return -mu * torch.log(u)
