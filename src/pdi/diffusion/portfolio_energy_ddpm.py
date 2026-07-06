"""Portfolio-optimisation energy-guided DDPM.

Subclasses ``EnergyDDPM`` to implement constrained mean-variance
portfolio allocation via Lagrangian energy guidance.

The Lagrangian energy mirrors the WRA formulation exactly:

    E(x, lambda; beta) = beta^{-1} [ -E_xi[r(xi)^T x]
                                       + sum_j lambda_j ((Sigma x)_j x_j - b_j) ]

where x are portfolio weights on the simplex, Sigma is the return covariance,
and b_j are per-asset risk budgets.

See ``docs/synthetic_experiment_portfolio_prompt.md`` for full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch_geometric.data import Data

from pdi.diffusion.energy_ddpm import EnergyDDPM

__all__ = [
    "PortfolioEnergyDDPM", "PortfolioContext", "make_portfolio_problem",
    "shortfall_contributions_np", "shortfall_contributions_torch",
]


def shortfall_contributions_np(w, scenarios, alpha):
    """Per-asset expected shortfall contribution (numpy).

    c_j(x) = x_j * E_xi[(alpha - r(xi)^T x)_+].  w: [B, N] or [N]; returns same.
    """
    import numpy as _np
    w = _np.atleast_2d(w)  # [B, N]
    port_ret = scenarios @ w.T  # [R, B]
    shortfall = _np.maximum(alpha - port_ret, 0.0)  # [R, B]
    S = shortfall.mean(axis=0)  # [B]
    c = w * S[:, None]
    return c


def shortfall_contributions_torch(w, scenarios, alpha):
    """Per-asset expected shortfall contribution (torch). w: [B, N], scenarios: [R, N]."""
    port_ret = torch.matmul(scenarios, w.t())  # [R, B]
    S = torch.clamp_min(alpha - port_ret, 0.0).mean(dim=0)  # [B]
    return w * S.unsqueeze(-1)  # [B, N]


def variance_contributions_np(w, Sigma):
    """Marginal variance contribution (numpy). c_j = x_j (Σx)_j. w: [B,N] or [N]."""
    import numpy as _np
    w = _np.atleast_2d(w)
    Sigma_x = w @ Sigma  # [B, N]
    return w * Sigma_x


def variance_contributions_torch(w, Sigma):
    """Marginal variance contribution (torch). c_j = x_j (Σx)_j. w: [B,N], Sigma: [N,N]."""
    Sigma_x = torch.matmul(w, Sigma)  # [B, N]
    return w * Sigma_x


# ---------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------

@dataclass
class PortfolioContext:
    """Per-batch tensors for portfolio expectation-constraint energy.

    Expected-shortfall constraint:
        E_xi[x_j * (alpha - r(xi)^T x)_+] <= b_j,  j = 1..N.

    The expectation is evaluated by averaging across the stored ``scenarios``.
    """

    mu: torch.Tensor           # [N]
    Sigma: torch.Tensor        # [N, N]  (kept for baselines/metrics; not used in E)
    scenarios: torch.Tensor    # [R, N]
    risk_budgets: torch.Tensor # [N]
    alpha: torch.Tensor        # scalar: shortfall threshold
    r_min: torch.Tensor        # [B, N] (negated budgets for sign trick)


# ---------------------------------------------------------------
# Synthetic problem generator
# ---------------------------------------------------------------

def make_portfolio_problem(
    N: int,
    K_factors: int = 5,
    R_scenarios: int = 2000,
    gamma: float = 3.0,
    alpha_percentile: float = 10.0,
    seed: int = 0,
    structure: str = "homogeneous",
    num_sectors: Optional[int] = None,
    df: float = 5.0,
    constraint_type: str = "variance",
    budget_type: str = "proportional",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a synthetic portfolio with per-asset risk constraints.

    Two constraint types:
        "variance":  c_j(x) = x_j (Σx)_j ≤ b_j  (marginal variance contribution)
        "shortfall": c_j(x) = x_j E_ξ[(α − rᵀx)₊] ≤ b_j  (shortfall contribution)

    Keyword args:
        n_high_risk_sectors (int): Number of "crypto-like" sectors with
            higher mean, higher variance, fatter tails, weaker market
            correlation. Default 2.

    Args:
        constraint_type: "variance" (default) or "shortfall".
        structure: Factor model structure.
            "homogeneous" — original: all assets load on all factors (dense B).
            "sectors" — block-sparse loadings with sector structure + global factor.
            "sectors_fat" — sector structure with t-distributed factors (fat tails).
        num_sectors: Number of sectors (default: K_factors for sectors/sectors_fat).
        df: Degrees of freedom for t-distribution (only used by sectors_fat).

    Returns (mu, Sigma, scenarios, budgets, alpha).
    """
    rng = np.random.RandomState(seed)

    if structure == "homogeneous":
        B = np.abs(rng.randn(N, K_factors)) * 0.3
        factor_cov = np.diag(rng.uniform(0.01, 0.1, K_factors))
        idio_var = rng.uniform(0.005, 0.05, N)
        Sigma_log = B @ factor_cov @ B.T + np.diag(idio_var)
        mu_log = rng.uniform(0.02, 0.15, N)
        log_returns = rng.multivariate_normal(mu_log, Sigma_log, size=R_scenarios)

    elif structure in ("sectors", "sectors_fat"):
        S = num_sectors if num_sectors is not None else K_factors
        n_high_risk = kwargs.get("n_high_risk_sectors", 2)
        sector_ids = np.arange(N) % S
        factors_per_sector = max(K_factors // S, 1)
        total_factors = S * factors_per_sector + 1  # +1 for global market factor

        # Pick which sectors are "high-risk" (e.g., crypto-like)
        high_risk_sectors = set(rng.choice(S, size=n_high_risk, replace=False))

        B = np.zeros((N, total_factors))
        # Global market factor: high-risk sectors load less (less correlated with market)
        for j in range(N):
            if sector_ids[j] in high_risk_sectors:
                B[j, 0] = np.abs(rng.randn()) * 0.05  # weak market coupling
            else:
                B[j, 0] = np.abs(rng.randn()) * 0.15
        # Sector factors: each asset loads only on its sector's factors
        for s in range(S):
            mask = sector_ids == s
            n_s = mask.sum()
            cols = range(1 + s * factors_per_sector, 1 + (s + 1) * factors_per_sector)
            loading_scale = 0.5 if s in high_risk_sectors else 0.3
            B[np.ix_(mask, cols)] = np.abs(rng.randn(n_s, factors_per_sector)) * loading_scale

        # Sector-dependent factor variances
        sector_risk = np.zeros(S)
        for s in range(S):
            if s in high_risk_sectors:
                sector_risk[s] = rng.uniform(4.0, 6.0)  # 3-5x more volatile
            else:
                sector_risk[s] = rng.uniform(0.5, 2.0)
        factor_var = np.zeros(total_factors)
        factor_var[0] = rng.uniform(0.02, 0.08)  # global factor
        for s in range(S):
            cols = range(1 + s * factors_per_sector, 1 + (s + 1) * factors_per_sector)
            factor_var[list(cols)] = rng.uniform(0.01, 0.1, factors_per_sector) * sector_risk[s]
        factor_cov = np.diag(factor_var)

        # Idiosyncratic variance: higher for high-risk sectors
        idio_var = np.zeros(N)
        for j in range(N):
            if sector_ids[j] in high_risk_sectors:
                idio_var[j] = rng.uniform(0.02, 0.15)
            else:
                idio_var[j] = rng.uniform(0.005, 0.05)
        Sigma_log = B @ factor_cov @ B.T + np.diag(idio_var)

        # Sector-dependent means with wide within-sector spread
        # High-risk sectors have higher mean (high risk, high reward)
        sector_mu_base = np.zeros(S)
        for s in range(S):
            if s in high_risk_sectors:
                sector_mu_base[s] = rng.uniform(0.15, 0.35)
            else:
                sector_mu_base[s] = rng.uniform(0.02, 0.15)
        mu_log = np.array([sector_mu_base[sector_ids[j]] + rng.uniform(-0.06, 0.06)
                           for j in range(N)])
        mu_log = np.clip(mu_log, 0.005, 0.40)

        if structure == "sectors":
            log_returns = rng.multivariate_normal(mu_log, Sigma_log, size=R_scenarios)
        else:
            # Fat tails: sample factors from multivariate t, then transform
            from scipy.stats import chi2
            L = np.linalg.cholesky(Sigma_log + 1e-8 * np.eye(N))
            z_normal = rng.randn(R_scenarios, N)
            chi2_samples = rng.chisquare(df, size=(R_scenarios, 1))
            log_returns = mu_log[None, :] + (z_normal @ L.T) * np.sqrt(df / chi2_samples)

    else:
        raise ValueError(f"Unknown structure: {structure!r}")

    scenarios = np.exp(log_returns) - 1.0  # [R, N], bounded below by -1

    mu = scenarios.mean(axis=0)
    Sigma = np.cov(scenarios, rowvar=False)

    x_eq = np.ones(N) / N
    port_ret_eq = scenarios @ x_eq
    alpha = float(np.percentile(port_ret_eq, alpha_percentile))

    # Calibrate budgets at equal weight for the chosen constraint type.
    # With normalize_constraints=True (default), the dual operates on
    # (c_j/b_j - 1), which is O(1) at equal weight.
    if constraint_type == "variance":
        Sigma_x_eq = Sigma @ x_eq  # [N]
        c_eq = x_eq * Sigma_x_eq   # [N] marginal variance contribution
    elif constraint_type == "variance_sector":
        Sigma_x_eq = Sigma @ x_eq  # [N]
        c_per_asset = x_eq * Sigma_x_eq  # [N]
        S = num_sectors if num_sectors is not None else 10
        sector_ids = np.arange(N) % S
        sector_totals = np.bincount(sector_ids, weights=c_per_asset, minlength=S)
        c_eq = sector_totals[sector_ids]
    elif constraint_type == "shortfall":
        shortfall_eq = np.maximum(alpha - port_ret_eq, 0.0).mean()
        c_eq = x_eq * shortfall_eq  # [N] (uniform since x_eq is uniform)
    elif constraint_type == "variance_band":
        Sigma_x_eq = Sigma @ x_eq
        c_eq = x_eq * Sigma_x_eq
        b_max = gamma * c_eq
        b_min = c_eq / gamma
        budgets = np.concatenate([
            b_max,      # upper bounds: c_j <= gamma * c_eq_j
            -b_min,     # lower bounds encoded: -c_j <= -c_eq_j / gamma
        ])
        return mu, Sigma, scenarios, budgets, alpha
    elif constraint_type == "dual":
        Sigma_x_eq = Sigma @ x_eq
        c_var = x_eq * Sigma_x_eq
        shortfall_eq = np.maximum(alpha - port_ret_eq, 0.0).mean()
        c_short = x_eq * shortfall_eq
        if budget_type == "uniform":
            bud_var = gamma * np.full(N, c_var.mean())
            bud_short = gamma * np.full(N, c_short.mean())
        else:
            bud_var = gamma * c_var
            bud_short = gamma * c_short
        budgets = np.concatenate([bud_var, bud_short])
        return mu, Sigma, scenarios, budgets, alpha
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")
    if budget_type == "uniform":
        budgets = gamma * np.full(N, c_eq.mean())
    else:
        budgets = gamma * c_eq

    hr_scale = kwargs.get("high_risk_budget_scale", None)
    if hr_scale is not None and structure in ("sectors", "sectors_fat"):
        S = num_sectors if num_sectors is not None else K_factors
        n_high_risk = kwargs.get("n_high_risk_sectors", 2)
        rng_hr = np.random.RandomState(seed)
        high_risk_sectors = set(rng_hr.choice(S, size=n_high_risk, replace=False))
        sector_ids = np.arange(N) % S
        for j in range(N):
            if sector_ids[j] in high_risk_sectors:
                budgets[j] *= hr_scale

    return mu, Sigma, scenarios, budgets, alpha


# ---------------------------------------------------------------
# PortfolioEnergyDDPM
# ---------------------------------------------------------------

class PortfolioEnergyDDPM(EnergyDDPM):
    """Energy-based DDPM sampler for constrained portfolio optimisation.

    Subclasses ``EnergyDDPM`` (so it passes the trainer's isinstance check)
    and overrides the WRA-specific energy/rate/context methods.

    Operates in **z-space** (unconstrained): the DDPM diffuses over z in R^N
    and portfolio weights are recovered via ``x = softmax(z)``.
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 500,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        loss_type: str = "l2",
        parameterization: str = "eps",
        cosine_s: float = 0.008,
        compile_model: bool = False,
        *,
        # Portfolio problem data (numpy)
        portfolio_mu: Optional[np.ndarray] = None,
        portfolio_Sigma: Optional[np.ndarray] = None,
        portfolio_scenarios: Optional[np.ndarray] = None,
        portfolio_risk_budgets: Optional[np.ndarray] = None,
        portfolio_alpha: Optional[float] = None,
        # Energy params
        energy_mc_samples: int = 8,
        inverse_beta: float = 1.0,
        inverse_beta_schedule: str = "constant",
        inverse_beta_start: Optional[float] = None,
        inverse_beta_end: Optional[float] = None,
        dual_update_mode: str = "x0_pred",
        dual_step_size: float = 1.0,
        dual_step_size_schedule: str = "constant",
        dual_step_size_start: Optional[float] = None,
        dual_step_size_end: Optional[float] = None,
        dual_num_outer_iterations: int = 1,
        dual_lambda_init: float = 0.0,
        dual_lambda_max: Optional[float] = None,
        langevin_step_size: float = 1e-4,
        langevin_noise_scale: float = 1.0,
        min_energy_sigma: float = 1e-4,
        shared_lambda: bool = False,
        normalize_constraints: bool = True,
        objective_scale: float = 1.0,
        constraint_type: str = "variance",
        dual_lambda_decay: float = 0.0,
        **kwargs,
    ):
        self.shared_lambda = bool(shared_lambda)
        self.normalize_constraints = bool(normalize_constraints)
        self.dual_lambda_decay = float(dual_lambda_decay)
        self.objective_scale = float(objective_scale)
        self.constraint_type = constraint_type
        self._num_sectors = int(kwargs.pop("num_sectors", 10))
        self._constraint_scale = float(kwargs.pop("constraint_scale", 1.0))
        self.use_dps_guidance = bool(kwargs.pop("use_dps_guidance", False))
        # clip_denoised=False: z-space is unconstrained.
        # Pass dummy WRA-specific params that won't be used (overridden methods
        # never touch channels/gains).
        super().__init__(
            model=model,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            loss_type=loss_type,
            parameterization=parameterization,
            clip_denoised=False,
            cosine_s=cosine_s,
            compile_model=compile_model,
            energy_mc_samples=energy_mc_samples,
            energy_num_channel_realizations=1,
            inverse_beta=inverse_beta,
            inverse_beta_schedule=inverse_beta_schedule,
            inverse_beta_start=inverse_beta_start,
            inverse_beta_end=inverse_beta_end,
            dual_update_mode=dual_update_mode,
            dual_step_size=dual_step_size,
            dual_step_size_schedule=dual_step_size_schedule,
            dual_step_size_start=dual_step_size_start,
            dual_step_size_end=dual_step_size_end,
            dual_num_outer_iterations=dual_num_outer_iterations,
            dual_lambda_init=dual_lambda_init,
            dual_lambda_max=dual_lambda_max,
            langevin_step_size=langevin_step_size,
            langevin_noise_scale=langevin_noise_scale,
            dataset_root=None,
            default_p_max=1.0,
            default_noise_var=1.0,
            default_r_min=0.0,
            r_min_override=None,
            min_energy_sigma=min_energy_sigma,
            use_precomputed_channels=False,
            allow_sparse_graph_fallback=True,
            **kwargs,
        )

        # Store problem data as buffers (moved with .to(device))
        if portfolio_mu is not None:
            self.register_buffer("_port_mu", torch.as_tensor(portfolio_mu, dtype=torch.float32))
            self.register_buffer("_port_Sigma", torch.as_tensor(portfolio_Sigma, dtype=torch.float32))
            self.register_buffer("_port_scenarios", torch.as_tensor(portfolio_scenarios, dtype=torch.float32))
            self.register_buffer("_port_risk_budgets", torch.as_tensor(portfolio_risk_budgets, dtype=torch.float32))
            self.register_buffer("_port_alpha", torch.tensor(float(portfolio_alpha if portfolio_alpha is not None else 0.0), dtype=torch.float32))
        else:
            self.register_buffer("_port_mu", torch.empty(0))
            self.register_buffer("_port_Sigma", torch.empty(0))
            self.register_buffer("_port_scenarios", torch.empty(0))
            self.register_buffer("_port_risk_budgets", torch.empty(0))
            self.register_buffer("_port_alpha", torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Context builder (overrides WRA channel-based context)
    # ------------------------------------------------------------------

    def _build_energy_context(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PortfolioContext:
        mu = self._port_mu.to(device=device, dtype=dtype)
        Sigma = self._port_Sigma.to(device=device, dtype=dtype)
        scenarios = self._port_scenarios.to(device=device, dtype=dtype)
        risk_budgets = self._port_risk_budgets.to(device=device, dtype=dtype)
        alpha = self._port_alpha.to(device=device, dtype=dtype)

        # Sign trick: r_min = -budget so inherited dual ascent computes
        # violation = r_min - (-c_j) = c_j - b_j (correct sign).
        r_min = (-risk_budgets).unsqueeze(0).expand(batch_size, -1)  # [B, N]

        return PortfolioContext(
            mu=mu, Sigma=Sigma, scenarios=scenarios,
            risk_budgets=risk_budgets, alpha=alpha, r_min=r_min,
        )

    def _init_dual_lambda(self, batch_size, num_nodes, *, device, dtype, init_value):
        n_constraints = len(self._port_risk_budgets)
        return torch.full((batch_size, n_constraints), float(init_value),
                          device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # MC candidate projection
    # ------------------------------------------------------------------

    def _clip_mc_candidate(self, x0: torch.Tensor) -> torch.Tensor:
        B, _, N, _ = x0.shape
        w = torch.softmax(x0[:, 0, :, 0], dim=-1)
        z = torch.log(w.clamp_min(1e-12))
        z = z - z.mean(dim=-1, keepdim=True)
        return z.view(B, 1, N, 1)

    # ------------------------------------------------------------------
    # z -> x conversion (softmax parameterization)
    # ------------------------------------------------------------------

    @staticmethod
    def _z_to_weights(z: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained z [B, 1, N, 1] to simplex weights [B, N]."""
        z_flat = z[:, 0, :, 0]  # [B, N]
        return torch.softmax(z_flat, dim=-1)

    # ------------------------------------------------------------------
    # Risk contribution
    # ------------------------------------------------------------------

    def _risk_contributions(
        self,
        x: torch.Tensor,  # [B, N] portfolio weights
        Sigma: torch.Tensor,  # [N, N] -- unused in expectation-constraint mode
    ) -> torch.Tensor:
        """Deterministic variance-contribution (kept for metrics/baselines).

        For the *expectation* constraint used by the trained/MC sampler, see
        ``_shortfall_contributions`` instead.
        """
        Sigma_x = torch.matmul(x, Sigma)
        return Sigma_x * x

    def _shortfall_contributions(
        self,
        x: torch.Tensor,        # [B, N] portfolio weights
        scenarios: torch.Tensor,  # [R, N]
        alpha: torch.Tensor,     # scalar
    ) -> torch.Tensor:
        """Per-asset expected shortfall contribution:
            c_j(x) = x_j * E_xi[(alpha - r(xi)^T x)_+]

        Returns [B, N]. Differentiable w.r.t. x.
        """
        return shortfall_contributions_torch(x, scenarios, alpha)

    # ------------------------------------------------------------------
    # Ergodic rates override (sign-tricked for dual ascent compatibility)
    # ------------------------------------------------------------------

    def _variance_contributions(
        self,
        x: torch.Tensor,        # [B, N] portfolio weights
        Sigma: torch.Tensor,     # [N, N]
    ) -> torch.Tensor:
        """Marginal variance contribution: c_j(x) = x_j (Σx)_j.

        Returns [B, N]. This is the Euler decomposition of portfolio variance:
        sum_j c_j(x) = xᵀΣx. Fully differentiable (polynomial in x).
        """
        Sigma_x = torch.matmul(x, Sigma)  # [B, N]
        return x * Sigma_x  # [B, N]

    def _ergodic_rates_from_samples(
        self,
        x: torch.Tensor,  # [B, 1, N, 1] (z-space)
        context: PortfolioContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-asset constraint contributions + expected return.

        Returns (negated_c_j, expected_return). Sign trick: negated_c so that
        inherited dual ascent computes violation = r_min - (-c) = c - b.
        """
        weights = self._z_to_weights(x)  # [B, N]

        if self.constraint_type == "variance":
            c = self._variance_contributions(weights, context.Sigma) * self._constraint_scale
        elif self.constraint_type == "variance_sector":
            c_per_asset = self._variance_contributions(weights, context.Sigma) * self._constraint_scale
            B, N = weights.shape
            S = self._num_sectors
            sector_ids = torch.arange(N, device=weights.device) % S
            # Scatter-add per-asset contributions to sector totals [B, S]
            sector_totals = torch.zeros(B, S, device=weights.device, dtype=weights.dtype)
            sector_totals.scatter_add_(1, sector_ids.unsqueeze(0).expand(B, -1), c_per_asset)
            # Gather back: each asset gets its sector's total
            c = sector_totals.gather(1, sector_ids.unsqueeze(0).expand(B, -1))
        elif self.constraint_type == "shortfall":
            c = self._shortfall_contributions(weights, context.scenarios, context.alpha)
        elif self.constraint_type == "variance_band":
            c_var = self._variance_contributions(weights, context.Sigma) * self._constraint_scale
            negated_c = torch.cat([-c_var, c_var], dim=1)  # [B, 2N]
            ret_per_scenario = torch.matmul(context.scenarios, weights.t())
            expected_return = ret_per_scenario.mean(dim=0)
            return negated_c, expected_return
        elif self.constraint_type == "dual":
            c_var = self._variance_contributions(weights, context.Sigma)
            c_short = self._shortfall_contributions(weights, context.scenarios, context.alpha)
            c = torch.cat([c_var, c_short], dim=1)  # [B, 2N]
        else:
            raise ValueError(f"Unknown constraint_type: {self.constraint_type!r}")

        negated_c = -c  # [B, N] or [B, 2N]
        ret_per_scenario = torch.matmul(context.scenarios, weights.t())  # [R, B]
        expected_return = ret_per_scenario.mean(dim=0)  # [B]

        return negated_c, expected_return

    # ------------------------------------------------------------------
    # Energy override
    # ------------------------------------------------------------------

    def _energy_from_samples(
        self,
        x: torch.Tensor,  # [B, 1, N, 1] (z-space)
        context: PortfolioContext,
        dual_lambda: Optional[torch.Tensor] = None,  # [B, N]
        t: Optional[torch.Tensor] = None,
        inverse_beta_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lagrangian energy for portfolio optimisation.

        E = beta^{-1} [ -E[r^T x] + sum_j lambda_j ((Sigma x)_j x_j - b_j) ]
        """
        negated_risk, expected_return = self._ergodic_rates_from_samples(x, context)
        # negated_risk = -risk, so risk = -negated_risk
        # violation_j = risk_j - b_j = -negated_risk_j - b_j
        # Using the sign trick with r_min = -b: violation = r_min - negated_risk

        if dual_lambda is None:
            dual_lambda = torch.zeros(
                x.shape[0], x.shape[2], device=x.device, dtype=x.dtype,
            )

        # Lagrangian: sum_j lambda_j * (risk_j - b_j)
        # = sum_j lambda_j * (r_min_j - negated_risk_j)   [via sign trick]
        violation = context.r_min - negated_risk  # [B, N]
        if self.normalize_constraints:
            violation = violation / context.risk_budgets.unsqueeze(0).abs().clamp_min(1e-12)
        lagrangian_term = (violation * dual_lambda).sum(dim=1)  # [B]

        # Resolve inverse_beta
        if inverse_beta_override is not None:
            ib = inverse_beta_override.to(device=x.device, dtype=x.dtype).view(-1)
            if ib.numel() == 1:
                ib = ib.expand(x.shape[0])
        elif t is None:
            ib = self.inverse_beta_by_t[0].to(device=x.device, dtype=x.dtype).expand(x.shape[0])
        else:
            t_safe = t.to(device=x.device, dtype=torch.long).clamp(0, self.num_timesteps - 1)
            ib = self._gather(self.inverse_beta_by_t.to(device=x.device), t_safe).to(dtype=x.dtype)

        energy = ib * (-self.objective_scale * expected_return + lagrangian_term)
        return energy

    # ------------------------------------------------------------------
    # Score-to-x0 override (no clamping in z-space)
    # ------------------------------------------------------------------

    def _score_to_x0_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp_min(1e-12))
        sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))

        eps_pred = -score * sqrt_one_minus_alpha_bar
        x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar
        # Simplex projection: map z₀_pred → softmax → log back to z-space.
        # Prevents x0_pred explosion when sqrt(alpha_bar) → 0 (cosine schedule)
        # and ensures the predicted clean sample is always a valid portfolio.
        B, _, N, _ = x0_pred.shape
        z_flat = x0_pred[:, 0, :, 0]  # [B, N]
        w = torch.softmax(z_flat, dim=-1)
        z_proj = torch.log(w.clamp_min(1e-12))
        z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)  # center
        x0_pred = z_proj.view(B, 1, N, 1)
        eps_pred = (x_t - sqrt_alpha_bar * x0_pred) / sqrt_one_minus_alpha_bar
        return x0_pred, eps_pred

    # ------------------------------------------------------------------
    # DPS-style score estimation (autograd through Tweedie estimate)
    # ------------------------------------------------------------------

    def _estimate_score(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context,
        dual_lambda=None,
        inverse_beta_override=None,
    ) -> torch.Tensor:
        if not self.use_dps_guidance:
            return super()._estimate_score(
                x_t, t, context,
                dual_lambda=dual_lambda,
                inverse_beta_override=inverse_beta_override,
            )
        # DPS: differentiate energy through the Tweedie estimate.
        with torch.enable_grad():
            z_t = x_t.detach().requires_grad_(True)
            alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
            sqrt_omb = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
            # Model prediction (eps); for NoOp model this is zeros.
            with torch.no_grad():
                eps_pred = self.model(z_t, t, edge_index=None)[0]
                if eps_pred is None:
                    eps_pred = torch.zeros_like(z_t)
            # Tweedie estimate
            z0_hat = (z_t - sqrt_omb * eps_pred) / sqrt_ab
            # Simplex projection (differentiable)
            B, _, N, _ = z0_hat.shape
            w = torch.softmax(z0_hat[:, 0, :, 0], dim=-1)
            z0_proj = torch.log(w.clamp_min(1e-12))
            z0_proj = z0_proj - z0_proj.mean(dim=-1, keepdim=True)
            z0_proj = z0_proj.view(B, 1, N, 1)
            # Energy at the projected Tweedie estimate
            energy = self._energy_from_samples(
                z0_proj, context,
                dual_lambda=dual_lambda, t=t,
                inverse_beta_override=inverse_beta_override,
            )  # [B]
            grad_zt = torch.autograd.grad(energy.sum(), z_t)[0]
        # Score = -grad (score points downhill in energy)
        score = -grad_zt
        return score.detach()

    # ------------------------------------------------------------------
    # Rates summary (portfolio-specific metrics)
    # ------------------------------------------------------------------

    def _rates_summary_from_samples(
        self,
        x: torch.Tensor,
        context: PortfolioContext,
        n_samples_per_input: int = 1,
        dual_lambda: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        inverse_beta_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            negated_c, expected_return = self._ergodic_rates_from_samples(x, context)
            c = -negated_c
            violation = (c - context.risk_budgets.unsqueeze(0)).clamp_min(0.0)

            sr = float(expected_return.mean().item())
            max_vio = float(violation.max().item())
            mean_vio = float(violation.mean().item())
            feas = float((violation.max(dim=1).values <= 1e-8).float().mean().item())

        return {
            "sum_rate": sr,
            "p5": feas,
            "energy": 0.0,
            "per_net_sum_rates": [sr],
            "per_net_p5s": [feas],
            "per_net_energies": [0.0],
        }

    # ------------------------------------------------------------------
    # Dual ascent (uses inherited _dual_ascent_step with sign trick)
    # ------------------------------------------------------------------
    # The inherited _dual_ascent_step computes:
    #   violation = context.r_min.view(-1, 1) - ergodic_rates
    # With our sign trick (r_min = -budget, ergodic_rates = -risk):
    #   violation = (-budget) - (-risk) = risk - budget  ✓

    def _dual_ascent_step(
        self,
        dual_lambda: torch.Tensor,
        ergodic_rates: torch.Tensor,
        context: PortfolioContext,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dual ascent: lambda <- (lambda + eta * violation)+.

        Per-sample (``shared_lambda=False``): violation is [B, N] per-sample.
        Shared (``shared_lambda=True``, default): violation is reduced via
        batch-mean to [1, N] and applied uniformly, preventing per-sample
        dual-ascent drift (HANDOFF §1).
        """
        if t is None:
            dual_step_size_t = self.dual_step_size_by_t[0].to(
                device=dual_lambda.device, dtype=dual_lambda.dtype,
            ).expand(dual_lambda.shape[0])
        else:
            t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(
                0, self.num_timesteps - 1,
            )
            dual_step_size_t = self._gather(
                self.dual_step_size_by_t.to(device=dual_lambda.device), t_safe,
            ).to(dtype=dual_lambda.dtype)

        # r_min is [B, N] for portfolio (not [B, 1, 1] like WRA)
        violation = context.r_min - ergodic_rates  # [B, N]
        if self.normalize_constraints:
            violation = violation / context.risk_budgets.unsqueeze(0).clamp_min(1e-12)
        step = dual_step_size_t.view(-1, 1)  # [B, 1]

        decay = self.dual_lambda_decay

        if self.shared_lambda:
            mean_violation = violation.mean(dim=0, keepdim=True)  # [1, N]
            step_scalar = step[0:1]  # [1, 1]
            lam_decayed = (1.0 - decay) * dual_lambda[0:1]
            lam_shared = (lam_decayed + step_scalar * mean_violation).clamp_min(0.0)
            if self.dual_lambda_max is not None:
                lam_shared = lam_shared.clamp_max(float(self.dual_lambda_max))
            return lam_shared.expand_as(dual_lambda).contiguous()

        lam_decayed = (1.0 - decay) * dual_lambda
        updated = (lam_decayed + step * violation).clamp_min(0.0)
        if self.dual_lambda_max is not None:
            updated = updated.clamp_max(float(self.dual_lambda_max))
        return updated

    # ------------------------------------------------------------------
    # Training loss (not implemented — sampling only for now)
    # ------------------------------------------------------------------

    def training_loss(self, data: Data) -> torch.Tensor:
        raise NotImplementedError(
            "PortfolioEnergyDDPM training is not implemented. Use EnergyScoreTrainer."
        )

    # ------------------------------------------------------------------
    # Sample entry point
    # ------------------------------------------------------------------

    def sample(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        data: Optional[Data] = None,
        use_amp: bool = False,
        return_selector_sampling_diagnostics: bool = False,
        selector_probe_timesteps: Optional[List[int]] = None,
        return_diffusion_rate_trace: bool = False,
        n_samples_per_input: int = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        del use_amp, selector_probe_timesteps

        batch_size, num_nodes = int(shape[0]), int(shape[2])

        # Build a minimal Data object if none provided
        if data is None:
            data = Data()

        x_init = torch.randn(shape, device=device, dtype=torch.float32)
        context = self._build_energy_context(
            data=data,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=x_init.dtype,
        )

        dual_lambda = self._init_dual_lambda(
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            dtype=x_init.dtype,
            init_value=self.dual_lambda_init,
        )

        if self.dual_update_mode == "x0_pred":
            x_t, dual_lambda, rate_trace = self._run_single_backward_pass(
                x_init=x_init,
                context=context,
                dual_lambda=dual_lambda,
                n_samples_per_input=n_samples_per_input,
                return_diffusion_rate_trace=return_diffusion_rate_trace,
                update_dual_each_timestep=True,
                show_progress=True,
                progress_desc="Portfolio Energy-DDPM Sampling",
                inverse_beta_override=None,
            )
        elif self.dual_update_mode == "langevin":
            x_t, dual_lambda, rate_trace = self._run_single_langevin_pass(
                x_init=x_init,
                context=context,
                dual_lambda=dual_lambda,
                n_samples_per_input=n_samples_per_input,
                return_diffusion_rate_trace=return_diffusion_rate_trace,
                update_dual_each_timestep=True,
                show_progress=True,
                progress_desc="Portfolio Energy-DDPM Langevin",
                inverse_beta_override=None,
            )
        else:
            # full_backward / hybrid modes
            hybrid_mode = self.dual_update_mode == "hybrid"
            mode_label = "hybrid" if hybrid_mode else "full_backward"
            ib_const = torch.tensor(
                float(self.inverse_beta), device=x_init.device, dtype=x_init.dtype,
            )
            if hybrid_mode:
                tqdm.tqdm.write(
                    "[Portfolio hybrid] running x0_pred warm-start for dual variable."
                )
                _, dual_lambda, _ = self._run_single_backward_pass(
                    x_init=x_init, context=context, dual_lambda=dual_lambda,
                    n_samples_per_input=n_samples_per_input,
                    return_diffusion_rate_trace=False,
                    update_dual_each_timestep=True,
                    show_progress=True,
                    progress_desc="Portfolio Hybrid Warmup",
                    inverse_beta_override=ib_const,
                )

            total_passes = int(self.dual_num_outer_iterations) + 1
            for outer_idx in tqdm.tqdm(
                range(self.dual_num_outer_iterations),
                desc="Portfolio Dual Updates", unit="iter",
            ):
                if hybrid_mode:
                    ib_outer = ib_const
                else:
                    ib_outer = self._inverse_beta_for_full_backward_pass(
                        pass_index=outer_idx, total_passes=total_passes,
                        device=x_init.device, dtype=x_init.dtype,
                    )
                x_candidate, _, _ = self._run_single_backward_pass(
                    x_init=x_init, context=context, dual_lambda=dual_lambda,
                    n_samples_per_input=n_samples_per_input,
                    return_diffusion_rate_trace=False,
                    update_dual_each_timestep=False,
                    show_progress=False, progress_desc="",
                    inverse_beta_override=ib_outer,
                )
                ergodic_rates, _ = self._ergodic_rates_from_samples(
                    x=x_candidate, context=context,
                )
                dual_lambda = self._dual_ascent_step(
                    dual_lambda=dual_lambda, ergodic_rates=ergodic_rates,
                    context=context,
                    t=torch.zeros(batch_size, device=dual_lambda.device, dtype=torch.long),
                )

            if hybrid_mode:
                ib_final = ib_const
            else:
                ib_final = self._inverse_beta_for_full_backward_pass(
                    pass_index=int(self.dual_num_outer_iterations),
                    total_passes=total_passes,
                    device=x_init.device, dtype=x_init.dtype,
                )
            x_t, dual_lambda, rate_trace = self._run_single_backward_pass(
                x_init=x_init, context=context, dual_lambda=dual_lambda,
                n_samples_per_input=n_samples_per_input,
                return_diffusion_rate_trace=return_diffusion_rate_trace,
                update_dual_each_timestep=False,
                show_progress=True,
                progress_desc="Portfolio Energy-DDPM Sampling",
                inverse_beta_override=ib_final,
            )

        if rate_trace is not None:
            rate_trace["final_dual_lambda_mean"] = float(dual_lambda.mean().item())
            rate_trace["final_dual_lambda_max"] = float(dual_lambda.max().item())
            if return_selector_sampling_diagnostics:
                return x_t, None, rate_trace
            return x_t, rate_trace

        if return_selector_sampling_diagnostics:
            return x_t, None
        return x_t

    # ------------------------------------------------------------------
    # Convenience: extract final portfolio weights from z-space samples
    # ------------------------------------------------------------------

    def z_to_portfolio_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Convert raw z-space DDPM output [B, 1, N, 1] to simplex weights [B, N]."""
        return self._z_to_weights(z)
