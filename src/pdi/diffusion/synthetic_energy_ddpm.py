"""Synthetic constrained Boltzmann sampler -- MoG potential with halfspace constraints.

Subclasses EnergyDDPM, overriding only the domain-specific pieces:
  - _build_energy_context   (MoG + constraint params instead of channel data)
  - _energy_from_samples    (Boltzmann potential + Lagrangian instead of SINR)
  - _ergodic_rates_from_samples  (constraint satisfaction values instead of rates)
  - _rates_summary_from_samples  (synthetic metrics for trace logging)

All other machinery (MC score estimator, dual ascent, noise schedule, backward
pass, Langevin pass) is inherited unchanged from EnergyDDPM / _EnergyScoreMixin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pdi.diffusion.energy_ddpm import EnergyDDPM


# ---------------------------------------------------------------------------
# Context dataclass -- replaces _EnergySamplerContext for synthetic problems
# ---------------------------------------------------------------------------

@dataclass
class SyntheticContext:
    """Stores MoG + halfspace constraint parameters for a batch of samples.

    All constraint arrays are padded to length d (= num "nodes") so that the
    per-node dual-lambda convention (``[B, d]``) is compatible with the
    inherited ``_dual_ascent_step`` which broadcasts ``context.r_min``
    against ``[B, N]`` ergodic_rates.
    """

    means: torch.Tensor            # [K, d]
    cov_diag: torch.Tensor         # [K, d]  diagonal covariances
    precision_diag: torch.Tensor   # [K, d]  = 1 / cov_diag
    log_norm_const: torch.Tensor   # [K]     -d/2 log(2pi) - 1/2 sum log(sigma^2_ki)
    log_weights: torch.Tensor      # [K]     log(w_k)
    constraint_A: torch.Tensor     # [d, d]  padded (first m rows are real constraints)
    constraint_b: torch.Tensor     # [d]     padded
    r_min: torch.Tensor            # [B, 1, 1]  = 0  (threshold for dual ascent)
    num_real_constraints: int      # m (before padding)


# ---------------------------------------------------------------------------
# Stub model (mc_sampler placeholder -- never called during energy sampling)
# ---------------------------------------------------------------------------

class _NoOpModel(nn.Module):
    """Placeholder model for EnergyDDPM -- analytical score replaces model."""

    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


# ---------------------------------------------------------------------------
# SyntheticEnergyDDPM
# ---------------------------------------------------------------------------

class SyntheticEnergyDDPM(EnergyDDPM):
    """Energy-guided DDPM for constrained MoG Boltzmann distributions.

    The energy is:

        E(x, lambda; beta) = beta^{-1} [ U(x) + sum_j lambda_j h_j(x) ]

    where ``U(x) = -log sum_k w_k N(x; mu_k, Sigma_k)`` is the MoG potential
    and ``h_j(x) = b_j - a_j^T x <= 0`` are halfspace constraints.

    Parameters
    ----------
    means : Tensor [K, d]
        Gaussian mixture component means.
    cov_diag : Tensor [K, d]
        Diagonal covariance entries for each component.
    weights : Tensor [K]
        Mixture weights (must sum to 1).
    constraint_A : Tensor [m, d]
        Constraint normal vectors (unit-normalised rows).
    constraint_b : Tensor [m]
        Constraint thresholds.  Constraint j is satisfied when a_j^T x >= b_j.
    """

    def __init__(
        self,
        *,
        means: torch.Tensor,
        cov_diag: torch.Tensor,
        weights: torch.Tensor,
        constraint_A: torch.Tensor,
        constraint_b: torch.Tensor,
        model: Optional[nn.Module] = None,
        num_timesteps: int = 500,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        energy_mc_samples: int = 32,
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
        clip_range: Optional[float] = None,
        potential_divisor: Optional[float] = None,
        constraint_term_clamp: Optional[float] = None,
    ):
        d = int(means.shape[1])
        m = int(constraint_A.shape[0])

        if model is None:
            model = _NoOpModel()

        # Auto-compute clip range from mode positions if not given.
        # Covers the furthest mode centre + 4σ of the widest mode.
        if clip_range is None:
            max_abs_mean = float(means.detach().abs().max().item())
            max_std = float(cov_diag.detach().sqrt().max().item())
            clip_range = max_abs_mean + 8.0 * max_std
        self._clip_range = float(clip_range)
        self._potential_divisor = (
            float(potential_divisor) if potential_divisor is not None else 1.0
        )
        # Per-element cap on ``lambda_j * h_j(x)`` during energy evaluation.
        # Prevents the constraint term from exploding when both inverse_beta
        # and lambda are large, which would otherwise underflow the Boltzmann
        # weight exp(-E) to zero and degenerate the MC score estimate.
        # Set only on the *training* sampler; leave None on inference samplers
        # so the true tempered posterior is used at inference.
        self._constraint_term_clamp = (
            float(constraint_term_clamp) if constraint_term_clamp is not None else None
        )

        # ------------------------------------------------------------------
        # Pre-compute and store MoG / constraint tensors (plain attributes;
        # _build_energy_context moves them to the target device on each call).
        # ------------------------------------------------------------------
        self._syn_d = d
        self._syn_m = m
        self._syn_means = means.detach().float().clone()
        self._syn_cov_diag = cov_diag.detach().float().clone()
        self._syn_precision_diag = (1.0 / cov_diag.detach().float().clone())
        self._syn_log_norm_const = (
            -0.5 * d * math.log(2.0 * math.pi)
            - 0.5 * cov_diag.detach().float().clone().log().sum(dim=1)
        )
        self._syn_log_weights = weights.detach().float().clone().log()

        # Pad constraints from [m, d] → [d, d] with zero rows so that
        # dual_lambda has the standard [B, N=d] layout expected everywhere.
        if m < d:
            self._syn_A = torch.cat(
                [constraint_A.detach().float().clone(),
                 torch.zeros(d - m, d)], dim=0,
            )
            self._syn_b = torch.cat(
                [constraint_b.detach().float().clone(),
                 torch.zeros(d - m)], dim=0,
            )
        elif m == d:
            self._syn_A = constraint_A.detach().float().clone()
            self._syn_b = constraint_b.detach().float().clone()
        else:
            raise ValueError(
                f"m ({m}) > d ({d}): more constraints than coordinates is not "
                "supported with per-coordinate lambda padding."
            )

        # ------------------------------------------------------------------
        # Initialise parent (EnergyDDPM → _EnergyScoreMixin + DDPM).
        # Irrelevant WRA params get harmless dummy values.
        # ------------------------------------------------------------------
        super().__init__(
            model=model,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            clip_denoised=True,
            energy_mc_samples=energy_mc_samples,
            energy_num_channel_realizations=1,  # unused
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
        )

    # ------------------------------------------------------------------
    # Per-sample dual variable recording
    # ------------------------------------------------------------------

    def enable_dual_recording(self, sample_indices: list[int]):
        """Enable per-sample dual variable recording for given sample indices."""
        self._record_dual = True
        self._record_indices = list(sample_indices)
        self._dual_traces: list[torch.Tensor] = []  # each entry: [n_indices, m]

    def disable_dual_recording(self):
        self._record_dual = False

    def get_dual_traces(self) -> torch.Tensor:
        """Return [T_recorded, n_indices, m] tensor of recorded dual trajectories."""
        return torch.stack(self._dual_traces, dim=0)

    def _dual_ascent_step(self, dual_lambda, ergodic_rates, context, t=None):
        updated = super()._dual_ascent_step(dual_lambda, ergodic_rates, context, t)
        if getattr(self, "_record_dual", False):
            m = self._syn_m
            self._dual_traces.append(
                updated[self._record_indices, :m].detach().cpu()
            )
        return updated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_dummy_data(batch_size: int, d: int) -> Data:
        """Create a minimal PyG Data batch for ``sample()`` / trainer rollout."""
        from torch_geometric.data import Batch

        graphs = [
            Data(
                y=torch.zeros(d, 1, 1),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                num_nodes=d,
            )
            for _ in range(batch_size)
        ]
        return Batch.from_data_list(graphs)

    def _mog_potential(
        self, x_flat: torch.Tensor, ctx: SyntheticContext,
    ) -> torch.Tensor:
        """U(x) = -log sum_k w_k N(x; mu_k, Sigma_k).  x_flat [B,d] -> [B].

        If ``potential_divisor`` is set, U is divided by that constant.
        Division by d gives per-dimension normalized log-density, whose
        variance across the data support is O(1) regardless of d.
        """
        diff = x_flat.unsqueeze(1) - ctx.means.unsqueeze(0)          # [B, K, d]
        mahal = (diff.square() * ctx.precision_diag.unsqueeze(0)).sum(dim=2)  # [B, K]
        log_components = (
            ctx.log_norm_const + ctx.log_weights - 0.5 * mahal       # [B, K]
        )
        U = -torch.logsumexp(log_components, dim=1)                   # [B]
        return U / self._potential_divisor

    def _constraint_values(
        self, x_flat: torch.Tensor, ctx: SyntheticContext,
    ) -> torch.Tensor:
        """f_j(x) - b_j  for each (padded) constraint j.

        Returns [B, d].  Positive means satisfied (a_j^T x >= b_j).
        """
        return x_flat @ ctx.constraint_A.T - ctx.constraint_b        # [B, d]

    def _resolve_inverse_beta(
        self, x: torch.Tensor, t: Optional[torch.Tensor],
        inverse_beta_override: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return per-sample inverse_beta as a [B] tensor."""
        B = x.shape[0]
        if inverse_beta_override is not None:
            ib = inverse_beta_override.to(device=x.device, dtype=x.dtype).view(-1)
            return ib.expand(B) if ib.numel() == 1 else ib
        if t is None:
            return self.inverse_beta_by_t[0].to(
                device=x.device, dtype=x.dtype,
            ).expand(B)
        t_safe = t.to(device=x.device, dtype=torch.long).clamp(
            0, self.num_timesteps - 1,
        )
        return self._gather(
            self.inverse_beta_by_t.to(device=x.device), t_safe,
        ).to(dtype=x.dtype)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _score_to_x0_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override to clamp x0_pred to the data support.

        Analogous to ``clip_denoised=True`` in WRA (which clamps to
        [-0.5, 0.5]).  Without clamping, the 1/sqrt(alpha_bar) factor
        amplifies large scores into x0_pred values far outside the MoG
        support, causing runaway divergence.
        """
        alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp_min(1e-12))
        sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))

        eps_pred = -score * sqrt_one_minus
        x0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar
        x0_pred = x0_pred.clamp(-self._clip_range, self._clip_range)
        return x0_pred, eps_pred

    def _clip_mc_candidate(self, x0: torch.Tensor) -> torch.Tensor:
        """Clamp MC x0 candidates to the MoG data support.

        Mirrors ``_score_to_x0_eps``: without this clamp, far-tail draws from
        ``N(x_t, sigma_t^2 I)`` produce x0 with huge negative log-density,
        which blows up the MC-score softmax (ESS → 1).
        """
        return x0.clamp(-self._clip_range, self._clip_range)

    def _build_energy_context(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SyntheticContext:
        """Build context from stored problem parameters (ignores ``data``)."""
        return SyntheticContext(
            means=self._syn_means.to(device=device, dtype=dtype),
            cov_diag=self._syn_cov_diag.to(device=device, dtype=dtype),
            precision_diag=self._syn_precision_diag.to(device=device, dtype=dtype),
            log_norm_const=self._syn_log_norm_const.to(device=device, dtype=dtype),
            log_weights=self._syn_log_weights.to(device=device, dtype=dtype),
            constraint_A=self._syn_A.to(device=device, dtype=dtype),
            constraint_b=self._syn_b.to(device=device, dtype=dtype),
            r_min=torch.zeros(batch_size, 1, 1, device=device, dtype=dtype),
            num_real_constraints=self._syn_m,
        )

    def _ergodic_rates_from_samples(
        self,
        x: torch.Tensor,           # [B, 1, d, 1]
        context: SyntheticContext,  # type: ignore[override]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (constraint_satisfaction [B, d], neg_potential [B]).

        ``constraint_satisfaction`` plays the role of per-user ergodic rates:
        the dual-ascent step computes ``violation = r_min(=0) - rates``, which
        equals ``b_j - a_j^T x = h_j(x)``—positive when the constraint is
        violated, exactly as needed.

        ``neg_potential = -U(x)`` is returned as the "sum_rate" analogue for
        trace logging (higher is better, like sum rate).
        """
        x_flat = x[:, 0, :, 0]                                       # [B, d]
        rates = self._constraint_values(x_flat, context)              # [B, d]
        neg_potential = -self._mog_potential(x_flat, context)         # [B]
        return rates, neg_potential

    def _energy_from_samples(
        self,
        x: torch.Tensor,                                   # [B, 1, d, 1]
        context: SyntheticContext,                          # type: ignore[override]
        dual_lambda: Optional[torch.Tensor] = None,        # [B, d]
        t: Optional[torch.Tensor] = None,                  # [B]
        inverse_beta_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Lagrangian energy  E = ib * [ U(x) + sum_j lambda_j h_j(x) ]."""
        x_flat = x[:, 0, :, 0]                                       # [B, d]
        B = x_flat.shape[0]

        U = self._mog_potential(x_flat, context)                      # [B]
        rates = self._constraint_values(x_flat, context)              # [B, d]

        if dual_lambda is None:
            dual_lambda = torch.zeros(B, x.shape[2], device=x.device, dtype=x.dtype)

        # Lagrangian:  sum_j lambda_j (r_min - rates_j)
        #            = sum_j lambda_j (0 - (a_j^T x - b_j))
        #            = sum_j lambda_j h_j(x)
        per_term = (context.r_min.view(-1, 1) - rates) * dual_lambda  # [B, d]
        if self._constraint_term_clamp is not None:
            # Cap each lambda_j * h_j(x) to avoid exp(-ib * lambda * h) underflow.
            per_term = per_term.clamp(max=float(self._constraint_term_clamp))
        lagrangian = per_term.sum(dim=1)                              # [B]

        ib = self._resolve_inverse_beta(x, t, inverse_beta_override)  # [B]
        return ib * (U + lagrangian)                                  # [B]

    def _rates_summary_from_samples(
        self,
        x: torch.Tensor,                                   # [B, 1, d, 1]
        context: SyntheticContext,                          # type: ignore[override]
        n_samples_per_input: int = 1,
        dual_lambda: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        inverse_beta_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute synthetic-specific summary metrics for trace logging."""
        with torch.no_grad():
            B = x.shape[0]
            x_flat = x[:, 0, :, 0]
            m = context.num_real_constraints

            U = self._mog_potential(x_flat, context)
            rates = self._constraint_values(x_flat, context)

            # Feasibility: fraction of real constraints satisfied per sample
            feas = (rates[:, :m] >= -1e-6).float().mean(dim=1)       # [B]
            violation = (-rates[:, :m]).clamp_min(0.0)

            if dual_lambda is None:
                dual_lambda = torch.zeros(B, x.shape[2], device=x.device, dtype=x.dtype)
            per_term = (context.r_min.view(-1, 1) - rates) * dual_lambda
            if self._constraint_term_clamp is not None:
                per_term = per_term.clamp(max=float(self._constraint_term_clamp))
            lagrangian = per_term.sum(dim=1)
            ib = self._resolve_inverse_beta(x, t, inverse_beta_override)
            energy = ib * (U + lagrangian)

            neg_U = float((-U).mean().item())
            feas_p5 = float(feas.quantile(0.05).item()) if B > 1 else float(feas.mean().item())

        return {
            "sum_rate": neg_U,
            "p5": feas_p5,
            "energy": float(energy.mean().item()),
            "per_net_sum_rates": [neg_U],
            "per_net_p5s": [feas_p5],
            "per_net_energies": [float(energy.mean().item())],
        }

    def sample(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        data: Optional[Data] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Override to auto-create dummy Data when ``data`` is None."""
        if data is None:
            B, _T, N, _F = shape
            data = self.make_dummy_data(batch_size=B, d=N)
        return super().sample(shape=shape, device=device, data=data, **kwargs)
