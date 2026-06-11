"""Energy-score training loop (iDEM-style with constrained extension).

Implements Algorithm 3 of the design document:

    - Outer loop: run the *current* x0_pred sampler on a network and push the
      full (x_t, t, lambda_t) trajectory into a replay buffer.
    - Inner loop: sample entries from the buffer, optionally decouple lambda
      via pi_lambda(. | t), compute the MC target, and regress the score net.

The implementation is intentionally minimal and focused on the smoke-test
regime.  Efficiency improvements (better batching across entries, on-GPU
storage, etc.) are deferred.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pdi.diffusion.energy_ddpm import EnergyDDPM
from pdi.trainers.energy_score.lambda_prior import (
    ExponentialLambdaPrior,
    LambdaPrior,
)
from pdi.trainers.energy_score.score_net import ScoreNetWithLambda
from pdi.trainers.energy_score.trajectory_buffer import (
    TrajectoryBuffer,
    TrajectoryEntry,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EnergyScoreTrainConfig:
    """Hyperparameters for ``EnergyScoreTrainer``."""

    # Optimiser
    lr: float = 1e-4
    weight_decay: float = 0.0
    # "unit" | "one_minus_alpha_bar" | "min_snr"
    # min_snr: w(t) = min(SNR_t, gamma) / SNR_t for eps-prediction
    # (Hang et al. 2023, "Efficient Diffusion Training via Min-SNR Weighting")
    loss_weighting: str = "one_minus_alpha_bar"
    min_snr_gamma: float = 5.0
    grad_clip_norm: Optional[float] = 1.0  # None to disable

    # Target normalization: rescale eps_pred/eps_target by per-sample target
    # magnitude before MSE, to tame 5-orders-of-magnitude variance across t,lambda.
    target_normalize: bool = True
    target_normalize_eps: float = 1e-3

    # Clip MC score target to this max norm (per sample). Prevents loss spikes
    # from rare high-magnitude MC estimates. iDEM uses 20 for d≈30–40.
    target_clip_norm: float = 0.0  # 0 = disabled

    # MC target
    k_train: int = 16

    # Inner loop
    inner_steps_per_outer: int = 5
    minibatch_size: int = 4  # number of (record, step) triples per SGD step

    # Mixing between on-policy and decoupled lambda
    rho_max: float = 0.5
    warmup_iters: int = 50  # iters over which rho ramps from 0 to rho_max

    # Optional schedule on the MC sampler's ``inverse_beta`` across outer
    # iterations. When enabled, rewrites ``mc_sampler.inverse_beta_by_t`` to a
    # constant tensor of the current value before each outer rollout — so the
    # MC target tempering anneals over training. Default disabled.
    ib_schedule_enabled: bool = False
    ib_schedule_start: float = 0.1
    ib_schedule_end: float = 3.0
    ib_schedule_shape: str = "linear"  # "linear" | "log"
    ib_schedule_warmup_outer_iters: Optional[int] = None  # if set, ramp over first N outer iters, then hold at ib_schedule_end


    # Replay buffer
    buffer_capacity: int = 4096
    mc_score_chunk: int = 16

    # Lambda prior (decoupled component). Calibrated from the 2026-03-16
    # best x0_pred MC run: see scripts/wra/diffusion/calibrate_lambda_prior.py.
    lambda_prior_mu_min: float = 5.0
    lambda_prior_mu_max: float = 80.0

    # Logging
    log_every_n_iters: int = 1

    # Rollout
    use_progress_bar: bool = False
    mc_rollout_fraction: float = 0.0  # fraction of rollouts using MC ceiling instead of score_net
    perturb_fraction: float = 0.0  # fraction of train samples to perturb x_t and λ
    perturb_x_std: float = 0.1     # std of Gaussian noise added to x_t
    perturb_lambda_std: float = 0.5  # std of Gaussian noise added to λ (clamped ≥0)
    num_rollouts_per_outer: int = 1
    n_samples_per_network: int = 1  # K independent trajectories per network in rollout


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EnergyScoreTrainer:
    """Trains a ``ScoreNetWithLambda`` against MC-score targets.

    The trainer does not subclass :class:`BaseTrainer` to keep the interface
    narrow and avoid coupling to the main ``DiffusionTrainer``.  The outer
    API is ``fit(data_loader, num_iters)``.
    """

    def __init__(
        self,
        score_net: ScoreNetWithLambda,
        mc_sampler: EnergyDDPM,
        config: EnergyScoreTrainConfig,
        device: torch.device,
        *,
        lambda_prior: Optional[LambdaPrior] = None,
        rng_seed: int = 0,
    ):
        if not isinstance(score_net, ScoreNetWithLambda):
            raise TypeError(
                "score_net must be a ScoreNetWithLambda instance; got "
                f"{type(score_net).__name__}"
            )
        if not isinstance(mc_sampler, EnergyDDPM):
            raise TypeError(
                f"mc_sampler must be EnergyDDPM; got {type(mc_sampler).__name__}"
            )

        self.score_net = score_net.to(device)
        self.mc_sampler = mc_sampler.to(device)
        self.mc_sampler.eval()  # MC sampler is used for targets only; no gradients
        for p in self.mc_sampler.parameters():
            p.requires_grad_(False)

        self.cfg = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.score_net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # K_MC for inner-loop score targets (high for quality targets).
        # Rollouts use the sampler's original K (low, just needs reasonable trajectories).
        self._k_train = int(self.cfg.k_train)
        self._k_rollout = int(self.mc_sampler.energy_mc_samples)
        self.mc_sampler.energy_mc_samples = self._k_train

        self.buffer = TrajectoryBuffer(
            capacity=self.cfg.buffer_capacity,
            rng=random.Random(rng_seed),
        )

        if lambda_prior is None:
            lambda_prior = ExponentialLambdaPrior(
                sqrt_alphas_cumprod=self.mc_sampler.sqrt_alphas_cumprod.detach(),
                mu_min=self.cfg.lambda_prior_mu_min,
                mu_max=self.cfg.lambda_prior_mu_max,
            )
        self.lambda_prior = lambda_prior

        self._rng = random.Random(rng_seed)
        self._iteration = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        return int(self.mc_sampler.num_timesteps)

    def _extract_cond_from_data(
        self,
        data: Data,
        batch_size: int,
        num_nodes: int,
    ) -> Optional[torch.Tensor]:
        """Reshape ``data.x`` (if any) to ``[B, T_cond, N, F_cond]``.

        Mirrors the convention used by ``DDPM.training_loss``.
        """
        if not hasattr(data, "x") or data.x is None:
            return None
        x = data.x
        if x.dim() < 2:
            raise ValueError(f"Unexpected data.x rank: {x.dim()}")
        return x.view(batch_size, num_nodes, *x.shape[1:]).swapaxes(1, 2).contiguous()

    def _loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weighting schedule."""
        if self.cfg.loss_weighting == "unit":
            return torch.ones_like(t, dtype=torch.float32)
        if self.cfg.loss_weighting == "one_minus_alpha_bar":
            ab = self.mc_sampler.alphas_cumprod.to(t.device)[t]
            return (1.0 - ab).clamp_min(1e-8)
        if self.cfg.loss_weighting == "min_snr":
            ab = self.mc_sampler.alphas_cumprod.to(t.device)[t].clamp(1e-8, 1.0 - 1e-8)
            snr = ab / (1.0 - ab)
            gamma = float(self.cfg.min_snr_gamma)
            return (torch.minimum(snr, torch.full_like(snr, gamma)) / snr).clamp_min(1e-8)
        raise ValueError(f"Unknown loss_weighting: {self.cfg.loss_weighting!r}")

    def _current_rho(self) -> float:
        if self.cfg.warmup_iters <= 0:
            return float(self.cfg.rho_max)
        ratio = min(1.0, self._iteration / float(self.cfg.warmup_iters))
        return float(self.cfg.rho_max) * ratio

    # ------------------------------------------------------------------
    # Rollout (outer loop body)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, data: Data) -> None:
        """Run reverse diffusion and push trajectory to buffer.

        The batch contains n_nets unique networks, each tiled K times
        (K = n_samples_per_network). Lambda is shared per network.
        All graphs are processed in a single forward pass per timestep.
        """
        from torch_geometric.data import Batch as PyGBatch

        data = data.to(self.device)
        batch_size = int(data.num_graphs)  # n_nets * K
        num_nodes = int(data.y.size(0) // batch_size)
        seq_len = int(data.y.size(1))
        feat_dim = int(data.y.size(2))
        K = max(1, self.cfg.n_samples_per_network)
        n_nets = batch_size // K

        # Build per-network dual contexts (needed for lambda update)
        all_graphs = data.to_data_list()
        net_dual_ctxs = []
        for i in range(n_nets):
            nb = PyGBatch.from_data_list(
                all_graphs[i * K : (i + 1) * K],
                follow_batch=["y"], exclude_keys=["info"],
            ).to(self.device)
            ctx = self.mc_sampler._build_dual_context(
                data=nb, batch_size=K, num_nodes=num_nodes,
                device=self.device, dtype=torch.float32,
            )
            if ctx is None:
                ctx = self.mc_sampler._build_energy_context(
                    data=nb, batch_size=K, num_nodes=num_nodes,
                    device=self.device, dtype=torch.float32,
                )
            net_dual_ctxs.append(ctx)

        x_t = torch.randn(batch_size, seq_len, num_nodes, feat_dim,
                           device=self.device, dtype=torch.float32)
        cond = self._extract_cond_from_data(data, batch_size, num_nodes)
        dual_lambda_net = self.mc_sampler._init_dual_lambda(
            batch_size=n_nets, num_nodes=num_nodes,
            device=self.device, dtype=torch.float32,
            init_value=float(self.mc_sampler.dual_lambda_init),
        )  # [n_nets, N]

        # Pre-convert sparse edge_index to dense adj once
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        if edge_index.dim() == 2:
            A = torch.zeros(batch_size, num_nodes, num_nodes,
                            device=edge_index.device, dtype=torch.float32)
            row, col = edge_index
            graph_idx = row // num_nodes
            local_row = row % num_nodes
            local_col = col % num_nodes
            if edge_weight is not None:
                A[graph_idx, local_row, local_col] = edge_weight
            else:
                A[graph_idx, local_row, local_col] = 1.0
            edge_index = A
            edge_weight = None

        T = self.num_timesteps
        x_by_t: List[torch.Tensor] = []
        lambda_by_t: List[torch.Tensor] = []
        t_values: List[int] = []

        sqrt_alphas = self.mc_sampler.sqrt_alphas_cumprod
        sqrt_one_minus = self.mc_sampler.sqrt_one_minus_alphas_cumprod
        post_var = self.mc_sampler.posterior_variance
        dual_step_size_by_t = self.mc_sampler.dual_step_size_by_t
        clip = self.mc_sampler.clip_denoised

        for t_int in reversed(range(T)):
            # Broadcast shared lambda
            dual_lambda = dual_lambda_net.unsqueeze(1).expand(
                n_nets, K, num_nodes).reshape(batch_size, num_nodes)

            # Record before update
            x_by_t.append(x_t.detach().cpu())
            lambda_by_t.append(dual_lambda.detach().cpu())
            t_values.append(t_int)

            # Batched score net forward (all networks at once)
            t_full = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)
            eps_pred = self.score_net(
                x=x_t, timesteps=t_full, dual_lambda=dual_lambda,
                edge_index=edge_index, edge_weight=edge_weight,
                cond=cond, return_intermediates=False,
            )

            sqrt_ab = sqrt_alphas[t_int]
            sqrt_omb = sqrt_one_minus[t_int]
            x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab.clamp_min(1e-12)
            if clip:
                x0_pred = x0_pred.clamp(-0.5, 0.5)

            # Dual ascent: per-network, average over K samples
            step = dual_step_size_by_t[t_int]
            for i in range(n_nets):
                s, e = i * K, (i + 1) * K
                ergodic_rates_i, _ = self.mc_sampler._ergodic_rates_from_samples(
                    x=x0_pred[s:e], context=net_dual_ctxs[i],
                )  # [K, N]
                avg_rates = ergodic_rates_i.mean(dim=0)  # [N]
                r_min_i = net_dual_ctxs[i].r_min[0]
                violation = r_min_i - avg_rates
                dual_lambda_net[i] = (dual_lambda_net[i] + step * violation).clamp_min(0.0)
                if self.mc_sampler.dual_lambda_max is not None:
                    dual_lambda_net[i] = dual_lambda_net[i].clamp_max(
                        float(self.mc_sampler.dual_lambda_max))

            # DDPM posterior transition
            mean = self.mc_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t_full)
            if t_int > 0:
                x_t = mean + torch.sqrt(post_var[t_int].clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        self.buffer.push_trajectory(
            data=data,
            x_by_t=torch.stack(x_by_t, dim=0),       # [T, B, T_s, N, F]
            lambda_by_t=torch.stack(lambda_by_t, dim=0),  # [T, B, N]
            t_values=torch.tensor(t_values, dtype=torch.long),
        )

        if not hasattr(self, "_recent_lambda_norms"):
            self._recent_lambda_norms = []
        self._recent_lambda_norms.append(
            dual_lambda_net.detach().cpu().norm(dim=-1).numpy())

    def pop_lambda_stats(self):
        """Return and clear λ-norm stats from recent rollouts."""
        if not hasattr(self, "_recent_lambda_norms") or not self._recent_lambda_norms:
            return None
        import numpy as np
        all_norms = np.concatenate(self._recent_lambda_norms)
        stats = {
            "mean": float(all_norms.mean()),
            "q50": float(np.median(all_norms)),
            "q95": float(np.percentile(all_norms, 95)),
            "max": float(all_norms.max()),
            "n_rollouts": len(self._recent_lambda_norms),
        }
        self._recent_lambda_norms.clear()
        return stats

    @torch.no_grad()
    def mc_rollout(self, data: Data) -> None:
        """Run a rollout using the MC ceiling (analytic energy) and push to buffer.

        Identical to ``rollout()`` except the score at each step comes from
        ``mc_sampler._estimate_score`` instead of the learned ``score_net``.
        This provides clean training signal independent of the net's quality.
        """
        self.mc_sampler.energy_mc_samples = self._k_rollout
        data = data.to(self.device)
        batch_size = int(data.num_graphs)
        num_nodes = int(data.y.size(0) // batch_size)
        seq_len = int(data.y.size(1))
        feat_dim = int(data.y.size(2))
        shape = (batch_size, seq_len, num_nodes, feat_dim)

        context = self.mc_sampler._build_energy_context(
            data=data, batch_size=batch_size, num_nodes=num_nodes,
            device=self.device, dtype=torch.float32,
        )
        mc_dual_context = self.mc_sampler._build_dual_context(
            data=data, batch_size=batch_size, num_nodes=num_nodes,
            device=self.device, dtype=torch.float32,
        )
        mc_dual_ctx = mc_dual_context if mc_dual_context is not None else context

        x_t = torch.randn(shape, device=self.device, dtype=torch.float32)
        dual_lambda = self.mc_sampler._init_dual_lambda(
            batch_size=batch_size, num_nodes=num_nodes,
            device=self.device, dtype=torch.float32,
            init_value=float(self.mc_sampler.dual_lambda_init),
        )

        T = self.num_timesteps
        x_by_t: List[torch.Tensor] = []
        lambda_by_t: List[torch.Tensor] = []
        t_values: List[int] = []

        for t_int in reversed(range(T)):
            t = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)

            mc_score = self.mc_sampler._estimate_score(
                x_t=x_t, t=t, context=context, dual_lambda=dual_lambda,
            )
            alpha_bar = self.mc_sampler.alphas_cumprod.to(self.device)[t].view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
            sqrt_omb = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
            eps_pred = -mc_score * sqrt_omb
            x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab
            if self.mc_sampler.clip_denoised:
                x0_pred = x0_pred.clamp(-0.5, 0.5)
            if hasattr(self.mc_sampler, '_z_to_weights'):
                Bp, _, Np, _ = x0_pred.shape
                z_flat = x0_pred[:, 0, :, 0]
                w = torch.softmax(z_flat, dim=-1)
                z_proj = torch.log(w.clamp_min(1e-12))
                z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
                x0_pred = z_proj.view(Bp, 1, Np, 1)

            x_by_t.append(x_t.detach().cpu())
            lambda_by_t.append(dual_lambda.detach().cpu())
            t_values.append(t_int)

            if self.mc_sampler.dual_lambda_mode == "shared_per_network":
                ergodic_rates = self.mc_sampler._time_shared_ergodic_rates_broadcast(
                    x=x0_pred, context=mc_dual_ctx,
                    n_samples_per_input=batch_size,
                )
            else:
                ergodic_rates, _ = self.mc_sampler._ergodic_rates_from_samples(
                    x=x0_pred, context=mc_dual_ctx,
                )
            dual_lambda = self.mc_sampler._dual_ascent_step(
                dual_lambda=dual_lambda, ergodic_rates=ergodic_rates,
                context=mc_dual_ctx, t=t,
            )

            mean = self.mc_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
            if t_int > 0:
                var = self.mc_sampler.posterior_variance.to(self.device)[t].view(-1, 1, 1, 1)
                x_t = mean + torch.sqrt(var.clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        self.mc_sampler.energy_mc_samples = self._k_train
        self.buffer.push_trajectory(
            data=data,
            x_by_t=torch.stack(x_by_t, dim=0),
            lambda_by_t=torch.stack(lambda_by_t, dim=0),
            t_values=torch.tensor(t_values, dtype=torch.long),
            sample_weights=None,
        )

    @torch.no_grad()
    def ddim_rollout(self, ddim_model, data: Data) -> None:
        """Run a DDIM model rollout and push trajectory to buffer.

        Uses the same shared-lambda-per-network scheme as rollout().
        """
        data = data.to(self.device)
        batch_size = int(data.num_graphs)
        num_nodes = int(data.y.size(0) // batch_size)
        seq_len = int(data.y.size(1))
        feat_dim = int(data.y.size(2))
        shape = (batch_size, seq_len, num_nodes, feat_dim)
        K = max(1, self.cfg.n_samples_per_network)
        n_nets = batch_size // K

        cond = self._extract_cond_from_data(data, batch_size, num_nodes)
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None

        x_t = torch.randn(shape, device=self.device, dtype=torch.float32)

        T = self.num_timesteps
        x_by_t: List[torch.Tensor] = []
        lambda_by_t: List[torch.Tensor] = []
        t_values: List[int] = []

        dual_lambda_net = self.mc_sampler._init_dual_lambda(
            batch_size=n_nets, num_nodes=num_nodes,
            device=self.device, dtype=torch.float32,
            init_value=float(self.mc_sampler.dual_lambda_init),
        )

        context = self.mc_sampler._build_energy_context(
            data=data, batch_size=batch_size, num_nodes=num_nodes,
            device=self.device, dtype=torch.float32,
        )

        sqrt_alphas = self.mc_sampler.sqrt_alphas_cumprod
        sqrt_one_minus = self.mc_sampler.sqrt_one_minus_alphas_cumprod
        post_var = self.mc_sampler.posterior_variance
        dual_step_size_by_t = self.mc_sampler.dual_step_size_by_t
        clip = self.mc_sampler.clip_denoised

        for t_int in reversed(range(T)):
            t = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)

            dual_lambda = dual_lambda_net.unsqueeze(1).expand(
                n_nets, K, num_nodes).reshape(batch_size, num_nodes)

            pred, _ = ddim_model.model(
                x=x_t, timesteps=t, edge_index=edge_index,
                edge_weight=edge_weight, cond=cond,
                return_intermediates=False,
            )
            sqrt_ab = sqrt_alphas[t_int]
            sqrt_omb = sqrt_one_minus[t_int]
            x0_pred = (x_t - sqrt_omb * pred) / sqrt_ab.clamp_min(1e-12)
            if clip:
                x0_pred = x0_pred.clamp(-0.5, 0.5)

            x_by_t.append(x_t.detach().cpu())
            lambda_by_t.append(dual_lambda.detach().cpu())
            t_values.append(t_int)

            ergodic_rates, _ = self.mc_sampler._ergodic_rates_from_samples(
                x=x0_pred, context=context,
            )
            per_net_rates = ergodic_rates.view(n_nets, K, num_nodes).mean(dim=1)

            step = dual_step_size_by_t[t_int]
            violation = context.r_min[:n_nets].view(-1, 1) - per_net_rates
            dual_lambda_net = (dual_lambda_net + step * violation).clamp_min(0.0)
            if self.mc_sampler.dual_lambda_max is not None:
                dual_lambda_net = dual_lambda_net.clamp_max(
                    float(self.mc_sampler.dual_lambda_max))

            mean = self.mc_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
            if t_int > 0:
                x_t = mean + torch.sqrt(post_var[t_int].clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        self.buffer.push_trajectory(
            data=data,
            x_by_t=torch.stack(x_by_t, dim=0),
            lambda_by_t=torch.stack(lambda_by_t, dim=0),
            t_values=torch.tensor(t_values, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Inner loop: one SGD step
    # ------------------------------------------------------------------

    def _gather_minibatch(self, entries: List[TrajectoryEntry]) -> Dict[str, Any]:
        """Extract individual (sample, timestep) entries and re-batch them.

        Each entry yields one graph's x_t, lambda, and PyG data at one
        timestep. All M entries are collated into a single PyG batch of
        M graphs so the score net sees them in one forward pass.
        """
        from torch_geometric.data import Batch as PyGBatch

        x_list = []
        lam_list = []
        t_list = []
        data_list = []

        rec_graph_cache: Dict[int, list] = {}

        for e in entries:
            rec = e.record
            si = e.step_index
            bi = e.sample_index

            x_list.append(rec.x_by_t[si, bi])       # [T_s, N, F]
            lam_list.append(rec.lambda_by_t[si, bi])  # [N]
            t_list.append(int(rec.t_values[si].item()))

            rec_id = id(rec)
            if rec_id not in rec_graph_cache:
                rec_graph_cache[rec_id] = rec.data.to_data_list()
            data_list.append(rec_graph_cache[rec_id][bi])

        data = PyGBatch.from_data_list(data_list, follow_batch=["y"],
                                       exclude_keys=["info"]).to(self.device)
        M = len(entries)
        num_nodes = int(data.y.size(0) // M)

        return {
            "data": data,
            "x_t": torch.stack(x_list).to(self.device),        # [M, T_s, N, F]
            "lambda_on_policy": torch.stack(lam_list).to(self.device),  # [M, N]
            "t": torch.tensor(t_list, device=self.device, dtype=torch.long),
            "cond_full": self._extract_cond_from_data(data, M, num_nodes),
            "edge_index": data.edge_index,
            "edge_weight": data.edge_weight if hasattr(data, "edge_weight") else None,
            "num_nodes": num_nodes,
            "batch_size": M,
        }

    def _build_micro_context(self, data: Data):
        """Build the full-batch ``_EnergySamplerContext`` for the given record."""
        batch_size = int(data.num_graphs)
        num_nodes = int(data.y.size(0) // batch_size)
        return self.mc_sampler._build_energy_context(
            data=data,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=self.device,
            dtype=torch.float32,
        )

    def _resolve_lambda(
        self,
        mb: Dict[str, Any],
        rho: float,
    ) -> torch.Tensor:
        """Mix on-policy and decoupled lambda per sample."""
        M = int(mb["x_t"].shape[0])
        if rho >= 1.0:
            return mb["lambda_on_policy"]
        if rho <= 0.0:
            return self.lambda_prior.sample(
                t=mb["t"], num_nodes=mb["num_nodes"],
                device=self.device, dtype=torch.float32,
            )
        lam_decoupled = self.lambda_prior.sample(
            t=mb["t"], num_nodes=mb["num_nodes"],
            device=self.device, dtype=torch.float32,
        )
        use_on = (torch.rand(M, device=self.device) < rho).view(M, 1)
        return torch.where(use_on, mb["lambda_on_policy"], lam_decoupled)

    def train_step(self) -> Dict[str, float]:
        """One SGD step on M individual (x_t, t, lambda, graph) samples."""
        if self.buffer.num_records == 0:
            raise RuntimeError("train_step called before any rollout populated the buffer.")

        entries = self.buffer.sample(self.cfg.minibatch_size)
        mb = self._gather_minibatch(entries)
        rho = self._current_rho()
        lam = self._resolve_lambda(mb, rho).detach()

        # Perturb a fraction of (x_t, λ) pairs
        if self.cfg.perturb_fraction > 0:
            M = mb["x_t"].shape[0]
            mask = torch.rand(M, device=self.device) < self.cfg.perturb_fraction
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]
                mb["x_t"][idx] += self.cfg.perturb_x_std * torch.randn_like(mb["x_t"][idx])
                lam[idx] = (lam[idx] * (1.0 + self.cfg.perturb_lambda_std * torch.randn_like(lam[idx]))).clamp_min(0.0)

        # MC score target (no gradient), chunked to avoid OOM
        with torch.no_grad():
            context = self._build_micro_context(mb["data"])
            M_full = mb["x_t"].shape[0]
            mc_chunk = max(1, self.cfg.mc_score_chunk)
            alphas_cumprod = self.mc_sampler.alphas_cumprod
            eps_parts = []
            from dataclasses import fields as _dc_fields, replace as _dc_replace
            ctx_fields = _dc_fields(context)
            for c0 in range(0, M_full, mc_chunk):
                c1 = min(c0 + mc_chunk, M_full)
                ctx_updates = {}
                for f in ctx_fields:
                    v = getattr(context, f.name)
                    if isinstance(v, torch.Tensor) and v.shape[0] == M_full:
                        ctx_updates[f.name] = v[c0:c1]
                ctx_chunk = _dc_replace(context, **ctx_updates) if ctx_updates else context
                mc_score_c = self.mc_sampler._estimate_score(
                    x_t=mb["x_t"][c0:c1], t=mb["t"][c0:c1],
                    context=ctx_chunk, dual_lambda=lam[c0:c1],
                )
                ab_c = alphas_cumprod[mb["t"][c0:c1]].view(-1, 1, 1, 1)
                sqrt_omb_c = torch.sqrt((1.0 - ab_c).clamp_min(1e-12))
                eps_c = -mc_score_c * sqrt_omb_c
                if self.cfg.target_clip_norm > 0:
                    flat_c = eps_c.reshape(c1 - c0, -1)
                    norms_c = flat_c.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    scale_c = (self.cfg.target_clip_norm / norms_c).clamp_max(1.0)
                    eps_c = (flat_c * scale_c).view_as(eps_c)
                eps_parts.append(eps_c)
            eps_target = torch.cat(eps_parts, dim=0)

        # Score-net forward
        eps_pred = self.score_net(
            x=mb["x_t"], timesteps=mb["t"], dual_lambda=lam,
            edge_index=mb["edge_index"], edge_weight=mb["edge_weight"],
            cond=mb["cond_full"], return_intermediates=False,
        )

        w = self._loss_weight(mb["t"]).view(-1, 1, 1, 1)
        if self.cfg.target_normalize:
            M = eps_target.shape[0]
            tgt_rms = eps_target.detach().reshape(M, -1).pow(2).mean(dim=1).clamp_min(
                float(self.cfg.target_normalize_eps) ** 2
            ).sqrt().view(M, 1, 1, 1)
            resid = (eps_pred - eps_target) / tgt_rms
        else:
            resid = eps_pred - eps_target
        loss = (resid ** 2 * w).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_net.parameters(), float(self.cfg.grad_clip_norm))
        self.optimizer.step()

        self._iteration += 1

        # Diagnostics
        with torch.no_grad():
            M = eps_target.shape[0]
            p = eps_pred.detach().reshape(M, -1)
            tgt = eps_target.detach().reshape(M, -1)
            p_n = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            t_n = tgt / tgt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos_vals = (p_n * t_n).sum(dim=1)
            rms_floor = float(self.cfg.target_normalize_eps) if self.cfg.target_normalize else 1e-12
            rms_ratio = p.pow(2).mean(dim=1).sqrt() / tgt.pow(2).mean(dim=1).sqrt().clamp_min(rms_floor)

        import statistics as _st
        return {
            "loss": float(loss.detach().item()),
            "rho": rho,
            "iter": self._iteration,
            "cos_mean": float(cos_vals.mean()),
            "cos_median": float(cos_vals.median()),
            "pred_rms_ratio_mean": float(rms_ratio.mean()),
            "t_mean": float(mb["t"].float().mean()),
        }

    # ------------------------------------------------------------------
    # Inverse-beta schedule across outer iters
    # ------------------------------------------------------------------

    def _apply_ib_schedule(self, outer: int, num_outer_iters: int) -> None:
        """Overwrite ``mc_sampler.inverse_beta_by_t`` with the scheduled ib.

        Interpolates from ``ib_schedule_start`` at ``outer=0`` to
        ``ib_schedule_end`` at ``outer=num_outer_iters-1``. Linear by default,
        log-linear when ``ib_schedule_shape="log"``.
        """
        if self.cfg.ib_schedule_warmup_outer_iters is not None:
            denom = max(1, int(self.cfg.ib_schedule_warmup_outer_iters))
        else:
            denom = max(1, num_outer_iters - 1)
        u = min(1.0, outer / denom)
        ib0 = float(self.cfg.ib_schedule_start)
        ib1 = float(self.cfg.ib_schedule_end)
        shape = str(self.cfg.ib_schedule_shape).strip().lower()
        if shape == "log":
            ib_cur = float(math.exp(math.log(ib0) + u * (math.log(ib1) - math.log(ib0))))
        else:
            ib_cur = ib0 + u * (ib1 - ib0)

        buf = getattr(self.mc_sampler, "inverse_beta_by_t", None)
        if buf is None:
            return
        new_buf = torch.full_like(buf, ib_cur).clamp_min(1e-12)
        self.mc_sampler.inverse_beta_by_t = new_buf
        self.mc_sampler.inverse_beta = ib_cur
        if hasattr(self.mc_sampler, "inverse_beta_start"):
            self.mc_sampler.inverse_beta_start = ib_cur
        if hasattr(self.mc_sampler, "inverse_beta_end"):
            self.mc_sampler.inverse_beta_end = ib_cur
        if outer % max(1, self.cfg.log_every_n_iters) == 0:
            log.info("ib_schedule: outer=%d ib=%.4g", outer, ib_cur)

    # ------------------------------------------------------------------
    # Top-level fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data_loader: DataLoader,
        num_outer_iters: int,
        *,
        log_fn: Optional[Any] = None,
        pre_rollout_fn: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Run the full outer/inner loop for ``num_outer_iters`` outer iterations.

        Each outer iteration consumes one batch from ``data_loader`` (cycled
        if exhausted), runs one rollout, then performs
        ``inner_steps_per_outer`` SGD steps.

        If ``pre_rollout_fn`` is provided, it is called with ``(outer_idx,)``
        before each rollout batch — use it to rotate problem instances.
        """
        self.score_net.train()
        history: List[Dict[str, float]] = []
        it = iter(data_loader)

        for outer in range(num_outer_iters):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            if pre_rollout_fn is not None:
                pre_rollout_fn(outer)
            if self.cfg.ib_schedule_enabled:
                self._apply_ib_schedule(outer, num_outer_iters)
            for _r in range(self.cfg.num_rollouts_per_outer):
                if self.cfg.mc_rollout_fraction > 0 and self._rng.random() < self.cfg.mc_rollout_fraction:
                    self.mc_rollout(batch)
                else:
                    self.rollout(batch)

            for _ in range(self.cfg.inner_steps_per_outer):
                step_info = self.train_step()
                if log_fn is not None:
                    log_fn(step_info)
                elif self._iteration % max(1, self.cfg.log_every_n_iters) == 0:
                    log.info(
                        "outer=%d iter=%d loss=%.4g rho=%.3f",
                        outer,
                        step_info["iter"],
                        step_info["loss"],
                        step_info["rho"],
                    )
                history.append(step_info)
        return history
