"""Energy-score training with outer-loop dual λ update (no λ conditioning).

Unlike EnergyScoreTrainer (which conditions the score net on λ), this trainer
maintains a persistent per-network λ state that evolves between training phases.
The model learns to predict the score at the *current* λ — it never sees λ as
input.

Training loop:
    For each outer iteration:
        1. Sample a training network
        2. Rollout: run reverse diffusion with current model → generate samples
        3. Dual update: λ[net] += dual_lr * (r_min - ergodic_rates(samples))
        4. Push trajectory to replay buffer
        5. Inner loop: train model against MC score targets at current λ[net]
"""

from __future__ import annotations

import logging
import random
import statistics as _st
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pdi.diffusion.energy_ddpm import EnergyDDPM
from pdi.trainers.energy_score.trajectory_buffer import (
    TrajectoryBuffer,
    TrajectoryEntry,
)

log = logging.getLogger(__name__)


@dataclass
class DualUpdateTrainConfig:
    """Hyperparameters for DualUpdateTrainer."""

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Loss
    loss_weighting: str = "one_minus_alpha_bar"
    target_clip_norm: float = 20.0
    target_normalize: bool = False
    target_normalize_eps: float = 0.1

    # MC target
    k_train: int = 256

    # Inner loop
    inner_steps_per_outer: int = 10
    minibatch_size: int = 4

    # Replay buffer
    buffer_capacity: int = 4096
    mc_score_chunk: int = 16

    # Dual update
    dual_lr: float = 0.1
    lambda_init: float = 10.0
    lambda_max: Optional[float] = None

    # Perturbation (applied during inner loop on buffer samples)
    perturb_fraction: float = 0.5
    perturb_x_std: float = 0.1

    # Rollout
    n_samples_per_network: int = 1

    # Logging
    log_every_n_iters: int = 1


class DualUpdateTrainer:
    """Trains a score net (no λ input) with outer-loop dual variable updates.

    The key difference from EnergyScoreTrainer: the model signature is
    (x, timesteps, edge_index, edge_weight, cond) with NO dual_lambda argument.
    λ is maintained externally per network and used only to compute MC score
    targets.
    """

    def __init__(
        self,
        score_net: nn.Module,
        mc_sampler: EnergyDDPM,
        config: DualUpdateTrainConfig,
        device: torch.device,
        *,
        rng_seed: int = 0,
    ):
        self.score_net = score_net.to(device)
        self.mc_sampler = mc_sampler.to(device)
        self.mc_sampler.eval()
        for p in self.mc_sampler.parameters():
            p.requires_grad_(False)

        self.cfg = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.score_net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self._k_train = int(self.cfg.k_train)
        self._k_rollout = int(self.mc_sampler.energy_mc_samples)
        self.mc_sampler.energy_mc_samples = self._k_train

        self.buffer = TrajectoryBuffer(
            capacity=self.cfg.buffer_capacity,
            rng=random.Random(rng_seed),
        )

        # Per-network λ state: {(dataset_name, network_id): Tensor[N]}
        self.lambda_state: Dict[Tuple[str, int], torch.Tensor] = {}

        self._rng = random.Random(rng_seed)
        self._iteration = 0
        self._recent_lambda_norms: List = []

    # ------------------------------------------------------------------
    # Lambda state management
    # ------------------------------------------------------------------

    def _get_net_key(self, data: Data) -> Tuple[str, int]:
        """Extract (dataset_name, network_id) from a PyG batch."""
        ds = getattr(data, "dataset_name", None)
        if isinstance(ds, (list, tuple)):
            ds = str(ds[0])
        else:
            ds = str(ds) if ds is not None else "unknown"
        nid = getattr(data, "network_id", None)
        if isinstance(nid, torch.Tensor):
            nid = int(nid[0])
        elif isinstance(nid, (list, tuple)):
            nid = int(nid[0])
        else:
            nid = 0
        return (ds, nid)

    def _get_lambda(self, net_key: Tuple[str, int], num_nodes: int) -> torch.Tensor:
        """Get or initialize λ for a network."""
        if net_key not in self.lambda_state:
            self.lambda_state[net_key] = torch.full(
                (num_nodes,), self.cfg.lambda_init, dtype=torch.float32
            )
        return self.lambda_state[net_key]

    def _update_lambda(
        self, net_key: Tuple[str, int], ergodic_rates: torch.Tensor, r_min: float
    ) -> torch.Tensor:
        """Dual ascent step on λ for a network."""
        num_nodes = ergodic_rates.shape[0]
        lam = self._get_lambda(net_key, num_nodes).to(ergodic_rates.device)
        violation = r_min - ergodic_rates  # [N], positive = infeasible
        lam = (lam + self.cfg.dual_lr * violation).clamp_min(0.0)
        if self.cfg.lambda_max is not None:
            lam = lam.clamp_max(self.cfg.lambda_max)
        self.lambda_state[net_key] = lam.detach().cpu()
        return lam

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        return int(self.mc_sampler.num_timesteps)

    def _extract_cond(self, data: Data, batch_size: int, num_nodes: int):
        if not hasattr(data, "x") or data.x is None:
            return None
        x = data.x
        return x.view(batch_size, num_nodes, *x.shape[1:]).swapaxes(1, 2).contiguous()

    def _loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_weighting == "unit":
            return torch.ones_like(t, dtype=torch.float32)
        if self.cfg.loss_weighting == "one_minus_alpha_bar":
            ab = self.mc_sampler.alphas_cumprod.to(t.device)[t]
            return (1.0 - ab).clamp_min(1e-8)
        raise ValueError(f"Unknown loss_weighting: {self.cfg.loss_weighting!r}")

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, data: Data) -> Dict[str, Any]:
        """Run reverse diffusion with current model and push trajectory to buffer.

        The batch contains n_nets unique networks, each tiled K times.
        All graphs are processed in a single forward pass per timestep.
        Does NOT update λ — that happens separately in the dual update phase.
        """
        from torch_geometric.data import Batch as PyGBatch

        data = data.to(self.device)
        batch_size = int(data.num_graphs)  # n_nets * K
        num_nodes = int(data.y.size(0) // batch_size)
        seq_len = int(data.y.size(1))
        feat_dim = int(data.y.size(2))
        K = max(1, self.cfg.n_samples_per_network)
        n_nets = batch_size // K

        # Extract net keys directly from batch attributes (avoid to_data_list)
        ds_names = getattr(data, "dataset_name", None)
        net_ids = getattr(data, "network_id", None)
        net_keys = []
        for i in range(n_nets):
            idx = i * K
            ds = str(ds_names[idx]) if isinstance(ds_names, (list, tuple)) else str(ds_names) if ds_names is not None else "unknown"
            nid = int(net_ids[idx]) if isinstance(net_ids, (list, tuple, torch.Tensor)) else 0
            net_keys.append((ds, nid))
        net_lambdas = [self._get_lambda(nk, num_nodes) for nk in net_keys]

        x_t = torch.randn(batch_size, seq_len, num_nodes, feat_dim,
                           device=self.device, dtype=torch.float32)
        cond = self._extract_cond(data, batch_size, num_nodes)
        # Pre-convert sparse edge_index to dense adj once (avoids per-timestep conversion)
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
        t_values: List[int] = []

        sqrt_alphas = self.mc_sampler.sqrt_alphas_cumprod
        sqrt_one_minus = self.mc_sampler.sqrt_one_minus_alphas_cumprod
        post_var = self.mc_sampler.posterior_variance
        clip = self.mc_sampler.clip_denoised

        for t_int in reversed(range(T)):
            x_by_t.append(x_t.detach().cpu())
            t_values.append(t_int)

            t_full = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)
            out = self.score_net(
                x=x_t, timesteps=t_full,
                edge_index=edge_index, edge_weight=edge_weight,
                cond=cond, return_intermediates=False,
            )
            eps_pred = out[0] if isinstance(out, tuple) else out

            sqrt_ab = sqrt_alphas[t_int]
            sqrt_omb = sqrt_one_minus[t_int]
            x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab.clamp_min(1e-12)
            if clip:
                x0_pred = x0_pred.clamp(-0.5, 0.5)

            mean = self.mc_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t_full)
            if t_int > 0:
                x_t = mean + torch.sqrt(post_var[t_int].clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        # Store current λ in buffer (same value for all timesteps — fixed during training phase)
        lam_by_t_list = []
        for i in range(n_nets):
            lam_by_t_list.append(net_lambdas[i].cpu().unsqueeze(0).expand(K, num_nodes))
        lam_flat = torch.cat(lam_by_t_list, dim=0)  # [batch_size, N]
        lam_expanded = lam_flat.unsqueeze(0).expand(T, batch_size, num_nodes)

        self.buffer.push_trajectory(
            data=data,
            x_by_t=torch.stack(x_by_t, dim=0),
            lambda_by_t=lam_expanded.contiguous(),
            t_values=torch.tensor(t_values, dtype=torch.long),
        )

        avg_lam = float(torch.stack([l.mean() for l in net_lambdas]).mean())
        return {"net_keys": net_keys, "lambda_mean": avg_lam}

    def pop_lambda_stats(self):
        """Return and clear λ stats from recent rollouts."""
        if not self._recent_lambda_norms:
            return None
        import numpy as np
        all_lam = np.stack(self._recent_lambda_norms)
        stats = {
            "mean": float(all_lam.mean()),
            "max": float(all_lam.max()),
        }
        self._recent_lambda_norms.clear()
        return stats

    # ------------------------------------------------------------------
    # Inner loop: one SGD step
    # ------------------------------------------------------------------

    def train_step(self) -> Dict[str, float]:
        """One SGD step on M individual (x_t, t, graph) samples."""
        if self.buffer.num_records == 0:
            raise RuntimeError("train_step called before any rollout.")

        entries = self.buffer.sample(self.cfg.minibatch_size)
        mb = self._gather_minibatch(entries)
        M = mb["x_t"].shape[0]
        num_nodes = mb["num_nodes"]

        # Build per-sample λ from current lambda_state
        lam_list = []
        for single_data in mb["data_list"]:
            nk = self._get_net_key(single_data)
            lam_list.append(self._get_lambda(nk, num_nodes).to(self.device))
        lam = torch.stack(lam_list)  # [M, N]

        # Perturb a fraction of x_t samples
        if self.cfg.perturb_fraction > 0:
            mask = torch.rand(M, device=self.device) < self.cfg.perturb_fraction
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]
                mb["x_t"][idx] += self.cfg.perturb_x_std * torch.randn_like(mb["x_t"][idx])

        # MC score target at current λ, chunked
        with torch.no_grad():
            context = self.mc_sampler._build_energy_context(
                data=mb["data"], batch_size=M, num_nodes=num_nodes,
                device=self.device, dtype=torch.float32,
            )
            mc_chunk = max(1, self.cfg.mc_score_chunk)
            alphas_cumprod = self.mc_sampler.alphas_cumprod
            eps_parts = []
            from dataclasses import fields as _dc_fields, replace as _dc_replace
            ctx_fields = _dc_fields(context)
            for c0 in range(0, M, mc_chunk):
                c1 = min(c0 + mc_chunk, M)
                ctx_updates = {}
                for f in ctx_fields:
                    v = getattr(context, f.name)
                    if isinstance(v, torch.Tensor) and v.shape[0] == M:
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

        # Forward pass (no λ input)
        out = self.score_net(
            x=mb["x_t"], timesteps=mb["t"],
            edge_index=mb["edge_index"],
            edge_weight=mb["edge_weight"],
            cond=mb["cond"], return_intermediates=False,
        )
        eps_pred = out[0] if isinstance(out, tuple) else out

        w = self._loss_weight(mb["t"]).view(-1, 1, 1, 1)
        if self.cfg.target_normalize:
            tgt_rms = eps_target.detach().reshape(M, -1).pow(2).mean(dim=1).clamp_min(
                self.cfg.target_normalize_eps ** 2
            ).sqrt().view(M, 1, 1, 1)
            resid = (eps_pred - eps_target) / tgt_rms
        else:
            resid = eps_pred - eps_target
        loss = (resid ** 2 * w).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.score_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        self._iteration += 1

        with torch.no_grad():
            p = eps_pred.detach().reshape(M, -1)
            tgt = eps_target.detach().reshape(M, -1)
            p_n = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            t_n = tgt / tgt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos_vals = (p_n * t_n).sum(dim=1)

        return {
            "loss": float(loss.detach().item()),
            "iter": self._iteration,
            "cos_mean": float(cos_vals.mean()),
            "t_mean": float(mb["t"].float().mean()),
        }

    # ------------------------------------------------------------------
    # Minibatch assembly
    # ------------------------------------------------------------------

    def _gather_minibatch(self, entries: List[TrajectoryEntry]) -> Dict[str, Any]:
        """Extract individual (sample, timestep) entries and re-batch them."""
        from torch_geometric.data import Batch as PyGBatch

        x_list = []
        t_list = []
        data_list = []

        # Cache to_data_list() per record to avoid repeated decomposition
        rec_graph_cache: Dict[int, list] = {}

        for e in entries:
            rec = e.record
            si = e.step_index
            bi = e.sample_index

            x_list.append(rec.x_by_t[si, bi])       # [T_s, N, F]
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
            "data_list": data_list,
            "x_t": torch.stack(x_list).to(self.device),
            "t": torch.tensor(t_list, device=self.device, dtype=torch.long),
            "cond": self._extract_cond(data, M, num_nodes),
            "edge_index": data.edge_index,
            "edge_weight": data.edge_weight if hasattr(data, "edge_weight") else None,
            "num_nodes": num_nodes,
            "batch_size": M,
        }
