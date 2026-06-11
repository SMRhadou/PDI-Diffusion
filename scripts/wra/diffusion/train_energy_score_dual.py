#!/usr/bin/env python3
"""Train energy-guided score network with outer-loop dual λ update.

Unlike train_energy_score.py (where the model is conditioned on λ), here
the model has NO λ input. Instead, a per-network λ is maintained externally
and updated via dual ascent between training phases based on constraint
violations of the generated samples.

Usage:
    micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score_dual.py \
        --dataset wra_medium_outdoor_high_density \
        --dual-lr 0.1 --lambda-init 10.0 \
        --label dual_update_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdi.datasets.wra.datamodule import WRABuilder
from pdi.diffusion.energy_ddpm import EnergyDDPM, _EnergyScoreMixin, _safe_torch_load
from pdi.models.portfolio_gnn_backbone import PortfolioGNNBackbone
from pdi.trainers.energy_score.trainer_dual import (
    DualUpdateTrainer,
    DualUpdateTrainConfig,
)
from _experiment_paths import experiment_dir
from wra_baselines import _evaluate_method, _load_ordered_channel_gains, _compute_full_ergodic_evolution

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _NoOpModel(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None,
                cond=None, return_intermediates=False):
        return torch.zeros_like(x), None


def _load_dataset_cfg(project_root, batch_size, n_samples_per_input,
                      dataset_config="wra_medium_outdoor_high_density"):
    base = OmegaConf.load(project_root / "src/pdi/conf/dataset/wra.yaml")
    overlay = OmegaConf.load(
        project_root / f"src/pdi/conf/dataset/{dataset_config}.yaml")
    cfg = OmegaConf.merge(base, overlay)
    cfg.root = str(project_root / "data" / "wra")
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.persistent_workers = False
    cfg.batch_size = int(batch_size)
    cfg.batch_size_val = 1
    cfg.n_samples_per_input = int(n_samples_per_input)
    cfg.model_cond_channels = 3
    return cfg


def _compose_diffusion_cfg(conf_dir, name="energy_ddpm_wra"):
    p = conf_dir / f"{name}.yaml"
    if p.exists():
        return OmegaConf.load(p)
    return OmegaConf.load(conf_dir / "energy_ddpm.yaml")


def _build_mc_sampler(cfg, project_root, device, *,
                      dual_step_size=0.1, dual_lambda_init=10.0,
                      inverse_beta=1.0, energy_mc_samples=16,
                      num_channel_realizations=10):
    ddpm_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(ddpm_cfg, dict):
        raise ValueError("Config must be dict-like")
    dataset_root_raw = ddpm_cfg.get("dataset_root", "")
    if dataset_root_raw and "${" not in str(dataset_root_raw):
        dataset_root = str(Path(dataset_root_raw).expanduser().resolve())
    else:
        dataset_root = str(project_root / "data" / "wra")
    R = num_channel_realizations or int(ddpm_cfg.get("energy_num_channel_realizations", 200))
    sampler = EnergyDDPM(
        model=_NoOpModel(),
        num_timesteps=int(ddpm_cfg.get("num_timesteps", 500)),
        beta_schedule=str(ddpm_cfg.get("beta_schedule", "cosine")),
        sigmoid_range=float(ddpm_cfg.get("sigmoid_range", 6.0)),
        clip_denoised=bool(ddpm_cfg.get("clip_denoised", True)),
        energy_mc_samples=energy_mc_samples,
        energy_num_channel_realizations=R,
        inverse_beta=inverse_beta,
        inverse_beta_schedule=str(ddpm_cfg.get("inverse_beta_schedule", "linear")),
        dual_update_mode=str(ddpm_cfg.get("dual_update_mode", "x0_pred")),
        dual_step_size=dual_step_size,
        dual_step_size_schedule=str(ddpm_cfg.get("dual_step_size_schedule", "constant")),
        dual_num_outer_iterations=int(ddpm_cfg.get("dual_num_outer_iterations", 15)),
        dual_lambda_init=dual_lambda_init,
        dual_lambda_max=None,
        dual_lambda_mode="shared_per_network",
        langevin_step_size=float(ddpm_cfg.get("langevin_step_size", 1e-4)),
        langevin_noise_scale=float(ddpm_cfg.get("langevin_noise_scale", 1.0)),
        dataset_root=dataset_root,
        min_energy_sigma=float(ddpm_cfg.get("min_energy_sigma", 1e-4)),
        use_precomputed_channels=bool(ddpm_cfg.get("use_precomputed_channels", True)),
        allow_sparse_graph_fallback=False,
    ).to(device)
    sampler.eval()
    return sampler


def _build_backbone(num_nodes, device, model_cond_channels=3,
                    hidden=128, num_layers=6, K=2, num_timesteps=500):
    """Build PortfolioGNNBackbone (no λ conditioning — uses model_cond_channels directly)."""
    backbone = PortfolioGNNBackbone(
        d=num_nodes, hidden=hidden, num_layers=num_layers,
        num_timesteps=num_timesteps, cond_channels=model_cond_channels, K=K,
    )
    print(f"Portfolio backbone (no λ cond): {sum(p.numel() for p in backbone.parameters()):,} params")
    return backbone.to(device)


def _update_all_lambdas(trainer, train_loader, mc_sampler, device,
                        n_samples_per_network=10, networks_per_chunk=32):
    """Run inference on all training networks and update λ.

    Networks are processed in large chunks (``networks_per_chunk`` at a time)
    with all K tiled copies batched into a single score-net forward pass per
    timestep.  This is ~30x faster than the per-network loop because GPU
    utilisation jumps from ~3 GB to ~12-15 GB.
    """
    from torch_geometric.data import Batch as PyGBatch

    trainer.score_net.eval()
    K = n_samples_per_network

    all_graphs = []
    for batch in train_loader:
        all_graphs.extend(batch.to_data_list())
    n_total = len(all_graphs)
    log.info(f"    λ-update: {n_total} networks, K={K}, chunk={networks_per_chunk}")

    all_feas = []
    all_lam_means = []
    all_rates = []
    all_r_min = 0.0

    with torch.no_grad():
        for c0 in range(0, n_total, networks_per_chunk):
            c1 = min(c0 + networks_per_chunk, n_total)
            chunk_graphs = all_graphs[c0:c1]
            n_nets = len(chunk_graphs)

            log.info(f"    λ-update {c0+1}-{c1}/{n_total} "
                     f"({n_nets} nets, {n_nets * K} graphs)")

            tiled_list = [g.clone() for g in chunk_graphs for _ in range(K)]
            tiled = PyGBatch.from_data_list(
                tiled_list, follow_batch=["y"], exclude_keys=["info"],
            ).to(device)

            B = n_nets * K
            num_nodes = int(tiled.y.size(0) // B)
            seq_len = int(tiled.y.size(1) if tiled.y.dim() > 1 else 1)
            feat_dim = int(tiled.y.size(2) if tiled.y.dim() > 2 else tiled.y.size(1))

            x_t = torch.randn(B, seq_len, num_nodes, feat_dim,
                               device=device, dtype=torch.float32)
            cond = trainer._extract_cond(tiled, B, num_nodes)
            edge_index = tiled.edge_index
            edge_weight = getattr(tiled, "edge_weight", None)
            if edge_index.dim() == 2:
                edge_index = _sparse_to_dense_adj(edge_index, edge_weight, num_nodes, B)
                edge_weight = None
            sqrt_alphas = mc_sampler.sqrt_alphas_cumprod
            sqrt_one_minus = mc_sampler.sqrt_one_minus_alphas_cumprod
            post_var = mc_sampler.posterior_variance

            for t_int in reversed(range(trainer.num_timesteps)):
                t_full = torch.full((B,), t_int, device=device, dtype=torch.long)
                out = trainer.score_net(
                    x=x_t, timesteps=t_full,
                    edge_index=edge_index, edge_weight=edge_weight,
                    cond=cond, return_intermediates=False,
                )
                eps_pred = out[0] if isinstance(out, tuple) else out

                sqrt_ab = sqrt_alphas[t_int]
                sqrt_omb = sqrt_one_minus[t_int]
                x0_pred = ((x_t - sqrt_omb * eps_pred) / sqrt_ab.clamp_min(1e-12)).clamp(-0.5, 0.5)
                mean = mc_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t_full)
                if t_int > 0:
                    x_t = mean + torch.sqrt(post_var[t_int].clamp_min(0.0)) * torch.randn_like(x_t)
                else:
                    x_t = mean

            for i in range(n_nets):
                s, e = i * K, (i + 1) * K
                nb = PyGBatch.from_data_list(
                    tiled_list[i * K : (i + 1) * K],
                    follow_batch=["y"], exclude_keys=["info"],
                ).to(device)
                net_key = trainer._get_net_key(nb)

                dual_ctx = mc_sampler._build_dual_context(
                    data=nb, batch_size=K, num_nodes=num_nodes,
                    device=device, dtype=torch.float32,
                )
                if dual_ctx is None:
                    dual_ctx = mc_sampler._build_energy_context(
                        data=nb, batch_size=K, num_nodes=num_nodes,
                        device=device, dtype=torch.float32,
                    )
                ergodic_rates, _ = mc_sampler._ergodic_rates_from_samples(
                    x=x_t[s:e], context=dual_ctx)
                avg_rates = ergodic_rates.mean(dim=0)
                r_min = float(dual_ctx.r_min[0])
                new_lam = trainer._update_lambda(net_key, avg_rates, r_min)
                all_feas.append(float((avg_rates >= r_min).float().mean()))
                all_lam_means.append(float(new_lam.mean()))
                all_rates.append(avg_rates.detach().cpu().numpy())
                all_r_min = r_min

    trainer.score_net.train()

    all_lam_tensors = list(trainer.lambda_state.values())
    flat = np.concatenate(all_rates) if all_rates else np.array([])
    return {
        "lambda_mean": float(np.mean(all_lam_means)) if all_lam_means else 0.0,
        "lambda_max": float(max(v.max().item() for v in all_lam_tensors)) if all_lam_tensors else 0.0,
        "avg_feasibility": float(np.mean(all_feas)) if all_feas else 0.0,
        "num_networks_updated": len(all_feas),
        "mean_rate": float(flat.mean()) if len(flat) else 0.0,
        "p5_rate": float(np.percentile(flat, 5)) if len(flat) else 0.0,
        "p1_rate": float(np.percentile(flat, 1)) if len(flat) else 0.0,
    }


def _update_batch_lambdas(trainer, batch, mc_sampler, device,
                          n_samples_per_network=10):
    """Run inference on the current batch's networks and update their λ.

    Unlike _update_all_lambdas which sweeps every training network, this only
    processes the networks in the current batch — much cheaper per outer
    iteration.
    """
    from torch_geometric.data import Batch as PyGBatch

    trainer.score_net.eval()
    K = n_samples_per_network

    all_graphs = batch.to_data_list()
    n_nets = len(all_graphs)

    tiled_list = [g.clone() for g in all_graphs for _ in range(K)]
    tiled = PyGBatch.from_data_list(
        tiled_list, follow_batch=["y"], exclude_keys=["info"],
    ).to(device)

    B = n_nets * K
    num_nodes = int(tiled.y.size(0) // B)
    seq_len = int(tiled.y.size(1) if tiled.y.dim() > 1 else 1)
    feat_dim = int(tiled.y.size(2) if tiled.y.dim() > 2 else tiled.y.size(1))

    x_t = torch.randn(B, seq_len, num_nodes, feat_dim,
                       device=device, dtype=torch.float32)
    cond = trainer._extract_cond(tiled, B, num_nodes)
    edge_index = tiled.edge_index
    edge_weight = getattr(tiled, "edge_weight", None)
    if edge_index.dim() == 2:
        edge_index = _sparse_to_dense_adj(edge_index, edge_weight, num_nodes, B)
        edge_weight = None

    all_feas = []
    all_lam_means = []
    all_rates = []
    all_r_min = 0.0

    sqrt_alphas = mc_sampler.sqrt_alphas_cumprod
    sqrt_one_minus = mc_sampler.sqrt_one_minus_alphas_cumprod
    post_var = mc_sampler.posterior_variance

    with torch.no_grad():
        for t_int in reversed(range(trainer.num_timesteps)):
            t_full = torch.full((B,), t_int, device=device, dtype=torch.long)
            out = trainer.score_net(
                x=x_t, timesteps=t_full,
                edge_index=edge_index, edge_weight=edge_weight,
                cond=cond, return_intermediates=False,
            )
            eps_pred = out[0] if isinstance(out, tuple) else out

            sqrt_ab = sqrt_alphas[t_int]
            sqrt_omb = sqrt_one_minus[t_int]
            x0_pred = ((x_t - sqrt_omb * eps_pred) / sqrt_ab.clamp_min(1e-12)).clamp(-0.5, 0.5)
            mean = mc_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t_full)
            if t_int > 0:
                x_t = mean + torch.sqrt(post_var[t_int].clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        for i in range(n_nets):
            s, e = i * K, (i + 1) * K
            nb = PyGBatch.from_data_list(
                tiled_list[i * K : (i + 1) * K],
                follow_batch=["y"], exclude_keys=["info"],
            ).to(device)
            net_key = trainer._get_net_key(nb)

            dual_ctx = mc_sampler._build_dual_context(
                data=nb, batch_size=K, num_nodes=num_nodes,
                device=device, dtype=torch.float32,
            )
            if dual_ctx is None:
                dual_ctx = mc_sampler._build_energy_context(
                    data=nb, batch_size=K, num_nodes=num_nodes,
                    device=device, dtype=torch.float32,
                )
            ergodic_rates, _ = mc_sampler._ergodic_rates_from_samples(
                x=x_t[s:e], context=dual_ctx)
            avg_rates = ergodic_rates.mean(dim=0)
            r_min = float(dual_ctx.r_min[0])
            new_lam = trainer._update_lambda(net_key, avg_rates, r_min)
            all_feas.append(float((avg_rates >= r_min).float().mean()))
            all_lam_means.append(float(new_lam.mean()))
            all_rates.append(avg_rates.detach().cpu().numpy())
            all_r_min = r_min

    trainer.score_net.train()

    all_lam_tensors = list(trainer.lambda_state.values())
    flat = np.concatenate(all_rates) if all_rates else np.array([])
    return {
        "lambda_mean": float(np.mean(all_lam_means)) if all_lam_means else 0.0,
        "lambda_max": float(max(v.max().item() for v in all_lam_tensors)) if all_lam_tensors else 0.0,
        "avg_feasibility": float(np.mean(all_feas)) if all_feas else 0.0,
        "num_networks_updated": len(all_feas),
        "mean_rate": float(flat.mean()) if len(flat) else 0.0,
        "p5_rate": float(np.percentile(flat, 5)) if len(flat) else 0.0,
        "p1_rate": float(np.percentile(flat, 1)) if len(flat) else 0.0,
    }


def _tile_batch(batch, K):
    """Repeat each graph in a PyG batch K times for independent trajectories."""
    from torch_geometric.data import Batch
    data_list = batch.to_data_list()
    tiled = [d.clone() for d in data_list for _ in range(K)]
    return Batch.from_data_list(tiled, follow_batch=["y"], exclude_keys=["info"])


def _sparse_to_dense_adj(edge_index, edge_weight, num_nodes, batch_size):
    """Convert PyG sparse edge_index to dense [B, N, N] adjacency."""
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
    return A


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(score_net, mc_sampler, test_loader, device,
                     n_samples_per_input, max_batches=4, eval_timeslots=200,
                     num_trials=5, dual_lambda=None, lambda_state=None):
    """Run inference with trained model on test set.

    For test networks not seen during training, we need λ. Options:
    - If lambda_state is provided and the test network was seen, use stored λ
    - Otherwise use dual_lambda as a fixed value (scalar broadcast to [N])
    """
    import types as _types

    score_net.eval()
    alphas_cumprod = mc_sampler.alphas_cumprod.to(device)

    def _trained_estimate_score(self, x_t, t, context, dual_lambda=None,
                                inverse_beta_override=None):
        B = x_t.shape[0]
        N = x_t.shape[2]
        data = self._current_eval_data
        cond = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
        edge_index = data.edge_index if hasattr(data, "edge_index") else None
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
        with torch.no_grad():
            out = score_net(
                x=x_t, timesteps=t,
                edge_index=edge_index, edge_weight=edge_weight,
                cond=cond, return_intermediates=False,
            )
            eps_pred = out[0] if isinstance(out, tuple) else out
        ab = alphas_cumprod[t].view(-1, 1, 1, 1)
        return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))

    orig_estimate = mc_sampler._estimate_score
    mc_sampler._estimate_score = _types.MethodType(_trained_estimate_score, mc_sampler)

    all_rates = []
    all_metrics = []
    r_min = None

    try:
        for bi, batch in enumerate(test_loader):
            if bi >= max_batches:
                break
            batch = batch.to(device)
            B = int(batch.num_graphs)
            N = int(batch.y.size(0) // B)
            shape = (B, int(batch.y.size(1)), N, int(batch.y.size(2)))
            K = n_samples_per_input

            mc_sampler._current_eval_data = batch

            context = mc_sampler._build_energy_context(
                data=batch, batch_size=B, num_nodes=N,
                device=device, dtype=torch.float32)
            if r_min is None:
                r_min = float(context.r_min[0])

            ds_name = str(getattr(batch, "dataset_name", ["unknown"])[0]) if isinstance(getattr(batch, "dataset_name", None), (list, tuple)) else "unknown"
            nid = int(getattr(batch, "network_id", [bi])[0]) if isinstance(getattr(batch, "network_id", None), (list, tuple, torch.Tensor)) else bi

            # Set dual_lambda for inference
            net_key = (ds_name, nid)
            if lambda_state is not None and net_key in lambda_state:
                lam_vec = lambda_state[net_key].to(device)
            elif dual_lambda is not None:
                lam_vec = torch.full((N,), dual_lambda, device=device, dtype=torch.float32)
            else:
                lam_vec = torch.full((N,), 10.0, device=device, dtype=torch.float32)

            # Override sampler's dual init and disable dual updates for net-based inference
            orig_init = mc_sampler._init_dual_lambda
            orig_dual_step = mc_sampler._dual_ascent_step
            mc_sampler._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=lam_vec: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
            mc_sampler._dual_ascent_step = lambda dual_lambda, ergodic_rates, context, t=None: dual_lambda

            torch.manual_seed(42 + bi)
            result = mc_sampler.sample(shape=shape, device=device, data=batch,
                                       n_samples_per_input=K)
            x = result[0] if isinstance(result, tuple) else result

            mc_sampler._init_dual_lambda = orig_init
            mc_sampler._dual_ascent_step = orig_dual_step

            m = _evaluate_method(x, context, mc_sampler, ds_name, nid, N,
                                 K, device, num_trials,
                                 max_timeslots=eval_timeslots)
            all_rates.append(m["per_user_rates"])
            all_metrics.append(m)
    finally:
        mc_sampler._estimate_score = orig_estimate
        if hasattr(mc_sampler, "_current_eval_data"):
            del mc_sampler._current_eval_data

    score_net.train()

    if not all_rates:
        return {}

    flat = np.concatenate(all_rates)
    mean_vio = float(np.maximum(r_min - flat, 0).mean())
    return {
        "mean_rate": float(np.mean(flat)),
        "p5_rate": float(np.percentile(flat, 5)),
        "p1_rate": float(np.percentile(flat, 1)),
        "mean_vio": mean_vio,
        "feasibility": float((flat >= r_min).mean()),
        "num_users": len(flat),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train energy score net with dual λ update (no λ conditioning)")
    parser.add_argument("--label", type=str, default="dual_update")
    parser.add_argument("--dataset", type=str, default="wra_medium_outdoor_high_density")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Backbone
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--gnn-K", type=int, default=2)

    # MC sampler config
    parser.add_argument("--inverse-beta", type=float, default=1.0)
    parser.add_argument("--energy-mc-samples", type=int, default=16)
    parser.add_argument("--num-channel-realizations", type=int, default=10)

    # Training structure
    parser.add_argument("--num-outer", type=int, default=400,
                        help="Number of outer iterations")
    parser.add_argument("--inner-steps", type=int, default=10,
                        help="SGD steps per outer iteration")
    parser.add_argument("--num-rollouts-per-outer", type=int, default=1,
                        help="Rollouts per outer iteration (each draws a fresh batch)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of unique networks per rollout batch")
    parser.add_argument("--n-samples-per-network", type=int, default=10,
                        help="Independent trajectories per network per rollout")
    parser.add_argument("--train-mc-samples", type=int, default=256)
    parser.add_argument("--mc-score-chunk", type=int, default=16,
                        help="Graphs per chunk in MC score target computation (higher = faster, more VRAM)")
    parser.add_argument("--target-clip-norm", type=float, default=20.0)
    parser.add_argument("--buffer-capacity", type=int, default=4096)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Perturbation
    parser.add_argument("--perturb-fraction", type=float, default=0.5)
    parser.add_argument("--perturb-x-std", type=float, default=0.1)

    # Dual update
    parser.add_argument("--dual-lr", type=float, default=0.1)
    parser.add_argument("--lambda-init", type=float, default=10.0)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--lambda-batch-size", type=int, default=None,
                        help="Networks per lambda update rollout (default: same as --batch-size)")
    parser.add_argument("--lambda-update-mode", type=str, default="per_batch",
                        choices=["per_batch", "sweep"],
                        help="Lambda update mode: per_batch (rollout current batch only, default) "
                             "or sweep (rollout all training networks)")
    parser.add_argument("--lambda-update-chunk", type=int, default=32,
                        help="Networks per chunk during λ sweep update (higher = faster, more VRAM)")
    parser.add_argument("--freeze-lambda", action="store_true",
                        help="Freeze λ values (skip dual updates entirely)")

    # Eval
    parser.add_argument("--eval-every", type=int, default=20,
                        help="Evaluate every N outer iterations")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--eval-networks", type=int, default=4)
    parser.add_argument("--eval-timeslots", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-lr", action="store_true",
                        help="Start LR schedule fresh (ignore resumed sgd_count)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    project_root = _project_root()

    output_dir = experiment_dir("score_net_train", args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    print(f"Output: {output_dir}")
    print(f"Args: {vars(args)}")

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------------------------------------------------------
    # Build MC sampler
    # ---------------------------------------------------------------
    conf_dir = project_root / "src/pdi/conf/diffusion"
    ddpm_cfg = _compose_diffusion_cfg(conf_dir, "energy_ddpm_wra")

    mc_sampler = _build_mc_sampler(
        ddpm_cfg, project_root, device,
        dual_step_size=args.dual_lr,
        dual_lambda_init=args.lambda_init,
        inverse_beta=args.inverse_beta,
        energy_mc_samples=args.energy_mc_samples,
        num_channel_realizations=args.num_channel_realizations,
    )

    # ---------------------------------------------------------------
    # Build backbone (no λ conditioning)
    # ---------------------------------------------------------------
    model_cond_channels = 3

    # Peek at dataset to get num_nodes
    ds_cfg_peek = _load_dataset_cfg(project_root, 1, 1, dataset_config=args.dataset)
    builder_peek = WRABuilder()
    ds_peek = builder_peek.build_datasets(ds_cfg_peek)
    sample = ds_peek["train"][0]
    num_nodes = sample.num_nodes
    num_timesteps = mc_sampler.num_timesteps
    print(f"num_nodes={num_nodes}, num_timesteps={num_timesteps}")
    del ds_peek, builder_peek

    score_net = _build_backbone(
        num_nodes, device, model_cond_channels=model_cond_channels,
        hidden=args.hidden, num_layers=args.num_layers, K=args.gnn_K,
        num_timesteps=num_timesteps,
    )

    # Wrap forward to convert sparse → dense adjacency
    import types as _types
    _orig_forward = score_net.forward

    def _dense_adj_forward(self, x, timesteps, edge_index,
                           edge_weight=None, cond=None, return_intermediates=False):
        B, _, N, _ = x.shape
        if edge_index.dim() == 2:  # sparse [2, E] → dense [B, N, N]
            edge_index = _sparse_to_dense_adj(edge_index, edge_weight, N, B)
            edge_weight = None
        return _orig_forward(x=x, timesteps=timesteps,
                             edge_index=edge_index, edge_weight=edge_weight,
                             cond=cond, return_intermediates=return_intermediates)

    score_net.forward = _types.MethodType(_dense_adj_forward, score_net)

    resume_outer = 0
    resume_sgd = 0
    resume_lambda_state = None
    resume_optim_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        score_net.load_state_dict(ckpt["state_dict"])
        resume_outer = ckpt.get("outer", 0) + 1
        resume_sgd = ckpt.get("sgd_count", 0)
        if "lambda_state" in ckpt:
            raw_ls = ckpt["lambda_state"]
            resume_lambda_state = {}
            for k, v in raw_ls.items():
                if isinstance(k, str):
                    parts = k.rsplit("_", 1)
                    resume_lambda_state[(parts[0], int(parts[1]))] = v
                else:
                    resume_lambda_state[k] = v
        if "optimizer_state" in ckpt:
            resume_optim_state = ckpt["optimizer_state"]
        print(f"Resumed from {args.resume} (outer={resume_outer}, sgd={resume_sgd})")

    # ---------------------------------------------------------------
    # Build datasets (one item per unique network, sample_id=0)
    # ---------------------------------------------------------------
    ds_cfg = _load_dataset_cfg(project_root, args.batch_size, 1,
                                dataset_config=args.dataset)
    builder = WRABuilder()
    datasets = builder.build_datasets(ds_cfg)

    train_ds = datasets["train"]
    seen_nets = set()
    unique_indices = []
    for i, (ds_name, net_id, sample_id) in enumerate(train_ds.samples):
        key = (ds_name, net_id)
        if key not in seen_nets:
            seen_nets.add(key)
            unique_indices.append(i)
    train_ds.samples = [train_ds.samples[i] for i in unique_indices]
    print(f"Training on {len(train_ds.samples)} unique networks")

    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, follow_batch=["y"], exclude_keys=["info"],
    )

    eval_cfg = _load_dataset_cfg(project_root, 1, args.eval_samples,
                                  dataset_config=args.dataset)
    eval_datasets = builder.build_datasets(eval_cfg)
    eval_loaders = builder.build_loaders(eval_cfg, eval_datasets)
    test_loader = eval_loaders["val"]

    # ---------------------------------------------------------------
    # Build trainer
    # ---------------------------------------------------------------
    train_config = DualUpdateTrainConfig(
        lr=args.lr,
        weight_decay=1e-4,
        k_train=args.train_mc_samples,
        inner_steps_per_outer=args.inner_steps,
        minibatch_size=128,
        mc_score_chunk=args.mc_score_chunk,
        buffer_capacity=args.buffer_capacity,
        target_clip_norm=args.target_clip_norm,
        grad_clip_norm=args.grad_clip_norm,
        dual_lr=args.dual_lr,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
        perturb_fraction=args.perturb_fraction,
        perturb_x_std=args.perturb_x_std,
        n_samples_per_network=args.n_samples_per_network,
    )

    trainer = DualUpdateTrainer(
        score_net=score_net,
        mc_sampler=mc_sampler,
        config=train_config,
        device=device,
        rng_seed=args.seed,
    )

    # ---------------------------------------------------------------
    # Training loop — same structure as standard trainer.
    # Each outer iteration:
    #   1. Draw a batch of unique networks, tile K samples, rollout
    #   2. Inner SGD steps against MC targets at each network's current λ
    #   3. Update λ for the networks in this batch
    # ---------------------------------------------------------------
    print(f"\nStarting dual-update training:")
    print(f"  {args.num_outer} outer iters, {args.inner_steps} SGD steps each, "
          f"{args.num_rollouts_per_outer} rollouts/iter")
    print(f"  Dual: lr={args.dual_lr}, λ_init={args.lambda_init}, λ_max={args.lambda_max}")
    print(f"  Eval every {args.eval_every} outer iters on {args.eval_networks} val networks")
    print("=" * 70)

    # Restore lambda_state and optimizer from checkpoint
    if resume_lambda_state is not None:
        trainer.lambda_state = resume_lambda_state
        print(f"  Restored lambda_state: {len(resume_lambda_state)} networks")
    if resume_optim_state is not None:
        trainer.optimizer.load_state_dict(resume_optim_state)
        if args.reset_lr:
            for pg in trainer.optimizer.param_groups:
                pg['lr'] = args.lr
            print(f"  Restored optimizer state (lr reset to {args.lr})")
        else:
            print(f"  Restored optimizer state")

    best_metric = -float("inf")
    best_outer = -1
    best_obj = -float("inf")
    best_obj_outer = -1
    best_composite = -float("inf")
    best_composite_outer = -1
    train_history_path = output_dir / "train_history.jsonl"
    eval_history = []
    sgd_count = resume_sgd
    train_it = iter(train_loader)
    K = args.n_samples_per_network

    # Separate loader for lambda updates if lambda_batch_size differs
    lambda_batch_size = args.lambda_batch_size or args.batch_size
    if lambda_batch_size != args.batch_size:
        lambda_loader = PyGDataLoader(
            train_ds, batch_size=lambda_batch_size, shuffle=True,
            num_workers=0, follow_batch=["y"], exclude_keys=["info"],
        )
        lambda_it = iter(lambda_loader)
    else:
        lambda_loader = train_loader
        lambda_it = None

    total_sgd = args.num_outer * args.inner_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=total_sgd, eta_min=args.lr_min)
    # Fast-forward scheduler to resume point
    if resume_sgd > 0 and not args.reset_lr:
        scheduler.last_epoch = resume_sgd

    for outer in range(resume_outer, args.num_outer):
        t0 = time.time()

        # Get training batch
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch = next(train_it)

        # Per-batch mode: update λ first, then rollout with fresh λ
        t_lam = time.time()
        if args.lambda_update_mode == "per_batch" and outer > 0 and not args.freeze_lambda:
            if lambda_it is not None:
                try:
                    lam_batch = next(lambda_it)
                except StopIteration:
                    lambda_it = iter(lambda_loader)
                    lam_batch = next(lambda_it)
            else:
                lam_batch = batch
            dual_info = _update_batch_lambdas(
                trainer, lam_batch.to(device), mc_sampler, device,
                n_samples_per_network=args.n_samples_per_network)
            log.info(f"  λ-update: Mean={dual_info['mean_rate']:.4f} "
                     f"λ_mean={dual_info['lambda_mean']:.2f} "
                     f"feas={dual_info['avg_feasibility']:.0%}")
        dt_lam = time.time() - t_lam

        # Rollout(s) with up-to-date λ
        t_roll = time.time()
        for _r in range(args.num_rollouts_per_outer):
            if _r > 0:
                try:
                    batch = next(train_it)
                except StopIteration:
                    train_it = iter(train_loader)
                    batch = next(train_it)
            tiled = _tile_batch(batch, K).to(device) if K > 1 else batch.to(device)
            rollout_info = trainer.rollout(tiled)
        dt_roll = time.time() - t_roll

        # Inner SGD
        t_sgd = time.time()
        for _inner in range(args.inner_steps):
            step_info = trainer.train_step()
            with open(train_history_path, "a") as _fh:
                _fh.write(json.dumps(step_info) + "\n")
            scheduler.step()
            sgd_count += 1
            log.info(f"  [inner {_inner}/{args.inner_steps}] "
                     f"loss={step_info['loss']:.4f} cos={step_info['cos_mean']:.3f}")
        dt_sgd = time.time() - t_sgd

        # Sweep mode: update all λ after inner SGD (original behavior)
        if args.lambda_update_mode == "sweep" and not args.freeze_lambda:
            t_lam = time.time()
            log.info(f"  Updating λ for all networks...")
            dual_info = _update_all_lambdas(
                trainer, train_loader, mc_sampler, device,
                n_samples_per_network=args.n_samples_per_network,
                networks_per_chunk=args.lambda_update_chunk)
            dt_lam = time.time() - t_lam

        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        timing_str = f"lam={dt_lam:.1f}s roll={dt_roll:.1f}s sgd={dt_sgd:.1f}s total={dt:.1f}s"
        if args.freeze_lambda or (args.lambda_update_mode == "per_batch" and outer == 0):
            log.info(f"[outer {outer:4d}] iter={sgd_count:5d} loss={step_info['loss']:.4f} "
                     f"cos={step_info['cos_mean']:.3f} lr={lr_now:.1e} ({timing_str})")
        else:
            log.info(f"[outer {outer:4d}] iter={sgd_count:5d} loss={step_info['loss']:.4f} "
                     f"cos={step_info['cos_mean']:.3f} lr={lr_now:.1e} "
                     f"Mean={dual_info['mean_rate']:.4f} p5={dual_info['p5_rate']:.4f} "
                     f"p1={dual_info['p1_rate']:.4f} "
                     f"λ_mean={dual_info['lambda_mean']:.2f} λ_max={dual_info['lambda_max']:.1f} "
                     f"feas={dual_info['avg_feasibility']:.0%} ({timing_str})")

        # Periodic checkpoint
        if outer % 50 == 0 or outer == args.num_outer - 1:
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                "state_dict": score_net.state_dict(),
                "lambda_state": {
                    f"{k[0]}_{k[1]}": v for k, v in trainer.lambda_state.items()
                },
                "optimizer_state": trainer.optimizer.state_dict(),
                "outer": outer,
                "sgd_count": sgd_count,
                "args": vars(args),
            }, ckpt_dir / f"outer_{outer:05d}.pt")
            log.info(f"  Saved checkpoint: outer_{outer:05d}.pt")

        # Eval
        if (outer > resume_outer and outer >= 400 and outer % args.eval_every == 0) or outer == args.num_outer - 1:
            torch.cuda.empty_cache()
            log.info(f"\n--- Eval at outer={outer} ---")
            metrics = evaluate_on_test(
                score_net, mc_sampler, test_loader, device,
                n_samples_per_input=args.eval_samples,
                max_batches=args.eval_networks,
                eval_timeslots=args.eval_timeslots,
                lambda_state=trainer.lambda_state,
            )
            if metrics:
                metrics["outer"] = outer
                metrics["sgd_count"] = sgd_count
                eval_history.append(metrics)
                log.info(f"  Mean={metrics['mean_rate']:.4f}  p5={metrics['p5_rate']:.4f}  "
                         f"p1={metrics['p1_rate']:.4f}  Vio={metrics['mean_vio']:.4f}  "
                         f"Feas={metrics['feasibility']:.1%}")

                with open(output_dir / "eval_history.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                score = metrics["p5_rate"] if metrics["feasibility"] > 0.8 else -1.0
                if score > best_metric:
                    best_metric = score
                    best_outer = outer
                    torch.save({
                        "state_dict": score_net.state_dict(),
                        "lambda_state": {
                            f"{k[0]}_{k[1]}": v for k, v in trainer.lambda_state.items()
                        },
                        "outer": outer,
                        "sgd_count": sgd_count,
                        "metrics": metrics,
                        "args": vars(args),
                    }, output_dir / "best_model.pt")
                    log.info(f"  ** New best p5={metrics['p5_rate']:.4f}")

                # Best mean rate (objective)
                if metrics["mean_rate"] > best_obj:
                    best_obj = metrics["mean_rate"]
                    best_obj_outer = outer
                    torch.save({
                        "state_dict": score_net.state_dict(),
                        "lambda_state": {
                            f"{k[0]}_{k[1]}": v for k, v in trainer.lambda_state.items()
                        },
                        "outer": outer,
                        "sgd_count": sgd_count,
                        "metrics": metrics,
                        "args": vars(args),
                    }, output_dir / "best_model_obj.pt")
                    log.info(f"  ** New best obj (mean={metrics['mean_rate']:.4f})")

                # Best composite: mean + 5*p5
                composite = metrics["mean_rate"] + 5 * metrics["p5_rate"]
                if composite > best_composite:
                    best_composite = composite
                    best_composite_outer = outer
                    torch.save({
                        "state_dict": score_net.state_dict(),
                        "lambda_state": {
                            f"{k[0]}_{k[1]}": v for k, v in trainer.lambda_state.items()
                        },
                        "outer": outer,
                        "sgd_count": sgd_count,
                        "metrics": metrics,
                        "args": vars(args),
                    }, output_dir / "best_model_composite.pt")
                    log.info(f"  ** New best composite (mean={metrics['mean_rate']:.4f}, p5={metrics['p5_rate']:.4f}, score={composite:.4f})")

            torch.cuda.empty_cache()

            # Save latest
            torch.save({
                "state_dict": score_net.state_dict(),
                "lambda_state": {
                    f"{k[0]}_{k[1]}": v for k, v in trainer.lambda_state.items()
                },
                "outer": outer,
                "sgd_count": sgd_count,
                "metrics": metrics,
                "args": vars(args),
            }, output_dir / "latest_model.pt")
            log.info("")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    log.info("=" * 70)
    log.info(f"Training complete. Best p5 at outer={best_outer} (p5={best_metric:.4f})")
    log.info(f"                  Best obj at outer={best_obj_outer} (mean={best_obj:.4f})")
    log.info(f"                  Best composite at outer={best_composite_outer} (score={best_composite:.4f})")
    print(f"Output: {output_dir}")

    with open(output_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f)

    if trainer.lambda_state:
        all_lam = torch.stack(list(trainer.lambda_state.values()))
        log.info(f"Final λ state: {len(trainer.lambda_state)} networks, "
                 f"mean={all_lam.mean():.2f}, max={all_lam.max():.1f}")

    if eval_history:
        print(f"\n{'Dual':>6} {'SGD':>7} {'Mean':>7} {'p5':>7} {'p1':>7} "
              f"{'Vio':>7} {'Feas':>7}")
        print("-" * 55)
        for e in eval_history:
            marker = " *" if e["outer"] == best_outer else ""
            print(f"{e['outer']:6d} {e['sgd_count']:7d} {e['mean_rate']:7.4f} "
                  f"{e['p5_rate']:7.4f} {e['p1_rate']:7.4f} "
                  f"{e['mean_vio']:7.4f} {e['feasibility']:6.1%}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
