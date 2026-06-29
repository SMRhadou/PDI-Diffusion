#!/usr/bin/env python3
"""WRA baselines: PDL, PDM, DPS — compare against PDI MC ceiling.

Usage:
    micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
        --methods pdl,pdm,dps,pdi \
        --inverse-beta 1.0 --dual-step-size 0.1 --dual-lambda-init 10.0 \
        --max-batches 2 --n-samples-per-input 50 \
        --label shared_lambda_baselines
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import types as _types
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
})

import numpy as np
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

sys.path.insert(0, str(_project_root() / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdi.datasets.wra.utils import compute_ergodic_rates_batched

from pdi.datasets.wra.datamodule import WRABuilder
from pdi.diffusion.energy_ddpm import (
    EnergyDDPM,
    _EnergyScoreMixin,
    _safe_torch_load,
)
from pdi.diffusion.ddim import DDIM
from hydra.utils import instantiate
from _experiment_paths import experiment_dir


# ---------------------------------------------------------------------------
# Shared helpers (from test_energy_diffusion_medium.py)
# ---------------------------------------------------------------------------

class _NoOpModel(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None,
                cond=None, return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


def _compose_diffusion_cfg(conf_dir: Path, cfg_name: str, stack=None):
    if stack is None:
        stack = set()
    if cfg_name in stack:
        raise ValueError(f"Cyclic defaults at '{cfg_name}'.")
    stack.add(cfg_name)
    cfg_path = conf_dir / f"{cfg_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    merged = OmegaConf.create({})
    for entry in cfg.get("defaults", []):
        if entry == "_self_":
            continue
        child_name = None
        if isinstance(entry, str):
            child_name = entry
        elif isinstance(entry, dict):
            for _, v in entry.items():
                child_name = None if v == "_self_" else str(v)
                break
        if child_name:
            merged = OmegaConf.merge(
                merged, _compose_diffusion_cfg(conf_dir, child_name, stack=stack.copy()))
    container = OmegaConf.to_container(cfg, resolve=False)
    container.pop("defaults", None)
    return OmegaConf.merge(merged, OmegaConf.create(container))


def _resolve_path(raw, project_root):
    if raw is None:
        return project_root / "data" / "wra"
    return Path(str(raw).replace("${hydra:runtime.cwd}", str(project_root))).expanduser()


def _pick(override, cfg, key, fallback):
    if override is not None:
        return override
    if not isinstance(cfg, dict):
        return fallback
    v = cfg.get(key, fallback)
    return fallback if v is None else v


def _load_dataset_cfg(project_root, batch_size_val, n_samples_per_input,
                      dataset_config="wra_medium"):
    base = OmegaConf.load(project_root / "src/pdi/conf/dataset/wra.yaml")
    overlay = OmegaConf.load(
        project_root / f"src/pdi/conf/dataset/{dataset_config}.yaml")
    cfg = OmegaConf.merge(base, overlay)
    cfg.root = str(project_root / "data" / "wra")
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.persistent_workers = False
    cfg.batch_size_val = int(batch_size_val)
    cfg.n_samples_per_input = int(n_samples_per_input)
    return cfg


# ---------------------------------------------------------------------------
# Rate computation (standalone, grad-capable)
# ---------------------------------------------------------------------------

def _rate_tensor(x, context):
    """Compute per-realization rates [B, R, T, N] from x [B, T, N, F]."""
    power_norm = x[..., 0]
    power = (power_norm + 0.5) * context.p_max
    received = torch.einsum("brij,bti->brtj", context.gains, power)
    direct_gain = torch.diagonal(context.gains, dim1=2, dim2=3).unsqueeze(2)
    direct_signal = power.unsqueeze(1) * direct_gain
    interference = (received - direct_signal).clamp_min(0.0)
    noise = context.noise_var.clamp_min(1e-20).unsqueeze(1)
    sinr = direct_signal.clamp_min(0.0) / (interference + noise)
    return torch.log2(1.0 + sinr)


def _ergodic_rates(x, context):
    """Return (ergodic_rates [B,N], sum_rates [B])."""
    rates = _rate_tensor(x, context)
    erg = rates.mean(dim=(1, 2))
    return erg, erg.sum(dim=1)


# ---------------------------------------------------------------------------
# 1. PDL — Primal-Dual Langevin (no diffusion)
# ---------------------------------------------------------------------------

def pdl_wra(
    context,
    shape: Tuple[int, ...],
    device: torch.device,
    *,
    num_iters: int = 1000,
    primal_lr: float = 1e-3,
    dual_lr: float = 10.0,
    noise_scale: float = 1.0,
    lambda_init: float = 0.0,
    shared_lambda: bool = True,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict]:
    torch.manual_seed(seed)
    B, T_seq, N, F = shape
    x = torch.randn(shape, device=device) * 0.1
    x = x.clamp(-0.5, 0.5)

    if shared_lambda:
        lam = torch.full((1, N), lambda_init, device=device)
    else:
        lam = torch.full((B, N), lambda_init, device=device)

    r_min = context.r_min.view(-1, 1)

    for k in range(num_iters):
        x_req = x.detach().requires_grad_(True)
        erg, sum_r = _ergodic_rates(x_req, context)
        violation = r_min - erg
        lagrangian = (-sum_r + (lam * violation).sum(dim=1)).sum()
        grad = torch.autograd.grad(lagrangian, x_req)[0]

        noise_term = 0.0
        if k < num_iters - 1 and noise_scale > 0:
            noise_term = noise_scale * math.sqrt(2 * primal_lr) * torch.randn_like(x)
        x = (x_req - primal_lr * grad + noise_term).detach()
        x = x.clamp(-0.5, 0.5)

        with torch.no_grad():
            erg_now, _ = _ergodic_rates(x, context)
            v = r_min - erg_now
            if shared_lambda:
                lam = (lam + dual_lr * v.mean(dim=0, keepdim=True)).clamp_min(0.0)
            else:
                lam = (lam + dual_lr * v).clamp_min(0.0)

    return x.detach(), {
        "final_lambda_mean": float(lam.mean()),
        "final_lambda_max": float(lam.max()),
        "final_lambda": lam.detach(),
    }


# ---------------------------------------------------------------------------
# 2. PDM — Projected Diffusion Model (MC score + per-step projection)
# ---------------------------------------------------------------------------

def _project_wra(x, context, num_iters=5, lr=0.05, rho=10.0):
    """Project onto feasible set: min_w ||w - x||^2 + (rho/2)||[f(w)]_+||^2."""
    x0 = x.clone().detach()
    w = x0.clone()
    r_min = context.r_min.view(-1, 1)

    for _ in range(num_iters):
        w_req = w.detach().requires_grad_(True)
        erg, _ = _ergodic_rates(w_req, context)
        violation = (r_min - erg).clamp_min(0.0)
        proximity = ((w_req - x0) ** 2).sum()
        penalty = (rho / 2.0) * (violation ** 2).sum()
        loss = proximity + penalty
        if violation.max().item() < 1e-6:
            break
        grad = torch.autograd.grad(loss, w_req)[0]
        w = (w_req - lr * grad).detach()
        w = w.clamp(-0.5, 0.5)

    return w.detach()


def _final_project_wra(x, sampler, dataset_name, network_id, num_nodes,
                        device, p_max, noise_var, r_min,
                        num_iters=500, lr=0.05, rho=10.0,
                        max_timeslots=None):
    """Project onto the feasible set using full channel history.

    Solves: min_w ||w - w_0||^2 + (rho/2)||[f(w)]_+||^2
    where [f(w)]_+ = max(0, r_min - ergodic_rate).

    Uses the same rate formula as _compute_full_ergodic_evolution so that
    feasibility measured here matches evaluation.
    """
    B, T_seq, N, F = x.shape

    H_raw, associations = _load_raw_channel_gains(
        sampler, dataset_name, network_id, num_nodes, device, x.dtype)
    H_raw = _resize_h(H_raw, max_timeslots)

    H = H_raw.to(device=device, dtype=x.dtype)
    assoc = associations.to(device=device, dtype=x.dtype)

    def _erg_rates(w_in):
        power_rx = ((w_in + 0.5) * p_max).clamp_min(0.0)
        power_tx = power_rx @ assoc.T
        total_received = torch.einsum("tmn,bm->btn", H, power_tx)
        direct_signal = torch.einsum("tmn,mn,bm->btn", H, assoc, power_tx)
        interference = total_received - direct_signal
        sinr = direct_signal / (interference + noise_var)
        rates = torch.log2(1.0 + sinr)
        return rates.mean(dim=1)

    w0 = x[:, 0, :, 0].clone().detach()
    w = w0.clone()

    for it in range(num_iters):
        w_req = w.detach().requires_grad_(True)
        erg = _erg_rates(w_req)
        violation = (r_min - erg).clamp_min(0.0)
        proximity = ((w_req - w0) ** 2).sum(dim=1)
        penalty = (rho / 2.0) * (violation ** 2).sum(dim=1)
        loss = (proximity + penalty).mean()
        if violation.max().item() < 1e-6:
            break
        grad = torch.autograd.grad(loss, w_req)[0]
        w = (w_req - lr * grad).detach()
        w = w.clamp(-0.5, 0.5)

    result = x.clone()
    result[:, 0, :, 0] = w
    return result


def pdm_wra(
    sampler,
    shape: Tuple[int, ...],
    device: torch.device,
    data,
    context,
    *,
    projection_iters: int = 5,
    projection_lr: float = 0.05,
    projection_rho: float = 10.0,
    seed: int = 42,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    torch.manual_seed(seed)
    B = shape[0]
    N = shape[2]
    x_t = torch.randn(shape, device=device, dtype=torch.float32)

    is_ddim = isinstance(sampler, DDIM) and not isinstance(sampler, EnergyDDPM)

    if is_ddim:
        edge_index = data.edge_index if data is not None and hasattr(data, "edge_index") else None
        edge_weight = data.edge_weight if data is not None and hasattr(data, "edge_weight") else None
        u_0 = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if data is not None and hasattr(data, "x") and data.x is not None else None
        timesteps = sampler.ddim_timesteps.to(device)
        iterator = enumerate(timesteps)
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="PDM-DDIM", unit="step", total=len(timesteps))
        for i, t_curr in iterator:
            t = torch.full((B,), int(t_curr.item()), device=device, dtype=torch.long)
            t_next_val = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            t_next = torch.full((B,), t_next_val, device=device, dtype=torch.long)
            with torch.no_grad():
                pred, _ = sampler.model(x=x_t, timesteps=t, edge_index=edge_index,
                                        edge_weight=edge_weight, cond=u_0, return_intermediates=False)
                x0_pred, eps_pred = sampler._pred_to_x0_eps(x_t, t, pred)
                x_t = sampler._ddim_step(x_t=x_t, t=t, t_next=t_next,
                                         x0_pred=x0_pred, eps_pred=eps_pred, eta=sampler.ddim_eta)
            x_t = _project_wra(x_t, context, num_iters=projection_iters, lr=projection_lr, rho=projection_rho)
    else:
        iterator = reversed(range(sampler.num_timesteps))
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="PDM Sampling", unit="step")
        for t_int in iterator:
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            with torch.no_grad():
                score = sampler._estimate_score(
                    x_t=x_t, t=t, context=context)
                x0_pred, eps_pred = sampler._score_to_x0_eps(x_t=x_t, t=t, score=score)
                mean = sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
                if t_int > 0:
                    var = sampler._gather(sampler.posterior_variance, t).view(-1, 1, 1, 1)
                    x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
                else:
                    x_t = mean
            x_t = _project_wra(x_t, context, num_iters=projection_iters, lr=projection_lr, rho=projection_rho)

    return x_t.detach(), {"projection_iters": projection_iters}


# ---------------------------------------------------------------------------
# 3. DPS — Diffusion Posterior Sampling (MC score + gradient guidance)
# ---------------------------------------------------------------------------

def dps_wra(
    sampler,
    shape: Tuple[int, ...],
    device: torch.device,
    data,
    context,
    *,
    dps_scale: float = 1.0,
    n_samples_per_input: int = 50,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict]:
    torch.manual_seed(seed)

    _ac = sampler.alphas_cumprod
    _r_min = context.r_min
    _ctx = context
    is_ddim = isinstance(sampler, DDIM) and not isinstance(sampler, EnergyDDPM)

    def _dps_correct(x_t, t, eps_pred):
        x_t_g = x_t.detach().requires_grad_(True)
        alpha_bar = _ac[t].view(-1, 1, 1, 1)
        sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
        sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
        x0_tw = (x_t_g - sqrt_1mab * eps_pred.detach()) / sqrt_ab
        x0_tw = x0_tw.clamp(-0.5, 0.5)

        erg, _ = _ergodic_rates(x0_tw, _ctx)
        violation = (_r_min.view(-1, 1) - erg).clamp_min(0.0)
        loss = (violation ** 2).sum()
        grad_xt = torch.autograd.grad(loss, x_t_g)[0]
        sigma_t = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
        return (x_t - dps_scale * sigma_t * grad_xt).detach()

    if is_ddim:
        B, N = shape[0], shape[2]
        edge_index = data.edge_index if data is not None and hasattr(data, "edge_index") else None
        edge_weight = data.edge_weight if data is not None and hasattr(data, "edge_weight") else None
        u_0 = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if data is not None and hasattr(data, "x") and data.x is not None else None
        x_t = torch.randn(shape, device=device)
        timesteps = sampler.ddim_timesteps.to(device)
        for i, t_curr in enumerate(tqdm.tqdm(timesteps, desc="DPS-DDIM", unit="step")):
            t = torch.full((B,), int(t_curr.item()), device=device, dtype=torch.long)
            t_next_val = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            t_next = torch.full((B,), t_next_val, device=device, dtype=torch.long)
            with torch.no_grad():
                pred, _ = sampler.model(x=x_t, timesteps=t, edge_index=edge_index,
                                        edge_weight=edge_weight, cond=u_0, return_intermediates=False)
                x0_pred, eps_pred = sampler._pred_to_x0_eps(x_t, t, pred)
            x_t = _dps_correct(x_t, t, eps_pred)
            with torch.no_grad():
                pred2, _ = sampler.model(x=x_t, timesteps=t, edge_index=edge_index,
                                         edge_weight=edge_weight, cond=u_0, return_intermediates=False)
                x0_pred2, eps_pred2 = sampler._pred_to_x0_eps(x_t, t, pred2)
                x_t = sampler._ddim_step(x_t=x_t, t=t, t_next=t_next,
                                         x0_pred=x0_pred2, eps_pred=eps_pred2, eta=sampler.ddim_eta)
        x_out = x_t
    else:
        orig = sampler._score_to_x0_eps

        def _guide(self, x_t, t, score, _orig=orig):
            x0_pred, eps_pred = _orig(x_t, t, score)
            x_t_corr = _dps_correct(x_t, t, eps_pred)
            alpha_bar = _ac[t].view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
            sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
            x0_new = ((x_t_corr - sqrt_1mab * eps_pred) / sqrt_ab).clamp(-0.5, 0.5)
            eps_new = (x_t - sqrt_ab * x0_new) / sqrt_1mab
            return x0_new, eps_new

        sampler._score_to_x0_eps = _types.MethodType(_guide, sampler)
        try:
            result = sampler.sample(
                shape=shape, device=device, data=data,
                n_samples_per_input=n_samples_per_input,
            )
            x_out = result[0] if isinstance(result, tuple) else result
        finally:
            sampler._score_to_x0_eps = orig

    return x_out.detach(), {"dps_scale": dps_scale}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_full_ergodic_evolution(
    power_allocs, H_raw, associations, p_max, noise_var, num_trials=5,
):
    """Time-shared ergodic rate evolution.

    Uses the same rate formula as the DDIM training evaluator
    (compute_ergodic_rates_batched) but keeps per-timeslot rates for
    time-sharing assignment.

    Parameters
    ----------
    power_allocs : [K, N] normalized per-receiver powers in [-0.5, 0.5]
    H_raw : [T, m, n] raw channel gains (NOT tx_assoc-reindexed)
    associations : [m, n] TX-RX pairing matrix
    p_max, noise_var : scalars
    num_trials : number of random policy-to-timeslot assignments
    """
    K, N = power_allocs.shape
    T = H_raw.shape[0]
    device = power_allocs.device
    dtype = power_allocs.dtype

    power_rx = ((power_allocs + 0.5) * p_max).clamp_min(0.0)  # [K, N]
    assoc = associations.to(device=device, dtype=dtype)  # [m, n]
    power_tx = power_rx @ assoc.T  # [K, m]
    H = H_raw.to(device=device, dtype=dtype)  # [T, m, n]

    total_received = torch.einsum("tmn,pm->ptn", H, power_tx)  # [K, T, N]
    direct_signal = torch.einsum("tmn,mn,pm->ptn", H, assoc, power_tx)  # [K, T, N]
    interference = total_received - direct_signal
    sinr = direct_signal / (interference + noise_var)
    rates_all = torch.log2(1.0 + sinr)  # [K, T, N] per-timeslot

    t_range = torch.arange(T, device=device)
    counts = (t_range + 1).to(dtype)
    idx_all = torch.randint(0, K, (num_trials, T), device=device)
    inst_all = rates_all[idx_all, t_range.unsqueeze(0), :]
    ergodic_all = inst_all.cumsum(dim=1) / counts.view(1, T, 1)
    return inst_all, ergodic_all


def _evaluate_method(
    x, context, sampler, dataset_name, network_id, num_nodes,
    n_samples_per_input, device, num_trials=5, max_timeslots=None,
    dual_lambda=None,
):
    with torch.no_grad():
        erg, sum_r = _ergodic_rates(x, context)
    r_min = float(context.r_min[0])

    K = n_samples_per_input
    power_allocs = x[:K, 0, :, 0].to(device)

    H_raw, associations = _load_raw_channel_gains(
        sampler, dataset_name, network_id, num_nodes, device, x.dtype)
    H_raw = _resize_h(H_raw, max_timeslots)
    torch.manual_seed(network_id)
    _, erg_evo = _compute_full_ergodic_evolution(
        power_allocs, H_raw, associations,
        float(context.p_max[0, 0, 0]),
        float(context.noise_var[0, 0, 0]),
        num_trials=num_trials,
    )
    final_erg = erg_evo[:, -1, :].mean(dim=0).cpu().numpy()
    mean_rate = float(final_erg.mean())
    p5_rate = float(np.percentile(final_erg, 5))
    p1_rate = float(np.percentile(final_erg, 1))
    min_rate = float(final_erg.min())
    feas = float((final_erg >= r_min - 0.01).mean())
    # Policy diversity: effective rank of the [K, N] power matrix.
    # Measures how many linearly independent policies exist.
    p_centered = power_allocs - power_allocs.mean(dim=0, keepdim=True)  # center
    sv = torch.linalg.svdvals(p_centered.float())
    sv_norm = sv / sv.sum().clamp_min(1e-12)
    sv_pos = sv_norm[sv_norm > 1e-12]
    power_entropy = float(torch.exp(-torch.sum(sv_pos * torch.log(sv_pos))).item())
    # Complementary slackness: λ_j * max(rate_j - r_min, 0)
    # Measures how much λ is "wasted" on already-satisfied constraints.
    cs = 0.0
    if dual_lambda is not None:
        lam_mean = dual_lambda[:K].mean(dim=0).cpu().numpy()  # [N]
        slack = np.maximum(final_erg - r_min, 0.0)  # [N]
        cs = float((lam_mean * slack).sum())
    return {
        "mean_rate": mean_rate,
        "p5_rate": p5_rate,
        "p1_rate": p1_rate,
        "min_rate": min_rate,
        "feasibility": feas,
        "per_user_rates": final_erg,
        "power_entropy": power_entropy,
        "cs": cs,
    }


def _load_ordered_channel_gains(sampler, dataset_name, network_id, num_nodes,
                                 device, dtype):
    net_data = sampler._get_network_channel_data(dataset_name, int(network_id))
    paths = net_data.h_timeslot_paths
    if not paths:
        raise RuntimeError(f"No H_instantaneous for {dataset_name}/{network_id}")
    gains_list = []
    for p in paths:
        slot = _safe_torch_load(p, weights_only=False)
        h_t = _EnergyScoreMixin._decode_timeslot_tensor(slot)
        g = h_t.index_select(0, net_data.tx_assoc).contiguous()
        gains_list.append(g.to(device=device, dtype=dtype))
    return torch.stack(gains_list, dim=0)


def _load_raw_channel_gains(sampler, dataset_name, network_id, num_nodes,
                             device, dtype):
    """Load raw H [T, m, n] and associations [m, n] without tx_assoc reindexing."""
    net_data = sampler._get_network_channel_data(dataset_name, int(network_id))
    paths = net_data.h_timeslot_paths
    if not paths:
        raise RuntimeError(f"No H_instantaneous for {dataset_name}/{network_id}")
    gains_list = []
    for p in paths:
        slot = _safe_torch_load(p, weights_only=False)
        h_t = _EnergyScoreMixin._decode_timeslot_tensor(slot)
        gains_list.append(h_t.to(dtype=dtype))  # keep on CPU, moved per-slot in eval
    H_raw = torch.stack(gains_list, dim=0)  # [T, m, n]
    # Load associations from graph.pt
    graph_path = paths[0].parent.parent / "graph.pt"
    graph_data = _safe_torch_load(graph_path, weights_only=False)
    associations = graph_data["associations"].to(dtype=dtype)
    return H_raw, associations


def _resize_h(H_raw: torch.Tensor, target: int | None) -> torch.Tensor:
    """Truncate or tile H_raw along dim-0 to reach *target* timeslots.

    When target > available, the channel realizations are repeated cyclically.
    """
    if target is None:
        return H_raw
    T = H_raw.shape[0]
    if target <= T:
        return H_raw[:target]
    reps = -(-target // T)  # ceil division
    return H_raw.repeat(reps, 1, 1)[:target]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_power_scatter_comparison(
    method_samples: Dict[str, torch.Tensor],  # method -> [K, T, N, F]
    p_max: float,
    num_nodes: int,
    network_id: Any,
    output_dir: Path,
    num_pairs: int = 4,
    seed: int = 0,
):
    import matplotlib
    from matplotlib.ticker import FormatStrFormatter

    colors = {
        "pdi": "tab:blue", "pdl": "tab:orange",
        "pd_train": "tab:brown", "ddim": "tab:purple",
        "pdm": "tab:green", "dps": "tab:red",
        "pd_expert": "tab:brown",
    }
    labels = {
        "pdi": "PDI (MC)", "pdl": "PDL",
        "pd_train": "PD Expert", "ddim": r"DDIM $\mu^*$",
        "pdm": "PDM", "dps": "DPS",
        "pd_expert": "PD Expert",
    }
    # Also handle pdi_net_ and dt_net_ prefixed methods
    for m in method_samples:
        if m.startswith("pdi_net_"):
            colors[m] = "tab:blue"
            labels[m] = "PDI-Net"
        elif m.startswith("dt_net_"):
            colors[m] = "tab:cyan"
            labels[m] = m

    all_methods = [m for m in method_samples if method_samples[m] is not None]
    if not all_methods:
        return None

    n_panels = max(1, min(num_pairs, num_nodes - 1))
    rng = np.random.RandomState(seed)
    nodes_a = rng.choice(num_nodes, size=n_panels, replace=False)
    all_nodes = np.arange(num_nodes)
    nodes_b = np.array([
        int(rng.choice([n for n in all_nodes if n != nodes_a[k]]))
        for k in range(n_panels)
    ])

    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows),
                             squeeze=False)
    for idx in range(n_panels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ni, nj = int(nodes_a[idx]), int(nodes_b[idx])
        for method in all_methods:
            x = method_samples[method]
            n_show = min(250, x.shape[0])
            power_np = ((x[:n_show, 0, :, 0] + 0.5) * p_max).detach().cpu().numpy()
            pi = power_np[:, ni] / p_max
            pj = power_np[:, nj] / p_max
            ax.scatter(pi, pj, alpha=0.4, s=18, c=colors.get(method, "gray"),
                       edgecolors="none",
                       label=labels.get(method, method))
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(alpha=0.2, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        ax.set_xlabel(rf"$p_{{{ni}}}/P_{{\max}}$", fontsize=9)
        ax.set_ylabel(rf"$p_{{{nj}}}/P_{{\max}}$", fontsize=9)
        ax.legend(fontsize=6, framealpha=0.8, markerscale=1.3, loc="upper right")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout(pad=0.4)
    save_path = output_dir / f"power_scatter_net{network_id}.pdf"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter: {save_path}")
    return [save_path]


def run_temperature_scatter(sampler, shape, device, data, context, output_dir,
                            seed=42, sub_batch=0, n_samples_per_input=None,
                            ib_values=None, num_agent_pairs=2, chunk_size=0,
                            dataset_name=None, network_id=None, num_nodes=None,
                            eval_timeslots=200, any_sampler=None):
    """Scatter plots of PDI samples at different inverse_beta values."""
    from matplotlib.ticker import FormatStrFormatter

    if ib_values is None:
        ib_values = [0.1, 0.5, 1.0, 5.0, 10.0]

    K = n_samples_per_input or shape[0]
    B, T_seq, N, F = shape
    rng = np.random.RandomState(seed)
    num_agent_pairs = min(num_agent_pairs, N - 1)
    nodes_a = rng.choice(N, size=num_agent_pairs, replace=False)
    nodes_b = np.array([int(rng.choice([n for n in range(N) if n != nodes_a[k]]))
                        for k in range(num_agent_pairs)])

    p_max = float(context.p_max[0, 0, 0])
    cmap = plt.cm.viridis
    ib_colors = {ib: cmap(i / max(1, len(ib_values) - 1)) for i, ib in enumerate(ib_values)}

    n_cols = len(ib_values)
    n_rows = num_agent_pairs
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows),
                             squeeze=False)

    orig_ib = sampler.inverse_beta
    orig_ib_buf = sampler.inverse_beta_by_t.clone()

    if chunk_size > 0:
        from wra_baselines import _install_chunked_score
        _install_chunked_score(sampler, chunk_size)

    orig_ds = sampler.dual_step_size
    orig_ds_buf = sampler.dual_step_size_by_t.clone()

    for col, ib in enumerate(ib_values):
        sampler.inverse_beta = ib
        sampler.inverse_beta_by_t = torch.full_like(sampler.inverse_beta_by_t, ib)
        ds = 0.1 if ib > 1 else orig_ds
        sampler.dual_step_size_by_t = torch.full_like(orig_ds_buf, ds)

        # Install sub-batch if needed
        _orig_dual = None
        if sub_batch > 0 and B >= sub_batch:
            n_sub = B // sub_batch
            _orig_dual = sampler._dual_ascent_step
            import types as _types
            def _sb_dual(dual_lambda, ergodic_rates, context, t=None,
                         _ns=n_sub, _sb=sub_batch, _s=sampler):
                if t is None:
                    step = _s.dual_step_size_by_t[0].to(device=dual_lambda.device, dtype=dual_lambda.dtype)
                else:
                    t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(0, _s.num_timesteps - 1)
                    step = _s._gather(_s.dual_step_size_by_t.to(device=dual_lambda.device), t_safe).to(dtype=dual_lambda.dtype)[0]
                B_l, N_l = dual_lambda.shape
                violation = context.r_min.view(-1, 1) - ergodic_rates
                mean_viol = violation.view(_ns, _sb, N_l).mean(dim=1, keepdim=True)
                lam_rep = dual_lambda.view(_ns, _sb, N_l)[:, :1, :]
                updated = (lam_rep + step * mean_viol).clamp_min(0.0)

                if _s.dual_lambda_max is not None:
                    updated = updated.clamp_max(float(_s.dual_lambda_max))
                return updated.expand(_ns, _sb, N_l).reshape(B_l, N_l).contiguous()
            sampler._dual_ascent_step = _sb_dual
            sampler.dual_lambda_mode = "per_sample"

        torch.manual_seed(seed)
        result = sampler.sample(shape=shape, device=device, data=data, n_samples_per_input=K)
        x = result[0] if isinstance(result, tuple) else result

        if _orig_dual is not None:
            sampler._dual_ascent_step = _orig_dual
            sampler.dual_lambda_mode = "shared_per_network"

        n_show = min(250, x.shape[0])
        power_np = ((x[:n_show, 0, :, 0] + 0.5) * p_max).detach().cpu().numpy()

        eval_sampler = any_sampler or sampler
        N_eval = num_nodes or N
        if dataset_name is not None and network_id is not None:
            m_ib = _evaluate_method(x, context, eval_sampler, dataset_name,
                                    network_id, N_eval, K, device, 5,
                                    max_timeslots=eval_timeslots)
            mean_rate = m_ib["mean_rate"]
            p5_rate = m_ib["p5_rate"]
            feas = m_ib["feasibility"]
        else:
            with torch.no_grad():
                erg, _ = _ergodic_rates(x, context)
            erg_np = erg.cpu().numpy()
            mean_rate = float(erg_np.mean())
            p5_rate = float(np.percentile(erg_np.flatten(), 5))
            feas = float((erg_np.flatten() >= float(context.r_min[0])).mean())
        print(f"  [temp] ib={ib}, ds={ds}: mean={mean_rate:.3f}, "
              f"p5={p5_rate:.3f}, feas={feas:.1%}")

        for row in range(num_agent_pairs):
            ax = axes[row, col]
            ni, nj = int(nodes_a[row]), int(nodes_b[row])
            pi = power_np[:, ni] / p_max
            pj = power_np[:, nj] / p_max
            ax.scatter(pi, pj, alpha=0.45, s=22, c=[ib_colors[ib]],
                       edgecolors="none")
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.grid(alpha=0.2, linewidth=0.3)
            ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            if row == 0:
                ax.set_title(rf"$\beta^{{-1}}={ib}$" + "\n"
                             + rf"$\bar{{R}}={mean_rate:.2f}$, p5$={p5_rate:.2f}$",
                             fontsize=8)
            if col == 0:
                ax.set_ylabel(rf"$p_{{{nj}}}/P_{{\max}}$", fontsize=9)
            if row == num_agent_pairs - 1:
                ax.set_xlabel(rf"$p_{{{ni}}}/P_{{\max}}$", fontsize=9)

    # Restore
    sampler.inverse_beta = orig_ib
    sampler.inverse_beta_by_t = orig_ib_buf
    sampler.dual_step_size_by_t = orig_ds_buf

    fig.tight_layout(pad=0.4)
    save_path = output_dir / "fig_temperature_scatter.pdf"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temperature scatter: {save_path}")
    return save_path


def _plot_cdf_comparison(method_rates, r_min, output_dir):
    import matplotlib

    colors = {
        "pdi": "tab:blue", "pdi_unc": "tab:cyan", "pdl": "tab:orange", "pdm": "tab:green",
        "dps": "tab:red", "pd_train": "tab:brown", "ddim": "tab:purple",
        "full_power": "tab:gray",
    }
    linestyles = {
        "pdi": "-", "pdi_unc": "--", "pdl": "--", "pdm": "-.", "dps": ":", "pd_train": "-",
        "ddim": "-.", "full_power": "--",
    }
    labels = {
        "pdi": "PDI (MC)", "pdi_unc": r"PDI (Unc., $\lambda=0$)", "pdl": "PDL", "pdm": "PDM",
        "dps": "DPS", "pd_train": "PD Expert", "ddim": "DDIM",
        "full_power": "Full Power",
    }
    plot_order = ["pdi", "pdi_unc", "pdl", "ddim", "pd_train", "pdm", "dps"]

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))
    stats_lines = []
    for method in plot_order:
        if method not in method_rates:
            continue
        rates = method_rates[method]
        sorted_r = np.sort(rates)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        color = colors.get(method, "tab:gray")
        ls = linestyles.get(method, "-")
        label = labels.get(method, method)
        feas = float((rates >= r_min - 0.01).mean())
        p5 = float(np.percentile(rates, 5))
        ax.plot(sorted_r, cdf, linewidth=1.8, color=color, linestyle=ls,
                label=label)
        stats_lines.append(
            f"{label}: mean={np.mean(rates):.3f}, p5={p5:.3f}, feas={feas:.1%}")

    ax.axvline(r_min, color="red", linestyle=":", linewidth=1.2, alpha=0.7,
               label=r"$r_{\min}$")
    ax.set_xlabel(r"Per-user ergodic rate", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 1.05])

    save_path = output_dir / "baseline_cdf_comparison.pdf"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved CDF comparison: {save_path}")
    for s in stats_lines:
        print(f"    {s}")
    return save_path


def _plot_bar_comparison(method_metrics, output_dir, scorer_tag="MC"):
    import matplotlib


    methods = list(method_metrics.keys())
    labels_map = {
        "pdi": "PDI (MC)", "pdi_unc": r"PDI (Unc.)", "pdl": "PDL",
        "pdm": f"PDM ({scorer_tag})", "dps": f"DPS ({scorer_tag})",
        "pd_train": "PD Expert", "full_power": "Full Power",
    }
    x_labels = [labels_map.get(m, m) for m in methods]
    mean_rates = [method_metrics[m]["mean_rate"] for m in methods]
    p5_rates = [method_metrics[m]["p5_rate"] for m in methods]
    feas = [method_metrics[m]["feasibility"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.8))

    axes[0].bar(x_labels, mean_rates, color="steelblue", width=0.6)
    axes[0].set_ylabel(r"Mean rate $(\uparrow)$", fontsize=6)
    axes[0].tick_params(axis="x", rotation=30, labelsize=5)
    axes[0].tick_params(axis="y", labelsize=5)

    axes[1].bar(x_labels, p5_rates, color="darkorange", width=0.6)
    axes[1].set_ylabel(r"5th %ile $(\uparrow)$", fontsize=6)
    axes[1].tick_params(axis="x", rotation=30, labelsize=5)
    axes[1].tick_params(axis="y", labelsize=5)

    axes[2].bar(x_labels, feas, color="seagreen", width=0.6)
    axes[2].set_ylabel(r"Feasibility $(\uparrow)$", fontsize=6)
    axes[2].set_ylim([0, 1.05])
    axes[2].tick_params(axis="x", rotation=30, labelsize=5)
    axes[2].tick_params(axis="y", labelsize=5)

    fig.tight_layout(pad=0.3)
    save_path = output_dir / "baseline_bar_comparison.pdf"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar comparison: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Log-axis formatting helper
# ---------------------------------------------------------------------------

def _fmt_log_axis(ax):
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelsize=5)


# ---------------------------------------------------------------------------
# Tracing infrastructure
# ---------------------------------------------------------------------------

def _sample_with_trace(sampler, shape, device, data, context, sub_batch=0,
                       n_samples_per_input=None):
    """Run reverse process step-by-step, recording λ, violation, and rate at every step."""
    B = shape[0]
    N = shape[2]
    K = n_samples_per_input or B
    sampler._current_study_data = data
    x_t = torch.randn(shape, device=device, dtype=torch.float32)
    dual_lambda = sampler._init_dual_lambda(
        batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
        init_value=sampler.dual_lambda_init,
    )

    lam_vectors = []
    violation_traces = []
    rate_traces = []

    n_sub = B // sub_batch if sub_batch > 0 and B >= sub_batch else 0
    use_shared = (not n_sub) and getattr(sampler, 'dual_lambda_mode', '') == 'shared_per_network'

    for t_int in tqdm.tqdm(reversed(range(sampler.num_timesteps)),
                           total=sampler.num_timesteps, desc="PDI trace", unit="step"):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        score = sampler._estimate_score(
            x_t=x_t, t=t, context=context, dual_lambda=dual_lambda)
        x0_pred, _ = sampler._score_to_x0_eps(x_t=x_t, t=t, score=score)

        with torch.no_grad():
            erg, sum_r = _ergodic_rates(x0_pred, context)
            violation = (context.r_min.view(-1, 1) - erg).clamp_min(0.0)
            violation_traces.append(violation.mean(dim=0).cpu().numpy())
            rate_traces.append(float(sum_r.mean() / N))

        if n_sub > 0:
            sb = sub_batch
            violation_full = context.r_min.view(-1, 1) - erg
            step_val = sampler.dual_step_size_by_t[min(t_int, len(sampler.dual_step_size_by_t)-1)].to(
                device=dual_lambda.device, dtype=dual_lambda.dtype)
            mean_viol = violation_full.view(n_sub, sb, N).mean(dim=1, keepdim=True)
            lam_rep = dual_lambda.view(n_sub, sb, N)[:, :1, :]
            updated = (lam_rep + step_val * mean_viol).clamp_min(0.0)
            if sampler.dual_lambda_max is not None:
                updated = updated.clamp_max(float(sampler.dual_lambda_max))
            dual_lambda = updated.expand(n_sub, sb, N).reshape(B, N).contiguous()
        elif use_shared:
            shared_erg = sampler._time_shared_ergodic_rates_broadcast(
                x=x0_pred, context=context, n_samples_per_input=K)
            dual_lambda = sampler._dual_ascent_step(
                dual_lambda=dual_lambda, ergodic_rates=shared_erg, context=context, t=t)
        else:
            dual_lambda = sampler._dual_ascent_step(
                dual_lambda=dual_lambda, ergodic_rates=erg, context=context, t=t)

        lam_vectors.append(dual_lambda.mean(dim=0).cpu().numpy())

        mean = sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
        if t_int > 0:
            var = sampler._gather(sampler.posterior_variance, t).view(-1, 1, 1, 1)
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = mean

    traces = {
        "lam_vectors": np.stack(lam_vectors),
        "violations": np.stack(violation_traces),
        "rates": np.asarray(rate_traces),
        "final_dual_lambda": dual_lambda.cpu().numpy(),
    }
    return x_t.detach(), traces


# ---------------------------------------------------------------------------
# Schedule sweep
# ---------------------------------------------------------------------------

def _build_alphas_cumprod(name, T):
    k = torch.arange(T, dtype=torch.float64)
    tau = (T - 1 - k) / (T - 1)
    if name == "linear":
        snr = 1e-3 + (1e3 - 1e-3) * tau
    elif name == "poly2":
        snr = 1e-3 + (1e3 - 1e-3) * tau.pow(2)
    elif name == "ddpm_linear":
        betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
        return torch.cumprod(1.0 - betas, dim=0).clamp(1e-8, 1 - 1e-8).float()
    elif name == "cosine":
        s = 0.008
        u = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos((u / T + s) / (1 + s) * math.pi / 2).pow(2)
        ab = f / f[0]
        betas = (1.0 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)
        return torch.cumprod(1.0 - betas, dim=0).clamp(1e-8, 1 - 1e-8).float()
    else:
        raise ValueError(f"Unknown schedule: {name}")
    return (snr / (1.0 + snr)).clamp(1e-8, 1 - 1e-8).float()


def _override_schedule(sampler, alphas_cumprod):
    device = sampler.alphas_cumprod.device
    ac = alphas_cumprod.to(device=device, dtype=sampler.alphas_cumprod.dtype)
    alphas = torch.empty_like(ac)
    alphas[0] = ac[0]
    alphas[1:] = ac[1:] / ac[:-1].clamp_min(1e-12)
    alphas = alphas.clamp(1e-8, 1.0)
    betas = (1.0 - alphas).clamp(1e-8, 0.999)
    ac_prev = torch.cat([torch.ones_like(ac[:1]), ac[:-1]], dim=0)
    sampler.betas.copy_(betas)
    sampler.alphas.copy_(alphas)
    sampler.alphas_cumprod.copy_(ac)
    if hasattr(sampler, "alphas_cumprod_prev"):
        sampler.alphas_cumprod_prev.copy_(ac_prev)
    sampler.sqrt_alphas_cumprod.copy_(torch.sqrt(ac))
    if hasattr(sampler, "sqrt_one_minus_alphas_cumprod"):
        sampler.sqrt_one_minus_alphas_cumprod.copy_(torch.sqrt(1.0 - ac))
    if hasattr(sampler, "sqrt_recip_alphas"):
        sampler.sqrt_recip_alphas.copy_(torch.sqrt(1.0 / alphas))
    pv = betas * (1.0 - ac_prev) / (1.0 - ac).clamp_min(1e-12)
    if hasattr(sampler, "posterior_variance"):
        sampler.posterior_variance.copy_(pv)
    if hasattr(sampler, "posterior_log_variance_clipped"):
        sampler.posterior_log_variance_clipped.copy_(torch.log(pv.clamp_min(1e-20)))
    if hasattr(sampler, "posterior_mean_coef1"):
        sampler.posterior_mean_coef1.copy_(betas * torch.sqrt(ac_prev) / (1.0 - ac).clamp_min(1e-12))
    if hasattr(sampler, "posterior_mean_coef2"):
        sampler.posterior_mean_coef2.copy_((1.0 - ac_prev) * torch.sqrt(alphas) / (1.0 - ac).clamp_min(1e-12))


def run_schedule_sweep(sampler, shape, device, data, context, output_dir, seed=42, sub_batch=0, n_samples_per_input=None,
                       dataset_name=None, network_id=None, num_nodes=None, num_trials=5):
    import matplotlib


    schedules = ["cosine", "ddpm_linear", "linear", "poly2"]
    colors = {"cosine": "#9467bd", "ddpm_linear": "#d62728", "linear": "#1f77b4", "poly2": "#2ca02c"}
    names = {"cosine": "Cosine", "ddpm_linear": r"Linear $b_t$ (DDPM)", "linear": "Linear SNR", "poly2": r"Polynomial ($p\!=\!2$)"}
    T = sampler.num_timesteps
    K = n_samples_per_input or shape[0]
    traces_by_sched = {}
    final_samples = {}

    for sname in schedules:
        print(f"  [schedule] {sname}...")
        ac = _build_alphas_cumprod(sname, T)
        torch.manual_seed(seed)
        _override_schedule(sampler, ac)
        x_final, tr = _sample_with_trace(sampler, shape, device, data, context, sub_batch=sub_batch, n_samples_per_input=K)
        traces_by_sched[sname] = tr
        final_samples[sname] = x_final
        print(f"    vio={tr['violations'][-1].mean():.2e}  rate={tr['rates'][-1]:.4f}")

    _override_schedule(sampler, _build_alphas_cumprod("cosine", T))

    # Fig 1: violation + rate trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.4, 1.8))
    for sn in schedules:
        c = colors.get(sn, "gray")
        vio = traces_by_sched[sn]["violations"].mean(axis=1)
        ax1.plot(np.arange(len(vio)), vio, color=c, lw=0.8, label=names.get(sn, sn))
        ret = traces_by_sched[sn]["rates"]
        ax2.plot(np.arange(len(ret)), ret, color=c, lw=0.8, label=names.get(sn, sn))
    ax1.set_yscale("log")
    _fmt_log_axis(ax1)
    ax1.set_xlabel("Reverse step", fontsize=6)
    ax1.set_ylabel(r"Mean violation $(\downarrow)$", fontsize=6)
    ax1.legend(fontsize=4, framealpha=0.8)
    ax1.grid(alpha=0.2, linewidth=0.3)
    ax1.tick_params(axis="x", labelsize=5)
    ax2.set_xlabel("Reverse step", fontsize=6)
    ax2.set_ylabel(r"Mean rate $(\uparrow)$", fontsize=6)
    ax2.legend(fontsize=4, framealpha=0.8)
    ax2.grid(alpha=0.2, linewidth=0.3)
    ax2.tick_params(axis="both", labelsize=5)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_schedule_sweep.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Fig 2: CDF comparison across schedules (using full time-sharing evaluation)
    r_min = float(context.r_min[0])
    N_nodes = num_nodes or shape[2]
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))
    all_rates_list = []
    for sn in schedules:
        m = _evaluate_method(
            final_samples[sn], context, sampler,
            dataset_name, network_id, N_nodes,
            K, device, num_trials)
        rates_np = m["per_user_rates"]
        all_rates_list.append(rates_np)
        sorted_r = np.sort(rates_np)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        c = colors.get(sn, "gray")
        ax.plot(sorted_r, cdf, color=c, lw=1.8,
                label=f"{names.get(sn, sn)}")
    ax.axvline(r_min, color="red", ls=":", lw=1.2, alpha=0.7, label=r"$r_{\min}$")
    ax.set_xlabel("Per-user ergodic rate", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    path = output_dir / "fig_schedule_cdf.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Return per-schedule evaluation metrics
    sched_metrics = {}
    for sn in schedules:
        if dataset_name is not None and network_id is not None:
            m_sn = _evaluate_method(
                final_samples[sn], context, sampler, dataset_name, network_id,
                N_nodes, K, device, num_trials)
            sched_metrics[sn] = m_sn
    return sched_metrics


# ---------------------------------------------------------------------------
# Tail sweep
# ---------------------------------------------------------------------------

def run_tail_sweep(sampler, shape, device, data, context, output_dir, seed=42, sub_batch=0, n_samples_per_input=None,
                   dataset_name=None, network_id=None, num_nodes=None, num_trials=5):
    import matplotlib


    print("  [tail] running reference trajectory...")
    torch.manual_seed(seed)
    K = n_samples_per_input or shape[0]
    N_nodes = num_nodes or shape[2]
    x_ref, tr_ref = _sample_with_trace(sampler, shape, device, data, context, sub_batch=sub_batch, n_samples_per_input=K)
    if dataset_name is not None and network_id is not None:
        m_ref = _evaluate_method(x_ref, context, sampler, dataset_name, network_id,
                                 N_nodes, K, device, num_trials)
        tr_ref["p5"] = m_ref["p5_rate"]
        tr_ref["mean_eval"] = m_ref["mean_rate"]
    lam_all = tr_ref["lam_vectors"]  # [T, N]
    T_total = lam_all.shape[0]

    def _noop_dual(dual_lambda, ergodic_rates, context, t=None):
        return dual_lambda

    tail_fracs = [0.20, 0.40, 0.60, 0.80, 1.00]
    tail_traces = {}
    tail_rates = {}

    for frac in tail_fracs:
        start_idx = int(T_total * (1.0 - frac))
        lam_tail = torch.tensor(lam_all[start_idx:].mean(axis=0), device=device, dtype=torch.float32)
        print(f"  [tail] frac={int(frac*100)}%: λ_mean={lam_tail.mean():.2f}, λ_max={lam_tail.max():.1f}")

        orig_dual = sampler._dual_ascent_step
        orig_init = sampler._init_dual_lambda
        sampler._dual_ascent_step = _noop_dual
        _lv = lam_tail.clone()
        sampler._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=_lv: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)

        torch.manual_seed(seed)
        x_tail, tr_tail = _sample_with_trace(sampler, shape, device, data, context, sub_batch=0, n_samples_per_input=K)
        tail_traces[frac] = tr_tail
        tail_rates[frac] = tr_tail["rates"][-1]
        N_nodes = num_nodes or shape[2]
        if dataset_name is not None and network_id is not None:
            m_tail = _evaluate_method(
                x_tail, context, sampler, dataset_name, network_id, N_nodes,
                K, device, num_trials)
            tail_traces[frac]["p5"] = m_tail["p5_rate"]
            tail_traces[frac]["mean_eval"] = m_tail["mean_rate"]
            tail_traces[frac]["per_user_rates"] = m_tail["per_user_rates"]
            tail_traces[frac]["power_entropy"] = m_tail.get("power_entropy", 0)
        print(f"    rate={tr_tail['rates'][-1]:.4f}  vio={tr_tail['violations'][-1].mean():.2e}"
              f"  p5={tail_traces[frac].get('p5', 'N/A')}")

        sampler._dual_ascent_step = orig_dual
        sampler._init_dual_lambda = orig_init

    cmap = plt.cm.viridis
    fracs_sorted = sorted(tail_fracs)

    # Fig 1: violation + rate + p5 vs tail fraction
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.5, 1.8))
    pct = [int(f * 100) for f in fracs_sorted]
    vios = [float(tail_traces[f]["violations"][-1].mean()) for f in fracs_sorted]
    rates = [tail_rates[f] for f in fracs_sorted]
    ref_vio = float(tr_ref["violations"][-1].mean())
    ref_rate = tr_ref["rates"][-1]

    from matplotlib.ticker import LogFormatterSciNotation
    ax1.plot(pct, vios, "o-", color="#2c7fb8", lw=0.8, markersize=2, label=r"Fixed $\bar\lambda$")
    ax1.axhline(ref_vio, color="gray", ls="--", lw=0.6, label="PD from 0")
    ax1.set_xlabel(r"Tail \%", fontsize=6)
    ax1.set_ylabel(r"Mean vio $(\downarrow)$", fontsize=6)
    ax1.set_yscale("log")
    _fmt_log_axis(ax1)
    ax1.legend(fontsize=4, framealpha=0.8)
    ax1.grid(alpha=0.2, linewidth=0.3)
    ax1.tick_params(axis="x", labelsize=5)

    ax2.plot(pct, rates, "s-", color="#d95f02", lw=0.8, markersize=2, label=r"Fixed $\bar\lambda$")
    ax2.axhline(ref_rate, color="gray", ls="--", lw=0.6, label="PD from 0")
    ax2.set_xlabel("Tail %", fontsize=6)
    ax2.set_ylabel(r"Mean rate $(\uparrow)$", fontsize=6)
    ax2.legend(fontsize=4, framealpha=0.8)
    ax2.grid(alpha=0.2, linewidth=0.3)
    ax2.tick_params(axis="both", labelsize=5)

    p5s = [tail_traces[f].get("p5", 0) for f in fracs_sorted]
    ref_p5 = tr_ref.get("p5", 0)
    ax3.plot(pct, p5s, "^-", color="#7570b3", lw=0.8, markersize=2, label=r"Fixed $\bar\lambda$")
    ax3.axhline(ref_p5, color="gray", ls="--", lw=0.6, label="PD from 0")
    ax3.set_xlabel("Tail \\%", fontsize=6)
    ax3.set_ylabel(r"p5 rate $(\uparrow)$", fontsize=6)
    ax3.legend(fontsize=4, framealpha=0.8)
    ax3.grid(alpha=0.2, linewidth=0.3)
    ax3.tick_params(axis="both", labelsize=5)

    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_tail_avg_lambda.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Fig 2: violation trajectory per tail fraction
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.8))
    vio_ref = tr_ref["violations"].mean(axis=1)
    ax.plot(np.arange(len(vio_ref)), vio_ref, color="gray", ls="--", lw=0.8, label="PD from 0")
    for i, frac in enumerate(fracs_sorted):
        vio_t = tail_traces[frac]["violations"].mean(axis=1)
        ax.plot(np.arange(len(vio_t)), vio_t,
                color=cmap(i / max(len(fracs_sorted) - 1, 1)),
                lw=0.8, label=f"{int(frac*100)}%")
    ax.set_yscale("log")
    _fmt_log_axis(ax)
    ax.set_xlabel("Reverse step", fontsize=7)
    ax.set_ylabel(r"Mean violation $(\downarrow)$", fontsize=7)
    ax.legend(fontsize=4.5, ncol=2, framealpha=0.8)
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_tail_vio_trajectory.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Fig 3: rate trajectory per tail fraction
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.8))
    ax.plot(np.arange(len(tr_ref["rates"])), tr_ref["rates"], color="gray", ls="--", lw=0.8, label="PD from 0")
    for i, frac in enumerate(fracs_sorted):
        ax.plot(np.arange(len(tail_traces[frac]["rates"])), tail_traces[frac]["rates"],
                color=cmap(i / max(len(fracs_sorted) - 1, 1)),
                lw=0.8, label=f"{int(frac*100)}%")
    ax.set_xlabel("Reverse step", fontsize=7)
    ax.set_ylabel(r"Mean rate $(\uparrow)$", fontsize=7)
    ax.legend(fontsize=4.5, ncol=2, framealpha=0.8)
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_tail_rate_trajectory.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    tail_metrics = {}
    for frac in fracs_sorted:
        if "p5" in tail_traces[frac]:
            tail_metrics[f"Tail {int(frac*100)}\\%"] = {
                "mean_rate": tail_traces[frac].get("mean_eval", tail_rates[frac]),
                "p5_rate": tail_traces[frac]["p5"],
                "per_user_rates": tail_traces[frac].get("per_user_rates", np.array([])),
                "power_entropy": tail_traces[frac].get("power_entropy", 0),
            }
    return tr_ref, tail_metrics


# ---------------------------------------------------------------------------
# Lambda study + dual evolution figure
# ---------------------------------------------------------------------------

def run_lambda_study(sampler, shape, device, data, context, output_dir, seed=42, sub_batch=0, n_samples_per_input=None,
                     dataset_name=None, network_id=None, num_nodes=None, num_trials=5):
    import matplotlib


    B, _, N, _ = shape

    def _noop_dual(dual_lambda, ergodic_rates, context, t=None):
        return dual_lambda

    def _set_fixed_lam(s, lam_vec):
        s._dual_ascent_step = _noop_dual
        _lv = lam_vec.clone()
        s._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=_lv: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)

    def _set_init_lam(s, lam_vec):
        _lv = lam_vec.clone()
        s._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=_lv: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)

    # 1. Reference: PD from 0
    print("  [lam_study] reference PD from 0...")
    torch.manual_seed(seed)
    orig_dual = sampler._dual_ascent_step
    orig_init = sampler._init_dual_lambda
    K = n_samples_per_input or shape[0]
    x_ref, tr_ref = _sample_with_trace(sampler, shape, device, data, context, sub_batch=sub_batch, n_samples_per_input=K)
    lam_final = torch.tensor(tr_ref["lam_vectors"][-1], device=device, dtype=torch.float32)
    lam_avg = torch.tensor(tr_ref["lam_vectors"].mean(axis=0), device=device, dtype=torch.float32)
    print(f"    λ_final: mean={lam_final.mean():.2f}, max={lam_final.max():.1f}")
    print(f"    λ_avg:   mean={lam_avg.mean():.2f}, max={lam_avg.max():.1f}")

    variant_samples = {"PD from 0": x_ref}
    variant_traces = {"PD from 0": tr_ref}

    # 2. Fixed final λ
    print("  [lam_study] fixed final λ...")
    torch.manual_seed(seed)
    _set_fixed_lam(sampler, lam_final)
    x_ff, tr_fix_final = _sample_with_trace(sampler, shape, device, data, context, n_samples_per_input=K)
    variant_traces[r"Fixed $\lambda_T$"] = tr_fix_final
    variant_samples[r"Fixed $\lambda_T$"] = x_ff
    sampler._dual_ascent_step = orig_dual
    sampler._init_dual_lambda = orig_init

    # 3. Fixed avg λ
    print("  [lam_study] fixed avg λ...")
    torch.manual_seed(seed)
    _set_fixed_lam(sampler, lam_avg)
    x_fa, tr_fix_avg = _sample_with_trace(sampler, shape, device, data, context, n_samples_per_input=K)
    variant_traces[r"Fixed $\bar\lambda$"] = tr_fix_avg
    variant_samples[r"Fixed $\bar\lambda$"] = x_fa
    sampler._dual_ascent_step = orig_dual
    sampler._init_dual_lambda = orig_init

    # Warm-start variants removed — only fixed λ_T and fixed λ_avg kept

    # Print summary
    print("\n  Lambda study summary:")
    for name, tr in variant_traces.items():
        vio = tr["violations"][-1].mean()
        rate = tr["rates"][-1]
        print(f"    {name:25s}: rate={rate:.4f}, vio={vio:.2e}")

    # Figure 1: Dual evolution — best 5 and worst 5 users (from reference PD-from-0)
    lv_ref = tr_ref["lam_vectors"]  # [T, N]
    T_steps = lv_ref.shape[0]
    steps = np.arange(T_steps)
    final_lam_per_user = lv_ref[-1]  # [N]
    worst_5 = np.argsort(final_lam_per_user)[-5:]  # highest λ = most constrained
    best_5 = np.argsort(final_lam_per_user)[:5]    # lowest λ = easiest users

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.4, 1.8))
    for i, j in enumerate(best_5):
        ax1.plot(steps, lv_ref[:, j], lw=0.7, label=f"u{j}")
    ax1.set_xlabel("Reverse step", fontsize=6)
    ax1.set_ylabel(r"$\lambda_j$ (best 5)", fontsize=6)
    ax1.legend(fontsize=4, framealpha=0.8)
    ax1.grid(alpha=0.2, linewidth=0.3)
    ax1.tick_params(axis="both", labelsize=5)
    for i, j in enumerate(worst_5):
        ax2.plot(steps, lv_ref[:, j], lw=0.7, label=f"u{j}")
    ax2.set_xlabel("Reverse step", fontsize=6)
    ax2.set_ylabel(r"$\lambda_j$ (worst 5)", fontsize=6)
    ax2.legend(fontsize=4, framealpha=0.8)
    ax2.grid(alpha=0.2, linewidth=0.3)
    ax2.tick_params(axis="both", labelsize=5)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_dual_evolution.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Figure 1b: All variants — mean and max λ over steps
    colors_v = {"PD from 0": "tab:blue", r"Fixed $\lambda_T$": "tab:orange",
                r"Warm $\lambda_T$": "tab:green", r"Fixed $\bar\lambda$": "tab:red",
                r"Warm $\bar\lambda$": "tab:purple"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.4, 1.8))
    for name, tr in variant_traces.items():
        lv = tr["lam_vectors"]
        c = colors_v.get(name, "gray")
        ax1.plot(np.arange(lv.shape[0]), lv.mean(axis=1), color=c, lw=0.7, label=name)
        ax2.plot(np.arange(lv.shape[0]), lv.max(axis=1), color=c, lw=0.7, label=name)
    ax1.set_xlabel("Reverse step", fontsize=6)
    ax1.set_ylabel(r"$\bar{\lambda}$ $(\uparrow)$", fontsize=6)
    ax1.legend(fontsize=3.5, framealpha=0.8)
    ax1.grid(alpha=0.2, linewidth=0.3)
    ax1.tick_params(axis="both", labelsize=5)
    ax2.set_xlabel("Reverse step", fontsize=6)
    ax2.set_ylabel(r"$\max_j \lambda_j$", fontsize=6)
    ax2.legend(fontsize=3.5, framealpha=0.8)
    ax2.grid(alpha=0.2, linewidth=0.3)
    ax2.tick_params(axis="both", labelsize=5)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_dual_evolution_variants.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Figure 2: Violation trajectory for all variants
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.8))
    for name, tr in variant_traces.items():
        vio = tr["violations"].mean(axis=1)
        c = colors_v.get(name, "gray")
        ax.plot(np.arange(len(vio)), vio, color=c, lw=0.8, label=name)
    ax.set_yscale("log")
    _fmt_log_axis(ax)
    ax.set_xlabel("Reverse step", fontsize=7)
    ax.set_ylabel(r"Mean violation $(\downarrow)$", fontsize=7)
    ax.legend(fontsize=4, framealpha=0.8)
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_lambda_study_violation.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Figure 3: Rate trajectory for all variants
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.8))
    for name, tr in variant_traces.items():
        c = colors_v.get(name, "gray")
        ax.plot(np.arange(len(tr["rates"])), tr["rates"], color=c, lw=0.8, label=name)
    ax.set_xlabel("Reverse step", fontsize=7)
    ax.set_ylabel(r"Mean rate $(\uparrow)$", fontsize=7)
    ax.legend(fontsize=4, framealpha=0.8)
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout(pad=0.3)
    path = output_dir / "fig_lambda_study_rate.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    lam_metrics = {}
    if dataset_name is not None and network_id is not None:
        N_nodes = num_nodes or shape[2]
        for vname, x_v in variant_samples.items():
            if vname == "PD from 0":
                continue
            m_v = _evaluate_method(x_v, context, sampler, dataset_name, network_id,
                                   N_nodes, K, device, num_trials)
            lam_metrics[vname] = m_v
    return variant_traces, lam_metrics


# ---------------------------------------------------------------------------
# Chunked score estimation (process B in chunks to fit GPU memory)
# ---------------------------------------------------------------------------

def _install_chunked_score(sampler, chunk_size=50):
    """Monkey-patch _estimate_score to process B in chunks of chunk_size."""
    orig_fn = sampler._estimate_score

    def _chunked_estimate_score(self, x_t, t, context, dual_lambda=None,
                                 inverse_beta_override=None,
                                 _orig=orig_fn, _cs=chunk_size):
        B = x_t.shape[0]
        if B <= _cs:
            return _orig(x_t=x_t, t=t, context=context,
                        dual_lambda=dual_lambda,
                        inverse_beta_override=inverse_beta_override)

        scores = []
        for i in range(0, B, _cs):
            j = min(i + _cs, B)
            ctx_chunk = type(context)(
                gains=context.gains if context.gains.shape[0] == 1 else context.gains[i:j],
                p_max=context.p_max[i:j] if context.p_max.shape[0] > 1 else context.p_max,
                noise_var=context.noise_var[i:j] if context.noise_var.shape[0] > 1 else context.noise_var,
                r_min=context.r_min[i:j] if context.r_min.dim() > 0 and context.r_min.shape[0] > 1 else context.r_min,
            )
            dl_chunk = dual_lambda[i:j] if dual_lambda is not None else None
            ib_chunk = None
            if inverse_beta_override is not None:
                if inverse_beta_override.dim() == 0 or inverse_beta_override.numel() == 1:
                    ib_chunk = inverse_beta_override
                else:
                    ib_chunk = inverse_beta_override[i:j]
            score_chunk = _orig(
                x_t=x_t[i:j], t=t[i:j], context=ctx_chunk,
                dual_lambda=dl_chunk,
                inverse_beta_override=ib_chunk)
            scores.append(score_chunk)
        return torch.cat(scores, dim=0)

    sampler._estimate_score = _types.MethodType(_chunked_estimate_score, sampler)
    return orig_fn


# ---------------------------------------------------------------------------
# Build vanilla DDIM from trained checkpoint
# ---------------------------------------------------------------------------

def _build_vanilla_ddim(checkpoint_dir, device, checkpoint_epoch=None):
    """Load a trained UGNN + DDIM from checkpoint directory."""
    import re
    ckpt_dir = Path(checkpoint_dir)
    train_log = ckpt_dir / "train.log"
    if not train_log.exists():
        raise FileNotFoundError(f"train.log not found in {ckpt_dir}")

    # Parse resolved config from train.log
    cfg_lines = []
    capture = False
    with open(train_log) as f:
        for line in f:
            if "=== Resolved config ===" in line:
                capture = True
                continue
            if capture:
                m = re.match(r"\[\d{4}-\d{2}-\d{2}.*?\]\[.*?\]\[INFO\] - (.*)", line)
                if m:
                    cfg_lines.append(m.group(1))
                elif line.startswith("[") or line.startswith("Processing") or line.startswith("Skipping") or line.strip() == "":
                    break
                else:
                    cfg_lines.append(line.rstrip())
    cfg = OmegaConf.create("\n".join(cfg_lines).replace("graph_signal_diffusion.", "pdi."))

    model = instantiate(cfg.model)
    diff_dict = OmegaConf.to_container(cfg.diffusion, resolve=True)
    for drop_key in ("name", "loss_type", "parameterization", "_target_"):
        diff_dict.pop(drop_key, None)
    ddim = DDIM(model=model, **diff_dict)

    # Load checkpoint — use specific epoch if provided, else best_model.pt
    if checkpoint_epoch is not None:
        best_model = ckpt_dir / "trainer_chkpts" / "best_models" / f"best_model_epoch_{checkpoint_epoch}.pt"
    else:
        best_model = ckpt_dir / "trainer_chkpts" / "best_models" / "best_model.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_model}")
    ckpt = torch.load(best_model, map_location="cpu", weights_only=False)
    if "model" in ckpt and ckpt["model"] is not None:
        model.load_state_dict(ckpt["model"])
    if "diffusion" in ckpt:
        ddim.load_state_dict(ckpt["diffusion"], strict=False)

    ddim = ddim.to(device)
    ddim.eval()
    return ddim


# ---------------------------------------------------------------------------
# Build energy samplers
# ---------------------------------------------------------------------------

def _build_sampler(cfg, project_root, device, *,
                   dual_step_size=0.1, dual_lambda_init=10.0,
                   inverse_beta=1.0, dual_lambda_mode="shared_per_network",
                   energy_mc_samples=8, num_channel_realizations=None,
                   unconstrained=False, r_min_override=None,
                   dual_step_size_schedule="constant",
                   dual_step_size_start=None, dual_step_size_end=None,
                   dual_lambda_decay=0.0):
    ddpm_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(ddpm_cfg, dict):
        raise ValueError("Config must be dict-like")

    dataset_root = _resolve_path(ddpm_cfg.get("dataset_root"), project_root)

    if unconstrained:
        dual_step_size = 1e-12
        dual_lambda_init = 0.0

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
        dual_step_size_schedule=dual_step_size_schedule,
        dual_step_size_start=dual_step_size_start,
        dual_step_size_end=dual_step_size_end,
        dual_num_outer_iterations=int(ddpm_cfg.get("dual_num_outer_iterations", 15)),
        dual_lambda_init=dual_lambda_init,
        dual_lambda_max=None,
        dual_lambda_decay=dual_lambda_decay,
        dual_lambda_mode=dual_lambda_mode,
        langevin_step_size=float(ddpm_cfg.get("langevin_step_size", 1e-4)),
        langevin_noise_scale=float(ddpm_cfg.get("langevin_noise_scale", 1.0)),
        dataset_root=str(dataset_root),
        min_energy_sigma=float(ddpm_cfg.get("min_energy_sigma", 1e-4)),
        use_precomputed_channels=bool(ddpm_cfg.get("use_precomputed_channels", True)),
        allow_sparse_graph_fallback=False,
        r_min_override=r_min_override,
    ).to(device)
    sampler.eval()
    return sampler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WRA baselines: PDL, PDM, DPS vs PDI")
    parser.add_argument("--methods", type=str, default="pdi,pdl,pdm,dps",
                        help="Comma-separated list of methods to run")
    parser.add_argument("--label", type=str, default="baselines")
    parser.add_argument("--dataset", type=str, default="wra_medium",
                        help="Dataset config name (e.g. wra_medium, wra_medium_outdoor_high_density)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--batch-size-val", type=int, default=1)
    parser.add_argument("--n-samples-per-input", type=int, default=50)
    parser.add_argument("--num-evolution-trials", type=int, default=5)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # PDI / shared sampler hyperparams
    parser.add_argument("--inverse-beta", type=float, default=1.0)
    parser.add_argument("--dual-step-size", type=float, default=0.1)
    parser.add_argument("--dual-step-size-schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Schedule for dual step size over diffusion timesteps")
    parser.add_argument("--dual-step-size-start", type=float, default=None,
                        help="Dual step size at t=0 (end of reverse). Default: same as --dual-step-size")
    parser.add_argument("--dual-step-size-end", type=float, default=None,
                        help="Dual step size at t=T-1 (start of reverse). Default: same as --dual-step-size")
    parser.add_argument("--dual-lambda-init", type=float, default=10.0)
    parser.add_argument("--dual-lambda-decay", type=float, default=0.0,
                        help="Weight decay on dual lambda: lambda <- (1-decay)*lambda + step*violation")
    parser.add_argument("--energy-mc-samples", type=int, default=4)
    parser.add_argument("--num-channel-realizations", type=int, default=50)
    parser.add_argument("--sub-batch", type=int, default=0,
                        help="Sub-batch size for PDI shared lambda. 0 = single shared lambda "
                             "across all K samples. >0 = divide K into groups of this size, "
                             "each group gets its own lambda.")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Chunk size for score estimation. 0 = no chunking. "
                             ">0 = process B in chunks to fit GPU memory.")

    # PDL hyperparams
    parser.add_argument("--pdl-num-iters", type=int, default=1000)
    parser.add_argument("--pdl-primal-lr", type=float, default=1e-3)
    parser.add_argument("--pdl-dual-lr", type=float, default=10.0)
    parser.add_argument("--pdl-noise-scale", type=float, default=1.0)
    parser.add_argument("--pdl-lambda-init", type=float, default=0.0)

    # PDM hyperparams
    parser.add_argument("--pdm-iters", type=int, default=100,
                        help="Gradient steps per projection (both per-step and final)")
    parser.add_argument("--pdm-lr", type=float, default=0.05,
                        help="Step size for projection")
    parser.add_argument("--pdm-rho", type=float, default=10.0,
                        help="Penalty weight rho")

    # DPS hyperparams
    parser.add_argument("--dps-scale", type=float, default=1.0)

    # Vanilla DDIM
    parser.add_argument("--ddim-checkpoint-dir", type=str, default=None,
                        help="Path to trained DDIM checkpoint directory (contains train.log + trainer_chkpts/)")
    parser.add_argument("--ddim-epoch", type=int, default=None,
                        help="Specific epoch checkpoint to load (default: best_model.pt)")
    parser.add_argument("--eval-timeslots", type=int, default=None,
                        help="Limit evaluation to this many timeslots (default: use all)")
    parser.add_argument("--r-min-override", type=float, default=None,
                        help="Override r_min for all networks")

    # Trained PDI net
    parser.add_argument("--pdi-net-checkpoints", type=str, nargs="+", default=None,
                        help="Path(s) to trained PDI net checkpoint(s) (.pt files)")
    parser.add_argument("--pdi-net-hidden", type=int, default=128)
    parser.add_argument("--pdi-net-layers", type=int, default=6)
    parser.add_argument("--pdi-net-K", type=int, default=2)

    # Dual-trained score net (no λ conditioning)
    parser.add_argument("--dt-checkpoints", type=str, nargs="+", default=None,
                        help="Path(s) to dual-trained score net checkpoint(s) (.pt files)")
    parser.add_argument("--dt-hidden", type=int, default=None)
    parser.add_argument("--dt-layers", type=int, default=None)
    parser.add_argument("--dt-K", type=int, default=None)

    # PD Expert (forward pass)
    parser.add_argument("--pd-expert-checkpoint", type=str, default=None,
                        help="Path to PD expert model checkpoint (.pt file) for forward-pass evaluation")
    parser.add_argument("--pd-expert-hidden", type=int, default=64)
    parser.add_argument("--pd-expert-layers", type=int, default=3)
    parser.add_argument("--pd-expert-K", type=int, default=2)

    # Studies
    parser.add_argument("--schedule-sweep", action="store_true",
                        help="Run noise schedule comparison study")
    parser.add_argument("--tail-sweep", action="store_true",
                        help="Run tail-averaged lambda study")
    parser.add_argument("--lambda-study", action="store_true",
                        help="Run lambda initialization/freezing study + dual evolution plot")
    parser.add_argument("--temperature-scatter", action="store_true",
                        help="Generate temperature effect scatter plot for PDI")
    parser.add_argument("--trajectory-divergence", action="store_true",
                        help="Compute ||x_t^PDI - x_t^DT|| over diffusion time, averaged over networks")
    parser.add_argument("--studies-only", action="store_true",
                        help="Skip main methods loop, run only studies")

    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    project_root = _project_root()
    conf_dir = project_root / "src/pdi/conf/diffusion"
    ddpm_cfg = _compose_diffusion_cfg(conf_dir, "energy_ddpm_wra")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiment_dir("energy_sampler", args.label)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Build samplers
    need_pdi = "pdi" in methods or args.pdi_net_checkpoints or args.dt_checkpoints
    need_unconstrained = any(m in methods for m in ("pdm", "dps", "pdi_unc")) and not args.ddim_checkpoint_dir

    sampler_pdi = None
    if need_pdi:
        sampler_pdi = _build_sampler(
            ddpm_cfg, project_root, device,
            dual_step_size=args.dual_step_size,
            dual_lambda_init=args.dual_lambda_init,
            inverse_beta=args.inverse_beta,
            energy_mc_samples=args.energy_mc_samples,
            num_channel_realizations=args.num_channel_realizations,
            r_min_override=args.r_min_override,
            dual_step_size_schedule=args.dual_step_size_schedule,
            dual_step_size_start=args.dual_step_size_start,
            dual_step_size_end=args.dual_step_size_end,
            dual_lambda_decay=args.dual_lambda_decay,
        )

    sampler_unc = None
    if need_unconstrained:
        sampler_unc = _build_sampler(
            ddpm_cfg, project_root, device,
            inverse_beta=args.inverse_beta,
            energy_mc_samples=args.energy_mc_samples,
            num_channel_realizations=args.num_channel_realizations,
            unconstrained=True,
            r_min_override=args.r_min_override,
        )

    # Vanilla DDIM from trained checkpoint
    sampler_ddim = None
    if "ddim" in methods and args.ddim_checkpoint_dir:
        sampler_ddim = _build_vanilla_ddim(args.ddim_checkpoint_dir, device, checkpoint_epoch=args.ddim_epoch)
        print(f"Loaded vanilla DDIM: {args.ddim_checkpoint_dir}")
        print(f"  sampling_timesteps={sampler_ddim.sampling_timesteps}, eta={sampler_ddim.ddim_eta}")

    # Trained PDI nets
    pdi_net_models = {}
    if args.pdi_net_checkpoints:
        from train_energy_score import _build_portfolio_backbone, _sparse_to_dense_adj
        from pdi.trainers.energy_score.score_net import ScoreNetWithLambda
        import types as _pdi_types
        for ckpt_path in args.pdi_net_checkpoints:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            label = Path(ckpt_path).parent.name.split(" - ", 1)[-1] if " - " in Path(ckpt_path).parent.name else Path(ckpt_path).stem
            net = _build_portfolio_backbone(
                200, device, model_cond_channels=3,
                hidden=args.pdi_net_hidden, num_layers=args.pdi_net_layers,
                K=args.pdi_net_K,
            )
            _orig_fwd = net.forward
            def _make_dense_fwd(_orig):
                def _dense_adj_forward(self, x, timesteps, dual_lambda, edge_index,
                                       edge_weight=None, cond=None, return_intermediates=False):
                    B, _, N, _ = x.shape
                    if edge_index.dim() == 2:
                        edge_index = _sparse_to_dense_adj(edge_index, edge_weight, N, B)
                        edge_weight = None
                    return _orig(x=x, timesteps=timesteps, dual_lambda=dual_lambda,
                                 edge_index=edge_index, edge_weight=edge_weight,
                                 cond=cond, return_intermediates=return_intermediates)
                return _dense_adj_forward
            net.forward = _pdi_types.MethodType(_make_dense_fwd(_orig_fwd), net)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            method_name = f"pdi_net_{label}"
            if method_name not in methods:
                methods.append(method_name)
            pdi_net_models[method_name] = net
            print(f"Loaded PDI net '{method_name}' from {ckpt_path}")
            print(f"  outer={ckpt.get('outer')}, metrics={ckpt.get('metrics', {})}")

    # Dual-trained score nets (no λ conditioning)
    if args.dt_checkpoints:
        from train_energy_score import _sparse_to_dense_adj
        from pdi.models.portfolio_gnn_backbone import PortfolioGNNBackbone
        import types as _dt_types
        for ckpt_path in args.dt_checkpoints:
            ckpt_path = Path(ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            train_args = {}
            if "args" in ckpt and isinstance(ckpt["args"], dict):
                train_args = ckpt["args"]
            else:
                for candidate in [ckpt_path.parent / "args.json",
                                  ckpt_path.parent.parent / "args.json"]:
                    if candidate.exists():
                        with open(candidate) as f:
                            train_args = json.load(f)
                        break
            hidden = args.dt_hidden or int(train_args.get("hidden", 128))
            num_layers = args.dt_layers or int(train_args.get("num_layers", 6))
            gnn_K = args.dt_K or int(train_args.get("gnn_K", 2))
            num_nodes = int(train_args.get("num_nodes", 200))
            backbone = PortfolioGNNBackbone(
                d=num_nodes, hidden=hidden, num_layers=num_layers,
                num_timesteps=500, cond_channels=3, K=gnn_K,
            )
            backbone.load_state_dict(ckpt["state_dict"])
            _orig_fwd = backbone.forward
            def _make_dt_fwd(_orig):
                def _dense_adj_forward(self, x, timesteps, edge_index,
                                       edge_weight=None, cond=None,
                                       return_intermediates=False, dual_lambda=None):
                    B, _, N, _ = x.shape
                    if edge_index.dim() == 2:
                        edge_index = _sparse_to_dense_adj(edge_index, edge_weight, N, B)
                        edge_weight = None
                    return _orig(x=x, timesteps=timesteps,
                                 edge_index=edge_index, edge_weight=edge_weight,
                                 cond=cond, return_intermediates=return_intermediates)
                return _dense_adj_forward
            backbone.forward = _dt_types.MethodType(_make_dt_fwd(_orig_fwd), backbone)
            backbone.to(device).eval()
            label = ckpt_path.stem
            method_name = f"dt_net_{label}"
            if method_name not in methods:
                methods.append(method_name)
            pdi_net_models[method_name] = backbone
            print(f"Loaded DT net '{method_name}' from {ckpt_path}")
            print(f"  hidden={hidden}, layers={num_layers}, K={gnn_K}")
            print(f"  outer={ckpt.get('outer')}, metrics={ckpt.get('metrics', {})}")

    # PD Expert model (forward pass through multiple checkpoints → diverse policies)
    pd_expert_checkpoint_dir = None
    pd_expert_model = None
    if args.pd_expert_checkpoint:
        from pdi.models.power_allocation_gnn import PowerAllocationGNN
        pd_expert_checkpoint_dir = Path(args.pd_expert_checkpoint)
        ckpt_files = sorted(pd_expert_checkpoint_dir.glob("sample_checkpoint_epoch_*.pt"),
                            key=lambda p: int(p.stem.split("_")[-1]))
        if not ckpt_files:
            raise RuntimeError(f"No sample_checkpoint_epoch_*.pt found in {pd_expert_checkpoint_dir}")
        # Build model from first checkpoint to get architecture
        first_ckpt = torch.load(ckpt_files[0], map_location=device, weights_only=False)
        pd_expert_model = PowerAllocationGNN(
            input_dim=2,
            hidden_dim=args.pd_expert_hidden,
            num_layers=args.pd_expert_layers,
            K=args.pd_expert_K,
            P_max=1.0,
        ).to(device)
        pd_expert_model.eval()
        if "pd_expert" not in methods:
            methods.append("pd_expert")
        print(f"PD Expert: {len(ckpt_files)} checkpoints in {pd_expert_checkpoint_dir}")
        print(f"  params={sum(p.numel() for p in pd_expert_model.parameters()):,}")
        print(f"  epochs: {int(ckpt_files[0].stem.split('_')[-1])} to {int(ckpt_files[-1].stem.split('_')[-1])}")

    # Any sampler for context building and channel loading
    any_sampler = sampler_pdi or sampler_unc or _build_sampler(
        ddpm_cfg, project_root, device,
        energy_mc_samples=args.energy_mc_samples,
        num_channel_realizations=args.num_channel_realizations,
        r_min_override=args.r_min_override,
        unconstrained=True,
    )

    # Install chunked score estimation if needed
    if args.chunk_size > 0:
        if sampler_pdi is not None:
            _install_chunked_score(sampler_pdi, args.chunk_size)
        if sampler_unc is not None:
            _install_chunked_score(sampler_unc, args.chunk_size)

    cfg = _load_dataset_cfg(
        project_root, args.batch_size_val, args.n_samples_per_input,
        dataset_config=args.dataset)
    if "ddim" in methods and args.ddim_checkpoint_dir:
        cfg.model_cond_channels = 3
    builder = WRABuilder()
    datasets = builder.build_datasets(cfg)
    loaders = builder.build_loaders(cfg, datasets)
    test_loader = loaders["test"]

    num_batches = min(len(test_loader), args.max_batches or len(test_loader))
    print(f"Running {methods} on {num_batches} batch(es), "
          f"K={args.n_samples_per_input} samples/network")
    print(f"PDI: ib={args.inverse_beta}, α={args.dual_step_size}, "
          f"λ₀={args.dual_lambda_init}, K_MC={args.energy_mc_samples}, "
          f"R={args.num_channel_realizations}")

    _scorer_tag = "Net" if pdi_net_models else "MC"

    # Collect per-user rates across all batches
    all_rates: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_metrics: Dict[str, List[Dict]] = defaultdict(list)

    if args.studies_only:
        # Jump straight to studies — need r_min and context from first batch
        _first_b = next(iter(test_loader)).to(device)
        _B_f = int(_first_b.num_graphs)
        _N_f = int(_first_b.y.size(0) // _B_f)
        context = any_sampler._build_energy_context(data=_first_b, batch_size=_B_f, num_nodes=_N_f, device=device, dtype=torch.float32)
        r_min = float(context.r_min[0])
        method_rates_flat = {}
        method_agg = {}
        print("Skipping main methods loop (--studies-only)")

    for batch_idx, batch in enumerate(test_loader):
        if args.studies_only:
            break
        if batch_idx >= num_batches:
            break
        batch = batch.to(device)
        B = int(batch.num_graphs)  # already includes K replication from loader
        N = int(batch.y.size(0) // B)
        T_seq = int(batch.y.size(1))
        F = int(batch.y.size(2))
        K = args.n_samples_per_input
        shape = (B, T_seq, N, F)

        ds_names = getattr(batch, "dataset_name", None)
        if isinstance(ds_names, (list, tuple)):
            ds_name = str(ds_names[0])
        else:
            ds_name = str(ds_names) if ds_names else "unknown"
        net_ids = getattr(batch, "network_id", None)
        if isinstance(net_ids, torch.Tensor):
            net_id = int(net_ids[0])
        elif isinstance(net_ids, (list, tuple)):
            net_id = int(net_ids[0])
        else:
            net_id = batch_idx
        unique_nets = max(1, B // K)

        print(f"\n[Batch {batch_idx+1}/{num_batches}] "
              f"B={B} ({unique_nets} net × K={K}), N={N}, ds={ds_name}, net={net_id}")

        x_pdi = x_pdl = x_pdm = x_dps = None

        # Build context with a capped R to avoid OOM. The full R=500 evaluation
        # happens inside _evaluate_method which loads gains separately.
        _orig_R = any_sampler.energy_num_channel_realizations
        R_ctx = min(_orig_R, 50)  # cap context R to 50 for memory
        any_sampler.energy_num_channel_realizations = R_ctx
        context = any_sampler._build_energy_context(
            data=batch, batch_size=B, num_nodes=N,
            device=device, dtype=torch.float32,
        )
        any_sampler.energy_num_channel_realizations = _orig_R

        # --- PDI ---
        if "pdi" in methods:
            torch.manual_seed(args.seed + batch_idx)
            sb = args.sub_batch
            _orig_dual_step = None
            if sb > 0 and B >= sb:
                n_sub = B // sb
                _orig_dual_step = sampler_pdi._dual_ascent_step
                _sampler_ref = sampler_pdi

                def _sub_batch_dual(dual_lambda, ergodic_rates, context, t=None,
                                    _nsub=n_sub, _sb=sb, _s=_sampler_ref):
                    if t is None:
                        step = _s.dual_step_size_by_t[0].to(
                            device=dual_lambda.device, dtype=dual_lambda.dtype)
                    else:
                        t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(
                            0, _s.num_timesteps - 1)
                        step = _s._gather(
                            _s.dual_step_size_by_t.to(device=dual_lambda.device), t_safe
                        ).to(dtype=dual_lambda.dtype)[0]
                    B_loc, N_loc = dual_lambda.shape
                    violation = context.r_min.view(-1, 1) - ergodic_rates
                    mean_viol = violation.view(_nsub, _sb, N_loc).mean(dim=1, keepdim=True)
                    lam_rep = dual_lambda.view(_nsub, _sb, N_loc)[:, :1, :]
                    updated = (lam_rep + step * mean_viol).clamp_min(0.0)
                    if _s.dual_lambda_max is not None:
                        updated = updated.clamp_max(float(_s.dual_lambda_max))
                    return updated.expand(_nsub, _sb, N_loc).reshape(B_loc, N_loc).contiguous()

                sampler_pdi._dual_ascent_step = _sub_batch_dual
                sampler_pdi.dual_lambda_mode = "per_sample"

            t0 = time.time()
            pdi_result = sampler_pdi.sample(
                shape=shape, device=device, data=batch,
                n_samples_per_input=K,
            )
            x_pdi = pdi_result[0] if isinstance(pdi_result, tuple) else pdi_result
            dt = time.time() - t0

            if _orig_dual_step is not None:
                sampler_pdi._dual_ascent_step = _orig_dual_step
                sampler_pdi.dual_lambda_mode = "shared_per_network"
            _final_lam = getattr(sampler_pdi, '_last_dual_lambda', None)
            m = _evaluate_method(
                x_pdi, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots, dual_lambda=_final_lam)
            m["wall_time"] = dt
            all_rates["pdi"].append(m["per_user_rates"])
            all_metrics["pdi"].append(m)
            print(f"  PDI: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%}, cs={m['cs']:.3f} ({dt:.1f}s)")

        # --- Unconstrained PDI (λ=0 always) ---
        if "pdi_unc" in methods:
            torch.manual_seed(args.seed + batch_idx)
            if sampler_unc is None:
                sampler_unc = _build_sampler(
                    ddpm_cfg, project_root, device,
                    inverse_beta=args.inverse_beta,
                    energy_mc_samples=args.energy_mc_samples,
                    num_channel_realizations=args.num_channel_realizations,
                    r_min_override=args.r_min_override,
                    unconstrained=True,
                )
                if args.chunk_size > 0:
                    _install_chunked_score(sampler_unc, args.chunk_size)
            t0 = time.time()
            unc_result = sampler_unc.sample(
                shape=shape, device=device, data=batch,
                n_samples_per_input=K,
            )
            x_unc = unc_result[0] if isinstance(unc_result, tuple) else unc_result
            dt = time.time() - t0
            _final_lam_unc = getattr(sampler_unc, '_last_dual_lambda', None)
            m = _evaluate_method(
                x_unc, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots, dual_lambda=_final_lam_unc)
            m["wall_time"] = dt
            all_rates["pdi_unc"].append(m["per_user_rates"])
            all_metrics["pdi_unc"].append(m)
            print(f"  PDI-Unc: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- PDL ---
        if "pdl" in methods:
            torch.manual_seed(args.seed + batch_idx)
            t0 = time.time()
            x_pdl, diag = pdl_wra(
                context, shape, device,
                num_iters=args.pdl_num_iters,
                primal_lr=args.pdl_primal_lr,
                dual_lr=args.pdl_dual_lr,
                noise_scale=args.pdl_noise_scale,
                lambda_init=args.pdl_lambda_init,
                shared_lambda=False,
                seed=args.seed + batch_idx,
            )
            dt = time.time() - t0
            m = _evaluate_method(
                x_pdl, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots,
                dual_lambda=diag["final_lambda"])
            m["wall_time"] = dt
            all_rates["pdl"].append(m["per_user_rates"])
            all_metrics["pdl"].append(m)
            print(f"  PDL: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- Install net-based score (λ=0) for PDM/DPS when a trained net is available ---
        _net_patched_sampler = None
        _net_patch_orig_score = None
        if pdi_net_models and ("pdm" in methods or "dps" in methods):
            _bl_net = list(pdi_net_models.values())[0]
            _bl_sampler = sampler_pdi if sampler_pdi is not None else (sampler_unc if sampler_unc is not None else any_sampler)
            _bl_acp = _bl_sampler.alphas_cumprod.to(device)
            _net_patch_orig_score = _bl_sampler._estimate_score
            def _make_net_lam0(_net, _acp, _batch_ref):
                def _score_fn(self, x_t, t, context, dual_lambda=None,
                              inverse_beta_override=None):
                    B, N_loc = x_t.shape[0], x_t.shape[2]
                    dl = torch.zeros(B, N_loc, device=x_t.device, dtype=x_t.dtype)
                    data = _batch_ref
                    cond = data.x.view(B, N_loc, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
                    ei = data.edge_index if hasattr(data, "edge_index") else None
                    ew = data.edge_weight if hasattr(data, "edge_weight") else None
                    with torch.no_grad():
                        eps_pred = _net(x=x_t, timesteps=t, dual_lambda=dl,
                                        edge_index=ei, edge_weight=ew,
                                        cond=cond, return_intermediates=False)
                    ab = _acp[t].view(-1, 1, 1, 1)
                    return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))
                return _score_fn
            _bl_sampler._estimate_score = _types.MethodType(
                _make_net_lam0(_bl_net, _bl_acp, batch), _bl_sampler)
            _net_patched_sampler = _bl_sampler

        # --- PDM ---
        if "pdm" in methods:
            torch.manual_seed(args.seed + batch_idx)
            t0 = time.time()
            pdm_sampler = _net_patched_sampler or sampler_unc
            x_pdm, diag = pdm_wra(
                pdm_sampler, shape, device, batch, context,
                projection_iters=args.pdm_iters,
                projection_lr=args.pdm_lr,
                projection_rho=args.pdm_rho,
                seed=args.seed + batch_idx,
            )
            x_pdm = _final_project_wra(
                x_pdm, any_sampler, ds_name, net_id, N, device,
                p_max=float(context.p_max[0, 0, 0]),
                noise_var=float(context.noise_var[0, 0, 0]),
                r_min=float(context.r_min[0]),
                num_iters=args.pdm_iters,
                lr=args.pdm_lr,
                rho=args.pdm_rho,
                max_timeslots=args.eval_timeslots,
            )
            dt = time.time() - t0
            m = _evaluate_method(
                x_pdm, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots)
            m["wall_time"] = dt
            all_rates["pdm"].append(m["per_user_rates"])
            all_metrics["pdm"].append(m)
            print(f"  PDM: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- DPS ---
        if "dps" in methods:
            torch.manual_seed(args.seed + batch_idx)
            t0 = time.time()
            dps_sampler = _net_patched_sampler or sampler_unc
            x_dps, diag = dps_wra(
                dps_sampler, shape, device, batch, context,
                dps_scale=args.dps_scale,
                n_samples_per_input=K,
                seed=args.seed + batch_idx,
            )
            dt = time.time() - t0
            m = _evaluate_method(
                x_dps, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots)
            m["wall_time"] = dt
            all_rates["dps"].append(m["per_user_rates"])
            all_metrics["dps"].append(m)
            print(f"  DPS: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- Restore original score if patched ---
        if _net_patched_sampler is not None and _net_patch_orig_score is not None:
            _net_patched_sampler._estimate_score = _net_patch_orig_score

        # --- PD Train (dataset samples) ---
        if "pd_train" in methods:
            x_pd_train = batch.y.view(B, T_seq, N, F).to(device)
            m = _evaluate_method(
                x_pd_train, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots)
            all_rates["pd_train"].append(m["per_user_rates"])
            all_metrics["pd_train"].append(m)
            print(f"  PD-Train: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%}")

        # --- PD Expert (forward pass through multiple checkpoints → K policies) ---
        if "pd_expert" in methods and pd_expert_model is not None:
            t0 = time.time()
            P_max = float(context.p_max[0, 0, 0])
            pd_expert_model.P_max = P_max
            ckpt_files = sorted(pd_expert_checkpoint_dir.glob("sample_checkpoint_epoch_*.pt"),
                                key=lambda p: int(p.stem.split("_")[-1]))
            # Use up to K checkpoints (one policy per checkpoint)
            ckpts_to_use = ckpt_files[-K:] if len(ckpt_files) >= K else ckpt_files
            policies = []
            # Rebuild node features from H_ls² (power) to match PD training
            # Diffusion dataset stores H_ls as amplitude; PD training uses H_l = amplitude²
            H_raw_pd, assoc_pd = _load_raw_channel_gains(
                any_sampler, ds_name, net_id, N, device, torch.float32)
            graph_path = Path(any_sampler._get_network_channel_data(ds_name, int(net_id)).h_timeslot_paths[0]).parent.parent / "graph.pt"
            gdata = torch.load(graph_path, map_location=device, weights_only=False)
            H_ls_amp = gdata["H_ls"].cpu().numpy()  # amplitude
            H_ls_power = H_ls_amp ** 2  # convert to power (matches PD training)
            assoc_np = gdata["associations"].cpu().numpy()
            from pdi.utils.graph_builder import build_interference_graph as _big
            pd_graph = _big(H_ls=H_ls_power, associations=assoc_np, top_k=10,
                            normalize_method='log1p', spectral_normalize_edge_weights=False)
            pd_x = pd_graph.x.to(device)  # (N, 2) with power-scale features
            pd_ei = pd_graph.edge_index.to(device)
            pd_ew = pd_graph.edge_weight.to(device) if pd_graph.edge_weight is not None else None
            pd_batch_idx = torch.zeros(N, dtype=torch.long, device=device)
            with torch.no_grad():
                for cp in ckpts_to_use:
                    ckpt_data = torch.load(cp, map_location=device, weights_only=False)
                    pd_expert_model.load_state_dict(ckpt_data["model_state_dict"])
                    if len(policies) == 0:
                        print(f"    [DBG] model.P_max={pd_expert_model.P_max}, pd_x shape={pd_x.shape}, edge_index shape={batch.edge_index.shape}")
                        print(f"    [DBG] loading checkpoint: {cp.name}")
                        print(f"    [DBG] ckpts_to_use: {len(ckpts_to_use)} files, first={ckpts_to_use[0].name}, last={ckpts_to_use[-1].name}")
                        w = list(pd_expert_model.parameters())[0]
                        print(f"    [DBG] first param: shape={w.shape}, sum={w.sum().item():.6f}")
                        print(f"    [DBG] pd_x[:5, :] = {pd_x[:5]}")
                    power_per_receiver = pd_expert_model(
                        x=pd_x, edge_index=pd_ei,
                        edge_weight=pd_ew, batch=pd_batch_idx,
                    )
                    if len(policies) == 0:
                        print(f"    [DBG] power_per_receiver: shape={power_per_receiver.shape}, range=[{power_per_receiver.min():.6f}, {power_per_receiver.max():.6f}]")
                    # Single graph forward — power_per_receiver is (N,)
                    power_rx = power_per_receiver  # (N,) actual power [0, P_max]
                    # Normalize to [-0.5, 0.5] (what _evaluate_method expects)
                    power_rx_norm = power_rx / P_max - 0.5
                    policies.append(power_rx_norm)
            # Stack normalized receiver powers → (K_used, N)
            power_allocs = torch.stack(policies, dim=0)  # (K, N) normalized [-0.5, 0.5]
            dt = time.time() - t0
            # Evaluate using _compute_full_ergodic_evolution directly
            H_raw, associations_pd = _load_raw_channel_gains(
                any_sampler, ds_name, net_id, N, device, power_allocs.dtype)
            H_raw = _resize_h(H_raw, args.eval_timeslots)
            print(f"    [DBG] power_allocs: shape={power_allocs.shape} range=[{power_allocs.min():.4f}, {power_allocs.max():.4f}]")
            print(f"    [DBG] H_raw: shape={H_raw.shape}, P_max={P_max}, noise_var={float(context.noise_var[0,0,0]):.2e}")
            power_rx_check = ((power_allocs + 0.5) * P_max).clamp_min(0.0)
            print(f"    [DBG] power_rx after denorm: range=[{power_rx_check.min():.6f}, {power_rx_check.max():.6f}]")
            _, erg_evo = _compute_full_ergodic_evolution(
                power_allocs, H_raw, associations_pd,
                P_max, float(context.noise_var[0, 0, 0]),
                num_trials=args.num_evolution_trials,
            )
            final_erg = erg_evo[:, -1, :].mean(dim=0).cpu().numpy()
            r_min = float(context.r_min[0])
            m = {
                "mean_rate": float(final_erg.mean()),
                "p5_rate": float(np.percentile(final_erg, 5)),
                "p1_rate": float(np.percentile(final_erg, 1)),
                "min_rate": float(final_erg.min()),
                "feasibility": float((final_erg >= r_min - 0.01).mean()),
                "per_user_rates": final_erg,
                "power_entropy": 0.0,
            }
            m["wall_time"] = dt
            all_rates["pd_expert"].append(m["per_user_rates"])
            all_metrics["pd_expert"].append(m)
            print(f"  PD-Expert ({len(policies)} ckpts): mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- Vanilla DDIM (trained model) ---
        x_ddim = None
        if "ddim" in methods and sampler_ddim is not None:
            torch.manual_seed(args.seed + batch_idx)
            t0 = time.time()
            with torch.no_grad():
                res_ddim = sampler_ddim.sample(shape=shape, device=device, data=batch)
                x_ddim = res_ddim[0] if isinstance(res_ddim, tuple) else res_ddim
            dt = time.time() - t0
            m = _evaluate_method(
                x_ddim, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots)
            m["wall_time"] = dt
            all_rates["ddim"].append(m["per_user_rates"])
            all_metrics["ddim"].append(m)
            print(f"  DDIM: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- Trained PDI nets ---
        net_samples = {}
        for mname, score_net in pdi_net_models.items():
            torch.manual_seed(args.seed + batch_idx)
            import types as _mn_types
            alphas_cumprod = sampler_pdi.alphas_cumprod.to(device) if sampler_pdi else any_sampler.alphas_cumprod.to(device)
            _sampler_for_net = sampler_pdi if sampler_pdi is not None else any_sampler

            def _make_trained_score(_net, _acp, _batch_ref):
                def _trained_estimate_score(self, x_t, t, context, dual_lambda=None,
                                            inverse_beta_override=None):
                    B = x_t.shape[0]
                    N_loc = x_t.shape[2]
                    if dual_lambda is None:
                        dual_lambda = torch.zeros(B, N_loc, device=x_t.device, dtype=x_t.dtype)
                    data = _batch_ref
                    cond = data.x.view(B, N_loc, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
                    edge_index = data.edge_index if hasattr(data, "edge_index") else None
                    edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
                    with torch.no_grad():
                        eps_pred = _net(
                            x=x_t, timesteps=t, dual_lambda=dual_lambda,
                            edge_index=edge_index, edge_weight=edge_weight,
                            cond=cond, return_intermediates=False,
                        )
                    ab = _acp[t].view(-1, 1, 1, 1)
                    return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))
                return _trained_estimate_score

            orig_score = _sampler_for_net._estimate_score
            _sampler_for_net._estimate_score = _mn_types.MethodType(
                _make_trained_score(score_net, alphas_cumprod, batch), _sampler_for_net)

            is_dt = mname.startswith("dt_net_")

            if is_dt:
                # DT nets: plain diffusion sampling — no dual updates
                t0 = time.time()
                context_net = _sampler_for_net._build_energy_context(
                    data=batch, batch_size=B, num_nodes=N, device=device, dtype=torch.float32)
                x_init = torch.randn(shape, device=device, dtype=torch.float32)
                dual_lambda_zero = torch.zeros(B, N, device=device, dtype=torch.float32)
                x_net, _, _ = _sampler_for_net._run_single_backward_pass(
                    x_init=x_init,
                    context=context_net,
                    dual_lambda=dual_lambda_zero,
                    n_samples_per_input=K,
                    return_diffusion_rate_trace=False,
                    update_dual_each_timestep=False,
                    show_progress=True,
                    progress_desc=f"DT Sampling ({mname})",
                )
                dt = time.time() - t0
            else:
                # PDI nets: full primal-dual sampling with dual ascent
                sb = args.sub_batch
                _orig_dual_step_net = None
                if sb > 0 and B >= sb:
                    n_sub = B // sb
                    _orig_dual_step_net = _sampler_for_net._dual_ascent_step
                    _s_ref = _sampler_for_net
                    def _sub_batch_dual_net(dual_lambda, ergodic_rates, context, t=None,
                                            _nsub=n_sub, _sb=sb, _s=_s_ref):
                        if t is None:
                            step = _s.dual_step_size_by_t[0].to(device=dual_lambda.device, dtype=dual_lambda.dtype)
                        else:
                            t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(0, _s.num_timesteps - 1)
                            step = _s._gather(_s.dual_step_size_by_t.to(device=dual_lambda.device), t_safe).to(dtype=dual_lambda.dtype)[0]
                        B_loc, N_loc = dual_lambda.shape
                        violation = context.r_min.view(-1, 1) - ergodic_rates
                        mean_viol = violation.view(_nsub, _sb, N_loc).mean(dim=1, keepdim=True)
                        lam_rep = dual_lambda.view(_nsub, _sb, N_loc)[:, :1, :]
                        updated = (lam_rep + step * mean_viol).clamp_min(0.0)
                        if _s.dual_lambda_max is not None:
                            updated = updated.clamp_max(float(_s.dual_lambda_max))
                        return updated.expand(_nsub, _sb, N_loc).reshape(B_loc, N_loc).contiguous()
                    _sampler_for_net._dual_ascent_step = _sub_batch_dual_net
                    _sampler_for_net.dual_lambda_mode = "per_sample"

                t0 = time.time()
                res_net = _sampler_for_net.sample(shape=shape, device=device, data=batch,
                                                  n_samples_per_input=K)
                x_net = res_net[0] if isinstance(res_net, tuple) else res_net
                dt = time.time() - t0

                if _orig_dual_step_net is not None:
                    _sampler_for_net._dual_ascent_step = _orig_dual_step_net
                    _sampler_for_net.dual_lambda_mode = "shared_per_network"

            _sampler_for_net._estimate_score = orig_score

            _final_lam_net = getattr(_sampler_for_net, '_last_dual_lambda', None)
            m = _evaluate_method(
                x_net, context, any_sampler, ds_name, net_id, N,
                K, device, args.num_evolution_trials,
                max_timeslots=args.eval_timeslots,
                dual_lambda=_final_lam_net if not is_dt else None)
            m["wall_time"] = dt
            all_rates[mname].append(m["per_user_rates"])
            all_metrics[mname].append(m)
            net_samples[mname] = x_net
            print(f"  {mname}: mean={m['mean_rate']:.4f}, p5={m['p5_rate']:.4f}, "
                  f"feas={m['feasibility']:.1%} ({dt:.1f}s)")

        # --- Power scatter comparison for this batch ---
        batch_samples = {}
        for mname, mvar in [("pdi", x_pdi), ("pdl", x_pdl),
                            ("pdm", x_pdm), ("dps", x_dps),
                            ("ddim", x_ddim),
                            ("pd_train", x_pd_train if "pd_train" in methods else None)]:
            if mname in methods and mvar is not None:
                batch_samples[mname] = mvar
        # Add pd_expert samples (normalized rx power → [K, 1, N, 1])
        if "pd_expert" in methods and pd_expert_model is not None and 'power_allocs' in dir():
            x_pd_expert = power_allocs.unsqueeze(1).unsqueeze(-1)
            batch_samples["pd_expert"] = x_pd_expert
        # Add pdi_net / dt_net samples
        for mname, x_stored in net_samples.items():
            batch_samples[mname] = x_stored
        if batch_samples:
            _plot_power_scatter_comparison(
                batch_samples,
                p_max=float(context.p_max[0, 0, 0]),
                num_nodes=N,
                network_id=net_id,
                output_dir=output_dir,
                seed=args.seed,
            )


        # --- Lambda study (inline, using first pdi-net via _sample_with_trace) ---
        if args.lambda_study and pdi_net_models and sampler_pdi is not None:
            import types as _lam_types
            _lam_net = list(pdi_net_models.values())[0]
            _lam_acp = sampler_pdi.alphas_cumprod.to(device)
            def _make_lam_score_fn(_net, _acp, _batch_ref):
                def _fn(self, x_t, t, context, dual_lambda=None, inverse_beta_override=None):
                    B_l = x_t.shape[0]; N_l = x_t.shape[2]
                    if dual_lambda is None:
                        dual_lambda = torch.zeros(B_l, N_l, device=x_t.device, dtype=x_t.dtype)
                    data = _batch_ref
                    cond = data.x.view(B_l, N_l, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
                    ei = data.edge_index if hasattr(data, "edge_index") else None
                    ew = data.edge_weight if hasattr(data, "edge_weight") else None
                    with torch.no_grad():
                        eps = _net(x=x_t, timesteps=t, dual_lambda=dual_lambda,
                                   edge_index=ei, edge_weight=ew,
                                   cond=cond, return_intermediates=False)
                    ab = _acp[t].view(-1, 1, 1, 1)
                    return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))
                return _fn
            # Patch score, run traces, restore — only _estimate_score is touched
            _orig_score_ls = sampler_pdi._estimate_score
            sampler_pdi._estimate_score = _lam_types.MethodType(
                _make_lam_score_fn(_lam_net, _lam_acp, batch), sampler_pdi)
            # Reference trace
            torch.manual_seed(args.seed + batch_idx)
            _, tr_ref = _sample_with_trace(
                sampler_pdi, shape, device, batch, context,
                sub_batch=args.sub_batch, n_samples_per_input=K)
            lam_T = torch.tensor(tr_ref["lam_vectors"][-1], device=device, dtype=torch.float32)
            lam_avg = torch.tensor(tr_ref["lam_vectors"].mean(axis=0), device=device, dtype=torch.float32)
            # Fixed-λ variants via _sample_with_trace (avoids sampler.sample() state mutation)
            _orig_dual_ls = sampler_pdi._dual_ascent_step
            _orig_init_ls = sampler_pdi._init_dual_lambda
            _noop_dual_ls = lambda dual_lambda, ergodic_rates, context, t=None: dual_lambda
            for lname, lv in [(r"Fixed $\lambda_T$", lam_T),
                              (r"Fixed $\bar\lambda$", lam_avg)]:
                sampler_pdi._dual_ascent_step = _noop_dual_ls
                _lv = lv.clone()
                sampler_pdi._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=_lv: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
                torch.manual_seed(args.seed + batch_idx)
                x_lam, _ = _sample_with_trace(
                    sampler_pdi, shape, device, batch, context,
                    sub_batch=args.sub_batch, n_samples_per_input=K)
                m_lam = _evaluate_method(
                    x_lam, context, any_sampler, ds_name, net_id, N,
                    K, device, args.num_evolution_trials,
                    max_timeslots=args.eval_timeslots)
                all_rates[lname].append(m_lam["per_user_rates"])
                all_metrics[lname].append(m_lam)
                if lname not in methods:
                    methods.append(lname)
                print(f"  {lname}: mean={m_lam['mean_rate']:.4f}, "
                      f"p5={m_lam['p5_rate']:.4f}, feas={m_lam['feasibility']:.1%}")
            sampler_pdi._dual_ascent_step = _orig_dual_ls
            sampler_pdi._init_dual_lambda = _orig_init_ls
            sampler_pdi._estimate_score = _orig_score_ls

        # --- Schedule sweep (inline, using first pdi-net) ---
        if args.schedule_sweep and pdi_net_models and sampler_pdi is not None:
            import types as _sw_types
            _sw_net = list(pdi_net_models.values())[0]
            def _make_sw_score(_net, _batch_ref):
                def _fn(self, x_t, t, context, dual_lambda=None, inverse_beta_override=None):
                    B_s = x_t.shape[0]; N_s = x_t.shape[2]
                    if dual_lambda is None:
                        dual_lambda = torch.zeros(B_s, N_s, device=x_t.device, dtype=x_t.dtype)
                    data = _batch_ref
                    cond = data.x.view(B_s, N_s, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
                    ei = data.edge_index if hasattr(data, "edge_index") else None
                    ew = data.edge_weight if hasattr(data, "edge_weight") else None
                    with torch.no_grad():
                        eps = _net(x=x_t, timesteps=t, dual_lambda=dual_lambda,
                                   edge_index=ei, edge_weight=ew,
                                   cond=cond, return_intermediates=False)
                    ab = self.alphas_cumprod[t].view(-1, 1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
                    return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))
                return _fn
            _orig_score_sw = sampler_pdi._estimate_score
            sampler_pdi._estimate_score = _sw_types.MethodType(
                _make_sw_score(_sw_net, batch), sampler_pdi)
            T_sched = sampler_pdi.num_timesteps
            for sname in ["cosine", "ddpm_linear", "linear", "poly2"]:
                _override_schedule(sampler_pdi, _build_alphas_cumprod(sname, T_sched))
                torch.manual_seed(args.seed + batch_idx)
                x_sched, _ = _sample_with_trace(
                    sampler_pdi, shape, device, batch, context,
                    sub_batch=args.sub_batch, n_samples_per_input=K)
                m_sched = _evaluate_method(
                    x_sched, context, any_sampler, ds_name, net_id, N,
                    K, device, args.num_evolution_trials,
                    max_timeslots=args.eval_timeslots)
                sched_label = f"Sched: {sname}"
                all_rates[sched_label].append(m_sched["per_user_rates"])
                all_metrics[sched_label].append(m_sched)
                if sched_label not in methods:
                    methods.append(sched_label)
                print(f"  {sched_label}: mean={m_sched['mean_rate']:.4f}, "
                      f"p5={m_sched['p5_rate']:.4f}, feas={m_sched['feasibility']:.1%}")
            _override_schedule(sampler_pdi, _build_alphas_cumprod("cosine", T_sched))
            sampler_pdi._estimate_score = _orig_score_sw

        # --- Tail sweep (inline, using first pdi-net) ---
        if args.tail_sweep and pdi_net_models and sampler_pdi is not None:
            import types as _ts_types
            _ts_net = list(pdi_net_models.values())[0]
            _ts_acp = sampler_pdi.alphas_cumprod.to(device)
            def _make_ts_score(_net, _acp, _batch_ref):
                def _fn(self, x_t, t, context, dual_lambda=None, inverse_beta_override=None):
                    B_s = x_t.shape[0]; N_s = x_t.shape[2]
                    if dual_lambda is None:
                        dual_lambda = torch.zeros(B_s, N_s, device=x_t.device, dtype=x_t.dtype)
                    data = _batch_ref
                    cond = data.x.view(B_s, N_s, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
                    ei = data.edge_index if hasattr(data, "edge_index") else None
                    ew = data.edge_weight if hasattr(data, "edge_weight") else None
                    with torch.no_grad():
                        eps = _net(x=x_t, timesteps=t, dual_lambda=dual_lambda,
                                   edge_index=ei, edge_weight=ew,
                                   cond=cond, return_intermediates=False)
                    ab = _acp[t].view(-1, 1, 1, 1)
                    return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))
                return _fn
            _orig_score_ts = sampler_pdi._estimate_score
            sampler_pdi._estimate_score = _ts_types.MethodType(
                _make_ts_score(_ts_net, _ts_acp, batch), sampler_pdi)
            torch.manual_seed(args.seed + batch_idx)
            _, tr_tail_ref = _sample_with_trace(
                sampler_pdi, shape, device, batch, context,
                sub_batch=args.sub_batch, n_samples_per_input=K)
            lam_all_tail = tr_tail_ref["lam_vectors"]
            T_total = lam_all_tail.shape[0]
            _orig_dual_ts = sampler_pdi._dual_ascent_step
            _orig_init_ts = sampler_pdi._init_dual_lambda
            _noop_dual_ts = lambda dual_lambda, ergodic_rates, context, t=None: dual_lambda
            tail_fracs = [0.20, 0.40, 0.60, 0.80, 1.00]
            for frac in tail_fracs:
                start_idx = int(T_total * (1.0 - frac))
                lam_tail = torch.tensor(
                    lam_all_tail[start_idx:].mean(axis=0),
                    device=device, dtype=torch.float32)
                sampler_pdi._dual_ascent_step = _noop_dual_ts
                _lv_ts = lam_tail.clone()
                sampler_pdi._init_dual_lambda = (
                    lambda batch_size, num_nodes, *, device, dtype, init_value, _l=_lv_ts:
                    _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype))
                torch.manual_seed(args.seed + batch_idx)
                x_tail, _ = _sample_with_trace(
                    sampler_pdi, shape, device, batch, context,
                    sub_batch=args.sub_batch, n_samples_per_input=K)
                m_tail = _evaluate_method(
                    x_tail, context, any_sampler, ds_name, net_id, N,
                    K, device, args.num_evolution_trials,
                    max_timeslots=args.eval_timeslots)
                tail_label = f"Tail {int(frac*100)}\\%"
                all_rates[tail_label].append(m_tail["per_user_rates"])
                all_metrics[tail_label].append(m_tail)
                if tail_label not in methods:
                    methods.append(tail_label)
                print(f"  {tail_label}: mean={m_tail['mean_rate']:.4f}, "
                      f"p5={m_tail['p5_rate']:.4f}, feas={m_tail['feasibility']:.1%}")
            sampler_pdi._dual_ascent_step = _orig_dual_ts
            sampler_pdi._init_dual_lambda = _orig_init_ts
            sampler_pdi._estimate_score = _orig_score_ts

    # --- Trajectory divergence study ---
    if args.trajectory_divergence and pdi_net_models and dt_models:
        print("\n" + "=" * 60)
        print("Trajectory divergence: ||x_t^PDI - x_t^DT|| over diffusion time")
        print("=" * 60)
        import types as _td_types
        _td_pdi_net = list(pdi_net_models.values())[0]
        _td_sampler = sampler_pdi if sampler_pdi is not None else any_sampler
        _td_acp = _td_sampler.alphas_cumprod.to(device)
        T_steps = _td_sampler.num_timesteps

        for dt_name, _td_dt_net in dt_models.items():
            print(f"\n--- Trajectory divergence: PDI vs {dt_name} ---")
            all_dx_norms = []

            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= args.max_batches:
                    break
                batch = batch.to(device)
                B = int(batch.num_graphs)
                N = int(batch.y.size(0) // B)
                shape = (B, int(batch.y.size(1)), N, int(batch.y.size(2)))

                context = _td_sampler._build_energy_context(
                    data=batch, batch_size=B, num_nodes=N, device=device, dtype=torch.float32)
                dual_context = _td_sampler._build_dual_context(
                    data=batch, batch_size=B, num_nodes=N, device=device, dtype=torch.float32)
                dual_ctx = dual_context if dual_context is not None else context

                edge_index = batch.edge_index
                edge_weight = getattr(batch, 'edge_weight', None)
                A_dense = _sparse_to_dense_adj(edge_index, edge_weight, N, B)
                cond = batch.x.view(B, N, *batch.x.shape[1:]).swapaxes(1, 2) if batch.x is not None else None

                torch.manual_seed(args.seed + batch_idx)
                x_init = torch.randn(shape, device=device)
                noise_seq = [torch.randn(shape, device=device) for _ in range(T_steps)]

                def _run_coupled(score_net_fn, do_dual=True):
                    x_t = x_init.clone()
                    dl = _td_sampler._init_dual_lambda(
                        batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
                        init_value=float(_td_sampler.dual_lambda_init) if do_dual else 0.0)
                    x_trace = []
                    for t_int in reversed(range(T_steps)):
                        t = torch.full((B,), t_int, device=device, dtype=torch.long)
                        with torch.no_grad():
                            score = score_net_fn(x_t, t, dl)
                            x0_pred, _ = _td_sampler._score_to_x0_eps(x_t=x_t, t=t, score=score)
                        if do_dual:
                            erg, _ = _td_sampler._ergodic_rates_from_samples(x=x0_pred, context=dual_ctx)
                            dl = _td_sampler._dual_ascent_step(
                                dual_lambda=dl, ergodic_rates=erg, context=dual_ctx, t=t)
                        mean = _td_sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
                        if t_int > 0:
                            var = _td_sampler._gather(_td_sampler.posterior_variance, t).view(-1, 1, 1, 1)
                            x_t = mean + torch.sqrt(var) * noise_seq[t_int]
                        else:
                            x_t = mean
                        x_trace.append(x_t[:, 0, :, 0].cpu())
                    return x_trace

                def _pdi_score(x_t, t, dl):
                    eps = _td_pdi_net(x=x_t, timesteps=t, dual_lambda=dl,
                                      edge_index=A_dense[:x_t.shape[0]], edge_weight=None,
                                      cond=cond[:x_t.shape[0]])
                    if isinstance(eps, tuple): eps = eps[0]
                    ab = _td_acp[t].view(-1, 1, 1, 1)
                    return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))

                def _dt_score(x_t, t, dl):
                    dl_zero = torch.zeros_like(dl)
                    eps = _td_dt_net(x=x_t, timesteps=t, dual_lambda=dl_zero,
                                     edge_index=A_dense[:x_t.shape[0]], edge_weight=None,
                                     cond=cond[:x_t.shape[0]])
                    if isinstance(eps, tuple): eps = eps[0]
                    ab = _td_acp[t].view(-1, 1, 1, 1)
                    return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))

                tr_pdi = _run_coupled(_pdi_score, do_dual=True)
                tr_dt = _run_coupled(_dt_score, do_dual=False)

                dx_norms = []
                for ti in range(T_steps):
                    diff = tr_pdi[ti].float() - tr_dt[ti].float()
                    dx_norms.append(diff.reshape(diff.shape[0], -1).norm(dim=1).mean().item())
                all_dx_norms.append(dx_norms)

                print(f"  [{batch_idx+1}/{args.max_batches}] final ||Δx||={dx_norms[-1]:.3f}")
                del batch, context, x_init, noise_seq, tr_pdi, tr_dt
                torch.cuda.empty_cache()

            all_dx_norms = np.array(all_dx_norms)
            mean_dx = all_dx_norms.mean(axis=0)
            std_dx = all_dx_norms.std(axis=0)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "text.usetex": True, "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })

            fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
            steps = np.arange(T_steps)
            ax.plot(steps, mean_dx, color='#333333', linewidth=1.5)
            ax.fill_between(steps, mean_dx - std_dx, mean_dx + std_dx,
                            alpha=0.2, color='#333333')
            ax.set_xlabel("Diffusion step", fontsize=12)
            ax.set_ylabel(r"$\frac{1}{K}\sum_k\|\mathbf{x}_t^{\mathrm{PDI}}(k) - \mathbf{x}_t^{\mathrm{DT}}(k)\|$", fontsize=10)
            ax.grid(True, alpha=0.2, linewidth=0.3)
            ax.tick_params(labelsize=10)
            fig.tight_layout()

            dt_stem = dt_name.replace('/', '_')
            out_div = output_dir / f"trajectory_divergence_{dt_stem}.pdf"
            fig.savefig(str(out_div), bbox_inches='tight')
            fig.savefig(str(out_div).replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
            plt.close(fig)

            np.savez(output_dir / f"trajectory_divergence_{dt_stem}.npz",
                     mean=mean_dx, std=std_dx, all=all_dx_norms)
            print(f"Saved: {out_div}")

    # --- Aggregate and plot ---
    if not args.studies_only:
        print("\n" + "=" * 60)
        print("Aggregate results:")
    method_rates_flat = {}
    method_agg = {}
    if not args.studies_only:
        r_min = float(context.r_min[0])

    for method in methods:
        if method not in all_rates:
            continue
        flat = np.concatenate(all_rates[method])
        method_rates_flat[method] = flat
        agg = {
            "mean_rate": float(np.mean(flat)),
            "p5_rate": float(np.percentile(flat, 5)),
            "p1_rate": float(np.percentile(flat, 1)),
            "min_rate": float(np.min(flat)),
            "feasibility": float((flat >= r_min - 0.01).mean()),
        }
        # Compute entropy from stored power samples (per-network avg)
        ent_list = []
        for met_list in all_metrics.get(method, []):
            if "power_entropy" in met_list:
                ent_list.append(met_list["power_entropy"])
        agg["entropy"] = float(np.mean(ent_list)) if ent_list else 0.0
        method_agg[method] = agg
        print(f"  {method:6s}: mean={agg['mean_rate']:.4f}, "
              f"p5={agg['p5_rate']:.4f}, feas={agg['feasibility']:.1%}")

    # Figures
    if method_rates_flat:
        _plot_cdf_comparison(method_rates_flat, r_min, output_dir)
        _plot_bar_comparison(method_agg, output_dir, scorer_tag=_scorer_tag)

    # Save summary
    summary = {
        "methods": methods,
        "args": vars(args),
        "aggregate": {k: {kk: vv for kk, vv in v.items() if kk != "per_user_rates"}
                      for k, v in method_agg.items()},
        "r_min": r_min,
        "num_batches": num_batches,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_dir / 'summary.json'}")

    # Save LaTeX-ready table (with std across networks)
    if method_agg:
        labels_tab = {
            "pdi": "PDI (MC)", "pdi_unc": r"PDI (Unc.)", "pdl": "PDL",
            "pdm": f"PDM ({_scorer_tag})", "dps": f"DPS ({_scorer_tag})",
            "pd_train": "PD Expert", "ddim": "DDIM",
            "full_power": "Full Power",
        }
        # Compute per-network metrics for std
        method_per_net = {}
        for m in methods:
            if m not in all_rates or m not in method_rates_flat:
                continue
            net_means, net_p5s, net_p1s, net_feas, net_vios, net_sum_vios, net_css, net_ents = [], [], [], [], [], [], [], []
            for bi, net_rates in enumerate(all_rates[m]):
                net_means.append(float(np.mean(net_rates)))
                net_p5s.append(float(np.percentile(net_rates, 5)))
                net_p1s.append(float(np.percentile(net_rates, 1)))
                net_feas.append(float((net_rates >= r_min - 0.01).mean()))
                net_vios.append(float(np.maximum(r_min - net_rates, 0).mean()))
                net_sum_vios.append(float(np.maximum(r_min - net_rates, 0).sum()))
                if bi < len(all_metrics.get(m, [])):
                    net_css.append(all_metrics[m][bi].get("cs", 0.0))
                    net_ents.append(all_metrics[m][bi].get("power_entropy", 0.0))
            net_times = [mm.get("wall_time", 0.0) for mm in all_metrics.get(m, []) if "wall_time" in mm]
            method_per_net[m] = {
                "mean": (np.mean(net_means), np.std(net_means)),
                "p5": (np.mean(net_p5s), np.std(net_p5s)),
                "p1": (np.mean(net_p1s), np.std(net_p1s)),
                "feas": (np.mean(net_feas), np.std(net_feas)),
                "vio": (np.mean(net_vios), np.std(net_vios)),
                "sum_vio": (np.mean(net_sum_vios), np.std(net_sum_vios)),
                "cs": (np.mean(net_css), np.std(net_css)) if net_css else (0.0, 0.0),
                "ent": (np.mean(net_ents), np.std(net_ents)) if net_ents else (0.0, 0.0),
                "time": (np.mean(net_times), np.std(net_times)) if net_times else (0.0, 0.0),
            }

        # Table without std
        table_lines = [
            r"\begin{tabular}{lccccccccc}",
            r"\toprule",
            r"Method & Mean $(\uparrow)$ & p5 $(\uparrow)$ & p1 $(\uparrow)$ & Vio $(\downarrow)$ & Sum Vio $(\downarrow)$ & Feas $(\uparrow)$ & CS $(\downarrow)$ & Entropy $(\uparrow)$ & Time (s) \\",
            r"\midrule",
        ]
        for m in methods:
            if m not in method_agg:
                continue
            a = method_agg[m]
            flat = method_rates_flat[m]
            mean_vio = float(np.maximum(r_min - flat, 0).mean())
            sum_vio = float(np.maximum(r_min - flat, 0).sum())
            avg_cs = method_per_net[m]["cs"][0] if m in method_per_net else 0.0
            avg_time = method_per_net[m]["time"][0] if m in method_per_net else 0.0
            table_lines.append(
                f"{labels_tab.get(m, m)} & {a['mean_rate']:.3f} & {a['p5_rate']:.3f} "
                f"& {a['p1_rate']:.3f} & {mean_vio:.4f} & {sum_vio:.2f} & {a['feasibility']:.1%} "
                f"& {avg_cs:.2f} & {a['entropy']:.2f} & {avg_time:.1f} \\\\")
        table_lines += [r"\bottomrule", r"\end{tabular}"]
        table_str = "\n".join(table_lines)
        with open(output_dir / "results_table.tex", "w") as f:
            f.write(table_str)

        # Table with std
        table_std_lines = [
            r"\begin{tabular}{lccccccccc}",
            r"\toprule",
            r"Method & Mean $(\uparrow)$ & p5 $(\uparrow)$ & p1 $(\uparrow)$ & Vio $(\downarrow)$ & Sum Vio $(\downarrow)$ & Feas $(\uparrow)$ & CS $(\downarrow)$ & Entropy $(\uparrow)$ & Time (s) \\",
            r"\midrule",
        ]
        for m in methods:
            if m not in method_per_net:
                continue
            pn = method_per_net[m]
            def _pm(mu, sd):
                return f"${mu:.3f}" + r" {\scriptstyle\pm " + f"{sd:.3f}" + r"}$"
            def _pm4(mu, sd):
                return f"${mu:.4f}" + r" {\scriptstyle\pm " + f"{sd:.4f}" + r"}$"
            def _pm2(mu, sd):
                return f"${mu:.2f}" + r" {\scriptstyle\pm " + f"{sd:.2f}" + r"}$"
            def _pmp(mu, sd):
                return f"${100*mu:.1f}" + r" {\scriptstyle\pm " + f"{100*sd:.1f}" + r"}\%$"
            table_std_lines.append(
                f"{labels_tab.get(m, m)} & {_pm(*pn['mean'])} & {_pm(*pn['p5'])} "
                f"& {_pm(*pn['p1'])} & {_pm4(*pn['vio'])} & {_pm2(*pn['sum_vio'])} & {_pmp(*pn['feas'])} "
                f"& {_pm2(*pn['cs'])} & {_pm(*pn['ent'])} & {_pm(*pn['time'])} \\\\")
        table_std_lines += [r"\bottomrule", r"\end{tabular}"]
        table_std_str = "\n".join(table_std_lines)
        with open(output_dir / "results_table_std.tex", "w") as f:
            f.write(table_std_str)

        print(f"Saved table to {output_dir / 'results_table.tex'}")
        print(f"Saved table (with std) to {output_dir / 'results_table_std.tex'}")
        print(table_std_str)

    # --- Helper: run a sampler variant across all test networks for table ---
    def _run_variant_all_nets(variant_sampler, label, sb_override=None):
        """Run variant across all test batches, return aggregate metrics."""
        var_rates = []
        var_metrics = []
        sb_v = sb_override if sb_override is not None else args.sub_batch
        _orig_ds_v = None
        if sb_v > 0:
            _orig_ds_v = variant_sampler._dual_ascent_step
            n_sub_v = args.n_samples_per_input // sb_v
            def _sb_dual_v(dual_lambda, ergodic_rates, context, t=None,
                           _ns=n_sub_v, _sb=sb_v, _s=variant_sampler):
                if t is None:
                    step = _s.dual_step_size_by_t[0].to(device=dual_lambda.device, dtype=dual_lambda.dtype)
                else:
                    t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(0, _s.num_timesteps - 1)
                    step = _s._gather(_s.dual_step_size_by_t.to(device=dual_lambda.device), t_safe).to(dtype=dual_lambda.dtype)[0]
                B_l, N_l = dual_lambda.shape
                violation = context.r_min.view(-1, 1) - ergodic_rates
                mean_viol = violation.view(_ns, _sb, N_l).mean(dim=1, keepdim=True)
                lam_rep = dual_lambda.view(_ns, _sb, N_l)[:, :1, :]
                updated = (lam_rep + step * mean_viol).clamp_min(0.0)

                if _s.dual_lambda_max is not None:
                    updated = updated.clamp_max(float(_s.dual_lambda_max))
                return updated.expand(_ns, _sb, N_l).reshape(B_l, N_l).contiguous()
            variant_sampler._dual_ascent_step = _sb_dual_v
            variant_sampler.dual_lambda_mode = "per_sample"

        for bi, batch_v in enumerate(test_loader):
            if bi >= num_batches:
                break
            batch_v = batch_v.to(device)
            B_v = int(batch_v.num_graphs)
            N_v = int(batch_v.y.size(0) // B_v)
            shape_v = (B_v, int(batch_v.y.size(1)), N_v, int(batch_v.y.size(2)))
            ds_v = str(getattr(batch_v, "dataset_name", ["unknown"])[0]) if isinstance(getattr(batch_v, "dataset_name", None), (list, tuple)) else "unknown"
            nid_v = int(getattr(batch_v, "network_id", [bi])[0]) if isinstance(getattr(batch_v, "network_id", None), (list, tuple, torch.Tensor)) else bi
            torch.manual_seed(args.seed + bi)
            res_v = variant_sampler.sample(shape=shape_v, device=device, data=batch_v, n_samples_per_input=K)
            x_v = res_v[0] if isinstance(res_v, tuple) else res_v
            ctx_v = any_sampler._build_energy_context(data=batch_v, batch_size=B_v, num_nodes=N_v, device=device, dtype=torch.float32)
            m_v = _evaluate_method(x_v, ctx_v, any_sampler, ds_v, nid_v, N_v, K, device, args.num_evolution_trials)
            var_rates.append(m_v["per_user_rates"])
            var_metrics.append(m_v)

        if _orig_ds_v is not None:
            variant_sampler._dual_ascent_step = _orig_ds_v
            variant_sampler.dual_lambda_mode = "shared_per_network"

        if var_rates:
            flat = np.concatenate(var_rates)
            rate_sum = max(float(flat.sum()), 1e-12)
            p_v = np.clip(flat / rate_sum, 1e-12, None)
            ent = float(-np.sum(p_v * np.log(p_v)))
            if np.isnan(ent): ent = 0.0
            ent_avg = float(np.mean([m.get("power_entropy", 0) for m in var_metrics]))
            return {
                "mean_rate": float(np.mean(flat)),
                "p5_rate": float(np.percentile(flat, 5)),
                "p1_rate": float(np.percentile(flat, 1)),
                "min_rate": float(np.min(flat)),
                "feasibility": float((flat >= r_min - 0.01).mean()),
                "entropy": ent_avg,
                "per_user_rates": flat,
            }
        return None

    # --- Studies ---
    any_study = args.schedule_sweep or args.tail_sweep or args.lambda_study or args.temperature_scatter or args.trajectory_divergence
    if any_study and any_sampler is not None:
        study_sampler = sampler_pdi or _build_sampler(
            ddpm_cfg, project_root, device,
            dual_step_size=args.dual_step_size,
            dual_lambda_init=args.dual_lambda_init,
            inverse_beta=args.inverse_beta,
            energy_mc_samples=args.energy_mc_samples,
            num_channel_realizations=args.num_channel_realizations,
            r_min_override=args.r_min_override,
            dual_lambda_decay=args.dual_lambda_decay,
        )
        # Install trained PDI net on study_sampler if available
        if pdi_net_models:
            import types as _study_types
            _study_net = list(pdi_net_models.values())[0]  # use first loaded net
            _study_acp = study_sampler.alphas_cumprod.to(device)
            def _make_study_score(_net, _acp):
                def _fn(self, x_t, t, context, dual_lambda=None, inverse_beta_override=None):
                    B = x_t.shape[0]; N_loc = x_t.shape[2]
                    if dual_lambda is None:
                        dual_lambda = torch.zeros(B, N_loc, device=x_t.device, dtype=x_t.dtype)
                    data = self._current_study_data
                    cond = data.x.view(B, N_loc, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
                    edge_index = data.edge_index if hasattr(data, "edge_index") else None
                    edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
                    with torch.no_grad():
                        eps = _net(x=x_t, timesteps=t, dual_lambda=dual_lambda,
                                   edge_index=edge_index, edge_weight=edge_weight,
                                   cond=cond, return_intermediates=False)
                    ab = _acp[t].view(-1, 1, 1, 1)
                    return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))
                return _fn
            study_sampler._estimate_score = _study_types.MethodType(
                _make_study_score(_study_net, _study_acp), study_sampler)
            print("Studies using trained PDI-Net score estimation")

        # Reload first batch for studies
        first_batch = next(iter(test_loader)).to(device)
        B_study = int(first_batch.num_graphs)
        N_study = int(first_batch.y.size(0) // B_study)
        shape_study = (B_study, int(first_batch.y.size(1)), N_study, int(first_batch.y.size(2)))
        ctx_study = study_sampler._build_energy_context(
            data=first_batch, batch_size=B_study, num_nodes=N_study,
            device=device, dtype=torch.float32,
        )
        sb = args.sub_batch
        K_study = args.n_samples_per_input

        # Extract dataset_name and network_id for the first batch
        ds_names_study = getattr(first_batch, "dataset_name", None)
        study_ds = str(ds_names_study[0]) if isinstance(ds_names_study, (list, tuple)) else str(ds_names_study) if ds_names_study else "unknown"
        net_ids_study = getattr(first_batch, "network_id", None)
        if isinstance(net_ids_study, torch.Tensor):
            study_net = int(net_ids_study[0])
        elif isinstance(net_ids_study, (list, tuple)):
            study_net = int(net_ids_study[0])
        else:
            study_net = 0

        study_table_rows = {}

        # Full power and average power baselines (aggregated)
        print("\n[Baselines] Full power / Avg power...")
        for tag, fill_val in [("Full power", 0.5), ("Avg power", 0.0)]:
            bp_rates = []
            for bi, batch_bp in enumerate(test_loader):
                if bi >= num_batches: break
                batch_bp = batch_bp.to(device)
                B_bp = int(batch_bp.num_graphs)
                N_bp = int(batch_bp.y.size(0) // B_bp)
                shape_bp = (B_bp, int(batch_bp.y.size(1)), N_bp, int(batch_bp.y.size(2)))
                x_bp = torch.full(shape_bp, fill_val, device=device)
                ctx_bp = any_sampler._build_energy_context(data=batch_bp, batch_size=B_bp, num_nodes=N_bp, device=device, dtype=torch.float32)
                ds_bp = str(getattr(batch_bp, "dataset_name", ["unknown"])[0]) if isinstance(getattr(batch_bp, "dataset_name", None), (list, tuple)) else "unknown"
                nid_bp = int(getattr(batch_bp, "network_id", [bi])[0]) if isinstance(getattr(batch_bp, "network_id", None), (list, tuple, torch.Tensor)) else bi
                m_bp = _evaluate_method(x_bp, ctx_bp, any_sampler, ds_bp, nid_bp, N_bp, K_study, device, 5)
                bp_rates.append(m_bp["per_user_rates"])
            flat_bp = np.concatenate(bp_rates)
            study_table_rows[tag] = {
                "mean_rate": float(np.mean(flat_bp)), "p5_rate": float(np.percentile(flat_bp, 5)),
                "p1_rate": float(np.percentile(flat_bp, 1)), "feasibility": float((flat_bp >= r_min - 0.01).mean()),
                "per_user_rates": flat_bp, "power_entropy": 0.0,
            }
            print(f"  {tag}: mean={np.mean(flat_bp):.4f}, p5={np.percentile(flat_bp,5):.4f}")

        if args.schedule_sweep:
            print("\n" + "=" * 60)
            print("Schedule Sweep")
            # Plots from first network
            run_schedule_sweep(
                study_sampler, shape_study, device, first_batch,
                ctx_study, output_dir, seed=args.seed, sub_batch=sb,
                n_samples_per_input=K_study,
                dataset_name=study_ds, network_id=study_net,
                num_nodes=N_study)
            # Table: aggregate across all networks
            T_sched = study_sampler.num_timesteps
            for sn in ["ddpm_linear", "linear", "poly2"]:
                print(f"  [sched table] {sn} across all networks...")
                ac = _build_alphas_cumprod(sn, T_sched)
                _override_schedule(study_sampler, ac)
                sm = _run_variant_all_nets(study_sampler, sn)
                if sm: study_table_rows[f"Sched: {sn}"] = sm
            _override_schedule(study_sampler, _build_alphas_cumprod("cosine", T_sched))

        if args.tail_sweep:
            print("\n" + "=" * 60)
            print("Tail Sweep")
            # Plots from first network
            run_tail_sweep(
                study_sampler, shape_study, device, first_batch,
                ctx_study, output_dir, seed=args.seed, sub_batch=sb,
                n_samples_per_input=K_study,
                dataset_name=study_ds, network_id=study_net,
                num_nodes=N_study)
            # Table: aggregate tail variants across all networks
            # Per-network: compute reference trace on THAT network, then re-run with fixed tail lambda
            def _noop_d(dual_lambda, ergodic_rates, context, t=None): return dual_lambda
            tail_fracs_agg = [0.20, 0.40, 0.60, 0.80, 1.00]
            tail_agg_rates = {f: [] for f in tail_fracs_agg}
            tail_agg_metrics = {f: [] for f in tail_fracs_agg}
            tail_ref_rate_traces = []  # per-network reference rate trajectories
            tail_ref_vio_traces = []
            tail_variant_rate_traces = {f: [] for f in tail_fracs_agg}
            tail_variant_vio_traces = {f: [] for f in tail_fracs_agg}
            for bi, batch_tail in enumerate(test_loader):
                if bi >= num_batches:
                    break
                batch_tail = batch_tail.to(device)
                B_tail = int(batch_tail.num_graphs)
                N_tail = int(batch_tail.y.size(0) // B_tail)
                shape_tail = (B_tail, int(batch_tail.y.size(1)), N_tail, int(batch_tail.y.size(2)))
                ctx_tail = study_sampler._build_energy_context(
                    data=batch_tail, batch_size=B_tail, num_nodes=N_tail,
                    device=device, dtype=torch.float32)
                ds_tail = str(getattr(batch_tail, "dataset_name", ["unknown"])[0]) if isinstance(getattr(batch_tail, "dataset_name", None), (list, tuple)) else "unknown"
                nid_tail = int(getattr(batch_tail, "network_id", [bi])[0]) if isinstance(getattr(batch_tail, "network_id", None), (list, tuple, torch.Tensor)) else bi
                print(f"  [tail table] net {nid_tail}: computing reference trace...")
                torch.manual_seed(args.seed + bi)
                _, tr_tail_net = _sample_with_trace(study_sampler, shape_tail, device,
                    batch_tail, ctx_tail, sub_batch=sb, n_samples_per_input=K_study)
                tail_ref_rate_traces.append(np.array(tr_tail_net["rates"]))
                tail_ref_vio_traces.append(np.array(tr_tail_net["violations"]).mean(axis=1) if len(tr_tail_net["violations"]) > 0 else np.zeros(1))
                lam_all_t = tr_tail_net["lam_vectors"]
                T_total = lam_all_t.shape[0]
                for frac in tail_fracs_agg:
                    start_idx = int(T_total * (1.0 - frac))
                    lam_tail_v = torch.tensor(lam_all_t[start_idx:].mean(axis=0), device=device, dtype=torch.float32)
                    orig_d = study_sampler._dual_ascent_step
                    orig_i = study_sampler._init_dual_lambda
                    study_sampler._dual_ascent_step = _noop_d
                    _lv_t = lam_tail_v.clone()
                    study_sampler._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=_lv_t: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
                    torch.manual_seed(args.seed + bi)
                    x_tail_v, tr_tail_v = _sample_with_trace(study_sampler, shape_tail, device,
                        batch_tail, ctx_tail, sub_batch=0, n_samples_per_input=K_study)
                    tail_variant_rate_traces[frac].append(np.array(tr_tail_v["rates"]))
                    tail_variant_vio_traces[frac].append(np.array(tr_tail_v["violations"]).mean(axis=1) if len(tr_tail_v["violations"]) > 0 else np.zeros(1))
                    m_tail_v = _evaluate_method(x_tail_v, ctx_tail, any_sampler, ds_tail, nid_tail, N_tail, K_study, device, args.num_evolution_trials)
                    tail_agg_rates[frac].append(m_tail_v["per_user_rates"])
                    tail_agg_metrics[frac].append(m_tail_v)
                    study_sampler._dual_ascent_step = orig_d
                    study_sampler._init_dual_lambda = orig_i
            for frac in tail_fracs_agg:
                if tail_agg_rates[frac]:
                    per_net_p5 = [float(np.percentile(r, 5)) for r in tail_agg_rates[frac]]
                    per_net_mean = [float(np.mean(r)) for r in tail_agg_rates[frac]]
                    per_net_p1 = [float(np.percentile(r, 1)) for r in tail_agg_rates[frac]]
                    per_net_feas = [float((r >= r_min - 0.01).mean()) for r in tail_agg_rates[frac]]
                    per_net_vio = [float(np.maximum(r_min - r, 0).mean()) for r in tail_agg_rates[frac]]
                    ent_avg = float(np.mean([m.get("power_entropy", 0) for m in tail_agg_metrics[frac]]))
                    ent_std = float(np.std([m.get("power_entropy", 0) for m in tail_agg_metrics[frac]]))
                    flat_t = np.concatenate(tail_agg_rates[frac])
                    study_table_rows[f"Tail {int(frac*100)}\\%"] = {
                        "mean_rate": float(np.mean(per_net_mean)),
                        "mean_rate_std": float(np.std(per_net_mean)),
                        "p5_rate": float(np.mean(per_net_p5)),
                        "p5_rate_std": float(np.std(per_net_p5)),
                        "p1_rate": float(np.mean(per_net_p1)),
                        "p1_rate_std": float(np.std(per_net_p1)),
                        "feasibility": float(np.mean(per_net_feas)),
                        "feasibility_std": float(np.std(per_net_feas)),
                        "mean_vio": float(np.mean(per_net_vio)),
                        "mean_vio_std": float(np.std(per_net_vio)),
                        "per_user_rates": flat_t,
                        "power_entropy": ent_avg,
                        "power_entropy_std": ent_std,
                    }
                    print(f"  [tail agg] {int(frac*100)}%: mean={np.mean(per_net_mean):.4f}+/-{np.std(per_net_mean):.4f}, p5={np.mean(per_net_p5):.4f}+/-{np.std(per_net_p5):.4f}, feas={np.mean(per_net_feas):.3f}")

            # --- Aggregated tail trajectory figures ---
            import matplotlib
            if tail_ref_rate_traces:
                ref_rates = np.array(tail_ref_rate_traces)  # [num_nets, T]
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                T_steps = np.arange(ref_rates.shape[1])
                # Rate trajectory
                ax = axes[0]
                ax.plot(T_steps, ref_rates.mean(0), label="PD (reference)", color="black", linewidth=1.5)
                ax.fill_between(T_steps, ref_rates.mean(0) - ref_rates.std(0),
                                ref_rates.mean(0) + ref_rates.std(0), alpha=0.15, color="black")
                tail_colors = {0.20: "tab:blue", 0.40: "tab:orange", 0.60: "tab:green", 0.80: "tab:red", 1.00: "tab:purple"}
                for frac in tail_fracs_agg:
                    if tail_variant_rate_traces[frac]:
                        vr = np.array(tail_variant_rate_traces[frac])
                        ax.plot(T_steps[:vr.shape[1]], vr.mean(0), label=f"Tail {int(frac*100)}%",
                                color=tail_colors.get(frac, "gray"), linewidth=1)
                        ax.fill_between(T_steps[:vr.shape[1]], vr.mean(0) - vr.std(0),
                                        vr.mean(0) + vr.std(0), alpha=0.1, color=tail_colors.get(frac, "gray"))
                ax.set_xlabel("Diffusion step"); ax.set_ylabel("Mean rate")
                ax.legend(fontsize=7); ax.grid(alpha=0.2)
                # Violation trajectory
                ax = axes[1]
                ref_vio = np.array(tail_ref_vio_traces)
                ax.plot(T_steps[:ref_vio.shape[1]], ref_vio.mean(0), label="PD (reference)", color="black", linewidth=1.5)
                ax.fill_between(T_steps[:ref_vio.shape[1]], ref_vio.mean(0) - ref_vio.std(0),
                                ref_vio.mean(0) + ref_vio.std(0), alpha=0.15, color="black")
                for frac in tail_fracs_agg:
                    if tail_variant_vio_traces[frac]:
                        vv = np.array(tail_variant_vio_traces[frac])
                        ax.plot(T_steps[:vv.shape[1]], vv.mean(0), label=f"Tail {int(frac*100)}%",
                                color=tail_colors.get(frac, "gray"), linewidth=1)
                        ax.fill_between(T_steps[:vv.shape[1]], vv.mean(0) - vv.std(0),
                                        vv.mean(0) + vv.std(0), alpha=0.1, color=tail_colors.get(frac, "gray"))
                ax.set_xlabel("Diffusion step"); ax.set_ylabel("Mean violation")
                ax.legend(fontsize=7); ax.grid(alpha=0.2)
                fig.tight_layout()
                agg_path = output_dir / "fig_tail_trajectory_aggregated.pdf"
                fig.savefig(agg_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved aggregated tail trajectory: {agg_path}")

        if args.temperature_scatter:
            print("\n" + "=" * 60)
            print("Temperature Scatter")
            run_temperature_scatter(
                study_sampler, shape_study, device, first_batch,
                ctx_study, output_dir, seed=args.seed, sub_batch=sb,
                n_samples_per_input=K_study,
                chunk_size=args.chunk_size,
                dataset_name=study_ds, network_id=study_net,
                num_nodes=N_study, eval_timeslots=args.eval_timeslots,
                any_sampler=any_sampler)

        # Append study rows to table
        if study_table_rows:
            table_path = output_dir / "results_table.tex"
            existing = table_path.read_text() if table_path.exists() else ""
            if not existing or r"\bottomrule" not in existing:
                existing = (r"\begin{tabular}{lcccccc}" + "\n"
                    + r"\toprule" + "\n"
                    + r"Method & Mean $(\uparrow)$ & p5 $(\uparrow)$ & p1 $(\uparrow)$ & Vio $(\downarrow)$ & Feas $(\uparrow)$ & Entropy $(\uparrow)$ \\" + "\n"
                    + r"\midrule" + "\n"
                    + r"\bottomrule" + "\n"
                    + r"\end{tabular}")
            if r"\bottomrule" in existing:
                extra_lines = [r"\midrule"]
                extra_lines.append(r"\multicolumn{7}{c}{\textit{Ablation studies}} \\")
                extra_lines.append(r"\midrule")
                def _fmt_pm(mu, sd, decimals=3):
                    if sd is None or sd == 0:
                        return f"${mu:.{decimals}f}$"
                    return f"${mu:.{decimals}f}" + r" {\scriptstyle\pm " + f"{sd:.{decimals}f}" + r"}$"
                def _fmt_pct(mu, sd):
                    if sd is None or sd == 0:
                        return f"${100*mu:.1f}\\%$"
                    return f"${100*mu:.1f}" + r" {\scriptstyle\pm " + f"{100*sd:.1f}" + r"}\%$"
                for rname, rm in study_table_rows.items():
                    m_mean = rm.get("mean_rate", 0); m_std = rm.get("mean_rate_std", 0)
                    p5_mean = rm.get("p5_rate", 0); p5_std = rm.get("p5_rate_std", 0)
                    p1_mean = rm.get("p1_rate", 0); p1_std = rm.get("p1_rate_std", 0)
                    vio_mean = rm.get("mean_vio", 0); vio_std = rm.get("mean_vio_std", 0)
                    feas_mean = rm.get("feasibility", 0); feas_std = rm.get("feasibility_std", 0)
                    ent_mean = rm.get("power_entropy", 0); ent_std = rm.get("power_entropy_std", 0)
                    if np.isnan(ent_mean): ent_mean = 0.0
                    # Fallback for rows without std
                    if m_std == 0 and "per_user_rates" in rm:
                        rates = rm["per_user_rates"]
                        vio_mean = float(np.maximum(r_min - rates, 0).mean())
                        p1_mean = float(np.percentile(rates, 1)) if len(rates) > 1 else 0
                        feas_mean = float((rates >= r_min - 0.01).mean())
                    extra_lines.append(
                        f"{rname} & {_fmt_pm(m_mean, m_std)} & {_fmt_pm(p5_mean, p5_std)} "
                        f"& {_fmt_pm(p1_mean, p1_std)} & {_fmt_pm(vio_mean, vio_std, 4)} "
                        f"& {_fmt_pct(feas_mean, feas_std)} & {_fmt_pm(ent_mean, ent_std, 2)} \\\\")
                new_table = existing.replace(r"\bottomrule", "\n".join(extra_lines) + "\n" + r"\bottomrule")
                table_path.write_text(new_table)
                print(f"Updated table with {len(study_table_rows)} study rows")

    print("Done.")


if __name__ == "__main__":
    main()
