#!/usr/bin/env python3
"""Graph visualization of portfolio weight and risk evolution during diffusion.

Analogous to ``scripts/wra/diffusion/graph_rate_evolution.py`` for portfolio.

Shows assets as nodes on the correlation graph, colored by portfolio weight
(row 1) and per-asset risk contribution vs budget (row 2) as the reverse
diffusion proceeds.  A histogram row (row 3) shows the distribution of
normalised constraint violations.

Usage:
    micromamba run -n gdiff python scripts/portfolio/graph_weight_evolution.py \
        --size crypto --ib 2000 --dual-step 500 \
        --ced-ckpt outputs/portfolio/score_net_train/.../score_net_best_feas.pt \
        --methods unc,ced,mc \
        --problem-seed 3000 \
        --label graph_evo
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import types as _types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import experiment_dir, _project_root

_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (
    PortfolioEnergyDDPM, make_portfolio_problem,
    variance_contributions_torch, shortfall_contributions_torch,
)
from pdi.models.portfolio_gnn_backbone import (
    PortfolioGNNBackbone, build_dense_adjacency, build_correlation_graph,
)
from pdi.models.portfolio_backbone import PortfolioScoreBackbone
from pdi.trainers.energy_score import ScoreNetWithLambda
import baselines as bl

from torch_geometric.data import Data, Batch


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=200, gamma=1.0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5),
    "crypto": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0,
                   structure="sectors", num_sectors=10,
                   constraint_type="variance", budget_type="uniform"),
    "sectors_shortfall_g3": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0,
                                 structure="sectors", num_sectors=10,
                                 constraint_type="shortfall"),
}


class _NoOpModel(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


def _make_batch(B, N):
    graphs = [Data(x=None, y=torch.zeros(N, 1, 1),
                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                   num_nodes=N) for _ in range(B)]
    return Batch.from_data_list(graphs)


def _compute_node_positions(Sigma, edge_index, edge_weight, N,
                            structure="homogeneous", num_sectors=None):
    """Compute 2D layout for graph nodes.

    For sector structure: arrange sectors radially with assets spread within
    each sector arc.  For homogeneous: spectral layout from the adjacency.
    """
    if structure in ("sectors", "sectors_fat") and num_sectors is not None:
        S = num_sectors
        sector_ids = np.arange(N) % S
        pos = np.zeros((N, 2))
        for s in range(S):
            angle_center = 2 * np.pi * s / S
            members = np.where(sector_ids == s)[0]
            n_s = len(members)
            radius_base = 1.0
            for i, j in enumerate(members):
                angle_offset = (i - n_s / 2) * 0.015
                r = radius_base + 0.15 * (i % 5) / 5
                pos[j, 0] = r * np.cos(angle_center + angle_offset)
                pos[j, 1] = r * np.sin(angle_center + angle_offset)
        return pos

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        A_sp = csr_matrix((edge_weight, (edge_index[0], edge_index[1])),
                          shape=(N, N))
        A_sym = (A_sp + A_sp.T) / 2
        eigvals, eigvecs = eigsh(A_sym, k=3, which="SM")
        pos = eigvecs[:, 1:3]
        pos = pos / np.abs(pos).max(axis=0, keepdims=True).clip(1e-8)
        return pos
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _build_sampler(mu_np, Sigma_np, scen_np, bud_np, alpha, *,
                   T, ib, dual_step, dual_lambda_max, dual_lambda_decay,
                   shared_lambda, normalize_constraints,
                   constraint_type, num_sectors, mc_samples,
                   beta_schedule="cosine", lam0=0.0,
                   use_dps_guidance=False):
    return PortfolioEnergyDDPM(
        model=_NoOpModel(),
        num_timesteps=T, beta_schedule=beta_schedule,
        portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
        portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_np,
        portfolio_alpha=alpha,
        energy_mc_samples=mc_samples,
        inverse_beta=ib, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=dual_step,
        dual_lambda_init=lam0,
        dual_lambda_max=dual_lambda_max,
        dual_lambda_decay=dual_lambda_decay,
        shared_lambda=shared_lambda,
        normalize_constraints=normalize_constraints,
        constraint_type=constraint_type,
        num_sectors=num_sectors,
        use_dps_guidance=use_dps_guidance,
    )


def _run_sampling_with_trace(sampler, shape, device, score_fn=None,
                             seed=42, x_init=None, noise_seq=None,
                             record_every=1):
    """Run reverse diffusion recording x0_pred and dual_lambda at each step."""
    B, _, N, _ = shape
    if x_init is not None:
        x_t = x_init.clone()
    else:
        torch.manual_seed(seed)
        x_t = torch.randn(shape, device=device)

    context = sampler._build_energy_context(
        data=Data(), batch_size=B, num_nodes=N, device=device,
        dtype=torch.float32)
    dual_lambda = sampler._init_dual_lambda(
        batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
        init_value=float(sampler.dual_lambda_init))

    if score_fn is not None:
        orig_score = sampler._estimate_score
        sampler._estimate_score = _types.MethodType(score_fn, sampler)

    trace_weights = []
    trace_lambda = []
    trace_risk = []

    for t_int in reversed(range(sampler.num_timesteps)):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        score = sampler._estimate_score(
            x_t=x_t, t=t, context=context,
            dual_lambda=dual_lambda)
        x0_pred, _ = sampler._score_to_x0_eps(x_t=x_t, t=t, score=score)

        erg, _ = sampler._ergodic_rates_from_samples(x=x0_pred, context=context)
        dual_lambda = sampler._dual_ascent_step(
            dual_lambda=dual_lambda, ergodic_rates=erg, context=context, t=t)

        if t_int % record_every == 0 or t_int == 0:
            w = sampler._z_to_weights(x0_pred)
            trace_weights.append(w.detach().cpu().numpy())
            trace_lambda.append(dual_lambda.detach().cpu().numpy())
            trace_risk.append((-erg).detach().cpu().numpy())

        mean = sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
        if t_int > 0:
            var = sampler._gather(sampler.posterior_variance, t).view(-1, 1, 1, 1)
            noise = noise_seq[t_int] if noise_seq is not None else torch.randn_like(x_t)
            x_t = mean + torch.sqrt(var) * noise
        else:
            x_t = mean

    if score_fn is not None:
        sampler._estimate_score = orig_score

    final_w = sampler._z_to_weights(x_t).detach().cpu().numpy()
    return {
        "weights_trace": np.array(trace_weights),
        "lambda_trace": np.array(trace_lambda),
        "risk_trace": np.array(trace_risk),
        "final_weights": final_w,
        "final_lambda": dual_lambda.detach().cpu().numpy(),
    }


def _make_ced_score_fn(score_net, alphas_cumprod, A_dense):
    """Build a score function using the trained CED network."""
    def _fn(self, x_t, t, context, dual_lambda=None, inverse_beta_override=None):
        del context, inverse_beta_override
        B = x_t.shape[0]
        N = x_t.shape[2]
        dl = dual_lambda if dual_lambda is not None else torch.zeros(B, N, device=x_t.device)
        with torch.no_grad():
            eps = score_net(x=x_t, timesteps=t, dual_lambda=dl,
                            edge_index=A_dense[:B], edge_weight=None,
                            cond=None)
        if isinstance(eps, tuple):
            eps = eps[0]
        ab = alphas_cumprod[t].view(-1, 1, 1, 1)
        return -eps / torch.sqrt((1.0 - ab).clamp_min(1e-12))
    return _fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="graph_weight_evo")
    parser.add_argument("--size", type=str, default="crypto",
                        choices=list(PROBLEM_CONFIGS.keys()))
    parser.add_argument("--problem-seed", type=int, default=3000)
    parser.add_argument("--ib", type=float, default=2000.0)
    parser.add_argument("--dual-step", type=float, default=500.0)
    parser.add_argument("--dual-lambda-max", type=float, default=50.0)
    parser.add_argument("--dual-lambda-decay", type=float, default=0.0)
    parser.add_argument("--lam0", type=float, default=0.1)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--beta-schedule", type=str, default="cosine")
    parser.add_argument("--B", type=int, default=128,
                        help="Batch size (number of portfolio samples)")
    parser.add_argument("--mc-samples", type=int, default=512)
    parser.add_argument("--ced-ckpt", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="gnn",
                        choices=["mlp", "gnn"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--tagconv-K", type=int, default=2)
    parser.add_argument("--methods", type=str, default="unc,ced,mc")
    parser.add_argument("--n-frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pdl-num-iters", type=int, default=500)
    parser.add_argument("--pdl-primal-lr", type=float, default=1e-2)
    parser.add_argument("--pdl-dual-lr", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.methods.split(",")]

    output_dir = experiment_dir("diagnostics", args.label)
    print(f"Output: {output_dir}")

    # --- Problem setup ---
    pcfg = dict(PROBLEM_CONFIGS[args.size])
    pcfg["seed"] = args.problem_seed
    constraint_type = pcfg.pop("constraint_type", "shortfall")
    structure = pcfg.get("structure", "homogeneous")
    num_sectors = pcfg.get("num_sectors", None)
    mu_np, Sigma_np, scen_np, bud_np, alpha = make_portfolio_problem(
        constraint_type=constraint_type, **pcfg)
    N = pcfg["N"]
    print(f"Problem: {args.size}, N={N}, constraint={constraint_type}, "
          f"seed={args.problem_seed}")

    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float32, device=device)
    scenarios = torch.tensor(scen_np, dtype=torch.float32, device=device)
    budgets = torch.tensor(bud_np, dtype=torch.float32, device=device)

    # --- Graph ---
    ei_np, ew_np = build_correlation_graph(Sigma_np, top_k=args.top_k)
    A_np = build_dense_adjacency(Sigma_np, top_k=args.top_k)
    A_dense = torch.tensor(A_np, dtype=torch.float32, device=device)
    A_dense = A_dense.unsqueeze(0).expand(args.B, -1, -1)

    node_pos = _compute_node_positions(
        Sigma_np, ei_np, ew_np, N,
        structure=structure, num_sectors=num_sectors)
    print(f"Graph: {ei_np.shape[1]} edges, top_k={args.top_k}")

    # --- Shared noise for coupling ---
    shape = (args.B, 1, N, 1)
    torch.manual_seed(args.seed)
    shared_x_init = torch.randn(shape, device=device)
    shared_noise = [torch.randn(shape, device=device) for _ in range(args.T)]
    data = _make_batch(args.B, N)

    record_every = max(1, args.T // args.n_frames)

    # --- Load CED score net ---
    ced_net = None
    if args.ced_ckpt:
        ckpt = torch.load(args.ced_ckpt, map_location=device, weights_only=False)
        if args.backbone == "gnn":
            backbone = PortfolioGNNBackbone(
                d=N, hidden=args.hidden, num_layers=args.num_layers,
                num_timesteps=args.T, K=args.tagconv_K)
        else:
            backbone = PortfolioScoreBackbone(
                d=N, hidden=args.hidden, num_layers=args.num_layers,
                num_timesteps=args.T, cond_channels=1)
        ced_net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0)
        ced_net.load_state_dict(ckpt["state_dict"])
        ced_net.to(device).eval()
        print(f"Loaded CED: outer={ckpt.get('outer')}")

    # --- Sampling ---
    sampler_kwargs = dict(
        T=args.T, ib=args.ib, dual_step=args.dual_step,
        dual_lambda_max=args.dual_lambda_max,
        dual_lambda_decay=args.dual_lambda_decay,
        shared_lambda=True,
        normalize_constraints=True,
        constraint_type=constraint_type,
        num_sectors=num_sectors or 10,
        mc_samples=args.mc_samples,
        beta_schedule=args.beta_schedule,
        lam0=args.lam0,
    )

    method_results = {}
    method_names = {
        "unc": "Unconstrained",
        "ced": "CED (Ours)",
        "mc": "MC Ceiling",
        "pdl": "PD-Langevin",
        "eq": "Equal Weight",
    }

    for method in methods:
        print(f"\nSampling: {method}...")
        t0 = time.time()

        if method == "unc" and ced_net is not None:
            sampler_unc = _build_sampler(
                mu_np, Sigma_np, scen_np, bud_np, alpha,
                **{**sampler_kwargs, "dual_step": 1e-12, "lam0": 0.0})
            sampler_unc.to(device)
            score_fn = _make_ced_score_fn(
                ced_net, sampler_unc.alphas_cumprod.to(device), A_dense)
            result = _run_sampling_with_trace(
                sampler_unc, shape, device, score_fn=score_fn,
                x_init=shared_x_init, noise_seq=shared_noise,
                record_every=record_every)

        elif method == "ced" and ced_net is not None:
            sampler_ced = _build_sampler(
                mu_np, Sigma_np, scen_np, bud_np, alpha, **sampler_kwargs)
            sampler_ced.to(device)
            score_fn = _make_ced_score_fn(
                ced_net, sampler_ced.alphas_cumprod.to(device), A_dense)
            result = _run_sampling_with_trace(
                sampler_ced, shape, device, score_fn=score_fn,
                x_init=shared_x_init, noise_seq=shared_noise,
                record_every=record_every)

        elif method == "mc":
            sampler_mc = _build_sampler(
                mu_np, Sigma_np, scen_np, bud_np, alpha, **sampler_kwargs)
            sampler_mc.to(device)
            result = _run_sampling_with_trace(
                sampler_mc, shape, device, score_fn=None,
                x_init=shared_x_init, noise_seq=shared_noise,
                record_every=record_every)

        elif method == "unc_mc":
            sampler_unc_mc = _build_sampler(
                mu_np, Sigma_np, scen_np, bud_np, alpha,
                **{**sampler_kwargs, "dual_step": 1e-12, "lam0": 0.0})
            sampler_unc_mc.to(device)
            result = _run_sampling_with_trace(
                sampler_unc_mc, shape, device, score_fn=None,
                x_init=shared_x_init, noise_seq=shared_noise,
                record_every=record_every)

        elif method == "pdl":
            w_pdl, _ = bl.pd_langevin(
                mu, Sigma, scenarios, budgets, alpha, args.B,
                num_iters=args.pdl_num_iters,
                primal_lr=args.pdl_primal_lr,
                dual_lr=args.pdl_dual_lr,
                device=device, seed=args.seed,
                constraint_type=constraint_type,
                shared_lambda=True)
            w_np = w_pdl.cpu().numpy()
            if constraint_type == "variance":
                c = variance_contributions_torch(w_pdl, Sigma).cpu().numpy()
            else:
                c = shortfall_contributions_torch(w_pdl, scenarios, alpha).cpu().numpy()
            result = {
                "weights_trace": w_np[np.newaxis],
                "risk_trace": c[np.newaxis],
                "lambda_trace": np.zeros((1, args.B, N)),
                "final_weights": w_np,
                "final_lambda": np.zeros((args.B, N)),
            }

        elif method == "eq":
            w_eq = np.ones((args.B, N)) / N
            if constraint_type == "variance":
                c = variance_contributions_torch(
                    torch.tensor(w_eq, dtype=torch.float32, device=device),
                    Sigma).cpu().numpy()
            else:
                c = shortfall_contributions_torch(
                    torch.tensor(w_eq, dtype=torch.float32, device=device),
                    scenarios, alpha).cpu().numpy()
            result = {
                "weights_trace": w_eq[np.newaxis],
                "risk_trace": c[np.newaxis],
                "lambda_trace": np.zeros((1, args.B, N)),
                "final_weights": w_eq,
                "final_lambda": np.zeros((args.B, N)),
            }
        else:
            print(f"  Skipping unknown method: {method}")
            continue

        method_results[method] = result

        # Summary of final state
        w_final = result["final_weights"]
        r_final = result["risk_trace"][-1]
        c_mean = r_final.mean(axis=0)
        feas = float((c_mean <= bud_np + 1e-8).mean())
        max_vio = float(np.maximum(c_mean - bud_np, 0).max())
        ret = float((scen_np @ w_final.T).mean())
        print(f"  Done ({time.time()-t0:.1f}s). feas={feas*100:.0f}%, "
              f"max_vio={max_vio:.4f}, ret={ret:.4f}")

        torch.cuda.empty_cache()

    if not method_results:
        print("No methods produced results. Exiting.")
        return

    # =================================================================
    # Plotting
    # =================================================================
    print("\nGenerating graph animation...")
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'cm',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    })

    n_methods = len(method_results)

    # Edge segments for graph overlay
    mask = (ei_np[0] < N) & (ei_np[1] < N)
    edges_src = ei_np[0][mask]
    edges_dst = ei_np[1][mask]
    # Subsample edges for visual clarity
    if len(edges_src) > 2000:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(edges_src), 2000, replace=False)
        edges_src, edges_dst = edges_src[idx], edges_dst[idx]

    # Determine number of animation frames
    first_key = list(method_results.keys())[0]
    n_trace_steps = method_results[first_key]["weights_trace"].shape[0]
    frame_indices = np.unique(
        np.geomspace(1, n_trace_steps, min(args.n_frames, n_trace_steps),
                     dtype=int)) - 1

    # Colormaps and norms
    weight_cmap = plt.cm.YlGnBu
    weight_norm = Normalize(vmin=0, vmax=None)

    # For risk: diverging around budget
    risk_cmap = plt.cm.RdBu_r

    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_methods,
                             figsize=(3.5 * n_methods, 3.5 * n_rows), dpi=150)
    if n_methods == 1:
        axes = axes.reshape(n_rows, 1)
    axes_weight = axes[0]
    axes_risk = axes[1]
    axes_hist = axes[2]

    def update(fi):
        frame_idx = frame_indices[fi]
        diff_step = args.T - 1 - frame_idx * record_every

        for mi, (method, data) in enumerate(method_results.items()):
            wt = data["weights_trace"]
            rt = data["risk_trace"]
            n_steps = wt.shape[0]
            idx = min(frame_idx, n_steps - 1)

            # --- Weight panel ---
            ax_w = axes_weight[mi]
            ax_w.clear()
            w_batch = wt[idx]
            w_mean = w_batch.mean(axis=0)

            segs = np.stack([node_pos[edges_src], node_pos[edges_dst]], axis=1)
            lc = LineCollection(segs, colors='#d0d0d0', linewidths=0.3, zorder=1)
            ax_w.add_collection(lc)

            vmax_w = max(w_mean.max(), 1.0 / N * 3)
            sc_w = ax_w.scatter(
                node_pos[:, 0], node_pos[:, 1],
                c=w_mean, cmap=weight_cmap,
                norm=Normalize(vmin=0, vmax=vmax_w),
                s=30 if N > 200 else 60, edgecolors='#333333',
                linewidths=0.3, alpha=0.9, zorder=3)
            name = method_names.get(method, method)
            ax_w.set_title(f"{name}", fontsize=12)
            ax_w.set_aspect('equal')
            ax_w.set_xticks([])
            ax_w.set_yticks([])
            if mi == 0:
                ax_w.set_ylabel("Weights $w_j$", fontsize=11)

            # --- Risk contribution panel ---
            ax_r = axes_risk[mi]
            ax_r.clear()
            c_batch = rt[idx]
            c_mean = c_batch.mean(axis=0)

            # Normalised violation: (c_j - b_j) / b_j
            rel_viol = (c_mean - bud_np) / np.maximum(bud_np, 1e-12)
            vmax_rv = max(np.abs(rel_viol).max(), 0.1)

            segs2 = np.stack([node_pos[edges_src], node_pos[edges_dst]], axis=1)
            lc2 = LineCollection(segs2, colors='#d0d0d0', linewidths=0.3, zorder=1)
            ax_r.add_collection(lc2)
            risk_norm = TwoSlopeNorm(vmin=-vmax_rv, vcenter=0, vmax=vmax_rv)
            ax_r.scatter(
                node_pos[:, 0], node_pos[:, 1],
                c=rel_viol, cmap=risk_cmap, norm=risk_norm,
                s=30 if N > 200 else 60, edgecolors='#333333',
                linewidths=0.3, alpha=0.9, zorder=3)
            ax_r.set_aspect('equal')
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            if mi == 0:
                ax_r.set_ylabel(r"$(c_j - b_j)/b_j$", fontsize=11)

            # --- Histogram of constraint violation ---
            ax_h = axes_hist[mi]
            ax_h.clear()
            feas_frac = float((c_mean <= bud_np + 1e-8).mean())
            max_vio = float(np.maximum(c_mean - bud_np, 0).max())

            weights_h = np.ones_like(rel_viol) / len(rel_viol)
            ax_h.hist(rel_viol, bins=40, range=(-1.0, max(1.0, vmax_rv)),
                      weights=weights_h,
                      color='#78909c', alpha=0.7, edgecolor='white', linewidth=0.3)
            ax_h.axvline(0.0, color='#b91c1c', linestyle='--', linewidth=1.5)
            ax_h.text(0.95, 0.92,
                      f'feas = {feas_frac*100:.0f}%\n'
                      rf'max vio $ = {max_vio:.3f}$',
                      transform=ax_h.transAxes, ha='right', va='top', fontsize=9,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax_h.set_xlim(-1.0, max(1.0, vmax_rv))
            ax_h.tick_params(labelsize=7)
            ax_h.set_xlabel(r"$(c_j - b_j)/b_j$", fontsize=10)
            if mi == 0:
                ax_h.set_ylabel("Frequency", fontsize=10)

        if axes_hist is not None:
            ymax_hist = max(ax.get_ylim()[1] for ax in axes_hist)
            for ax in axes_hist:
                ax.set_ylim(0, ymax_hist)

        fig.suptitle(f"Diffusion step $t = {max(diff_step, 0)}$", fontsize=13, y=0.98)

    update(0)

    cbar_ax_w = fig.add_axes([0.92, 0.7, 0.012, 0.18])
    w0 = method_results[first_key]["weights_trace"][0].mean(axis=0)
    vmax_w0 = max(w0.max(), 1.0 / N * 3)
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=vmax_w0), cmap=weight_cmap),
                 cax=cbar_ax_w, label="$w_j$")
    cbar_ax_w.tick_params(labelsize=7)

    cbar_ax_r = fig.add_axes([0.92, 0.4, 0.012, 0.18])
    fig.colorbar(ScalarMappable(norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5),
                                cmap=risk_cmap),
                 cax=cbar_ax_r, label=r"$(c_j-b_j)/b_j$")
    cbar_ax_r.axhline(0.0, color='black', linewidth=1.5, linestyle='--')
    cbar_ax_r.tick_params(labelsize=7)

    fig.subplots_adjust(right=0.90, wspace=0.12, hspace=0.15)

    # --- Render frames ---
    n_frames = len(frame_indices)
    frames_pil = []
    for fi in range(n_frames):
        update(fi)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames_pil.append(Image.open(buf).copy())
        buf.close()
        if (fi + 1) % 20 == 0:
            print(f"  Frame {fi+1}/{n_frames}")

    durations = [100] * n_frames
    durations[-1] = 3000

    # Full 3-row GIF
    out_gif = output_dir / "graph_weight_evo_hist.gif"
    frames_pil[0].save(str(out_gif), save_all=True, append_images=frames_pil[1:],
                       duration=durations, loop=0)
    print(f"Saved: {out_gif} ({n_frames} frames)")

    # Cropped 2-row GIF (weights + risk only)
    crop_h = int(frames_pil[0].height * 2 / 3)
    frames_cropped = [f.crop((0, 0, f.width, crop_h)) for f in frames_pil]
    out_gif_crop = output_dir / "graph_weight_evo.gif"
    frames_cropped[0].save(str(out_gif_crop), save_all=True,
                           append_images=frames_cropped[1:],
                           duration=durations, loop=0)
    print(f"Saved: {out_gif_crop} ({n_frames} frames)")

    plt.close(fig)

    # --- Final frame as PDF ---
    fig_final, axes_final = plt.subplots(2, n_methods,
                                         figsize=(3.5 * n_methods, 7), dpi=200)
    if n_methods == 1:
        axes_final = axes_final.reshape(2, 1)
    for mi, (method, data) in enumerate(method_results.items()):
        w_final = data["final_weights"].mean(axis=0)
        c_final = data["risk_trace"][-1].mean(axis=0)
        rel_viol = (c_final - bud_np) / np.maximum(bud_np, 1e-12)
        vmax_rv = max(np.abs(rel_viol).max(), 0.1)

        segs = np.stack([node_pos[edges_src], node_pos[edges_dst]], axis=1)

        ax_w = axes_final[0, mi]
        ax_w.add_collection(LineCollection(segs, colors='#d0d0d0', linewidths=0.3, zorder=1))
        vmax_w = max(w_final.max(), 1.0 / N * 3)
        ax_w.scatter(node_pos[:, 0], node_pos[:, 1],
                     c=w_final, cmap=weight_cmap,
                     norm=Normalize(vmin=0, vmax=vmax_w),
                     s=30 if N > 200 else 60, edgecolors='#333333',
                     linewidths=0.3, alpha=0.9, zorder=3)
        ax_w.set_title(method_names.get(method, method), fontsize=12)
        ax_w.set_aspect('equal')
        ax_w.set_xticks([])
        ax_w.set_yticks([])
        if mi == 0:
            ax_w.set_ylabel("Weights $w_j$", fontsize=11)

        ax_r = axes_final[1, mi]
        ax_r.add_collection(LineCollection(segs.copy(), colors='#d0d0d0',
                                           linewidths=0.3, zorder=1))
        ax_r.scatter(node_pos[:, 0], node_pos[:, 1],
                     c=rel_viol, cmap=risk_cmap,
                     norm=TwoSlopeNorm(vmin=-vmax_rv, vcenter=0, vmax=vmax_rv),
                     s=30 if N > 200 else 60, edgecolors='#333333',
                     linewidths=0.3, alpha=0.9, zorder=3)
        ax_r.set_aspect('equal')
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if mi == 0:
            ax_r.set_ylabel(r"$(c_j - b_j)/b_j$", fontsize=11)

    fig_final.subplots_adjust(right=0.90, wspace=0.12, hspace=0.12)
    cbar_w = fig_final.add_axes([0.92, 0.55, 0.015, 0.35])
    fig_final.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=vmax_w),
                                      cmap=weight_cmap),
                       cax=cbar_w, label="$w_j$")
    cbar_r = fig_final.add_axes([0.92, 0.1, 0.015, 0.35])
    fig_final.colorbar(ScalarMappable(norm=TwoSlopeNorm(vmin=-vmax_rv, vcenter=0,
                                                         vmax=vmax_rv),
                                      cmap=risk_cmap),
                       cax=cbar_r, label=r"$(c_j-b_j)/b_j$")
    out_pdf = output_dir / "graph_weight_evo_final.pdf"
    fig_final.savefig(str(out_pdf), bbox_inches='tight')
    fig_final.savefig(str(out_pdf).replace('.pdf', '.png'),
                      bbox_inches='tight', dpi=200)
    plt.close(fig_final)
    print(f"Saved: {out_pdf}")

    # --- Lambda graph (final dual multipliers) ---
    has_lambda = any(
        d["final_lambda"].max() > 0.01 for d in method_results.values())
    if has_lambda:
        print("\nGenerating lambda graph...")
        fig_lam, axes_lam = plt.subplots(1, n_methods,
                                          figsize=(3.5 * n_methods, 3.5), dpi=150)
        if n_methods == 1:
            axes_lam = [axes_lam]

        lam_max = max(
            d["final_lambda"].mean(axis=0).max() for d in method_results.values()
        ) or 1.0
        lam_norm = Normalize(vmin=0, vmax=lam_max)
        lam_cmap = plt.cm.YlOrRd

        for mi, (method, data) in enumerate(method_results.items()):
            ax = axes_lam[mi]
            lam_mean = data["final_lambda"].mean(axis=0)
            segs = np.stack([node_pos[edges_src], node_pos[edges_dst]], axis=1)
            ax.add_collection(LineCollection(segs, colors='#d0d0d0',
                                             linewidths=0.3, zorder=1))
            ax.scatter(node_pos[:, 0], node_pos[:, 1],
                       c=lam_mean, cmap=lam_cmap, norm=lam_norm,
                       s=30 if N > 200 else 60, edgecolors='#333333',
                       linewidths=0.3, alpha=0.9, zorder=3)
            name = method_names.get(method, method)
            n_active = int((lam_mean > 0.01).sum())
            ax.set_title(f"{name}\n{n_active} active, "
                         rf"$\bar\lambda$={lam_mean.mean():.1f}", fontsize=8)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        cbar_lam = fig_lam.add_axes([0.92, 0.15, 0.015, 0.7])
        fig_lam.colorbar(ScalarMappable(norm=lam_norm, cmap=lam_cmap),
                         cax=cbar_lam, label=r"$\lambda_j$")
        fig_lam.subplots_adjust(right=0.90, wspace=0.12)
        out_lam = output_dir / "graph_lambda.pdf"
        fig_lam.savefig(str(out_lam), bbox_inches='tight')
        fig_lam.savefig(str(out_lam).replace('.pdf', '.png'),
                        bbox_inches='tight', dpi=200)
        plt.close(fig_lam)
        print(f"Saved: {out_lam}")

    print("\nDone.")


if __name__ == "__main__":
    main()