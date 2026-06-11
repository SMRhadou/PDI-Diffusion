#!/usr/bin/env python3
"""Calibrate lambda prior by observing MC sampler's lambda trajectory.

HANDOFF §2: before training a score net with lambda conditioning, probe
the observed lambda distribution at the chosen (ib, dual_step) and set the
exponential prior mu_min/mu_max to match.

Runs the MC PortfolioEnergyDDPM with shared-lambda and records lambda at
each reverse-diffusion step. Reports per-t quantiles (mean, q50, q95, q99,
max). Saves the trajectory to disk.

Usage:
    python scripts/portfolio/calibrate_lambda.py --size small --ib 100 --dual-step 1000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import experiment_dir, _project_root  # noqa: E402

_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (  # noqa: E402
    PortfolioEnergyDDPM, make_portfolio_problem,
)


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=200, gamma=1.0, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5, seed=0),
    "crypto": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                   structure="sectors", num_sectors=10,
                   constraint_type="variance", budget_type="uniform"),
}


class _NoOp(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


def _make_batch(batch_size, N):
    graphs = [Data(x=None, y=torch.zeros(N, 1, 1),
                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                   num_nodes=N)
              for _ in range(batch_size)]
    return Batch.from_data_list(graphs)


def run_and_record_lambda(ib: float, dual_step: float, mu, Sigma, scenarios, budgets,
                           alpha, T: int, K: int, B: int, N: int,
                           device, seed: int = 42, shared_lambda: bool = True,
                           normalize_constraints: bool = True,
                           beta_schedule: str = "cosine",
                           dual_lambda_decay: float = 0.0,
                           lam0: float = 0.0,
                           constraint_type: str = "shortfall",
                           num_sectors: int = 10) -> dict:
    """Run the MC reverse pass and record lambda at every timestep."""
    torch.manual_seed(seed)
    sampler = PortfolioEnergyDDPM(
        model=_NoOp(), num_timesteps=T, beta_schedule=beta_schedule,
        portfolio_mu=mu, portfolio_Sigma=Sigma,
        portfolio_scenarios=scenarios, portfolio_risk_budgets=budgets,
        portfolio_alpha=alpha,
        energy_mc_samples=K,
        inverse_beta=ib, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=dual_step,
        dual_num_outer_iterations=1,
        dual_lambda_init=lam0,
        dual_lambda_max=1e6,
        dual_lambda_decay=dual_lambda_decay,
        shared_lambda=shared_lambda,
        normalize_constraints=normalize_constraints,
        constraint_type=constraint_type,
        num_sectors=num_sectors,
    ).to(device)

    data = _make_batch(B, N).to(device)
    shape = (B, 1, N, 1)

    # Manually run reverse pass, recording lambda
    x_init = torch.randn(shape, device=device, dtype=torch.float32)
    x_t = x_init.clone()
    context = sampler._build_energy_context(
        data=data, batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
    )
    dl = sampler._init_dual_lambda(batch_size=B, num_nodes=N, device=device,
                                    dtype=torch.float32, init_value=0.0)

    lambda_by_t: list = []  # [T, N] (shared-lambda so just take row 0)
    violation_by_t: list = []  # [T, N]
    t_values: list = []
    for t_int in reversed(range(T)):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        score = sampler._estimate_score(x_t=x_t, t=t, context=context, dual_lambda=dl)
        x0_pred, _ = sampler._score_to_x0_eps(x_t=x_t, t=t, score=score)

        # Record lambda BEFORE update (matches inference ordering)
        lambda_by_t.append(dl[0].detach().cpu().numpy())
        t_values.append(t_int)

        # Compute violation for inspection
        erg_rates, _ = sampler._ergodic_rates_from_samples(x=x0_pred, context=context)
        violation = (context.r_min - erg_rates).detach().cpu().numpy()  # [B, N]
        violation_by_t.append(violation.mean(axis=0))

        # Update dual
        dl = sampler._dual_ascent_step(
            dual_lambda=dl, ergodic_rates=erg_rates, context=context, t=t,
        )

        # DDPM posterior
        mean = sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t)
        if t_int > 0:
            var = sampler._gather(sampler.posterior_variance, t).view(-1, 1, 1, 1)
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = mean

    lambda_trace = np.stack(lambda_by_t, axis=0)   # [T, N]
    violation_trace = np.stack(violation_by_t, axis=0)  # [T, N]
    t_values = np.array(t_values)

    # Per-t statistics (across the N constraints)
    # Note: t_values are in reverse order (T-1, T-2, ..., 0)
    per_t_stats = {
        "t_values": t_values.tolist(),
        "lambda_mean": lambda_trace.mean(axis=1).tolist(),
        "lambda_q50": np.quantile(lambda_trace, 0.50, axis=1).tolist(),
        "lambda_q95": np.quantile(lambda_trace, 0.95, axis=1).tolist(),
        "lambda_q99": np.quantile(lambda_trace, 0.99, axis=1).tolist(),
        "lambda_max": lambda_trace.max(axis=1).tolist(),
        "violation_mean": violation_trace.mean(axis=1).tolist(),
    }

    # Aggregate stats
    overall = {
        "lambda_global_mean": float(lambda_trace.mean()),
        "lambda_global_max": float(lambda_trace.max()),
        "lambda_global_q99": float(np.quantile(lambda_trace, 0.99)),
        "lambda_at_t_last_noise": lambda_trace[0].tolist(),  # t=T-1 (near noise)
        "lambda_at_t_first_data": lambda_trace[-1].tolist(),  # t=0 (near data)
    }

    return {
        "per_t": per_t_stats,
        "overall": overall,
        "raw_lambda_trace": lambda_trace.tolist(),
        "raw_violation_trace": violation_trace.tolist(),
    }


def propose_prior(overall: dict) -> dict:
    """Propose exponential prior mu_min, mu_max from observed lambda distribution.

    HANDOFF §2 recipe:
        mu_min ~ observed mean at t=0 (near data)
        mu_max ~ observed max / 4.6    (so q99 = 4.6 * mu matches observed max)

    Also recommend the 10x and 100x widened mu_max per HANDOFF §13.1.
    """
    lam_t0 = np.asarray(overall["lambda_at_t_first_data"])
    lam_tT = np.asarray(overall["lambda_at_t_last_noise"])
    lam_max = overall["lambda_global_max"]

    mu_min = max(float(lam_t0.mean()), 1e-4)
    mu_max_q99match = max(float(lam_max / 4.6), mu_min * 2)
    return {
        "mu_min": mu_min,
        "mu_max_q99match": mu_max_q99match,
        "mu_max_10x": mu_max_q99match * 10,
        "mu_max_100x": mu_max_q99match * 100,
        "lambda_t0_mean": float(lam_t0.mean()),
        "lambda_tT_mean": float(lam_tT.mean()),
        "lambda_peak": float(lam_max),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(PROBLEM_CONFIGS.keys()), default="small")
    parser.add_argument("--ib", type=float, default=100.0)
    parser.add_argument("--dual-step", type=float, default=1000.0)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--beta-schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-lambda", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_true", default=False)
    parser.add_argument("--dual-lambda-decay", type=float, default=0.0)
    parser.add_argument("--lam0", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    pcfg = dict(PROBLEM_CONFIGS[args.size])
    N = pcfg["N"]
    constraint_type = pcfg.pop("constraint_type", "shortfall")
    mu, Sigma, scenarios, budgets, alpha = make_portfolio_problem(
        constraint_type=constraint_type, **pcfg)

    print(f"[calibrate] size={args.size} N={N} ib={args.ib} ds={args.dual_step} T={args.T} B={args.B}")
    print(f"  constraint_type={constraint_type} normalize={not args.no_normalize}")
    print(f"  budgets shape={budgets.shape} range=[{budgets.min():.4g}, {budgets.max():.4g}]")
    t0 = time.time()
    result = run_and_record_lambda(
        ib=args.ib, dual_step=args.dual_step,
        mu=mu, Sigma=Sigma, scenarios=scenarios, budgets=budgets,
        alpha=alpha,
        T=args.T, K=args.K, B=args.B, N=N, device=device, seed=args.seed,
        shared_lambda=args.shared_lambda,
        normalize_constraints=not args.no_normalize,
        beta_schedule=args.beta_schedule,
        dual_lambda_decay=args.dual_lambda_decay,
        lam0=args.lam0,
        constraint_type=constraint_type,
        num_sectors=pcfg.get("num_sectors", 10),
    )
    wall = time.time() - t0

    out_dir = experiment_dir("diagnostics",
                              f"lambda_calib_{args.size}_ib{args.ib:g}_ds{args.dual_step:g}")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== lambda trajectory stats ===")
    print(f"global mean = {result['overall']['lambda_global_mean']:.4g}")
    print(f"global max  = {result['overall']['lambda_global_max']:.4g}")
    print(f"global q99  = {result['overall']['lambda_global_q99']:.4g}")
    print(f"\nlambda at t=T-1 (noise):  mean={np.asarray(result['overall']['lambda_at_t_last_noise']).mean():.4g}")
    print(f"lambda at t=0 (data):      mean={np.asarray(result['overall']['lambda_at_t_first_data']).mean():.4g}")

    # Per-t trajectory (show every 50 steps)
    per_t = result["per_t"]
    print(f"\n{'t':>4s} {'lam_mean':>10s} {'lam_q95':>10s} {'lam_max':>10s} {'vio_mean':>10s}")
    for i, t in enumerate(per_t["t_values"]):
        if t % 50 == 0 or i < 3 or i > len(per_t["t_values"]) - 4:
            print(f"{t:>4d} {per_t['lambda_mean'][i]:>10.4g} "
                  f"{per_t['lambda_q95'][i]:>10.4g} {per_t['lambda_max'][i]:>10.4g} "
                  f"{per_t['violation_mean'][i]:>10.4g}")

    prior = propose_prior(result["overall"])
    print(f"\n=== Proposed exponential lambda prior ===")
    print(f"  mu_min         = {prior['mu_min']:.4g}  (from lambda@t=0 mean)")
    print(f"  mu_max (q99)   = {prior['mu_max_q99match']:.4g}  (from lambda_max / 4.6)")
    print(f"  mu_max (10x)   = {prior['mu_max_10x']:.4g}")
    print(f"  mu_max (100x)  = {prior['mu_max_100x']:.4g}  (HANDOFF §13.1 recommends sweep)")
    (out_dir / "proposed_prior.json").write_text(json.dumps(prior, indent=2))

    # Plot lambda trajectory
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    # t=0 is noise, t=T is data per project convention; reverse x-axis
    t_arr = np.asarray(per_t["t_values"])
    # Convert to display-t: display = T-1 - t_int
    display_t = (args.T - 1) - t_arr
    axes[0].plot(display_t, per_t["lambda_mean"], label="mean", color="C0", lw=2)
    axes[0].plot(display_t, per_t["lambda_q95"], label="q95", color="C1", lw=1.5, alpha=0.7)
    axes[0].plot(display_t, per_t["lambda_max"], label="max", color="C3", lw=1, alpha=0.6)
    axes[0].set_ylabel("lambda")
    axes[0].set_title(f"Shared-lambda trajectory  (size={args.size}, ib={args.ib}, ds={args.dual_step})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(display_t, per_t["violation_mean"], color="C4", lw=2)
    axes[1].axhline(0, ls="--", color="k", alpha=0.5)
    axes[1].set_ylabel("mean violation (risk - budget)")
    axes[1].set_xlabel("display-t  (0=noise, T-1=data)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "lambda_trajectory.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: {out_dir}")
    print(f"Trajectory plot: {out_dir / 'lambda_trajectory.png'}")


if __name__ == "__main__":
    main()
