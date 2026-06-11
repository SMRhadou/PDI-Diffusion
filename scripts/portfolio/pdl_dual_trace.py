#!/usr/bin/env python3
"""Run PD-Langevin with dual variable tracing and plot per-constraint panels."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import experiment_dir, _project_root

_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (
    make_portfolio_problem,
)
from baselines import _compute_constraints


def pd_langevin_traced(mu, Sigma, scenarios, budgets, alpha, B,
                       num_iters=500, primal_lr=0.01, dual_lr=1000.0,
                       noise_scale=0.01, device=None, seed=42,
                       constraint_type="variance", shared_lambda=False):
    torch.manual_seed(seed)
    N = len(mu)
    z = torch.randn(B, N, device=device)
    if shared_lambda:
        lam = torch.zeros(1, N, device=device)
    else:
        lam = torch.zeros(B, N, device=device)

    lam_trace = []

    for k in range(num_iters):
        z_req = z.detach().requires_grad_(True)
        x = torch.softmax(z_req, dim=-1)
        c = _compute_constraints(x, Sigma, scenarios, alpha, constraint_type)
        expected_return = torch.matmul(scenarios, x.t()).mean(dim=0)
        violation = c - budgets.unsqueeze(0)
        lag_pen = (lam * violation).sum(dim=1)
        energy = -expected_return + lag_pen

        grad = torch.autograd.grad(energy.sum(), z_req)[0]
        noise = math.sqrt(2 * primal_lr) * torch.randn_like(z) if k < num_iters - 1 else 0.0
        z = (z_req - primal_lr * grad + noise).detach()

        with torch.no_grad():
            x_now = torch.softmax(z, dim=-1)
            c_now = _compute_constraints(x_now, Sigma, scenarios, alpha, constraint_type)
            v = c_now - budgets.unsqueeze(0)
            if shared_lambda:
                v_mean = v.mean(dim=0, keepdim=True)
                lam = (lam + dual_lr * v_mean).clamp_min(0.0)
            else:
                lam = (lam + dual_lr * v).clamp_min(0.0)

        lam_trace.append(lam.detach().cpu().clone())

    w = torch.softmax(z, dim=-1).detach()
    lam_trace = torch.stack(lam_trace)  # [T, B or 1, N]
    return w, lam_trace


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu, Sigma, scen, bud, alpha = make_portfolio_problem(
        N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
        structure="sectors", num_sectors=10,
        constraint_type="variance", budget_type="uniform")

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    Sigma_t = torch.tensor(Sigma, dtype=torch.float32, device=device)
    scen_t = torch.tensor(scen, dtype=torch.float32, device=device)
    bud_t = torch.tensor(bud, dtype=torch.float32, device=device)

    B = 64
    w, lam_trace = pd_langevin_traced(
        mu_t, Sigma_t, scen_t, bud_t, alpha, B=B,
        num_iters=500, primal_lr=0.1, dual_lr=1000.0,
        noise_scale=0.01, device=device, seed=42,
        constraint_type="variance", shared_lambda=False)

    tr = lam_trace.numpy()  # [T, B, N]
    T_steps = tr.shape[0]

    # Pick 8 constraints: highest final λ mean across batch
    final_lam_mean = tr[-1].mean(axis=0)  # [N]
    top_constraints = np.argsort(final_lam_mean)[-8:][::-1]

    # Pick 10 samples from batch
    sample_idx = np.linspace(0, B - 1, 10, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    for i, j in enumerate(top_constraints):
        ax = axes[i]
        for s in sample_idx:
            ax.plot(np.arange(T_steps), tr[:, s, j], alpha=0.5, linewidth=0.8)
        ax.set_title(f"constraint j={j}\n" + r"$\bar{\lambda}_{final}=$" + f"{final_lam_mean[j]:.2f}")
        ax.set_ylabel(r"$\lambda_j$")
        ax.grid(alpha=0.3)
        if i >= 4:
            ax.set_xlabel("PDL iteration")

    fig.suptitle(f"PD-Langevin dual traces (B={B}, 10 samples shown per panel)", fontsize=14)
    fig.tight_layout()

    out_dir = experiment_dir("diagnostics", "pdl_dual_trace_crypto")
    fig.savefig(out_dir / "fig_pdl_dual_trace.png", dpi=150)
    fig.savefig(out_dir / "fig_pdl_dual_trace.pdf")
    plt.close(fig)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
