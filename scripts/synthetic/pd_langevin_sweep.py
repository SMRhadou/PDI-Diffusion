#!/usr/bin/env python3
"""Sweep PD Langevin primal x dual learning rates.

Produces in `--run-dir`:
  fig_pd_metrics_heatmap.png  (4 panels: feas/U/vio/|M_feas| heatmaps)
  fig_pd_sample_evolution_dual_{eta}.png (one per dual lr; rows=primal, cols=snapshots)
  fig_pd_dual_evolution.png   (panels per dual lr; lines per primal lr)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
})

# Ensure scripts/synthetic/ is on sys.path for sibling imports
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _shared import build_problem, experiment_dir, PRESETS  # noqa: E402
from legacy.generate_figures import _draw_constraints, _draw_modes  # noqa: E402


def _pd_langevin_with_trace(problem, B, T, lr, dual_lr, device, seed,
                            snapshot_steps, lam0=0.0):
    """Canonical PD-Langevin: π(x) ∝ exp(-U(x))·1{Ax>=b}, no temperature factor."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    means = problem["means"].to(device).float()
    cov = problem["cov_diag"].to(device).float()
    log_w = torch.log(problem["weights"].to(device).float())
    A = problem["constraint_A"].to(device).float()
    b = problem["constraint_b"].to(device).float()
    m_real = int((A.abs().sum(dim=1) > 0).sum().item())
    A_real = A[:m_real]
    b_real = b[:m_real]
    d = int(problem["d"])

    x = torch.randn((B, d), generator=g, device=device)
    lam = torch.full((B, m_real), float(lam0), device=device)
    log_norm = -0.5 * d * math.log(2 * math.pi) - 0.5 * torch.log(cov).sum(dim=1)

    snaps = {}
    if 0 in snapshot_steps:
        snaps[0] = x.detach().cpu().clone()
    lam_trace = [lam.detach().cpu().clone()]
    for step in range(1, T + 1):
        diff = x.unsqueeze(1) - means.unsqueeze(0)
        mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)
        log_comp = log_w + log_norm - 0.5 * mahal
        resp = torch.softmax(log_comp, dim=1)
        grad_logp = (resp.unsqueeze(-1) * (-diff) / cov.unsqueeze(0)).sum(dim=1)
        grad_U = -grad_logp
        drift = -(grad_U - lam @ A_real)
        noise = torch.randn_like(x)
        x = x + lr * drift + math.sqrt(2 * lr) * noise
        h = b_real - (x @ A_real.T)
        lam = torch.clamp(lam + dual_lr * h, min=0.0)
        if step in snapshot_steps:
            snaps[step] = x.detach().cpu().clone()
        lam_trace.append(lam.detach().cpu().clone())
    return snaps, torch.stack(lam_trace, dim=0).numpy()


def _apply_tweaks(problem, copy_mode_cov, shift_mode):
    if copy_mode_cov is not None:
        src, dst = (int(x) for x in copy_mode_cov.split(":"))
        problem["cov_diag"][dst] = problem["cov_diag"][src].clone()
        problem["cov_diag_original"][dst] = problem["cov_diag_original"][src].clone()
        print(f"[sweep] tweak: copied cov of mode {src} onto mode {dst}")
    if shift_mode is not None:
        idx, delta = shift_mode.split(":")
        idx, delta = int(idx), float(delta)
        A = problem["constraint_A"]
        b = problem["constraint_b"]
        mu = problem["means"][idx]
        slacks = A @ mu - b
        tight_j = int(slacks.argmin().item())
        normal = A[tight_j] / A[tight_j].norm().clamp_min(1e-12)
        problem["means"][idx] = mu + delta * normal
        problem["means_original"][idx] = (
            problem["means"][idx] * problem["norm_scale"] + problem["norm_shift"]
        )
        print(f"[sweep] tweak: shifted mode {idx} by {delta}")


def _compute_metrics(xf, problem, device, d, K, mahal_threshold):
    A_dev = problem["constraint_A"].to(device).float()
    b_dev = problem["constraint_b"].to(device).float()
    m_real = int((A_dev.abs().sum(dim=1) > 0).sum().item())
    A_real = A_dev[:m_real]
    b_real = b_dev[:m_real]
    means = problem["means"].to(device).float()
    cov = problem["cov_diag"].to(device).float()
    weights = problem["weights"].to(device).float()
    log_w = torch.log(weights)
    log_norm = -0.5 * d * math.log(2 * math.pi) - 0.5 * torch.log(cov).sum(dim=1)
    centers_feas = ((means @ A_real.T) >= b_real).all(dim=1)

    h = b_real - (xf @ A_real.T)
    feas = (h.mean(dim=0) <= 0).float().mean().item()
    vio = torch.clamp(h.mean(dim=0), min=0).mean().item()
    diff = xf.unsqueeze(1) - means.unsqueeze(0)
    mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)
    log_comp = log_w + log_norm - 0.5 * mahal
    logp = torch.logsumexp(log_comp, dim=1)
    U_mean = (-logp).mean().item()
    assign = log_comp.argmax(dim=1)
    min_mahal = mahal.min(dim=1).values
    assign = torch.where(min_mahal > mahal_threshold,
                         torch.full_like(assign, -1), assign)
    per_sample_feas = (h <= 0).all(dim=1)
    feas_mode_counts = torch.zeros(K, dtype=torch.long, device=device)
    for k in range(K):
        mask_k = (assign == k)
        if bool(centers_feas[k]):
            feas_mode_counts[k] = mask_k.sum()
        else:
            feas_mode_counts[k] = (mask_k & per_sample_feas).sum()
    uniq_feas = int((feas_mode_counts > 0).sum().item())
    n_unlab = int((assign == -1).sum().item())
    return dict(feas=feas, U_mean=U_mean, vio_mean=vio,
                uniq_feas=uniq_feas, n_unlab=n_unlab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True,
                    help="Label for the diagnostics experiment dir.")
    ap.add_argument("--difficulty", default="easy", choices=list(PRESETS))
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--n-snapshots", type=int, default=5)
    ap.add_argument("--primal-lrs", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2, 1e-1])
    ap.add_argument("--dual-lrs", type=float, nargs="+",
                    default=[3e-3, 3e-2, 1e-1, 3e-1])
    ap.add_argument("--lam0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-mode-cov", type=str, default=None)
    ap.add_argument("--shift-mode", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = PRESETS[args.difficulty]
    d = preset["d"]
    problem = build_problem(args.difficulty, seed=0)
    _apply_tweaks(problem, args.copy_mode_cov, args.shift_mode)

    run_dir = experiment_dir("diagnostics", args.label)
    print(f"[sweep] output: {run_dir}")

    snapshot_steps = sorted({
        int(round(k * args.T / (args.n_snapshots - 1)))
        for k in range(args.n_snapshots)
    })
    print(f"[sweep] snapshot steps: {snapshot_steps}")
    print(f"[sweep] primal_lrs={args.primal_lrs}  dual_lrs={args.dual_lrs}")

    K = int(problem["K"])
    try:
        from scipy.stats import chi2  # type: ignore
        mahal_threshold = float(chi2.ppf(0.95, d))
    except Exception:
        z = 1.6449
        mahal_threshold = d * (1 - 2 / (9 * d) + z * (2 / (9 * d)) ** 0.5) ** 3

    n_p = len(args.primal_lrs)
    n_d = len(args.dual_lrs)
    n_l = len(args.lam0)
    grid = {}
    # Heatmaps aggregate over lam0 by taking the best Pareto per (plr, dlr).
    feas_M = np.zeros((n_p, n_d))
    U_M = np.zeros((n_p, n_d))
    vio_M = np.zeros((n_p, n_d))
    cov_M = np.zeros((n_p, n_d))

    print(f"\n[sweep] metrics (final samples, N={args.N}):")
    print(f"{'primal':>8s} {'dual':>8s} {'lam0':>6s} {'feas':>6s} {'U':>9s} "
          f"{'vio':>9s} {'|M_f|':>7s}")
    for i, plr in enumerate(args.primal_lrs):
        for j, dlr in enumerate(args.dual_lrs):
            best_pareto = -1e9
            for l0 in args.lam0:
                snaps, lam = _pd_langevin_with_trace(
                    problem, args.N, args.T, plr, dlr,
                    device=device, seed=args.seed,
                    snapshot_steps=snapshot_steps, lam0=l0,
                )
                xf = snaps[snapshot_steps[-1]].to(device)
                m = _compute_metrics(xf, problem, device, d, K, mahal_threshold)
                grid[(plr, dlr, l0)] = dict(snaps=snaps, lam=lam, **m)
                pareto = m["feas"] - 0.01 * m["U_mean"]
                if pareto > best_pareto:
                    best_pareto = pareto
                    feas_M[i, j] = m["feas"]
                    U_M[i, j] = m["U_mean"]
                    vio_M[i, j] = m["vio_mean"]
                    cov_M[i, j] = m["uniq_feas"]
                print(f"{plr:>8g} {dlr:>8g} {l0:>6g} {m['feas']:>6.3f} "
                      f"{m['U_mean']:>9.3f} {m['vio_mean']:>9.4f} "
                      f"{m['uniq_feas']:>3d}/{K:<2d}")

    A_np = problem["constraint_A"].numpy()
    b_np = problem["constraint_b"].numpy()
    m_n = int((np.abs(A_np).sum(axis=1) > 0).sum())
    scale2 = problem["norm_scale"][:2].numpy()
    shift2 = problem["norm_shift"][:2].numpy()
    means_orig = problem["means_original"][:, :2].numpy()
    cov_orig = problem["cov_diag_original"][:, :2].numpy()
    K_tot = int(problem["K"])
    ns2 = problem["norm_scale"][:2].numpy()
    nsh2 = problem["norm_shift"][:2].numpy()

    mode_lo = means_orig - 2 * np.sqrt(cov_orig)
    mode_hi = means_orig + 2 * np.sqrt(cov_orig)
    pad = 2.0
    x_lo = mode_lo[:, 0].min() - pad
    x_hi = mode_hi[:, 0].max() + pad
    y_lo = mode_lo[:, 1].min() - pad
    y_hi = mode_hi[:, 1].max() + pad
    xr = x_hi - x_lo
    yr = y_hi - y_lo
    if xr > yr:
        mid = (y_lo + y_hi) / 2
        y_lo = mid - xr / 2
        y_hi = mid + xr / 2
    else:
        mid = (x_lo + x_hi) / 2
        x_lo = mid - yr / 2
        x_hi = mid + yr / 2

    # ---- Figure 1: metric heatmaps ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    specs = [
        (feas_M, r"Feasibility $\uparrow$", "viridis", "%.2f"),
        (U_M,    r"Objective $f_0(\mathbf{x})$ $\downarrow$", "viridis_r", "%.2f"),
        (vio_M,  r"Mean violation $\downarrow$", "viridis_r", "%.3f"),
        (cov_M,  r"$|M_{\mathrm{feas}}|$ $\uparrow$",  "viridis", "%d"),
    ]
    for ax, (M, title, cmap_name, fmt) in zip(axes, specs):
        im = ax.imshow(M, aspect="auto", cmap=cmap_name, origin="lower")
        ax.set_xticks(range(n_d))
        ax.set_xticklabels([f"{lr:g}" for lr in args.dual_lrs], fontsize=14)
        ax.set_yticks(range(n_p))
        ax.set_yticklabels([f"{lr:g}" for lr in args.primal_lrs], fontsize=14)
        ax.set_xlabel(r"dual lr $\eta$", fontsize=16)
        ax.set_ylabel(r"primal lr", fontsize=16)
        ax.set_title(title, fontsize=18)
        for i in range(n_p):
            for j in range(n_d):
                ax.text(j, i, fmt % M[i, j], ha="center", va="center",
                        fontsize=12, color="white" if M[i, j] < M.mean() else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p0 = run_dir / "fig_pd_metrics_heatmap.png"
    fig.savefig(p0, dpi=150, bbox_inches="tight")
    fig.savefig(p0.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p0}")

    # ---- Figure 2: one sample-evolution figure per dual lr ----
    n_panels = len(snapshot_steps)
    pw = 3.2
    for dlr in args.dual_lrs:
        fig, axes = plt.subplots(n_p, n_panels,
                                 figsize=(pw * n_panels, pw * n_p),
                                 sharex=True, sharey=True)
        if n_p == 1:
            axes = axes[None, :]
        for i, plr in enumerate(args.primal_lrs):
            for j, step in enumerate(snapshot_steps):
                ax = axes[i][j]
                _draw_constraints(ax, A_np, b_np, m_n, scale2, shift2,
                                  (x_lo, x_hi), (y_lo, y_hi))
                _draw_modes(ax, means_orig, cov_orig, K_tot, marker_size=6)
                snaps = grid[(plr, dlr, args.lam0[0])]["snaps"]
                if step in snaps:
                    pts = snaps[step][:, :2].numpy() * ns2 + nsh2
                    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.45,
                               color="#2166ac", edgecolors="none", zorder=5)
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
                ax.grid(alpha=0.15)
                if i == 0:
                    ax.set_title(f"step {step}", fontsize=15)
                if i == n_p - 1:
                    ax.set_xlabel(r"$x_1$", fontsize=13)
                if j == 0:
                    ax.set_ylabel(rf"primal$={plr:g}$", fontsize=14)
        fig.suptitle(rf"PD Langevin samples — dual lr $\eta={dlr:g}$",
                     fontsize=17)
        fig.tight_layout()
        p1 = run_dir / f"fig_pd_sample_evolution_dual{dlr:g}.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        fig.savefig(p1.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p1}")

    # ---- Figure 3: dual evolution — one panel per dual lr ----
    fig, axes = plt.subplots(1, n_d, figsize=(5.5 * n_d, 5.2), sharey=False)
    if n_d == 1:
        axes = [axes]
    cmap = plt.cm.viridis
    for j, dlr in enumerate(args.dual_lrs):
        ax = axes[j]
        for i, plr in enumerate(args.primal_lrs):
            lam = grid[(plr, dlr, args.lam0[0])]["lam"]
            T_rec = lam.shape[0]
            steps = np.arange(T_rec)
            mean_ = lam.reshape(T_rec, -1).mean(axis=1)
            color = cmap(i / max(n_p - 1, 1))
            ax.plot(steps, mean_, lw=2.0, color=color,
                    label=f"primal={plr:g}")
        ax.set_title(rf"$\eta={dlr:g}$", fontsize=18)
        ax.set_xlabel("Langevin step $k$", fontsize=14)
        if j == 0:
            ax.set_ylabel(r"mean $\lambda_{b,j}(k)$", fontsize=14)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="both", labelsize=12)
    axes[0].legend(fontsize=11, loc="best")
    fig.tight_layout()
    p2 = run_dir / "fig_pd_dual_evolution.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    fig.savefig(p2.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}")


if __name__ == "__main__":
    main()
