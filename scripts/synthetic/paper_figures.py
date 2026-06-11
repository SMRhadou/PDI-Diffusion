#!/usr/bin/env python3
"""Paper-ready figures for the synthetic CED experiment.

One CLI invocation produces all five figures used in the paper:

    1. fig_bars                — feasibility / objective / mode coverage / wall
    2. fig_dual_evolution      — mean |λ| over diffusion / MCMC time
    3. fig_sample_evolution    — LDA-2D snapshots, samples coloured red/blue
                                 by initial feasibility
    4. fig_vector_field        — same layout as (3) with drift quiver overlay
    5. fig_confusion           — initial-mode → final-mode transition matrix

PD-Langevin and CED-ceiling hyperparameters are loaded from
``paper_baselines.PAPER_BASELINES`` (one source-of-truth per preset). An
optional ``--ced-train-dir`` adds the trained CED student to every figure.

Usage:

    python -m scripts.synthetic.paper_figures \\
        --label medium_uniform_v1 \\
        --preset medium_uniform \\
        [--ced-train-dir runs/synthetic/ced_xxx]

Outputs land under ``outputs/synthetic_boltzmann/diagnostics/<ts> - <label>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from _shared import (  # noqa: E402
    build_problem,
    build_sampler,
    compute_metrics,
    load_score_net,
    wire_trained_net,
    project_polytope,
    _noop_dual,
    PRESETS,
)
from _experiment_paths import experiment_dir  # noqa: E402
from langevin_mc_confusion_combined import (  # noqa: E402
    _build_confusion,
    _ced_sample,
    _drift_ddpm_full,
    _drift_langevin_full,
    _draw_heatmap,
    _install_x_snap_hook,
    _lda_basis,
    _mc_sample_with_traces,
    _pd_langevin_with_traces,
)
from paper_baselines import PAPER_BASELINES  # noqa: E402
from noise_schedule_comparison import build_alphas_cumprod, override_schedule  # noqa: E402

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{bm}",
})

# Method styling — keep consistent across all figures.
_BLUE_ORDER = ["PDL", "CED-ceiling", "CED (ours)", "PDM", "DPS", "Unconstrained"]
DISPLAY_NAME = {
    "CED-ceiling": "PDI -- MC",
    "CED (ours)":  "PDI -- Net",
    "PDL":         "PDL",
    "PDM":         "PDM",
    "DPS":         "DPS",
    "Unconstrained": "Unconstrained",
}
def _dn(name: str) -> str:
    return DISPLAY_NAME.get(name, name)

def _blue_colors(methods: list[str]) -> dict[str, str]:
    n = max(len(methods), 1)
    return {m: plt.cm.Blues(0.35 + 0.5 * i / max(n - 1, 1))
            for i, m in enumerate(methods)}

COLOR = _blue_colors(_BLUE_ORDER)


# ---------------------------------------------------------------------------
# Sampler runners
# ---------------------------------------------------------------------------
def _run_rejection(problem: dict, n_target: int, device: torch.device,
                   seed: int = 99) -> tuple[torch.Tensor, float]:
    """Rejection-sample the constrained MoG (gold standard)."""
    means = problem["means"].cpu().numpy()
    cov_diag = problem["cov_diag"].cpu().numpy()
    weights = problem["weights"].cpu().numpy()
    A_full = problem["constraint_A"].cpu().numpy()
    b_full = problem["constraint_b"].cpu().numpy()
    m_real = int((np.abs(A_full).sum(1) > 0).sum())
    A = A_full[:m_real]; b = b_full[:m_real]
    K = problem["K"]

    accepted = []
    rng = np.random.RandomState(seed)
    t0 = time.time()
    max_tries = 200 * n_target
    tries = 0
    while len(accepted) < n_target and tries < max_tries:
        ks = rng.choice(K, size=10000, p=weights)
        xs = np.stack([
            rng.multivariate_normal(means[k], np.diag(cov_diag[k]))
            for k in ks
        ])
        feas_mask = (xs @ A.T >= b).all(axis=1)
        accepted.extend(xs[feas_mask].tolist())
        tries += len(ks)
    wall = time.time() - t0
    if len(accepted) == 0:
        print(f"[rejection] 0 feasible samples after {tries} tries ({wall:.1f}s)")
        x = torch.zeros((0, means.shape[1]), dtype=torch.float32, device=device)
        return x, wall
    x = torch.tensor(np.stack(accepted[:n_target]), dtype=torch.float32,
                     device=device)
    return x, wall


# ---------------------------------------------------------------------------
# 1. Bars figure
# ---------------------------------------------------------------------------
def fig_bars(summary: dict, out_dir: Path, **_kw) -> None:
    """Four panels: feasibility, objective U, feasible-mode coverage, wall."""
    method_order = [m for m in _BLUE_ORDER if m in summary]
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5))
    panels = [
        ("feas_avg",              r"Avg.-constraint feasibility $\uparrow$", (0, 1.05)),
        ("U",                     r"Objective $f_0(\bm{x})$ $\downarrow$",   None),
        ("modes_coverage",    r"Mode coverage",                          (0, 1.05)),
        ("wall",                  r"Wall time (s) $\downarrow$",             None),
    ]
    for ax, (key, title, ylim) in zip(axes, panels):
        vals = [summary[m][key] for m in method_order]
        cols = [COLOR.get(m, "#777777") for m in method_order]
        bars = ax.bar(range(len(method_order)), vals, color=cols,
                      edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels([_dn(m) for m in method_order], rotation=20, ha="right", fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(alpha=0.2, axis="y")
        if ylim is not None:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=18)
    fig.tight_layout()
    p_png = out_dir / "fig_bars.png"
    p_pdf = out_dir / "fig_bars.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


# ---------------------------------------------------------------------------
# 2. Dual evolution figure (mean |λ|)
# ---------------------------------------------------------------------------
def _install_violation_recorder(sampler):
    """Wrap _dual_ascent_step to also record per-step violation."""
    sampler._violation_traces = []
    orig = sampler._dual_ascent_step

    def _wrapper(self, dual_lambda, ergodic_rates, context, t=None):
        viol = (context.r_min.view(-1, 1) - ergodic_rates).mean(dim=0)  # [m]
        self._violation_traces.append(viol.detach().cpu())
        return orig(dual_lambda, ergodic_rates, context, t)

    sampler._dual_ascent_step = types.MethodType(_wrapper, sampler)


def _install_t0_gate(sampler, t0, T):
    """Skip dual updates for the first t0 user-time steps (t < t0).
    User-time t = T-1-k, so t < t0  <=>  DDPM index k > T-1-t0."""
    if t0 <= 0:
        return
    threshold_k = T - 1 - int(t0)
    orig = sampler._dual_ascent_step

    def _gated(self, dual_lambda, ergodic_rates, context, t=None):
        if t is not None and t.numel() > 0:
            if int(t[0].item()) > threshold_k:
                if getattr(self, "_record_dual", False):
                    m_rec = self._syn_m
                    self._dual_traces.append(
                        dual_lambda[self._record_indices, :m_rec].detach().cpu()
                    )
                return dual_lambda
        return orig(dual_lambda, ergodic_rates, context, t)

    sampler._dual_ascent_step = types.MethodType(_gated, sampler)


def _mc_sample_with_schedule(problem, B, T, ib, dual_lr, schedule_name,
                              device, seed, snap_steps=None, t0=0):
    """Run CED-ceiling with a custom noise schedule, return dual traces + snaps + violations."""
    d = int(problem["d"])
    s = build_sampler(
        problem, num_timesteps=T, inverse_beta=ib,
        dual_step_size=dual_lr, shared_lambda=True, device=device,
    )
    ac = build_alphas_cumprod(schedule_name, T)
    override_schedule(s, ac)
    _install_violation_recorder(s)
    if t0 > 0:
        _install_t0_gate(s, t0, T)
    shape = (B, 1, d, 1)
    torch.manual_seed(int(seed))
    s.enable_dual_recording(list(range(B)))
    x_snaps = _install_x_snap_hook(s, snap_steps or [])
    x0_out = s.sample(shape=shape, device=device)
    traces = s.get_dual_traces().cpu().numpy()
    violations = torch.stack(s._violation_traces).numpy()  # [T, m]
    s.disable_dual_recording()
    x0_flat = x0_out[:, 0, :, 0].detach()
    return traces, x_snaps, x0_flat, violations


SCHED_NAMES = {
    "linear": "Linear",
    "poly2": r"Polynomial ($p=2$)",
    "exp": "Exponential",
    "cosine": "Cosine (DDPM)",
}
SCHED_COLORS = {
    "linear": "#1f77b4",
    "poly2": "#2ca02c",
    "exp": "#ff7f0e",
    "cosine": "#9467bd",
}


def fig_dual_evolution(schedule_traces: dict[str, np.ndarray],
                       out_dir: Path) -> None:
    """Overlay ‖λ‖₂ for CED-ceiling under the 4 noise schedules."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for name in ["linear", "poly2", "exp", "cosine"]:
        trc = schedule_traces[name]                        # [T_rec, B, m]
        norms = np.linalg.norm(trc, axis=2)[:, 0]         # shared → all same
        step = np.arange(len(norms))
        ax.plot(step, norms, color=SCHED_COLORS[name], lw=2.2,
                label=SCHED_NAMES[name])
    ax.set_xlabel(r"iteration $t$", fontsize=22)
    ax.set_ylabel(r"$\|\bm{\lambda}\|_2$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=15, loc="best", framealpha=0.9)
    fig.tight_layout()
    p_png = out_dir / "fig_dual_evolution.png"
    p_pdf = out_dir / "fig_dual_evolution.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


def fig_dual_ced_vs_mc(traces_by_method: dict[str, np.ndarray],
                       out_dir: Path) -> None:
    """Compare ‖λ‖₂ between CED (trained) and CED-ceiling (MC)."""
    methods = [m for m in ("CED-ceiling", "CED (ours)") if m in traces_by_method]
    if len(methods) < 2:
        return
    mc_colors = {"CED-ceiling": "#2171b5", "CED (ours)": "#cb181d"}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for m in methods:
        trc = traces_by_method[m]                         # [T_rec, B, m]
        norms = np.linalg.norm(trc, axis=2)[:, 0]
        step = np.arange(len(norms))
        ax.plot(step, norms, color=mc_colors[m], lw=2.2, label=_dn(m))
    ax.set_xlabel(r"iteration $t$", fontsize=22)
    ax.set_ylabel(r"$\|\bm{\lambda}\|_2$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=15, loc="best", framealpha=0.9)
    fig.tight_layout()
    p_png = out_dir / "fig_dual_ced_vs_mc.png"
    p_pdf = out_dir / "fig_dual_ced_vs_mc.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


def fig_violation(schedule_violations: dict[str, np.ndarray],
                  schedule_traces: dict[str, np.ndarray],
                  out_dir: Path) -> None:
    """Fraction violated, mean clamped violation, and CS residual."""
    fig, (ax_frac, ax_mean, ax_cs) = plt.subplots(1, 3, figsize=(21, 5))
    for name in ["linear", "poly2", "exp", "cosine"]:
        viol_raw = schedule_violations[name]                 # [T, m]
        trc = schedule_traces[name]                          # [T, B, m]
        lam = trc[:, 0, :]                                   # shared → [T, m]
        m_dim = min(viol_raw.shape[1], lam.shape[1])
        viol_raw = viol_raw[:, :m_dim]
        lam = lam[:, :m_dim]
        viol = np.maximum(viol_raw, 0.0)
        step = np.arange(viol.shape[0])
        viol_max = viol.max(axis=1)
        viol_mean = viol.mean(axis=1)
        cs = np.abs(lam * viol_raw).mean(axis=1)
        ax_frac.plot(step, viol_max, color=SCHED_COLORS[name], lw=2.2,
                     label=SCHED_NAMES[name])
        ax_mean.plot(step, viol_mean, color=SCHED_COLORS[name], lw=2.2,
                     label=SCHED_NAMES[name])
        ax_cs.plot(step, cs, color=SCHED_COLORS[name], lw=2.2,
                   label=SCHED_NAMES[name])
    ax_frac.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_frac.set_xlabel(r"iteration $t$", fontsize=22)
    ax_frac.set_ylabel(r"$\max_j\; [v_j(t)]_+$", fontsize=24)
    ax_frac.tick_params(axis="both", labelsize=18)
    ax_frac.grid(alpha=0.3)
    ax_frac.legend(fontsize=13, loc="best", framealpha=0.9)
    ax_mean.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_mean.set_xlabel(r"iteration $t$", fontsize=22)
    ax_mean.set_ylabel(r"$\mathrm{mean}_j\; [v_j(t)]_+$", fontsize=24)
    ax_mean.tick_params(axis="both", labelsize=18)
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(fontsize=13, loc="best", framealpha=0.9)
    ax_cs.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_cs.set_xlabel(r"iteration $t$", fontsize=22)
    ax_cs.set_ylabel(r"$\mathrm{mean}_j\; |\lambda_j \cdot v_j|$", fontsize=24)
    ax_cs.tick_params(axis="both", labelsize=18)
    ax_cs.grid(alpha=0.3)
    ax_cs.legend(fontsize=13, loc="best", framealpha=0.9)
    fig.tight_layout()
    p_png = out_dir / "fig_violation.png"
    p_pdf = out_dir / "fig_violation.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


def fig_violation_ced_vs_mc(mc_violations: np.ndarray, mc_traces: np.ndarray,
                           ced_violations: np.ndarray | None,
                           ced_traces: np.ndarray | None,
                           out_dir: Path) -> None:
    """Max violation, mean violation, and complementary slackness: CED vs MC ceiling."""
    fig, (ax_max, ax_mean, ax_cs) = plt.subplots(1, 3, figsize=(21, 4))
    floor = 1e-4
    for label, viol_raw, trc, color in [
        (_dn("CED-ceiling"), mc_violations, mc_traces, "#4a90d9"),
        (_dn("CED (ours)"), ced_violations, ced_traces, "#d9534f"),
    ]:
        if viol_raw is None:
            continue
        lam = trc[:, 0, :]
        m_dim = min(viol_raw.shape[1], lam.shape[1])
        viol_raw_m = viol_raw[:, :m_dim]
        lam_m = lam[:, :m_dim]
        viol = np.maximum(viol_raw_m, 0.0)
        cs = np.abs(lam_m * viol_raw_m).mean(axis=1)
        step = np.arange(viol.shape[0])
        lam_norm = np.linalg.norm(lam_m, axis=1)
        ax_max.plot(step, np.clip(viol.max(axis=1), floor, None),
                    color=color, lw=3.0, label=label)
        ax_mean.plot(step, np.clip(lam_norm, floor, None),
                     color=color, lw=3.0, label=label)
        ax_cs.plot(step, np.clip(cs, floor, None),
                   color=color, lw=3.0, label=label)
    for ax, ylabel in [
        (ax_max, r"$\max_j\; [v_j(t)]_+$"),
        (ax_mean, r"$\|\bm{\lambda}\|_2$"),
        (ax_cs, r"$\mathrm{mean}_j\; |\lambda_j \cdot v_j|$"),
    ]:
        ax.set_yscale("log")
        ax.set_xlabel(r"iteration $t$", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=16, loc="best", framealpha=0.9)
    fig.tight_layout()
    p_png = out_dir / "fig_violation_ced_vs_mc.png"
    p_pdf = out_dir / "fig_violation_ced_vs_mc.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


def fig_t0_effect(problem, N, T, ib, dual_lr, device, seed, out_dir: Path,
                  t0_values=None) -> None:
    """Final mean constraint violation of output samples vs T0 warmup,
    one curve per noise schedule."""
    if t0_values is None:
        t0_values = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    A = problem["constraint_A"].to(device).float()
    b = problem["constraint_b"].to(device).float()
    m_real = int((A.abs().sum(1) > 0).sum().item())
    A = A[:m_real]; b = b[:m_real]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for sname in ["linear", "poly2", "exp", "cosine"]:
        feas_vals = []
        for t0_val in t0_values:
            print(f"[T0 sweep] {sname} T0={t0_val} ...", flush=True)
            _, _, x0, _ = _mc_sample_with_schedule(
                problem, N, T, ib, dual_lr, sname, device, seed, t0=t0_val,
            )
            ax_vals = x0.to(device) @ A.T
            fa = float(((ax_vals.mean(0) - b) >= 0).float().mean().item())
            feas_vals.append(fa)
        ax.plot(t0_values, feas_vals, "-o", color=SCHED_COLORS[sname],
                lw=2.2, ms=6, label=SCHED_NAMES[sname])
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(r"warmup $T_0$", fontsize=22)
    ax.set_ylabel(r"avg-constraint feasibility", fontsize=22)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14, loc="best", framealpha=0.9)
    fig.tight_layout()
    p_png = out_dir / "fig_t0_effect.png"
    p_pdf = out_dir / "fig_t0_effect.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


def fig_schedule_samples(schedule_snaps: dict[str, dict],
                         t_labels: list[int], T: int,
                         problem: dict, out_dir: Path) -> None:
    """LDA-2D sample evolution: one row per noise schedule, 5 time columns."""
    snaps_for_frame = {SCHED_NAMES[s]: sn for s, sn in schedule_snaps.items()}
    frame = _build_2d_frame(problem, snaps_for_frame)
    sched_order = ["linear", "poly2", "exp", "cosine"]
    n_rows = len(sched_order)
    n_cols = len(t_labels)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows),
        sharex=True, sharey=True,
    )
    for r, sname in enumerate(sched_order):
        snaps = schedule_snaps[sname]
        init_mask = _init_feas_mask(snaps, T - 1, frame)
        for c, t_label in enumerate(t_labels):
            ax = axes[r, c]
            _draw_bg(ax, frame)
            t_internal = T - 1 - t_label
            snap = snaps.get(int(t_internal))
            if snap is not None and init_mask is not None:
                xs = snap.reshape(snap.shape[0], -1) @ frame["W"]
                ax.scatter(xs[init_mask, 0], xs[init_mask, 1],
                           s=32, alpha=0.6, color="#d6604d",
                           edgecolors="none", zorder=6)
                ax.scatter(xs[~init_mask, 0], xs[~init_mask, 1],
                           s=32, alpha=0.6, color="#2166ac",
                           edgecolors="none", zorder=7)
            ax.set_xlim(frame["x_lo"], frame["x_hi"])
            ax.set_ylim(frame["y_lo"], frame["y_hi"])
            if r == 0:
                suffix = ""
                if c == 0:
                    suffix = r"  (noise)"
                elif c == n_cols - 1:
                    suffix = r"  (data)"
                ax.set_title(rf"$t={t_label}${suffix}", fontsize=20)
            if c == 0:
                ax.set_ylabel(SCHED_NAMES[sname], fontsize=20)
            ax.tick_params(axis="both", labelsize=14)
    for ax in axes[-1]:
        ax.set_xlabel(r"LDA axis 1", fontsize=18)
    fig.tight_layout()
    p_png = out_dir / "fig_schedule_samples.png"
    p_pdf = out_dir / "fig_schedule_samples.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


# ---------------------------------------------------------------------------
# Shared 2D-projection helpers (used by figures 3 and 4)
# ---------------------------------------------------------------------------
def _build_2d_frame(problem: dict, snaps_dict: dict[str, dict]) -> dict:
    """Build the LDA-2D frame and clip limits used by figs 3+4."""
    d_dim = int(problem["d"])
    means_np = problem["means"].cpu().numpy()
    A_np = problem["constraint_A"].cpu().numpy()
    b_np = problem["constraint_b"].cpu().numpy()
    m_real = int((np.abs(A_np).sum(1) > 0).sum())
    W = _lda_basis(means_np, A_np, b_np, m_real)            # [d, 2]
    K = means_np.shape[0]
    mode_ref = means_np[K - 1].copy()                       # last mode as anchor
    mode_ref_perp = mode_ref - W @ (W.T @ mode_ref)
    modes_2d = means_np @ W                                  # [K, 2]

    # Compute clip limits over all snapshots
    all_xs = [modes_2d[:, 0]]; all_ys = [modes_2d[:, 1]]
    for snaps in snaps_dict.values():
        if not snaps:
            continue
        for v in snaps.values():
            if v is None:
                continue
            v_flat = v.reshape(v.shape[0], -1)
            v_2d = v_flat @ W
            all_xs.append(v_2d[:, 0]); all_ys.append(v_2d[:, 1])
    x_lo = float(np.min(np.concatenate(all_xs)))
    x_hi = float(np.max(np.concatenate(all_xs)))
    y_lo = float(np.min(np.concatenate(all_ys)))
    y_hi = float(np.max(np.concatenate(all_ys)))
    pad_x = 0.2 * (x_hi - x_lo); pad_y = 0.2 * (y_hi - y_lo)
    x_lo -= pad_x; x_hi += pad_x; y_lo -= pad_y; y_hi += pad_y

    A_W = A_np[:m_real] @ W
    b_W = b_np[:m_real] - A_np[:m_real] @ mode_ref_perp

    cov_np = problem["cov_diag"].cpu().numpy()
    cov_2d_list = [(W.T @ (cov_np[k][:, None] * W)) for k in range(K)]

    return dict(
        W=W, mode_ref_perp=mode_ref_perp, modes_2d=modes_2d,
        x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
        A_W=A_W, b_W=b_W, m_real=m_real, K=K,
        cov_2d_list=cov_2d_list, A_np=A_np, b_np=b_np, d_dim=d_dim,
    )


def _draw_bg(ax, frame: dict) -> None:
    """Mode ellipses + constraint lines + infeasible shading on `ax`."""
    K = frame["K"]; m_real = frame["m_real"]
    A_W, b_W = frame["A_W"], frame["b_W"]
    modes_2d = frame["modes_2d"]; cov_2d_list = frame["cov_2d_list"]
    x_lo, x_hi = frame["x_lo"], frame["x_hi"]
    y_lo, y_hi = frame["y_lo"], frame["y_hi"]
    x_axis_line = np.linspace(x_lo - 2.0, x_hi + 2.0, 200)
    for j in range(m_real):
        a0, a1 = A_W[j, 0], A_W[j, 1]
        if abs(a1) > 1e-8:
            yy = (b_W[j] - a0 * x_axis_line) / a1
            ax.plot(x_axis_line, yy, color="#969696", lw=1.2, alpha=0.7,
                    zorder=1)
            lo_, hi_ = ((y_lo - 10, yy) if a1 > 0 else (yy, y_hi + 10))
            ax.fill_between(x_axis_line, lo_, hi_,
                            color="#d73027", alpha=0.06, zorder=0)
        elif abs(a0) > 1e-8:
            xv = b_W[j] / a0
            ax.axvline(xv, color="#969696", lw=1.2, alpha=0.7, zorder=1)
            if a0 > 0:
                ax.axvspan(x_lo - 10, xv, color="#d73027", alpha=0.06,
                           zorder=0)
            else:
                ax.axvspan(xv, x_hi + 10, color="#d73027", alpha=0.06,
                           zorder=0)
    for k in range(K):
        vals, vecs = np.linalg.eigh(cov_2d_list[k])
        order = np.argsort(vals)[::-1]
        vals = vals[order]; vecs = vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        for ns, alpha_e in [(2, 0.15), (1, 0.25)]:
            ell = Ellipse(
                xy=modes_2d[k],
                width=2 * ns * float(np.sqrt(max(vals[0], 0.0))),
                height=2 * ns * float(np.sqrt(max(vals[1], 0.0))),
                angle=angle,
                fill=True, facecolor="#fee0d2", edgecolor="#e31a1c",
                lw=1.2 if ns == 2 else 0,
                alpha=alpha_e, ls="--" if ns == 2 else "-",
                zorder=2,
            )
            ax.add_patch(ell)
        ax.plot(*modes_2d[k], "x", color="#e31a1c", ms=8, mew=2.5, zorder=10)


def _init_feas_mask(snaps: dict, init_key: int, frame: dict) -> np.ndarray | None:
    """Return [B] bool: True if the *initial* sample was infeasible."""
    if snaps is None or snaps.get(int(init_key)) is None:
        return None
    init_pts = snaps[int(init_key)].reshape(-1, frame["d_dim"])
    A = frame["A_np"][:frame["m_real"]]
    b = frame["b_np"][:frame["m_real"]]
    feas = (init_pts @ A.T >= b).all(axis=1)
    return ~feas


# ---------------------------------------------------------------------------
# 3. Sample evolution figure (samples-only, no quiver)
# ---------------------------------------------------------------------------
def fig_sample_evolution(snaps_by_method: dict, t_labels: list[int],
                         T: int, problem: dict, out_dir: Path,
                         T_by_method: dict | None = None) -> None:
    """Snapshots per method × timestep, samples coloured by initial
    feasibility (red = init infeasible, blue = init feasible).
    Uses original coordinates for d=2, LDA projection otherwise."""
    if T_by_method is None:
        T_by_method = {}
    d_dim = int(problem["d"])
    use_orig = (d_dim == 2)

    if use_orig:
        frame_2d = _orig_2d_frame(problem)
        A_feas = problem["constraint_A"].cpu().numpy()
        b_feas = problem["constraint_b"].cpu().numpy()
        m_feas = frame_2d["m_real"]
    else:
        frame = _build_2d_frame(problem, snaps_by_method)

    method_order = list(snaps_by_method.keys())
    n_rows = len(method_order); n_cols = len(t_labels)
    pw = 3.4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(pw * n_cols, pw * n_rows),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.18},
    )
    if n_rows == 1:
        axes = np.array([axes])

    for r, name in enumerate(method_order):
        snaps = snaps_by_method[name]
        kind = "langevin" if name == "PDL" else "ddpm"
        T_m = T_by_method.get(name, T)
        init_key = 0 if kind == "langevin" else (T_m - 1)

        # Compute init-feasibility mask
        init_snap = snaps.get(int(init_key)) if snaps else None
        init_mask = None
        if init_snap is not None:
            pts_flat = init_snap.reshape(init_snap.shape[0], -1)
            if use_orig:
                feas = (pts_flat @ A_feas[:m_feas].T >= b_feas[:m_feas]).all(axis=1)
            else:
                A_f = frame["A_np"][:frame["m_real"]]
                b_f = frame["b_np"][:frame["m_real"]]
                feas = (pts_flat @ A_f.T >= b_f).all(axis=1)
            init_mask = ~feas

        for c in range(n_cols):
            ax = axes[r, c]
            if use_orig:
                _draw_bg_orig(ax, frame_2d)
            else:
                _draw_bg(ax, frame)

            if kind == "langevin":
                frac = c / (n_cols - 1)
                t_internal = int(round(frac * (T_m - 1)))
            else:
                t_internal = T_m - 1 - t_labels[c]
            snap = snaps.get(int(t_internal)) if snaps is not None else None
            if snap is not None and init_mask is not None:
                pts_flat = snap.reshape(snap.shape[0], -1)
                if use_orig:
                    xs = pts_flat[:, :2] * frame_2d["scale"][:2] + frame_2d["shift"][:2]
                else:
                    xs = pts_flat @ frame["W"]
                ax.scatter(xs[init_mask, 0], xs[init_mask, 1],
                           s=32, alpha=0.6, color="#d6604d",
                           edgecolors="none", zorder=6,
                           label=r"init infeasible"
                                 if (r == 0 and c == n_cols - 1) else None)
                ax.scatter(xs[~init_mask, 0], xs[~init_mask, 1],
                           s=32, alpha=0.6, color="#2166ac",
                           edgecolors="none", zorder=7,
                           label=r"init feasible"
                                 if (r == 0 and c == n_cols - 1) else None)
            if use_orig:
                ax.set_xlim(frame_2d["xlim"]); ax.set_ylim(frame_2d["ylim"])
                ax.set_aspect("equal", adjustable="box")
            else:
                ax.set_xlim(frame["x_lo"], frame["x_hi"])
                ax.set_ylim(frame["y_lo"], frame["y_hi"])
            ax.grid(alpha=0.15)
            if r == 0:
                suffix = ""
                if c == 0:
                    suffix = r"  (noise)"
                elif c == n_cols - 1:
                    suffix = r"  (data)"
                ax.set_title(rf"$t={t_labels[c]}${suffix}", pad=6)
            if c == 0:
                ax.set_ylabel(f"{_dn(name)}\n" + (r"$x_2$" if use_orig else ""),
                              fontsize=16)
            else:
                ax.set_yticklabels([])
    for ax in axes[-1]:
        ax.set_xlabel(r"$x_1$" if use_orig else r"LDA axis 1 (feas vs infeas)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(handles, labels, fontsize=10, loc="lower right",
                           framealpha=0.9, markerscale=1.5)
    p_png = out_dir / "fig_sample_evolution.png"
    p_pdf = out_dir / "fig_sample_evolution.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


# ---------------------------------------------------------------------------
# 3b. fig_2d_panels — 2×N_ib grid (scatter top, shared-λ trajectory bottom)
# ---------------------------------------------------------------------------

def _orig_2d_frame(problem: dict):
    """Build axis limits and helpers for plotting in original 2D coordinates."""
    d = int(problem["d"])
    assert d == 2, f"fig_2d_panels requires d=2, got d={d}"
    scale = problem["norm_scale"].cpu().numpy()
    shift = problem["norm_shift"].cpu().numpy()
    means_orig = problem["means_original"].cpu().numpy()[:, :2]
    cov_orig = problem["cov_diag_original"].cpu().numpy()[:, :2]
    A_np = problem["constraint_A"].cpu().numpy()
    b_np = problem["constraint_b"].cpu().numpy()
    m_real = int((np.abs(A_np).sum(axis=1) > 0).sum())
    K = means_orig.shape[0]
    mode_sigma = np.sqrt(np.maximum(cov_orig, 0))
    mode_lo = means_orig - 2 * mode_sigma
    mode_hi = means_orig + 2 * mode_sigma
    pad = 4.0
    x_lo = mode_lo[:, 0].min() - pad; x_hi = mode_hi[:, 0].max() + pad
    y_lo = mode_lo[:, 1].min() - pad; y_hi = mode_hi[:, 1].max() + pad
    xr = x_hi - x_lo; yr = y_hi - y_lo
    if xr > yr:
        mid = (y_lo + y_hi) / 2; y_lo = mid - xr / 2; y_hi = mid + xr / 2
    else:
        mid = (x_lo + x_hi) / 2; x_lo = mid - yr / 2; x_hi = mid + yr / 2
    return dict(
        scale=scale, shift=shift, means_orig=means_orig, cov_orig=cov_orig,
        A_np=A_np, b_np=b_np, m_real=m_real, K=K,
        xlim=(x_lo, x_hi), ylim=(y_lo, y_hi),
    )


def _draw_bg_orig(ax, frame: dict):
    """Constraint half-planes + mode ellipses in original 2D coordinates."""
    A_np, b_np, m_real = frame["A_np"], frame["b_np"], frame["m_real"]
    scale2, shift2 = frame["scale"][:2], frame["shift"][:2]
    xlim, ylim = frame["xlim"], frame["ylim"]
    means_orig, cov_orig, K = frame["means_orig"], frame["cov_orig"], frame["K"]

    xx = np.linspace(xlim[0] - 5, xlim[1] + 5, 400)
    for j in range(m_real):
        a = A_np[j]; bv = b_np[j]
        a2 = a[:2] / scale2; b2 = bv + np.dot(a[:2] / scale2, shift2)
        if abs(a2[1]) > 1e-8:
            yy = (b2 - a2[0] * xx) / a2[1]
            vis = (yy >= ylim[0] - 5) & (yy <= ylim[1] + 5)
            ax.plot(xx[vis], yy[vis], color="#969696", lw=1.2, alpha=0.7)
            lo, hi = (ylim[0] - 10, yy) if a2[1] > 0 else (yy, ylim[1] + 10)
            ax.fill_between(xx, lo, hi, color="#d73027", alpha=0.04, zorder=0)
        elif abs(a2[0]) > 1e-8:
            xv = b2 / a2[0]
            ax.axvline(xv, color="#969696", lw=1.2, alpha=0.7)
            if a2[0] > 0:
                ax.axvspan(xlim[0] - 10, xv, color="#d73027", alpha=0.04, zorder=0)
            else:
                ax.axvspan(xv, xlim[1] + 10, color="#d73027", alpha=0.04, zorder=0)
    for k in range(K):
        for ns, alpha_e in [(2, 0.15), (1, 0.25)]:
            ell = Ellipse(
                xy=means_orig[k],
                width=2 * ns * np.sqrt(cov_orig[k, 0]),
                height=2 * ns * np.sqrt(cov_orig[k, 1]),
                fill=True, facecolor="#fee0d2", edgecolor="#e31a1c",
                lw=1.2 if ns == 2 else 0, alpha=alpha_e,
                ls="--" if ns == 2 else "-", zorder=2,
            )
            ax.add_patch(ell)
        ax.plot(*means_orig[k], "x", color="#e31a1c", ms=8, mew=2.5, zorder=10)


def _to_orig(xf_norm: np.ndarray, frame: dict) -> np.ndarray:
    """Map normalised samples [B, d] to original 2D coordinates."""
    return xf_norm[:, :2] * frame["scale"][:2] + frame["shift"][:2]


def fig_2d_panels(problem: dict, ibs: list[float], dual_step,
                  T: int, B: int, device: torch.device, seed: int,
                  out_dir: Path, beta_schedule: str = "linear") -> None:
    """2×len(ibs) figure.  Top: samples + modes + constraints in original 2D.
    Bottom: per-component dual variable over reverse steps.
    dual_step: float or dict {ib: ds} for per-ib dual step sizes."""
    frame = _orig_2d_frame(problem)
    xlim, ylim = frame["xlim"], frame["ylim"]

    samples_by_ib = {}
    traces_by_ib = {}

    for ib in ibs:
        ds = dual_step[ib] if isinstance(dual_step, dict) else dual_step
        print(f"[fig_2d_panels] ib={ib} ds={ds} ...", flush=True)
        _, xf, _, _, trc, _, _ = _mc_sample_with_traces(
            problem, B, T, ib, ds, device, seed,
            shared=True, beta_schedule=beta_schedule,
        )
        samples_by_ib[ib] = xf.cpu().numpy()
        traces_by_ib[ib] = trc

    n_ib = len(ibs)
    pw = 3.0
    fig, axes = plt.subplots(
        2, n_ib, figsize=(pw * n_ib, pw + pw * 0.5),
        gridspec_kw={"height_ratios": [1, 0.5],
                     "wspace": 0.05, "hspace": 0.30},
    )
    ax_top = axes[0]; ax_bot = axes[1]

    for i, ib in enumerate(ibs):
        ax = ax_top[i]
        _draw_bg_orig(ax, frame)
        pts = _to_orig(samples_by_ib[ib], frame)
        ax.scatter(pts[:, 0], pts[:, 1], s=32, alpha=0.6,
                   color="#2166ac", edgecolors="none", zorder=5)
        ax.set_title(rf"$\beta^{{-1}}={ib:g}$", pad=6)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(alpha=0.15)
        if i == 0:
            ax.set_ylabel(r"$x_2$")
        else:
            ax.set_yticklabels([])

    for i, ib in enumerate(ibs):
        ax = ax_bot[i]
        trc = traces_by_ib[ib]                      # [T, 1, m]
        ts = np.arange(trc.shape[0])
        m_dim = trc.shape[2]
        colors_lam = ["#2166ac", "#d6604d", "#2ca02c", "#9467bd"]
        for j in range(m_dim):
            ax.plot(ts, trc[:, 0, j], color=colors_lam[j % len(colors_lam)],
                    lw=1.5, label=rf"$\lambda_{j+1}$" if i == 0 else None)
        ax.set_xlabel("Reverse step")
        ax.grid(alpha=0.15)
        if i == 0:
            ax.set_ylabel(r"$\lambda$")
            ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        else:
            ax.set_yticklabels([])

    p_png = out_dir / "fig_2d_panels.png"
    p_pdf = out_dir / "fig_2d_panels.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


# ---------------------------------------------------------------------------
# 4. Vector field figure (samples + drift quiver)
# ---------------------------------------------------------------------------
def _build_vf_samplers(problem: dict, T: int, mc_cfg: dict, ced_cfg,
                       device: torch.device, args,
                       beta_schedule: str = "linear") -> tuple:
    """Build separate DDPM samplers for vector-field (so we can call drift at
    arbitrary t_internal). Returns (s_mc_vf, s_ced_vf or None)."""
    s_mc_vf = build_sampler(
        problem, num_timesteps=T, beta_schedule=beta_schedule,
        inverse_beta=mc_cfg["inverse_beta"],
        dual_step_size=mc_cfg["dual_step_size"],
        shared_lambda=True, device=device,
    )

    s_ced_vf = None
    if ced_cfg is not None:
        s_ced_vf = build_sampler(
            problem, num_timesteps=T, beta_schedule=beta_schedule,
            inverse_beta=ced_cfg["inverse_beta"],
            dual_step_size=ced_cfg["dual_step_size"],
            shared_lambda=True, device=device,
        )
        d_dim = int(problem["d"])
        ckpt_name = args.ced_ckpt_name or "score_net_best.pt"
        score_net = load_score_net(
            args.ced_train_dir, d_dim,
            hidden=args.hidden, num_layers=args.num_layers,
            num_timesteps=T, ckpt_name=ckpt_name, device=device,
        )
        wire_trained_net(s_ced_vf, score_net, d_dim)
    return s_mc_vf, s_ced_vf


def fig_vector_field(snaps_by_method: dict, traces_by_method: dict,
                     t_labels: list[int], T: int, problem: dict,
                     mc_cfg: dict, ced_cfg, device: torch.device,
                     out_dir: Path, args,
                     beta_schedule: str = "linear") -> None:
    """Drift vector field overlaid on snapshots.
    Uses original coordinates for d=2, LDA projection otherwise."""
    d_dim = int(problem["d"])
    use_orig = (d_dim == 2)

    if use_orig:
        frame_2d = _orig_2d_frame(problem)
        xlim, ylim = frame_2d["xlim"], frame_2d["ylim"]
        A_feas = problem["constraint_A"].cpu().numpy()
        b_feas = problem["constraint_b"].cpu().numpy()
        m_feas = frame_2d["m_real"]
    else:
        frame = _build_2d_frame(problem, snaps_by_method)

    method_order = list(snaps_by_method.keys())
    n_rows = len(method_order); n_cols = len(t_labels)

    s_mc_vf, s_ced_vf = _build_vf_samplers(problem, T, mc_cfg, ced_cfg,
                                            device, args,
                                            beta_schedule=beta_schedule)

    if use_orig:
        n_grid = 12
        gx = np.linspace(xlim[0], xlim[1], n_grid)
        gy = np.linspace(ylim[0], ylim[1], n_grid)
        GX, GY = np.meshgrid(gx, gy)
        grid_orig = np.stack([GX.flatten(), GY.flatten()], axis=1)
        scale, shift = frame_2d["scale"][:2], frame_2d["shift"][:2]
        pts = (grid_orig - shift) / scale
    else:
        n_grid = 12
        gx = np.linspace(frame["x_lo"], frame["x_hi"], n_grid)
        gy = np.linspace(frame["y_lo"], frame["y_hi"], n_grid)
        GX, GY = np.meshgrid(gx, gy)
        grid_2d = np.stack([GX.flatten(), GY.flatten()], axis=1)
        pts = grid_2d @ frame["W"].T + frame["mode_ref_perp"][None, :]

    drift_cache = {}
    for r, name in enumerate(method_order):
        kind = "langevin" if name == "PDL" else "ddpm"
        for c, t_label in enumerate(t_labels):
            t_internal = t_label if kind == "langevin" else (T - 1 - t_label)
            if kind == "langevin":
                trc = traces_by_method[name]
                k = min(trc.shape[0] - 1, int(t_internal))
                lam = trc[k].mean(axis=0)
                drift_d = _drift_langevin_full(problem, pts, lam, device)
            else:
                sampler = s_mc_vf if name == "CED-ceiling" else s_ced_vf
                drift_d = _drift_ddpm_full(sampler, pts, t_internal, device,
                                           d_dim)
            if use_orig:
                drift_cache[(r, c)] = drift_d[:, :2] * scale
            else:
                drift_cache[(r, c)] = drift_d @ frame["W"]

    x_range = (xlim[1] - xlim[0]) if use_orig else (frame["x_hi"] - frame["x_lo"])
    row_vmax = [
        max(float(np.max(np.linalg.norm(drift_cache[(r, c)], axis=1)))
            for c in range(n_cols))
        for r in range(n_rows)
    ]
    target_len = 0.12 * x_range

    pw = 3.4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(pw * n_cols, pw * n_rows),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.05, "hspace": 0.18},
    )
    if n_rows == 1:
        axes = np.array([axes])

    for r, name in enumerate(method_order):
        snaps = snaps_by_method[name]
        kind = "langevin" if name == "PDL" else "ddpm"
        init_key = 0 if kind == "langevin" else (T - 1)

        init_snap = snaps.get(int(init_key)) if snaps else None
        init_mask = None
        if init_snap is not None:
            pts_flat = init_snap.reshape(init_snap.shape[0], -1)
            if use_orig:
                feas = (pts_flat @ A_feas[:m_feas].T >= b_feas[:m_feas]).all(axis=1)
            else:
                A_f = frame["A_np"][:frame["m_real"]]
                b_f = frame["b_np"][:frame["m_real"]]
                feas = (pts_flat @ A_f.T >= b_f).all(axis=1)
            init_mask = ~feas

        row_scale = row_vmax[r] / max(target_len, 1e-12)
        for c, t_label in enumerate(t_labels):
            ax = axes[r, c]
            if use_orig:
                _draw_bg_orig(ax, frame_2d)
            else:
                _draw_bg(ax, frame)
            drift_2d = drift_cache[(r, c)] / max(row_scale, 1e-12)
            U = drift_2d[:, 0].reshape(GX.shape)
            V = drift_2d[:, 1].reshape(GY.shape)
            ax.quiver(GX, GY, U, V,
                      color="#888888", alpha=0.6,
                      scale_units="xy", scale=1.0, angles="xy",
                      width=0.0035, headwidth=3.5, headlength=4.5,
                      pivot="mid", zorder=3)
            t_internal = t_label if kind == "langevin" else (T - 1 - t_label)
            snap = snaps.get(int(t_internal)) if snaps is not None else None
            if snap is not None and init_mask is not None:
                pts_flat = snap.reshape(snap.shape[0], -1)
                if use_orig:
                    xs = pts_flat[:, :2] * scale + shift
                else:
                    xs = pts_flat @ frame["W"]
                ax.scatter(xs[init_mask, 0], xs[init_mask, 1],
                           s=32, alpha=0.6, color="#d6604d",
                           edgecolors="none", zorder=6,
                           label=r"init infeasible"
                                 if (r == 0 and c == n_cols - 1) else None)
                ax.scatter(xs[~init_mask, 0], xs[~init_mask, 1],
                           s=32, alpha=0.6, color="#2166ac",
                           edgecolors="none", zorder=7,
                           label=r"init feasible"
                                 if (r == 0 and c == n_cols - 1) else None)
            if use_orig:
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_aspect("equal", adjustable="box")
            else:
                ax.set_xlim(frame["x_lo"], frame["x_hi"])
                ax.set_ylim(frame["y_lo"], frame["y_hi"])
            ax.grid(alpha=0.15)
            if r == 0:
                suffix = ""
                if c == 0:
                    suffix = r"  (noise)"
                elif c == n_cols - 1:
                    suffix = r"  (data)"
                ax.set_title(rf"$t={t_label}${suffix}", pad=6)
            if c == 0:
                ax.set_ylabel(f"{_dn(name)}\n" + (r"$x_2$" if use_orig else ""),
                              fontsize=16)
            else:
                ax.set_yticklabels([])
    for ax in axes[-1]:
        ax.set_xlabel(r"$x_1$" if use_orig else r"LDA axis 1 (feas vs infeas)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(handles, labels, fontsize=10, loc="lower right",
                           framealpha=0.9, markerscale=1.5)
    p_png = out_dir / "fig_vector_field.png"
    p_pdf = out_dir / "fig_vector_field.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


# ---------------------------------------------------------------------------
# 5. Confusion matrix figure
# ---------------------------------------------------------------------------
def fig_confusion(conf_by_method: dict, problem: dict, A_real: torch.Tensor,
                  b_real: torch.Tensor, device: torch.device,
                  out_dir: Path) -> None:
    """One heatmap per method. Uses centers_feas / partial_feas tags on labels."""
    K = int(problem["means"].shape[0])
    d = int(problem["d"])
    means_t = problem["means"].to(device).float()
    cov_t = problem["cov_diag"].to(device).float()
    centers_feas = ((means_t @ A_real.T) >= b_real).all(dim=1).cpu().numpy()
    g = torch.Generator(device=device).manual_seed(0)
    feas_fraction = np.zeros(K)
    for k in range(K):
        eps = torch.randn((10000, d), generator=g, device=device)
        xs = means_t[k].unsqueeze(0) + torch.sqrt(cov_t[k]).unsqueeze(0) * eps
        feas_fraction[k] = float(
            ((xs @ A_real.T) >= b_real).all(dim=1).float().mean().item())
    partial_feas = (~centers_feas) & (feas_fraction >= 0.01)

    def _tag(k):
        if centers_feas[k]:
            return r"$^{*}$"
        if partial_feas[k]:
            return r"$^{\sim}$"
        return ""
    labels = [f"{k}{_tag(k)}" for k in range(K)]

    method_order = list(conf_by_method.keys())
    n_panels = len(method_order)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 7),
        gridspec_kw={"width_ratios": [1] * n_panels, "wspace": 0.08},
    )
    if n_panels == 1:
        axes = [axes]
    last_im = None
    for i, name in enumerate(method_order):
        conf = conf_by_method[name]
        last_im = _draw_heatmap(axes[i], conf, labels, labels, _dn(name))
        if i == 0:
            axes[i].set_ylabel(r"Init basin ($t=0$)", fontsize=22)
        else:
            axes[i].set_yticklabels([""] * K)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label(r"$P(\mathrm{final}=j \mid \mathrm{init}=i)$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    p_png = out_dir / "fig_confusion.png"
    p_pdf = out_dir / "fig_confusion.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


def _fig_dual_traces(traces_by_method: dict[str, np.ndarray],
                     out_dir: Path) -> None:
    """Plot ‖λ‖₂ over time for all available methods (PDL + CED-ceiling)."""
    methods = [m for m in ("CED-ceiling", "CED (ours)")
               if m in traces_by_method]
    if not methods:
        return
    colors = {"PDL": "#e6550d", "CED-ceiling": "#2171b5", "CED (ours)": "#cb181d"}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for m in methods:
        trc = traces_by_method[m]
        norms = np.linalg.norm(trc, axis=2).mean(axis=1)
        step = np.arange(len(norms))
        ax.plot(step, norms, color=colors.get(m, "gray"), lw=2.2, label=_dn(m))
    ax.set_xlabel(r"iteration $t$", fontsize=22)
    ax.set_ylabel(r"$\|\bm{\lambda}\|_2$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=15, loc="best", framealpha=0.9)
    fig.tight_layout()
    p_png = out_dir / "fig_dual_traces.png"
    p_pdf = out_dir / "fig_dual_traces.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True,
                    help="Run label (used in output dir name).")
    ap.add_argument("--preset", type=str, default="medium_uniform",
                    choices=list(PAPER_BASELINES))
    ap.add_argument("--N", type=int, default=1024,
                    help="Batch size for sampler runs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ced-train-dir", type=Path, default=None,
                    help="Optional CED checkpoint dir (score_net_best.pt).")
    ap.add_argument("--ced-ib", type=float, default=None,
                    help="Inverse-beta for CED inference (defaults to MC ib).")
    ap.add_argument("--ced-dual-lr", type=float, default=None,
                    help="Dual step for CED (defaults to MC ds).")
    ap.add_argument("--ced-ckpt-name", type=str, default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--ced-lambda-init", type=float, default=0.0,
                    help="Initial dual variable for CED inference.")
    ap.add_argument("--feas-tol", type=float, default=0.0,
                    help="Tolerance for avg-constraint feasibility check.")
    ap.add_argument("--dps-scale", type=float, default=1.0,
                    help="Step size for DPS constraint guidance.")
    ap.add_argument("--skip-schedule-figs", action="store_true",
                    help="Skip figures that sweep over noise schedules.")
    ap.add_argument("--ib-sweep", type=str, default=None,
                    help="Comma-separated IB values for fig_2d_panels "
                         "(e.g. '0.3,1,3,10').")
    ap.add_argument("--ds-sweep", type=str, default=None,
                    help="Comma-separated dual steps matching --ib-sweep "
                         "(e.g. '1,1,1,0.5'). Defaults to mc.dual_step_size.")
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = PAPER_BASELINES[args.preset]
    T = int(cfg["T"])
    mc_cfg = cfg["mc"]; pdl_cfg = cfg["pdl"]
    ced_cfg = None
    if args.ced_train_dir is not None:
        ced_cfg = {
            "inverse_beta":   float(args.ced_ib if args.ced_ib is not None
                                    else mc_cfg["inverse_beta"]),
            "dual_step_size": float(args.ced_dual_lr if args.ced_dual_lr is not None
                                    else mc_cfg["dual_step_size"]),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = build_problem(args.preset, args.seed)
    out_dir = experiment_dir("diagnostics", args.label)
    print(f"[paper_figures] preset={args.preset}  out_dir={out_dir}")
    print(f"[paper_figures] PDL cfg = {pdl_cfg}")
    print(f"[paper_figures] MC  cfg = {mc_cfg}")
    if ced_cfg is not None:
        print(f"[paper_figures] CED cfg = {ced_cfg}  ckpt={args.ced_train_dir}")

    T_pdl = cfg.get("T_pdl", T)

    # Snapshot times (user-time: t=0 noise, t=T-1 data).
    t_labels = [1, 100, 250, 350, 480]
    t_labels = [min(tl, T - 1) for tl in t_labels]
    n_cols = len(t_labels)
    t_labels_pdl = [int(round(c * (T_pdl - 1) / (n_cols - 1)))
                    for c in range(n_cols)]
    lang_snap_steps = list(t_labels_pdl)
    ddpm_snap_steps = [T - 1 - tl for tl in t_labels]
    if (T - 1) not in ddpm_snap_steps:
        ddpm_snap_steps.append(T - 1)  # init noise snap for feasibility colouring

    # ------------------------------------------------------------------
    # Run all samplers
    # ------------------------------------------------------------------
    summary = {}
    traces_by_method = {}
    snaps_by_method = {}
    conf_by_method = {}

    # PD-Langevin (per-sample λ — that is how it was proposed)
    t0 = time.time()
    x0_l, xf_l, A_l, b_l, trc_l, snaps_l = _pd_langevin_with_traces(
        problem, args.N, T_pdl, pdl_cfg["primal_lr"], pdl_cfg["dual_lr"],
        device, args.seed, snap_steps=lang_snap_steps, shared=False,
        lam0=pdl_cfg.get("lam0", 0.0),
    )
    wall_l = time.time() - t0
    summary["PDL"] = compute_metrics(xf_l, problem, device, feas_tol=args.feas_tol)
    summary["PDL"]["wall"] = wall_l
    traces_by_method["PDL"] = trc_l
    snaps_by_method["PDL"] = snaps_l
    conf_by_method["PDL"], _ = _build_confusion(
        problem, x0_l, xf_l, A_l, b_l, device,
    )

    # CED-ceiling (shared λ)
    t0 = time.time()
    beta_sched = cfg.get("beta_schedule", "linear")
    x0_m, xf_m, A_m, b_m, trc_m, snaps_m, mc_violations = _mc_sample_with_traces(
        problem, args.N, T, mc_cfg["inverse_beta"], mc_cfg["dual_step_size"],
        device, args.seed, snap_steps=ddpm_snap_steps, shared=True,
        beta_schedule=beta_sched,
    )
    wall_m = time.time() - t0
    summary["CED-ceiling"] = compute_metrics(xf_m, problem, device, feas_tol=args.feas_tol)
    summary["CED-ceiling"]["wall"] = wall_m
    traces_by_method["CED-ceiling"] = trc_m
    snaps_by_method["CED-ceiling"] = snaps_m
    conf_by_method["CED-ceiling"], _ = _build_confusion(
        problem, x0_m, xf_m, A_m, b_m, device,
    )

    # CED (optional)
    ced_violations = None
    if ced_cfg is not None:
        t0 = time.time()
        x0_c, xf_c, A_c, b_c, trc_c, snaps_c, ced_violations = _ced_sample(
            problem, args.N, T,
            ced_cfg["inverse_beta"], ced_cfg["dual_step_size"],
            args.ced_train_dir, device, args.seed,
            hidden=args.hidden, num_layers=args.num_layers,
            ckpt_name=args.ced_ckpt_name,
            snap_steps=ddpm_snap_steps, shared=True,
            beta_schedule=beta_sched,
            dual_lambda_init=args.ced_lambda_init,
        )
        wall_c = time.time() - t0
        summary["CED (ours)"] = compute_metrics(xf_c, problem, device, feas_tol=args.feas_tol)
        summary["CED (ours)"]["wall"] = wall_c
        traces_by_method["CED (ours)"] = trc_c
        snaps_by_method["CED (ours)"] = snaps_c
        conf_by_method["CED (ours)"], _ = _build_confusion(
            problem, x0_c, xf_c, A_c, b_c, device,
        )

    # PDM — trained net + per-step polytope projection, no dual ascent
    if ced_cfg is not None:
        d_dim = int(problem["d"])
        A_full = problem["constraint_A"].to(device).float()
        b_full = problem["constraint_b"].to(device).float()
        m_real = int((A_full.abs().sum(1) > 0).sum().item())
        A_real, b_real = A_full[:m_real], b_full[:m_real]

        s_pdm = build_sampler(
            problem, num_timesteps=T, beta_schedule=beta_sched,
            inverse_beta=ced_cfg["inverse_beta"],
            dual_step_size=ced_cfg["dual_step_size"],
            shared_lambda=True, device=device,
        )
        ckpt_name = args.ced_ckpt_name or "score_net_best.pt"
        score_net = load_score_net(
            args.ced_train_dir, d_dim,
            hidden=args.hidden, num_layers=args.num_layers,
            num_timesteps=T, ckpt_name=ckpt_name, device=device,
        )
        wire_trained_net(s_pdm, score_net, d_dim)
        s_pdm._dual_ascent_step = types.MethodType(_noop_dual, s_pdm)

        orig_score_to_x0 = s_pdm._score_to_x0_eps
        def _pdm_project(self, x_t, t, score):
            x0_pred, _eps = orig_score_to_x0(x_t, t, score)
            B = x0_pred.shape[0]
            x = x0_pred.view(B, d_dim)
            x = project_polytope(x, A_real, b_real, max_iters=80)
            x = x.clamp(-self._clip_range, self._clip_range)
            x0_new = x.view(B, 1, d_dim, 1)
            alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
            sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
            eps_new = (x_t - sqrt_ab * x0_new) / sqrt_1mab
            return x0_new, eps_new
        s_pdm._score_to_x0_eps = types.MethodType(_pdm_project, s_pdm)

        torch.manual_seed(int(args.seed))
        t0 = time.time()
        xf_pdm = s_pdm.sample(shape=(args.N, 1, d_dim, 1), device=device)
        wall_pdm = time.time() - t0
        xf_pdm = xf_pdm[:, 0, :, 0].detach()
        summary["PDM"] = compute_metrics(xf_pdm, problem, device, feas_tol=args.feas_tol)
        summary["PDM"]["wall"] = wall_pdm

    # DPS — trained net + fixed constraint gradient guidance, no dual ascent
    if ced_cfg is not None and args.dps_scale > 0:
        dps_scale = args.dps_scale
        s_dps = build_sampler(
            problem, num_timesteps=T, beta_schedule=beta_sched,
            inverse_beta=ced_cfg["inverse_beta"],
            dual_step_size=ced_cfg["dual_step_size"],
            shared_lambda=True, device=device,
        )
        ckpt_name_dps = args.ced_ckpt_name or "score_net_best.pt"
        score_net_dps = load_score_net(
            args.ced_train_dir, d_dim,
            hidden=args.hidden, num_layers=args.num_layers,
            num_timesteps=T, ckpt_name=ckpt_name_dps, device=device,
        )
        wire_trained_net(s_dps, score_net_dps, d_dim)
        s_dps._dual_ascent_step = types.MethodType(_noop_dual, s_dps)

        orig_dps_score_to_x0 = s_dps._score_to_x0_eps
        _dps_ac = s_dps.alphas_cumprod
        def _dps_guide(self, x_t, t, score):
            x0_pred, eps_pred = orig_dps_score_to_x0(x_t, t, score)
            B = x_t.shape[0]
            x_t_g = x_t.detach().requires_grad_(True)
            alpha_bar = _dps_ac[t].view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
            sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
            x0_tw = (x_t_g - sqrt_1mab * eps_pred.detach()) / sqrt_ab
            x0_flat = x0_tw.view(B, d_dim)
            viol = b_real.unsqueeze(0) - x0_flat @ A_real.T  # [B, m]
            loss = (torch.relu(viol) ** 2).sum()
            grad_xt = torch.autograd.grad(loss, x_t_g)[0]
            x_t_corrected = (x_t - dps_scale * grad_xt).detach()
            x0_new = (x_t_corrected - sqrt_1mab * eps_pred) / sqrt_ab
            x0_new = x0_new.clamp(-self._clip_range, self._clip_range)
            eps_new = eps_pred
            return x0_new, eps_new
        s_dps._score_to_x0_eps = types.MethodType(_dps_guide, s_dps)

        torch.manual_seed(int(args.seed))
        t0 = time.time()
        xf_dps = s_dps.sample(shape=(args.N, 1, d_dim, 1), device=device)
        wall_dps = time.time() - t0
        xf_dps = xf_dps[:, 0, :, 0].detach()
        summary["DPS"] = compute_metrics(xf_dps, problem, device, feas_tol=args.feas_tol)
        summary["DPS"]["wall"] = wall_dps

    # Unconstrained DDPM (no dual ascent)
    t0 = time.time()
    _, xf_unc, _, _, _, _, _ = _mc_sample_with_traces(
        problem, args.N, T, mc_cfg["inverse_beta"], 1e-6,
        device, args.seed, snap_steps=[], shared=True,
        beta_schedule=beta_sched,
    )
    wall_unc = time.time() - t0
    summary["Unconstrained"] = compute_metrics(xf_unc, problem, device, feas_tol=args.feas_tol)
    summary["Unconstrained"]["wall"] = wall_unc

    # ------------------------------------------------------------------
    # Save summary JSON + emit figures
    # ------------------------------------------------------------------
    summary_path = out_dir / f"summary_N{args.N}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {summary_path}")

    has_ced = "CED (ours)" in summary
    fig_bars(summary, out_dir, has_ced=has_ced)
    if not args.skip_schedule_figs:
        schedule_traces = {}
        schedule_snaps = {}
        schedule_violations = {}
        for sname in ["linear", "poly2", "exp", "cosine"]:
            print(f"[dual fig] CED-ceiling with {sname} schedule ...", flush=True)
            trc, snps, _, viol = _mc_sample_with_schedule(
                problem, args.N, T, mc_cfg["inverse_beta"],
                mc_cfg["dual_step_size"], sname, device, args.seed,
                snap_steps=ddpm_snap_steps,
            )
            schedule_traces[sname] = trc
            schedule_snaps[sname] = snps
            schedule_violations[sname] = viol
        fig_dual_evolution(schedule_traces, out_dir)
        fig_violation(schedule_violations, schedule_traces, out_dir)
        fig_schedule_samples(schedule_snaps, t_labels, T, problem, out_dir)
    fig_dual_ced_vs_mc(traces_by_method, out_dir)
    fig_violation_ced_vs_mc(
        mc_violations, trc_m,
        ced_violations, traces_by_method.get("CED (ours)"),
        out_dir,
    )
    _fig_dual_traces(traces_by_method, out_dir)
    if not args.skip_schedule_figs:
        fig_t0_effect(problem, args.N, T, mc_cfg["inverse_beta"],
                      mc_cfg["dual_step_size"], device, args.seed, out_dir)
    T_by_method = {"PDL": T_pdl}
    fig_sample_evolution(snaps_by_method, t_labels, T, problem, out_dir,
                         T_by_method=T_by_method)
    fig_vector_field(snaps_by_method, traces_by_method, t_labels, T,
                     problem, mc_cfg, ced_cfg, device, out_dir, args,
                     beta_schedule=beta_sched)
    fig_confusion(conf_by_method, problem, A_l, b_l, device, out_dir)

    if args.ib_sweep is not None:
        sweep_ibs = [float(x) for x in args.ib_sweep.split(",")]
        if args.ds_sweep is not None:
            sweep_ds_list = [float(x) for x in args.ds_sweep.split(",")]
            sweep_ds = dict(zip(sweep_ibs, sweep_ds_list))
        else:
            sweep_ds = mc_cfg["dual_step_size"]
        fig_2d_panels(problem, sweep_ibs, sweep_ds,
                      T, args.N, device, args.seed, out_dir,
                      beta_schedule=beta_sched)


if __name__ == "__main__":
    main()
