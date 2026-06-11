#!/usr/bin/env python3
"""Reload a trained score_net.pt and re-run post-training eval at larger N.
Writes an updated projection_2d.png (and optionally CDF + bars) alongside the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
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
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})
from matplotlib.patches import Ellipse, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from _shared import (  # noqa: E402
    PRESETS, build_problem, build_sampler, compute_metrics,
    _noop_dual, project_polytope,
    wire_trained_net, load_score_net,
    pd_langevin_sample,
)
from run_synthetic_experiment import _mode_counts  # noqa: E402
from legacy.generate_figures import (  # noqa: E402
    _draw_constraints, _draw_modes, _to_orig_2d,
)


def _install_pdm_projection(sampler, problem, d, max_project_iters=80):
    """Monkey-patch sampler._score_to_x0_eps to project x0_pred onto the
    feasible polytope after the standard clamp. PDM (Christopher 2024).
    """
    dev = sampler.alphas_cumprod.device
    A = problem["constraint_A"].to(dev).float()  # [m_pad, d]
    b = problem["constraint_b"].to(dev).float()  # [m_pad]
    m_real = int((A.abs().sum(dim=1) > 0).sum().item())
    A_real = A[:m_real]
    b_real = b[:m_real]

    orig = sampler._score_to_x0_eps

    def _patched(self, x_t, t, score):
        x0_pred, _eps_pred = orig(x_t, t, score)
        B = x0_pred.shape[0]
        x = x0_pred.view(B, d)
        x = project_polytope(x, A_real, b_real, max_iters=max_project_iters)
        x = x.clamp(-self._clip_range, self._clip_range)
        x0_new = x.view(B, 1, d, 1)
        alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
        sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
        sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
        eps_new = (x_t - sqrt_ab * x0_new) / sqrt_1mab
        return x0_new, eps_new

    sampler._score_to_x0_eps = types.MethodType(_patched, sampler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--difficulty", default="medium", choices=list(PRESETS))
    p.add_argument("--ib", type=float, required=True)
    p.add_argument("--dual-step-size", type=float, default=0.01)
    p.add_argument("--mc-ib", type=float, default=None,
                   help="Inverse-beta for MC Ceiling only (defaults to --ib).")
    p.add_argument("--mc-dual-step-size", type=float, default=None,
                   help="Dual step size for MC Ceiling only (defaults to --dual-step-size).")
    p.add_argument("--dual-lambda-init", type=float, default=0.0)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--mc", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy-mode-cov", type=str, default=None,
                   help="Post-process: 'src:dst' (must match training config).")
    p.add_argument("--shift-mode", type=str, default=None,
                   help="Post-process: 'idx:delta' (must match training config).")
    p.add_argument("--use-best", action="store_true",
                   help="Load score_net_best.pt instead of score_net.pt")
    p.add_argument("--ckpt-name", type=str, default=None,
                   help="Override checkpoint filename (e.g. score_net_best_u.pt). "
                        "Takes precedence over --use-best.")
    p.add_argument("--pd-langevin-lr", type=float, default=1e-2,
                   help="Primal step size for the PD Langevin baseline.")
    p.add_argument("--pd-langevin-dual-lr", type=float, default=None,
                   help="Dual step size for PD Langevin (defaults to "
                        "--dual-step-size if omitted).")
    p.add_argument("--shared-lambda", action="store_true",
                   help="Use a single batch-averaged dual variable for all "
                        "samplers (CED / MC Ceiling / PD Langevin) instead of "
                        "per-sample λ.")
    args = p.parse_args()

    preset = PRESETS[args.difficulty]
    d = preset["d"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = build_problem(args.difficulty, seed=args.seed)
    if args.copy_mode_cov is not None:
        src, dst = (int(x) for x in args.copy_mode_cov.split(":"))
        problem["cov_diag"][dst] = problem["cov_diag"][src].clone()
        problem["cov_diag_original"][dst] = problem["cov_diag_original"][src].clone()
        print(f"[rerun] tweak: copied cov of mode {src} onto mode {dst}")
    if args.shift_mode is not None:
        idx, delta = args.shift_mode.split(":")
        idx, delta = int(idx), float(delta)
        A = problem["constraint_A"]; b = problem["constraint_b"]
        mu = problem["means"][idx]
        slacks = A @ mu - b
        tight_j = int(slacks.argmin().item())
        normal = A[tight_j]
        normal = normal / normal.norm().clamp_min(1e-12)
        problem["means"][idx] = mu + delta * normal
        problem["means_original"][idx] = (
            problem["means"][idx] * problem["norm_scale"] + problem["norm_shift"]
        )
        print(f"[rerun] tweak: shifted mode {idx} by {delta}")

    # Common sampler kwargs
    _sampler_kw = dict(
        num_timesteps=args.T,
        energy_mc_samples=args.mc,
        inverse_beta=args.ib,
        dual_step_size=args.dual_step_size,
        dual_lambda_init=args.dual_lambda_init,
        device=device,
    )

    # Load score net
    if args.ckpt_name is not None:
        ckpt_name = args.ckpt_name
    else:
        ckpt_name = "score_net_best.pt" if args.use_best else "score_net.pt"
    score_net = load_score_net(
        args.run_dir, d=d,
        hidden=args.hidden, num_layers=args.num_layers,
        num_timesteps=args.T, ckpt_name=ckpt_name, device=device,
    )
    print(f"[rerun] loaded {ckpt_name}")

    shape = (args.N, 1, d, 1)

    results = {}

    # Build MC sampler up front (reused as the `sampler` handle for every entry's
    # mode-count computation); run it second in display order.
    mc_ib = args.mc_ib if args.mc_ib is not None else args.ib
    mc_dual = (args.mc_dual_step_size if args.mc_dual_step_size is not None
               else args.dual_step_size)
    s_mc = build_sampler(problem, **{**_sampler_kw, "inverse_beta": mc_ib,
                                     "dual_step_size": mc_dual,
                                     "shared_lambda": args.shared_lambda})

    # Baseline seed — reset BEFORE each .sample() call so every baseline is
    # independent of which checkpoint (and therefore which score_net forward
    # trajectory) ran before it. Without this, MC/Dual Disabled/PDM/PD Langevin
    # would inherit a checkpoint-dependent RNG state and jump around across
    # evals of the same (ib, dual_lr) config.
    BASE_SEED = 42

    # 1) Trained
    s_tn = build_sampler(problem, **{**_sampler_kw,
                                     "shared_lambda": args.shared_lambda})
    wire_trained_net(s_tn, score_net, d)
    n_track = min(32, args.N)
    s_tn.enable_dual_recording(list(range(n_track)))
    torch.manual_seed(BASE_SEED)
    t0 = time.time()
    x0_tn = s_tn.sample(shape=shape, device=device)
    ced_dual_traces = s_tn.get_dual_traces().numpy()  # [T_rec, n_track, m]
    s_tn.disable_dual_recording()
    results["CED (ours)"] = dict(x0=x0_tn, wall=time.time() - t0, sampler=s_mc)

    # 2) MC Ceiling (checkpoint-independent; must be identical across evals)
    torch.manual_seed(BASE_SEED)
    t0 = time.time()
    x0_mc = s_mc.sample(shape=shape, device=device)
    results["MC Ceiling"] = dict(x0=x0_mc, wall=time.time() - t0, sampler=s_mc)

    # 3) Unconstrained (trained score net, dual disabled)
    s_uc = build_sampler(problem, **_sampler_kw)
    wire_trained_net(s_uc, score_net, d)
    s_uc._dual_ascent_step = types.MethodType(_noop_dual, s_uc)
    torch.manual_seed(BASE_SEED)
    t0 = time.time()
    x0_uc = s_uc.sample(shape=shape, device=device)
    results["Dual Disabled"] = dict(x0=x0_uc, wall=time.time() - t0, sampler=s_mc)

    # 3b) PDM — trained score net + per-step polytope projection.
    # Independent of args.dual_step_size (dual is disabled), so results are
    # cached to disk keyed on (ckpt, ib, T, N, seed) and reused across
    # dual_lr sweeps within the same run_dir.
    ckpt_tag = ckpt_name.replace("score_net_", "").replace(".pt", "")
    pdm_cache = (args.run_dir /
                 f"pdm_cache_{ckpt_tag}_ib{args.ib}_T{args.T}_N{args.N}"
                 f"_seed{BASE_SEED}.pt")
    if pdm_cache.exists():
        cached = torch.load(pdm_cache, map_location=device)
        x0_pdm = cached["x0"].to(device)
        wall_pdm = float(cached["wall"])
        print(f"[rerun] PDM cache hit → {pdm_cache.name} (wall={wall_pdm:.2f}s)")
    else:
        s_pdm = build_sampler(problem, **_sampler_kw)
        wire_trained_net(s_pdm, score_net, d)
        s_pdm._dual_ascent_step = types.MethodType(_noop_dual, s_pdm)
        _install_pdm_projection(s_pdm, problem, d)
        torch.manual_seed(BASE_SEED)
        t0 = time.time()
        x0_pdm = s_pdm.sample(shape=shape, device=device)
        wall_pdm = time.time() - t0
        torch.save({"x0": x0_pdm.detach().cpu(), "wall": wall_pdm}, pdm_cache)
        print(f"[rerun] PDM cached → {pdm_cache.name} (wall={wall_pdm:.2f}s)")
    results["PDM"] = dict(x0=x0_pdm, wall=wall_pdm, sampler=s_mc)

    # 4) PD Langevin (primal-dual Langevin MCMC)
    pd_dual_lr = (args.pd_langevin_dual_lr
                  if args.pd_langevin_dual_lr is not None
                  else args.dual_step_size)
    torch.manual_seed(BASE_SEED)
    t0 = time.time()
    x0_pd = pd_langevin_sample(
        problem=problem, B=args.N, T=args.T,
        lr=args.pd_langevin_lr, dual_lr=pd_dual_lr,
        device=device, seed=BASE_SEED,
    )
    results["PD Langevin"] = dict(x0=x0_pd, wall=time.time() - t0, sampler=s_mc)

    # 5) Rejection
    torch.manual_seed(BASE_SEED)
    ctx = s_mc._build_energy_context(None, args.N, d, device, torch.float32)
    m_n = ctx.num_real_constraints
    means = ctx.means; cov = ctx.cov_diag; logw = ctx.log_weights
    weights = logw.exp()
    acc = []
    tries = 0
    max_tries = 200 * args.N
    t0 = time.time()
    while len(acc) < args.N and tries < max_tries:
        k = torch.multinomial(weights, args.N, replacement=True)
        mu = means[k]
        std = cov[k].sqrt()
        z = torch.randn_like(mu)
        x = mu + std * z
        rates = x @ ctx.constraint_A.T - ctx.constraint_b
        feas = (rates[:, :m_n] >= -1e-6).all(dim=1)
        acc.extend(x[feas].cpu().numpy().tolist())
        tries += args.N
    wall_rej = time.time() - t0
    if len(acc) == 0:
        x0_rej = torch.zeros(0, 1, d, 1, device=device)
    else:
        X = torch.tensor(acc[: args.N], dtype=torch.float32, device=device)
        x0_rej = X.view(-1, 1, d, 1)
    results["Rejection"] = dict(x0=x0_rej, wall=wall_rej, sampler=s_mc)

    # Mode-centre feasibility for tagging
    A_t = problem["constraint_A"]; b_t = problem["constraint_b"]
    mu_t = problem["means"]
    mode_rates = mu_t @ A_t.T - b_t
    mode_feas = (mode_rates >= 0).all(dim=1).cpu().numpy()
    K_tot = mode_feas.shape[0]

    # ---- Metrics (Mahalanobis-gated, feasibility-aware) ----
    print(f"\n=== {args.difficulty} d={d} ib={args.ib} (N={args.N}) ===")
    print(f"{'method':<14s} {'U(x)':>16s} {'feas_avg':>9s} "
          f"{'vio_mean':>9s} {'vio_max':>9s} {'|M|':>5s} "
          f"{'counts':>40s} {'wall':>7s}")
    summary = {}
    for name, res in results.items():
        x0 = res["x0"]
        if x0.shape[0] == 0:
            print(f"{name:<14s}  no samples")
            continue
        sampler = res["sampler"]
        xf = x0[:, 0, :, 0].detach()
        m_eval = compute_metrics(xf, problem, device)
        ctx = sampler._build_energy_context(None, x0.shape[0], d, device, torch.float32)
        with torch.no_grad():
            rates = sampler._constraint_values(xf, ctx)[:, :m_n].cpu().numpy()
            vio = np.clip(-rates, 0, None)
        summary[name] = dict(
            U_mean=m_eval["U"],
            feas_avg=m_eval["feas_avg"],
            vio_mean=float(vio.mean()), vio_max=float(vio.max()),
            wall=res["wall"],
            modes_used=m_eval["modes_used"],
            modes_coverage=m_eval["modes_coverage"],
            mode_counts=m_eval["mode_counts"],
            entropy=m_eval["entropy"],
        )
        print(f"{name:<14s} {m_eval['U']:>7.2f} "
              f"{m_eval['feas_avg']:>9.3f} {vio.mean():>9.4f} "
              f"{vio.max():>9.4f} {m_eval['modes_used']:>3d}/{K_tot} "
              f"{str(m_eval['mode_counts']):>40s} {res['wall']:>6.1f}s")

    # ---- 2D panels (constraints + mode ellipses) ----
    scale2 = problem["norm_scale"][:2].numpy()
    shift2 = problem["norm_shift"][:2].numpy()
    means_orig = problem["means_original"][:, :2].numpy()
    cov_orig = problem["cov_diag_original"][:, :2].numpy()
    A_np = problem["constraint_A"][:m_n].numpy()
    b_np = problem["constraint_b"][:m_n].numpy()
    # Square axis limits based on mode extents
    mode_lo = means_orig - 2 * np.sqrt(cov_orig)
    mode_hi = means_orig + 2 * np.sqrt(cov_orig)
    pad = 2.0
    x_lo = mode_lo[:, 0].min() - pad; x_hi = mode_hi[:, 0].max() + pad
    y_lo = mode_lo[:, 1].min() - pad; y_hi = mode_hi[:, 1].max() + pad
    xr = x_hi - x_lo; yr = y_hi - y_lo
    if xr > yr:
        mid = (y_lo + y_hi) / 2; y_lo = mid - xr / 2; y_hi = mid + xr / 2
    else:
        mid = (x_lo + x_hi) / 2; x_lo = mid - yr / 2; x_hi = mid + yr / 2
    xlim = (x_lo, x_hi); ylim = (y_lo, y_hi)

    n_panels = len(results)
    pw = 3.8
    fig, axes = plt.subplots(1, n_panels, figsize=(pw * n_panels, pw),
                             sharex=True, sharey=True)
    for i, (ax, (name, res)) in enumerate(zip(axes, results.items())):
        _draw_constraints(ax, A_np, b_np, m_n, scale2, shift2, xlim, ylim)
        _draw_modes(ax, means_orig, cov_orig, K_tot, marker_size=8)
        x0 = res["x0"]
        if x0.shape[0] > 0:
            pts = _to_orig_2d(x0, problem)
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.45, color="#2166ac",
                       edgecolors="none", zorder=5)
        info = summary.get(name, {})
        title_info = (rf" ($U={info['U_mean']:.1f}$, "
                      rf"feas$={info.get('feas_avg', 0):.2f}$, "
                      rf"$|M|={info.get('modes_used', 0)}$)") if info else ""
        ax.set_title(f"{name}{title_info}", pad=6, fontsize=13)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(alpha=0.15)
        ax.set_xlabel(r"$x_1$")
        if i > 0:
            ax.set_yticklabels([])
    axes[0].set_ylabel(r"$x_2$")
    extras = [
        Line2D([0], [0], marker="x", color="#e31a1c", ls="none",
               ms=9, mew=2.5, label="Mode centre"),
        Patch(facecolor="#fee0d2", edgecolor="#e31a1c", ls="--", alpha=0.4,
              label=r"MoG $2\sigma$"),
        Patch(facecolor="#d73027", alpha=0.15, label="Infeasible"),
    ]
    axes[-1].legend(handles=extras, fontsize=9, loc="lower right",
                    framealpha=0.9)
    fig.suptitle(rf"2D projection — {args.difficulty} $d={d}$, "
                 rf"$\beta^{{-1}}={args.ib:g}$ ($N={args.N}$)", fontsize=14)
    fig.tight_layout()
    out_path = args.run_dir / "projection_2d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(args.run_dir / "projection_2d.pdf", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_path}")

    # ---- Bars: feasibility / objective / feasible mode coverage / wall time ----
    method_order = [m for m in results if m in summary]
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5))
    metrics = [
        ("feas_avg",   r"Avg-constraint feasibility $\uparrow$"),
        ("U_mean",     r"Objective $f_0(\mathbf{x})$ $\downarrow$"),
        ("modes_cov",  r"Mode coverage"),
        ("wall",       r"Wall time $\downarrow$"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        if key == "feas_avg":
            vals = [summary[m].get("feas_avg", 0) for m in method_order]
        elif key == "U_mean":
            vals = [summary[m]["U_mean"] for m in method_order]
        elif key == "modes_cov":
            vals = [summary[m].get("modes_coverage", 0) for m in method_order]
        else:
            vals = [summary[m]["wall"] for m in method_order]
        cols = [plt.cm.Blues(0.35 + 0.5 * i / max(len(method_order) - 1, 1))
                for i in range(len(method_order))]
        bars = ax.bar(range(len(method_order)), vals, color=cols,
                      edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels(method_order, rotation=20, ha="right", fontsize=20)
        ax.set_title(title, fontsize=24)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(alpha=0.2, axis="y")
        if key in ("feas_avg", "modes_cov"):
            ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=18)
    fig.tight_layout()
    bars_path = args.run_dir / "fig_comparison_bars.png"
    fig.savefig(bars_path, dpi=150, bbox_inches="tight")
    fig.savefig(args.run_dir / "fig_comparison_bars.pdf", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved {bars_path}")

    summary_path = args.run_dir / f"summary_N{args.N}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {summary_path}")

    # Dual variable evolution plot (CED sampler only)
    if ced_dual_traces.shape[0] > 0:
        T_rec, n_samp, m = ced_dual_traces.shape
        step_idx = np.arange(T_rec)
        lam = ced_dual_traces[: , 0, :]  # [T_rec, m] — shared λ, all samples identical
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        cmap = plt.cm.tab20
        for j in range(m):
            axes[0].plot(step_idx, lam[:, j], color=cmap(j % 20),
                         lw=1.3, label=f"j={j}")
        axes[0].set_xlabel(r"reverse sampling step")
        axes[0].set_ylabel(r"$\lambda_j(t)$")
        axes[0].set_title(rf"Shared $\lambda$ per constraint (m={m})")
        if m <= 15:
            axes[0].legend(ncol=3, fontsize=8, loc="best", framealpha=0.9)
        axes[0].grid(alpha=0.3)
        lam_norm = np.linalg.norm(lam, axis=1)  # [T_rec]
        lam_mean = lam.mean(axis=1)
        axes[1].plot(step_idx, lam_norm, color="black", lw=2.0,
                     label=r"$\|\lambda\|_2$")
        axes[1].plot(step_idx, lam_mean, color="steelblue", lw=1.5,
                     ls="--", label=r"$\bar{\lambda}$")
        axes[1].set_xlabel(r"reverse sampling step")
        axes[1].set_ylabel(r"$\lambda$ aggregate")
        axes[1].set_title(r"Shared $\lambda$ norm \& mean")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=10, loc="best")
        fig.tight_layout()
        dual_path = args.run_dir / "dual_evolution.png"
        fig.savefig(dual_path, dpi=150, bbox_inches="tight")
        fig.savefig(args.run_dir / "dual_evolution.pdf", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"saved {dual_path}")


if __name__ == "__main__":
    main()
