#!/usr/bin/env python3
"""Combined init→final transition matrix for PD-Langevin and MC Ceiling.

Runs both samplers on the same problem/seed and produces a single figure with
two heatmaps side-by-side sharing a common colorbar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import math  # noqa: E402
import types  # noqa: E402

from _shared import (  # noqa: E402
    build_problem,
    build_sampler,
    load_score_net,
    wire_trained_net,
    PRESETS,
)
from _experiment_paths import experiment_dir  # noqa: E402
from legacy.langevin_init_confusion import _pd_langevin, _mc_sample, _assign_mode  # noqa: E402


def _lda_basis(means_np, A_np, b_np, m_real):
    """Compute a [d, 2] basis W: axis1 = feas-vs-infeas direction,
    axis2 = top PCA of mode centres in the subspace orthogonal to axis1."""
    feas_mask = ((means_np @ A_np[:m_real].T) >= b_np[:m_real]).all(axis=1)
    if feas_mask.any() and (~feas_mask).any():
        v1 = means_np[feas_mask].mean(axis=0) - means_np[~feas_mask].mean(axis=0)
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    else:
        mu_c = means_np - means_np.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(mu_c, full_matrices=False)
        v1 = Vt[0]
    mu_c = means_np - means_np.mean(axis=0, keepdims=True)
    mu_c_orth = mu_c - (mu_c @ v1[:, None]) * v1[None, :]
    _, _, Vt = np.linalg.svd(mu_c_orth, full_matrices=False)
    v2 = Vt[0]
    v2 = v2 - (v2 @ v1) * v1
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    return np.stack([v1, v2], axis=1)  # [d, 2]


def _build_2d_grid(problem, dims=(0, 1), n_grid=16, margin=0.4):
    """Build a 2D grid in (dim0, dim1), padding other dims with mode-11 center."""
    d = int(problem["d"])
    means_np = problem["means"].cpu().numpy()
    mode_ref = means_np[11].copy()                     # [d] — mode 11 center
    xs = means_np[:, dims[0]]; ys = means_np[:, dims[1]]
    x_lo, x_hi = xs.min() - margin, xs.max() + margin
    y_lo, y_hi = ys.min() - margin, ys.max() + margin
    gx = np.linspace(x_lo, x_hi, n_grid)
    gy = np.linspace(y_lo, y_hi, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    G = n_grid * n_grid
    pts = np.tile(mode_ref[None, :], (G, 1))
    pts[:, dims[0]] = GX.flatten()
    pts[:, dims[1]] = GY.flatten()
    return GX, GY, pts, mode_ref


def _drift_langevin_full(problem, pts, lam, device):
    d = int(problem["d"])
    means = problem["means"].to(device).float()
    cov = problem["cov_diag"].to(device).float()
    log_w = torch.log(problem["weights"].to(device).float())
    log_norm = (-0.5 * d * math.log(2 * math.pi)
                - 0.5 * torch.log(cov).sum(dim=1))
    A = problem["constraint_A"].to(device).float()
    m_real = int((A.abs().sum(dim=1) > 0).sum().item())
    A_real = A[:m_real]
    x = torch.from_numpy(pts).to(device).float()
    diff = x.unsqueeze(1) - means.unsqueeze(0)
    mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)
    log_comp = log_w + log_norm - 0.5 * mahal
    resp = torch.softmax(log_comp, dim=1)
    grad_logp = (resp.unsqueeze(-1) * (-diff) / cov.unsqueeze(0)).sum(dim=1)
    lam_t = torch.from_numpy(np.asarray(lam)).to(device).float()
    drift = grad_logp + lam_t @ A_real
    return drift.cpu().numpy()  # [G, d]


def _drift_ddpm_full(sampler, pts, t_idx, device, d):
    G = pts.shape[0]
    x_t = torch.from_numpy(pts).to(device).float().view(G, 1, d, 1)
    t_tensor = torch.full((G,), int(t_idx), dtype=torch.long, device=device)
    ctx = sampler._build_energy_context(None, G, d, device, torch.float32)
    with torch.no_grad():
        score = sampler._estimate_score(x_t, t_tensor, ctx)
        x0_pred, _ = sampler._score_to_x0_eps(x_t, t_tensor, score)
    drift = (x0_pred - x_t).view(G, d).cpu().numpy()
    return drift  # [G, d]


def _install_x_snap_hook(sampler, snap_steps):
    """Monkey-patch sampler to record x_t at the requested timesteps."""
    store = {int(t): None for t in snap_steps}
    orig = sampler._score_to_x0_eps

    def _hook(self, x_t, t, score):
        t_val = int(t[0].item()) if t.numel() > 0 else -1
        if t_val in store and store[t_val] is None:
            store[t_val] = x_t.detach().cpu().numpy().copy()
        return orig(x_t, t, score)

    sampler._score_to_x0_eps = types.MethodType(_hook, sampler)
    return store


def _pd_langevin_with_traces(problem, B, T, lr, dual_lr, device, seed,
                             snap_steps=None, shared=False, lam0=0.0):
    """Copy of _pd_langevin that also records per-iteration lambda traces."""
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
    log_norm = -0.5 * d * math.log(2 * math.pi) - 0.5 * torch.log(cov).sum(dim=1)
    x = torch.randn((B, d), generator=g, device=device)
    x0_init = x.detach().clone()
    lam = torch.full((B, m_real), float(lam0), device=device)
    traces = []
    x_snaps = {int(t): None for t in (snap_steps or [])}
    for k in range(T):
        diff = x.unsqueeze(1) - means.unsqueeze(0)
        mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)
        log_comp = log_w + log_norm - 0.5 * mahal
        resp = torch.softmax(log_comp, dim=1)
        grad_logp = (resp.unsqueeze(-1) * (-diff) / cov.unsqueeze(0)).sum(dim=1)
        grad_U = -grad_logp
        drift = -(grad_U - lam @ A_real)
        noise = torch.randn(x.shape, generator=g, device=device, dtype=x.dtype)
        x = x + lr * drift + math.sqrt(2 * lr) * noise
        h = b_real - (x @ A_real.T)
        if shared:
            lam_scalar = torch.clamp(
                lam.mean(0, keepdim=True) + dual_lr * h.mean(0, keepdim=True),
                min=0.0,
            )
            lam = lam_scalar.expand_as(lam).contiguous()
        else:
            lam = torch.clamp(lam + dual_lr * h, min=0.0)
        traces.append(lam.detach().cpu().numpy().copy())  # [B, m]
        if k in x_snaps:
            x_snaps[k] = x.detach().cpu().numpy().copy()
    return (x0_init, x.detach(), A_real, b_real,
            np.stack(traces, axis=0), x_snaps)


def _mc_sample_with_traces(problem, B, T, ib, dual_lr, device, seed,
                           snap_steps=None, shared=False,
                           beta_schedule="linear", lam0=0.0):
    """_mc_sample variant that records per-step lambda for all B samples."""
    d = int(problem["d"])
    s = build_sampler(
        problem, num_timesteps=T, beta_schedule=beta_schedule,
        inverse_beta=ib, dual_step_size=dual_lr,
        dual_lambda_init=lam0, shared_lambda=shared, device=device,
    )
    from paper_figures import _install_violation_recorder
    _install_violation_recorder(s)
    shape = (B, 1, d, 1)
    torch.manual_seed(int(seed))
    x_T = torch.randn(shape, device=device)[:, 0, :, 0].detach().clone()
    torch.manual_seed(int(seed))
    s.enable_dual_recording(list(range(B)))
    x_snaps = _install_x_snap_hook(s, snap_steps or [])
    x0_out = s.sample(shape=shape, device=device)
    traces = s.get_dual_traces().cpu().numpy()
    violations = torch.stack(s._violation_traces).numpy()  # [T, m]
    s.disable_dual_recording()
    x0_flat = x0_out[:, 0, :, 0].detach()
    A = problem["constraint_A"].to(device).float()
    b = problem["constraint_b"].to(device).float()
    m_real = int((A.abs().sum(dim=1) > 0).sum().item())
    return x_T, x0_flat, A[:m_real], b[:m_real], traces, x_snaps, violations


def _ced_sample(problem, B, T, ib, dual_lr, ced_train_dir, device, seed,
                hidden=256, num_layers=4, use_best=True, ckpt_name=None,
                snap_steps=None, shared=False, beta_schedule="linear",
                dual_lambda_init=0.0):
    """CED sampler: trained score net + dual ascent (same as rerun_eval)."""
    d = int(problem["d"])
    s = build_sampler(
        problem, num_timesteps=T, beta_schedule=beta_schedule,
        inverse_beta=ib, dual_step_size=dual_lr,
        dual_lambda_init=dual_lambda_init,
        shared_lambda=shared, device=device,
    )
    if ckpt_name is None:
        ckpt_name = "score_net_best.pt" if use_best else "score_net.pt"
    score_net = load_score_net(
        ced_train_dir, d, hidden=hidden, num_layers=num_layers,
        num_timesteps=T, ckpt_name=ckpt_name, device=device,
    )
    wire_trained_net(s, score_net, d)

    from paper_figures import _install_violation_recorder
    _install_violation_recorder(s)

    shape = (B, 1, d, 1)
    torch.manual_seed(int(seed))
    x_T = torch.randn(shape, device=device)[:, 0, :, 0].detach().clone()
    torch.manual_seed(int(seed))
    s.enable_dual_recording(list(range(B)))
    x_snaps = _install_x_snap_hook(s, snap_steps or [])
    x0_out = s.sample(shape=shape, device=device)
    dual_traces = s.get_dual_traces().cpu().numpy()  # [T_rec, B, m]
    violations = torch.stack(s._violation_traces).numpy()  # [T, m]
    s.disable_dual_recording()
    x0_flat = x0_out[:, 0, :, 0].detach()

    A = problem["constraint_A"].to(device).float()
    b = problem["constraint_b"].to(device).float()
    m_real = int((A.abs().sum(dim=1) > 0).sum().item())
    return x_T, x0_flat, A[:m_real], b[:m_real], dual_traces, x_snaps, violations

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{bm}",
})


def _build_confusion(problem, x0_init, x_final, A_real, b_real, device):
    K = int(problem["means"].shape[0])
    means = problem["means"].to(device).float()
    cov = problem["cov_diag"].to(device).float()
    log_w = torch.log(problem["weights"].to(device).float())

    init_assign = _assign_mode(x0_init, means, cov, log_w).cpu().numpy()
    final_assign = _assign_mode(x_final, means, cov, log_w).cpu().numpy()
    feas_mask = (b_real - (x_final @ A_real.T) <= 0).all(dim=1).cpu().numpy()

    conf = np.zeros((K, K), dtype=np.int64)
    for i, j in zip(init_assign, final_assign):
        conf[i, j] += 1
    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = conf / row_sums
    overall_feas = float(feas_mask.mean())
    return conf_norm, overall_feas


def _draw_heatmap(ax, conf_norm, row_labels, col_labels, title):
    K = conf_norm.shape[0]
    im = ax.imshow(conf_norm, cmap="magma", vmin=0, vmax=1, aspect="equal")
    for i in range(K):
        for j in range(K):
            val = conf_norm[i, j]
            if val >= 0.005:
                color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=12)
    ax.set_xticks(range(K)); ax.set_xticklabels(col_labels, fontsize=16)
    ax.set_yticks(range(K)); ax.set_yticklabels(row_labels, fontsize=16)
    ax.set_xlabel(r"Final basin ($t=T$)", fontsize=22)
    ax.set_title(title, fontsize=26)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--difficulty", default="medium_tight", choices=list(PRESETS))
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--primal-lr", type=float, default=1e-3)
    ap.add_argument("--langevin-dual-lr", type=float, default=10.0)
    ap.add_argument("--mc-ib", type=float, default=30.0)
    ap.add_argument("--mc-dual-lr", type=float, default=0.07)
    ap.add_argument("--ced-train-dir", type=Path, default=None,
                    help="Directory with score_net_best.pt; enables CED panel.")
    ap.add_argument("--ced-ib", type=float, default=10.0,
                    help="Inverse-beta for CED (match training).")
    ap.add_argument("--ced-dual-lr", type=float, default=0.01,
                    help="Dual step for CED inference.")
    ap.add_argument("--ced-ckpt-name", type=str, default=None,
                    help="Checkpoint filename under --ced-train-dir "
                         "(e.g. score_net_best_u.pt). Defaults to best.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared-lambda", action="store_true",
                    help="Use a single batch-averaged dual variable for all "
                         "samplers (Langevin / MC / CED) instead of per-sample λ.")
    args = ap.parse_args()

    d = PRESETS[args.difficulty]["d"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = build_problem(args.difficulty, args.seed)
    K = int(problem["means"].shape[0])

    out_dir = experiment_dir("diagnostics", args.label)
    print(f"[combined] output dir: {out_dir}")

    # User-convention column positions: t_label=0 is noise, t_label=T-1 is data.
    # Langevin snapshots at iteration=t_label (iter 0 = init noise).
    # DDPM snapshots at timestep T-1-t_label (t=T-1 = init noise).
    # These yield DIFFERENT snapshot keys per method so every column has data.
    n_cols_vf = 5
    t_labels = [int(round(c * (args.T - 1) / (n_cols_vf - 1)))
                for c in range(n_cols_vf)]                       # e.g. [0, 125, 250, 374, 499]
    lang_snap_steps = list(t_labels)
    ddpm_snap_steps = [args.T - 1 - tl for tl in t_labels]

    # Langevin
    x0_l, xf_l, A_l, b_l, trc_l, xsnap_l = _pd_langevin_with_traces(
        problem, args.N, args.T, args.primal_lr, args.langevin_dual_lr,
        device, args.seed, snap_steps=lang_snap_steps,
        shared=args.shared_lambda,
    )
    conf_l, feas_l = _build_confusion(problem, x0_l, xf_l, A_l, b_l, device)

    # MC
    x0_m, xf_m, A_m, b_m, trc_m, xsnap_m, _ = _mc_sample_with_traces(
        problem, args.N, args.T, args.mc_ib, args.mc_dual_lr, device, args.seed,
        snap_steps=ddpm_snap_steps, shared=args.shared_lambda,
    )
    conf_m, feas_m = _build_confusion(problem, x0_m, xf_m, A_m, b_m, device)

    # CED (optional)
    conf_c = None
    feas_c = None
    ced_dual_traces = None
    xsnap_c = None
    if args.ced_train_dir is not None:
        x0_c, xf_c, A_c, b_c, ced_dual_traces, xsnap_c = _ced_sample(
            problem, args.N, args.T, args.ced_ib, args.ced_dual_lr,
            args.ced_train_dir, device, args.seed,
            hidden=args.hidden, num_layers=args.num_layers,
            ckpt_name=args.ced_ckpt_name,
            snap_steps=ddpm_snap_steps, shared=args.shared_lambda,
        )
        conf_c, feas_c = _build_confusion(problem, x0_c, xf_c, A_c, b_c, device)

    # Class labels (centers_feas / partial_feas) — match standalone exactly by
    # doing the classification in torch on-device (float32).
    means_t = problem["means"].to(device).float()
    cov_t = problem["cov_diag"].to(device).float()
    centers_feas = ((means_t @ A_l.T) >= b_l).all(dim=1).cpu().numpy()
    g = torch.Generator(device=device).manual_seed(0)
    n_fm = 10000
    feas_fraction = np.zeros(K)
    for k in range(K):
        eps = torch.randn((n_fm, d), generator=g, device=device)
        xs = means_t[k].unsqueeze(0) + torch.sqrt(cov_t[k]).unsqueeze(0) * eps
        feas_fraction[k] = float(((xs @ A_l.T) >= b_l).all(dim=1).float().mean().item())
    partial_feas = (~centers_feas) & (feas_fraction >= 0.01)
    print(f"[combined] centers_feas = {centers_feas.astype(int)}")
    print(f"[combined] partial_feas = {partial_feas.astype(int)}")
    print(f"[combined] feas_fraction = {np.round(feas_fraction, 3)}")

    def _tag(k):
        if centers_feas[k]:
            return r"$^{*}$"
        if partial_feas[k]:
            return r"$^{\sim}$"
        return ""
    labels = [f"{k}{_tag(k)}" for k in range(K)]

    # Combined figure: 2 or 3 heatmaps + single colorbar
    n_panels = 3 if conf_c is not None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 7),
        gridspec_kw={"width_ratios": [1] * n_panels, "wspace": 0.08},
    )
    im_l = _draw_heatmap(axes[0], conf_l, labels, labels, r"PD Langevin")
    axes[0].set_ylabel(r"Init basin ($t=0$)", fontsize=22)
    im_r = _draw_heatmap(axes[1], conf_m, labels, labels, r"MC Ceiling")
    axes[1].set_yticklabels([""] * K)
    last_im = im_r
    if conf_c is not None:
        last_im = _draw_heatmap(axes[2], conf_c, labels, labels, r"CED (ours)")
        axes[2].set_yticklabels([""] * K)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label(r"$P(\mathrm{final}=j \mid \mathrm{init}=i)$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    p_png = out_dir / "fig_combined_confusion.png"
    p_pdf = out_dir / "fig_combined_confusion.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p_png}")
    print(f"saved {p_pdf}")
    msg = f"[combined] Langevin feas={feas_l:.3f}  MC feas={feas_m:.3f}"
    if feas_c is not None:
        msg += f"  CED feas={feas_c:.3f}"
    print(msg)

    # Per-sample dual variable evolution — one panel per method
    method_traces = [("PD Langevin", trc_l), ("MC Ceiling", trc_m)]
    if ced_dual_traces is not None and ced_dual_traces.shape[0] > 0:
        method_traces.append(("CED (ours)", ced_dual_traces))

    n_methods = len(method_traces)
    fig2, axes2 = plt.subplots(1, n_methods, figsize=(6.5 * n_methods, 4.2),
                               sharex=True)
    if n_methods == 1:
        axes2 = [axes2]
    for ax, (name, trc) in zip(axes2, method_traces):
        T_rec, B, m = trc.shape
        step_idx = np.arange(T_rec)
        per_sample = np.linalg.norm(trc, axis=2)  # [T_rec, B]  ||lambda||_2
        pop_mean = per_sample.mean(axis=1)         # [T_rec]
        pop_std = per_sample.std(axis=1)           # [T_rec]
        ax.fill_between(step_idx, pop_mean - pop_std, pop_mean + pop_std,
                        color="#d6604d", alpha=0.3,
                        label=r"$\pm 1\sigma$ across $N$")
        ax.plot(step_idx, pop_mean, color="#a50f15", lw=2.5,
                label=rf"mean over $N={B}$")
        ax.set_xlabel(r"iteration", fontsize=22)
        ax.set_title(name, fontsize=26)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(alpha=0.3)
        if name == "PD Langevin":
            ax.set_ylim(0, 10)
        ax.legend(fontsize=18, loc="best")
    axes2[0].set_ylabel(r"$\|\bm{\lambda}\|_2$", fontsize=24)
    fig2.tight_layout()

    d_png = out_dir / "fig_dual_evolution.png"
    d_pdf = out_dir / "fig_dual_evolution.pdf"
    fig2.savefig(d_png, dpi=150, bbox_inches="tight")
    fig2.savefig(d_pdf, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {d_png}")
    print(f"saved {d_pdf}")

    # Vector field evolution: 3 rows (method) x 5 cols (time step).
    # 2D projection: LDA-style axis1 = feas-vs-infeas, axis2 = orthogonal PCA.
    d_dim = int(problem["d"])
    means_np_full = problem["means"].cpu().numpy()
    A_np_full = problem["constraint_A"].cpu().numpy()
    b_np_full = problem["constraint_b"].cpu().numpy()
    m_real_full = int((np.abs(A_np_full).sum(axis=1) > 0).sum())
    W = _lda_basis(means_np_full, A_np_full, b_np_full, m_real_full)  # [d, 2]
    mode_ref = means_np_full[11].copy()
    # Orthogonal-complement component of mode_ref (kept fixed on the grid).
    mode_ref_perp = mode_ref - W @ (W.T @ mode_ref)

    modes_2d = means_np_full @ W  # [K, 2]
    all_xs = [modes_2d[:, 0]]; all_ys = [modes_2d[:, 1]]
    for xsn in (xsnap_l, xsnap_m, xsnap_c):
        if xsn is None:
            continue
        for v in xsn.values():
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
    n_grid = 12
    gx = np.linspace(x_lo, x_hi, n_grid)
    gy = np.linspace(y_lo, y_hi, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    G_pts = n_grid * n_grid
    grid_2d = np.stack([GX.flatten(), GY.flatten()], axis=1)  # [G, 2]
    # Lift to d-dim: x_d = W @ (2d) + mode_ref_perp
    pts = grid_2d @ W.T + mode_ref_perp[None, :]  # [G, d]

    # Build DDPM samplers for MC Ceiling and CED (vector-field only)
    s_mc_vf = build_sampler(
        problem, num_timesteps=args.T,
        inverse_beta=args.mc_ib, dual_step_size=args.mc_dual_lr,
        shared_lambda=args.shared_lambda, device=device,
    )
    s_ced_vf = None
    if args.ced_train_dir is not None:
        s_ced_vf = build_sampler(
            problem, num_timesteps=args.T,
            inverse_beta=args.ced_ib, dual_step_size=args.ced_dual_lr,
            shared_lambda=args.shared_lambda, device=device,
        )
        ckpt_name = args.ced_ckpt_name or "score_net_best.pt"
        score_net = load_score_net(
            args.ced_train_dir, d_dim,
            hidden=args.hidden, num_layers=args.num_layers,
            num_timesteps=args.T, ckpt_name=ckpt_name, device=device,
        )
        wire_trained_net(s_ced_vf, score_net, d_dim)

    T_l = trc_l.shape[0]

    # Precompute constraint lines in the LDA 2D frame.
    A_W = A_np_full[:m_real_full] @ W                      # [m, 2]
    b_W = b_np_full[:m_real_full] - A_np_full[:m_real_full] @ mode_ref_perp

    # 2D projected covariance for each mode (for mode ellipses).
    cov_np = problem["cov_diag"].cpu().numpy()  # [K, d]
    K_modes = means_np_full.shape[0]
    cov_2d_list = [(W.T @ (cov_np[k][:, None] * W)) for k in range(K_modes)]

    # Per-method initial feasibility masks (for blue/orange sample colouring).
    def _init_mask(xsnap_dict, init_key):
        if xsnap_dict is None or xsnap_dict.get(int(init_key)) is None:
            return None
        init_pts = xsnap_dict[int(init_key)].reshape(-1, d_dim)
        infeas = ~((init_pts @ A_np_full[:m_real_full].T)
                   >= b_np_full[:m_real_full]).all(axis=1)
        return infeas  # True = init infeasible

    init_mask_l = _init_mask(xsnap_l, 0)              # Langevin iter 0 = init
    init_mask_m = _init_mask(xsnap_m, args.T - 1)     # DDPM t=T-1 = init noise
    init_mask_c = _init_mask(xsnap_c, args.T - 1)

    method_defs = [
        ("PD Langevin", "langevin", None, xsnap_l, init_mask_l),
        ("MC Ceiling", "ddpm", s_mc_vf, xsnap_m, init_mask_m),
    ]
    if s_ced_vf is not None:
        method_defs.append(
            ("CED (ours)", "ddpm", s_ced_vf, xsnap_c, init_mask_c))

    n_rows = len(method_defs)
    n_cols = len(t_labels)
    fig3, axes3 = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows),
        sharex=True, sharey=True,
    )
    if n_rows == 1:
        axes3 = np.array([axes3])

    # t_labels already computed above (columns in user-convention: 0=noise, T-1=data).

    def _internal_t(kind, t_label):
        """Map column's user-convention t_label to sampler-specific index."""
        return t_label if kind == "langevin" else (args.T - 1 - t_label)

    # Pre-compute drifts (per-row normalisation for arrow length).
    drift_cache = {}
    for r, (_name, kind, sampler, _xsnap, _im) in enumerate(method_defs):
        for c, t_label in enumerate(t_labels):
            t_internal = _internal_t(kind, t_label)
            if kind == "langevin":
                lam_k = min(T_l - 1, int(t_internal))
                lam = trc_l[lam_k].mean(axis=0)
                drift_d = _drift_langevin_full(problem, pts, lam, device)
            else:
                drift_d = _drift_ddpm_full(sampler, pts, t_internal, device,
                                           d_dim)
            drift_cache[(r, c)] = drift_d @ W  # [G, 2]
    row_vmax = [
        max(float(np.max(np.linalg.norm(drift_cache[(r, c)], axis=1)))
            for c in range(n_cols))
        for r in range(n_rows)
    ]
    target_len = 0.12 * (x_hi - x_lo)

    x_axis_line = np.linspace(x_lo - 2.0, x_hi + 2.0, 200)

    def _draw_bg(ax):
        """Mode ellipses + constraint lines (grey) + infeasible shading (red).
        Mirrors plot_sample_evolution._draw_modes / _draw_constraints."""
        for j in range(m_real_full):
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
        for k in range(K_modes):
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
            ax.plot(*modes_2d[k], "x", color="#e31a1c", ms=8, mew=2.5,
                    zorder=10)

    for r, (name, kind, sampler, xsnap, init_mask) in enumerate(method_defs):
        row_scale = row_vmax[r] / max(target_len, 1e-12)
        for c, t_label in enumerate(t_labels):
            t_internal = _internal_t(kind, t_label)
            ax = axes3[r, c]
            _draw_bg(ax)
            drift_2d = drift_cache[(r, c)] / max(row_scale, 1e-12)
            U = drift_2d[:, 0].reshape(GX.shape)
            V = drift_2d[:, 1].reshape(GY.shape)
            ax.quiver(XX := GX, YY := GY, U, V,
                      color="#888888", alpha=0.6,
                      scale_units="xy", scale=1.0, angles="xy",
                      width=0.0035, headwidth=3.5, headlength=4.5,
                      pivot="mid", zorder=3)
            # Samples (init-feasibility coloured) on top.
            snap = xsnap.get(int(t_internal)) if xsnap is not None else None
            if snap is not None and init_mask is not None:
                xs = snap.reshape(snap.shape[0], -1) @ W
                ax.scatter(xs[init_mask, 0], xs[init_mask, 1],
                           s=32, alpha=0.6, color="#d6604d",
                           edgecolors="none", zorder=6,
                           label=r"init infeasible" if (r == 0 and c == n_cols - 1) else None)
                ax.scatter(xs[~init_mask, 0], xs[~init_mask, 1],
                           s=32, alpha=0.6, color="#2166ac",
                           edgecolors="none", zorder=7,
                           label=r"init feasible" if (r == 0 and c == n_cols - 1) else None)
            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
            if r == 0:
                suffix = ""
                if c == 0:
                    suffix = r"  (noise)"
                elif c == n_cols - 1:
                    suffix = r"  (data)"
                ax.set_title(rf"$t={t_label}${suffix}", fontsize=20)
            if c == 0:
                ax.set_ylabel(name, fontsize=22)
            ax.tick_params(axis="both", labelsize=14)
    for ax in axes3[-1]:
        ax.set_xlabel(r"LDA axis 1 (feas vs infeas)", fontsize=18)
    for ax in axes3[:, 0]:
        ax.set_ylabel(ax.get_ylabel() + "\nLDA axis 2", fontsize=18)
    handles, labels = axes3[0, -1].get_legend_handles_labels()
    if handles:
        axes3[0, -1].legend(
            handles, labels, fontsize=16, loc="lower right",
            framealpha=0.9, markerscale=1.5,
        )
    fig3.tight_layout()
    v_png = out_dir / "fig_vector_field.png"
    v_pdf = out_dir / "fig_vector_field.pdf"
    fig3.savefig(v_png, dpi=150, bbox_inches="tight")
    fig3.savefig(v_pdf, bbox_inches="tight")
    plt.close(fig3)
    print(f"saved {v_png}")
    print(f"saved {v_pdf}")


if __name__ == "__main__":
    main()
