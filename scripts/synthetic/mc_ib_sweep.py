#!/usr/bin/env python3
"""Sweep the MC sampler across ib values (no training) to pick a target ib."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from _shared import build_problem, build_sampler, PRESETS


def _make_mc_sampler(problem, T, ib, mc, dual_lr, device, mode="per_sample",
                     beta_schedule="linear", lam0=0.0):
    return build_sampler(
        problem,
        num_timesteps=T,
        beta_schedule=beta_schedule,
        inverse_beta=ib,
        dual_step_size=dual_lr,
        dual_lambda_init=lam0,
        energy_mc_samples=mc,
        shared_lambda=(mode == "shared"),
        device=device,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", default="medium", choices=list(PRESETS))
    p.add_argument("--ib", type=float, nargs="+",
                   default=[0.01, 0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--B", type=int, default=1024)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--mc", type=int, default=64)
    p.add_argument("--dual-lr", type=float, nargs="+", default=[0.002])
    p.add_argument("--lam0", type=float, nargs="+", default=[0.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta-schedule", type=str, default="linear",
                   choices=["linear", "cosine", "sigmoid"])
    p.add_argument("--mode", default="shared",
                   choices=["per_sample", "shared"])
    args = p.parse_args()

    preset = PRESETS[args.difficulty]
    d = preset["d"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[mc_ib_sweep] difficulty={args.difficulty}  d={d}  "
          f"K={preset['K']}  m={preset['m']}  device={device}  mode={args.mode}")
    problem = build_problem(args.difficulty, seed=args.seed)

    # Pointwise feasibility of each mode CENTRE (in normalised coords)
    mode_rates = problem["means"] @ problem["constraint_A"].T - problem["constraint_b"]
    mode_feas = (mode_rates >= 0).all(dim=1).cpu().numpy()     # [K] bool
    K_feas = int(mode_feas.sum())
    K_tot = mode_feas.shape[0]
    print(f"[mc_ib_sweep] pointwise-feasible mode centres: {K_feas}/{K_tot}  "
          f"(feas_ids={[i for i, f in enumerate(mode_feas) if f]})")

    print(f"[mc_ib_sweep] beta_schedule={args.beta_schedule}")
    print(f"[mc_ib_sweep] ib={args.ib}  dual_lr={args.dual_lr}  lam0={args.lam0}")

    print(f"\n{'ib':>8s}  {'dlr':>8s}  {'lam0':>6s}  {'U_mean':>9s}  {'U_std':>9s}  "
          f"{'feas_avg':>10s}  "
          f"{'vio_mean':>9s}  {'vio_max':>9s}  "
          f"{'uniq_modes':>10s}  {'cov':>26s}  {'wall':>6s}")
    print("-" * 140)
    rows = []
    for ib in args.ib:
        for dlr in args.dual_lr:
            for l0 in args.lam0:
                torch.manual_seed(42)
                sampler = _make_mc_sampler(
                    problem, args.T, ib, args.mc, dlr, device,
                    mode=args.mode, beta_schedule=args.beta_schedule,
                    lam0=l0,
                )
                t0 = time.time()
                shape = (args.B, 1, d, 1)
                x0 = sampler.sample(shape=shape, device=device)
                wall = time.time() - t0

                ctx = sampler._build_energy_context(
                    data=None, batch_size=args.B, num_nodes=d,
                    device=device, dtype=torch.float32,
                )
                m_n = ctx.num_real_constraints
                with torch.no_grad():
                    x_flat = x0[:, 0, :, 0]
                    U = sampler._mog_potential(x_flat, ctx).cpu().numpy()
                    rates = sampler._constraint_values(x_flat, ctx).cpu().numpy()[:, :m_n]
                    feas_avg = float((rates.mean(axis=0) >= -1e-6).mean())
                    vio = np.clip(-rates, 0, None)
                    vio_mean = float(vio.mean())
                    vio_max = float(vio.max())

                    diff = x_flat.unsqueeze(1) - ctx.means.unsqueeze(0)
                    mahal = (diff.square() * ctx.precision_diag.unsqueeze(0)).sum(dim=2)
                    log_comp = ctx.log_norm_const + ctx.log_weights - 0.5 * mahal
                    assign = log_comp.argmax(dim=1).cpu().numpy()
                    counts = np.bincount(assign, minlength=K_tot)
                    uniq = int((counts > 0).sum())
                    cov_str = "[" + " ".join(
                        f"{c}{'*' if mode_feas[k] else ''}" for k, c in enumerate(counts)
                    ) + "]"
                row = dict(ib=ib, dual_lr=dlr, lam0=l0,
                           U_mean=float(U.mean()), U_std=float(U.std()),
                           feas_avg=feas_avg,
                           vio_mean=vio_mean, vio_max=vio_max, wall=wall,
                           uniq_modes=uniq, counts=counts.tolist())
                rows.append(row)
                print(f"{ib:>8.3f}  {dlr:>8g}  {l0:>6g}  "
                      f"{row['U_mean']:>9.2f}  {row['U_std']:>9.2f}  "
                      f"{feas_avg:>10.3f}  "
                      f"{vio_mean:>9.4f}  {vio_max:>9.4f}  "
                      f"{uniq:>10d}  {cov_str:>26s}  {wall:>5.1f}s")

    # Also run rejection sampling as reference.
    print("-" * 140)
    print(f"{'rejection':>8s}", end="")
    t0 = time.time()
    s = _make_mc_sampler(problem, args.T, 1.0, args.mc, args.dual_lr[0], device,
                         beta_schedule=args.beta_schedule)
    with torch.no_grad():
        ctx = s._build_energy_context(
            data=None, batch_size=args.B, num_nodes=d, device=device,
            dtype=torch.float32,
        )
        acc = []
        tries = 0
        means = ctx.means; cov = ctx.cov_diag; logw = ctx.log_weights
        weights = logw.exp()
        while len(acc) < args.B and tries < 200 * args.B:
            k = torch.multinomial(weights, args.B, replacement=True)
            mu = means[k]
            std = cov[k].sqrt()
            z = torch.randn_like(mu)
            x = mu + std * z
            rates = x @ ctx.constraint_A.T - ctx.constraint_b
            feas_mask = (rates[:, :m_n] >= -1e-6).all(dim=1)
            acc.extend(x[feas_mask].cpu().numpy().tolist())
            tries += args.B
        if len(acc) == 0:
            print("   0 samples in %.1fs" % (time.time() - t0))
        else:
            X = np.array(acc[: args.B])
            Xt = torch.from_numpy(X).float().to(device)
            U_ref = s._mog_potential(Xt, ctx).cpu().numpy()
            rates = (Xt @ ctx.constraint_A.T - ctx.constraint_b).cpu().numpy()[:, :m_n]
            feas_avg = float((rates.mean(axis=0) >= -1e-6).mean())
            wall = time.time() - t0
            print(f"   U_mean={U_ref.mean():.2f}  feas_avg={feas_avg:.3f}"
                  f"  N={len(X)}  wall={wall:.1f}s")


if __name__ == "__main__":
    main()
