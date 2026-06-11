#!/usr/bin/env python3
"""Record the per-timestep distribution of per-sample lambda for the
synthetic medium preset at a given (ib, dual_lr).

Used to pick ExponentialLambdaPrior's (mu_min, mu_max) before training.
"""
from __future__ import annotations

import argparse
import types

import numpy as np
import torch

from _shared import build_problem, build_sampler, PRESETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", default="medium", choices=list(PRESETS))
    p.add_argument("--ib", type=float, default=3.0)
    p.add_argument("--dual-lr", type=float, default=0.01)
    p.add_argument("--B", type=int, default=1024)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--mc", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta-schedule", type=str, default="cosine",
                   choices=["linear", "cosine", "sigmoid"])
    p.add_argument("--shared-lambda", action="store_true")
    args = p.parse_args()

    preset = PRESETS[args.difficulty]
    d = preset["d"]
    is_avg = preset.get("avg_constraint_margins") is not None
    use_shared = args.shared_lambda or is_avg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = build_problem(args.difficulty, seed=args.seed)

    sampler = build_sampler(
        problem,
        num_timesteps=args.T,
        beta_schedule=args.beta_schedule,
        energy_mc_samples=args.mc,
        inverse_beta=args.ib,
        dual_step_size=args.dual_lr,
        dual_lambda_init=0.0,
        shared_lambda=use_shared,
        device=device,
    )

    # Capture full [B, m] lambda at each reverse step (mean/q50/q95/max)
    traces = []                              # list of [B, m]
    orig = sampler._dual_ascent_step

    def _hook(self, dual_lambda, ergodic_rates, context, t=None):
        updated = orig(dual_lambda, ergodic_rates, context, t=t)
        m = self._syn_m
        traces.append(updated[:, :m].detach().cpu())
        return updated

    sampler._dual_ascent_step = types.MethodType(_hook, sampler)

    torch.manual_seed(42)
    sampler.sample(shape=(args.B, 1, d, 1), device=device)

    # stack to [T_recorded, B, m]
    lam = torch.stack(traces, dim=0).numpy()
    T_rec, B, m = lam.shape

    # Map reverse-step index -> diffusion timestep t
    # reverse step r=0 is at t=T-1, r=T-1 is at t=0
    # Reports per-sample, per-constraint λ collapsed to scalar per (r): take the batch x constraint flat array
    rows = []
    for r in range(T_rec):
        vals = lam[r].ravel()
        rows.append(dict(
            step=r,
            t=args.T - 1 - r,
            mean=float(vals.mean()),
            q50=float(np.quantile(vals, 0.5)),
            q95=float(np.quantile(vals, 0.95)),
            q99=float(np.quantile(vals, 0.99)),
            mx=float(vals.max()),
        ))

    print(f"\n[calibrate] difficulty={args.difficulty}  ib={args.ib}  dual_lr={args.dual_lr}")
    print(f"{'t':>4s} {'mean':>8s} {'q50':>8s} {'q95':>8s} {'q99':>8s} {'max':>8s}")
    # Print only every ~50 steps for readability
    step_interval = max(1, T_rec // 12)
    for r in list(range(0, T_rec, step_interval)) + [T_rec - 1]:
        row = rows[r]
        print(f"{row['t']:>4d} {row['mean']:>8.4f} {row['q50']:>8.4f} "
              f"{row['q95']:>8.4f} {row['q99']:>8.4f} {row['mx']:>8.4f}")

    # Global summary
    all_vals = lam.ravel()
    print("\n[calibrate] global (over all t, b, j):")
    print(f"  mean = {all_vals.mean():.4f}")
    print(f"  q50  = {np.quantile(all_vals, 0.5):.4f}")
    print(f"  q95  = {np.quantile(all_vals, 0.95):.4f}")
    print(f"  q99  = {np.quantile(all_vals, 0.99):.4f}")
    print(f"  max  = {all_vals.max():.4f}")

    # Empirical t=T (sqrt_ab~0) and t=0 (sqrt_ab~1) means -> prior calibration
    # In our reverse-step index r:  r=0 <-> t=T-1 (noisy);  r=T-1 <-> t=0 (clean)
    mean_tT = float(lam[0].mean())
    mean_t0 = float(lam[-1].mean())
    peak = float(max(r["mean"] for r in rows))
    peak_t = next(r["t"] for r in rows if r["mean"] == peak)
    print(f"\n[calibrate] mean_t=T={mean_tT:.4f}  "
          f"mean_t=0={mean_t0:.4f}  peak={peak:.4f} at t={peak_t}")


if __name__ == "__main__":
    main()
