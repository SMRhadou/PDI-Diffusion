#!/usr/bin/env python3
"""MC-sampler probe: does the constraint bind for the diffusion sampler?

Runs the MC PortfolioEnergyDDPM at a grid of inverse_beta values, both with
and without dual ascent. Records:
- feasibility rate (constrained vs unconstrained)
- expected return, portfolio variance
- lambda trajectory quantiles (shared-lambda)
- max observed lambda

HANDOFF §5: we only have a non-trivial problem if unconstrained << constrained
in feasibility at our chosen ib.
"""
from __future__ import annotations

import argparse
import json
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
    PortfolioEnergyDDPM, make_portfolio_problem, shortfall_contributions_torch,
    variance_contributions_torch,
)


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=1000, gamma=1.0, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5, seed=0),
    "large": dict(N=500, K_factors=50, R_scenarios=1000, gamma=2.0, seed=0),
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


def _metrics(z, sampler, ctx, constraint_type="shortfall"):
    with torch.no_grad():
        w = sampler.z_to_portfolio_weights(z)
        if constraint_type in ("variance", "variance_sector"):
            risk = variance_contributions_torch(w, ctx.Sigma)
        else:
            risk = shortfall_contributions_torch(w, ctx.scenarios, ctx.alpha)
        budgets = ctx.risk_budgets.unsqueeze(0)
        violation = (risk - budgets).clamp_min(0.0)
        rel_violation = violation / budgets.clamp_min(1e-12)  # fractional over-budget
        ret = torch.matmul(ctx.scenarios, w.t()).mean(dim=0)
        Sx = torch.matmul(w, ctx.Sigma)
        var = (Sx * w).sum(dim=1)
        feas_full = (risk <= budgets + 1e-8).all(dim=1).float()
        feas_perconstraint = (risk <= budgets + 1e-8).float()  # [B, N]
        ent = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=1)
        # Option 2: E_x[c_j(x)] <= b_j (batch-mean feasibility)
        c_mean = risk.mean(dim=0)  # [N]
        bud = ctx.risk_budgets
        opt2_vio = (c_mean - bud).clamp_min(0.0)
        opt2_rel_vio = opt2_vio / bud.clamp_min(1e-12)
        # Feasibility at multiple relative tolerances
        tols = {"strict": 1e-8}
        for pct in [1, 2, 5, 10, 20]:
            tols[f"{pct}pct"] = pct / 100.0 * bud  # absolute tol = pct% of budget
        opt2_by_tol = {}
        for tname, tval in tols.items():
            sat = (c_mean <= bud + tval).float()
            opt2_by_tol[f"O2feas_{tname}"] = float(sat.all().item())
            opt2_by_tol[f"O2csat_{tname}"] = float(sat.mean().item())
    return {
        "expected_return": float(ret.mean().item()),
        "portfolio_var": float(var.mean().item()),
        "mean_violation": float(violation.mean().item()),
        "max_violation": float(violation.max().item()),
        "mean_rel_violation": float(rel_violation.mean().item()),
        "max_rel_violation": float(rel_violation.max().item()),
        "feasibility_rate": float(feas_full.mean().item()),
        "constraint_sat_rate": float(feas_perconstraint.mean().item()),
        "effective_N": float(torch.exp(ent).mean().item()),
        "hhi": float((w ** 2).sum(dim=1).mean().item()),
        "opt2_feas": float((c_mean <= bud + 1e-8).all().item()),
        "opt2_csat": float((c_mean <= bud + 1e-8).float().mean().item()),
        "opt2_mean_rel_vio": float(opt2_rel_vio.mean().item()),
        "opt2_max_rel_vio": float(opt2_rel_vio.max().item()),
        **opt2_by_tol,
    }


def _run_one(ib: float, dual_step: float, use_dual: bool,
              mu, Sigma, scenarios, budgets, alpha,
              T: int, K: int, B: int, N: int,
              device, seed: int = 42, shared_lambda: bool = True,
              normalize_constraints: bool = True,
              beta_schedule: str = "cosine",
              lam0: float = 0.0,
              use_dps: bool = False,
              constraint_type: str = "variance",
              num_sectors: int = 10,
              dual_lambda_decay: float = 0.0) -> dict:
    torch.manual_seed(seed)
    effective_step = dual_step if use_dual else 1e-12
    sampler = PortfolioEnergyDDPM(
        model=_NoOp(), num_timesteps=T, beta_schedule=beta_schedule,
        portfolio_mu=mu, portfolio_Sigma=Sigma,
        portfolio_scenarios=scenarios, portfolio_risk_budgets=budgets,
        portfolio_alpha=alpha,
        energy_mc_samples=K,
        inverse_beta=ib, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=effective_step,
        dual_num_outer_iterations=1,
        dual_lambda_init=lam0,
        dual_lambda_max=1e6,
        dual_lambda_decay=dual_lambda_decay,
        shared_lambda=shared_lambda,
        normalize_constraints=normalize_constraints,
        use_dps_guidance=use_dps,
        constraint_type=constraint_type,
        num_sectors=num_sectors,
    ).to(device)

    data = _make_batch(B, N).to(device)
    shape = (B, 1, N, 1)

    t0 = time.time()
    z = sampler.sample(shape=shape, device=device, data=data)
    wall = time.time() - t0

    ctx = sampler._build_energy_context(
        data=data, batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
    )
    m = _metrics(z, sampler, ctx, constraint_type=constraint_type)
    m["wall_time_sec"] = wall
    m["ib"] = ib
    m["dual_step"] = dual_step if use_dual else 0.0
    m["use_dual"] = use_dual
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(PROBLEM_CONFIGS.keys()), default="small")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--beta-schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--ib-grid", type=str, default="0.1,1.0,10.0,100.0")
    parser.add_argument("--dual-step-grid", type=str, default="0.001,0.01,0.1,1.0")
    parser.add_argument("--include-unconstrained", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-lambda", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="disable per-constraint budget normalization")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override gamma for budget scaling")
    parser.add_argument("--lam0-grid", type=str, default="0",
                        help="Comma-separated initial lambda values")
    parser.add_argument("--dual-lambda-decay", type=float, default=0.0)
    parser.add_argument("--decay-grid", type=str, default=None,
                        help="Comma-separated decay values to sweep (overrides --dual-lambda-decay)")
    parser.add_argument("--dps", action="store_true", default=False,
                        help="Use DPS-style autograd guidance instead of IS")
    parser.add_argument("--structure", type=str, default="homogeneous",
                        choices=["homogeneous", "sectors", "sectors_fat"],
                        help="Factor model structure")
    parser.add_argument("--num-sectors", type=int, default=None,
                        help="Number of sectors (default: K_factors)")
    parser.add_argument("--constraint-type", type=str, default="shortfall",
                        choices=["variance", "variance_sector", "shortfall"])
    parser.add_argument("--budget-type", type=str, default="proportional",
                        choices=["proportional", "uniform"])
    parser.add_argument("--high-risk-budget-scale", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MC PROBE — full parameter dump")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print("=" * 60)

    pcfg = PROBLEM_CONFIGS[args.size]
    if args.gamma is not None:
        pcfg = {**pcfg, "gamma": args.gamma}
    pcfg["structure"] = args.structure
    pcfg["constraint_type"] = args.constraint_type
    if args.num_sectors is not None:
        pcfg["num_sectors"] = args.num_sectors
    pcfg["budget_type"] = args.budget_type
    if args.high_risk_budget_scale is not None:
        pcfg["high_risk_budget_scale"] = args.high_risk_budget_scale
    mu, Sigma, scenarios, budgets, alpha = make_portfolio_problem(**pcfg)
    N = pcfg["N"]
    print(f"alpha threshold = {alpha:.5f}")

    ibs = [float(x) for x in args.ib_grid.split(",")]
    dual_steps = [float(x) for x in args.dual_step_grid.split(",")]

    results = []
    # Unconstrained baseline per ib
    if args.include_unconstrained:
        for ib in ibs:
            print(f"[unconstrained] ib={ib:g}")
            m = _run_one(ib, 0.0, False, mu, Sigma, scenarios, budgets, alpha,
                         T=args.T, K=args.K, B=args.B, N=N,
                         device=device, seed=args.seed,
                         shared_lambda=args.shared_lambda,
                         normalize_constraints=not args.no_normalize,
                         beta_schedule=args.beta_schedule,
                         use_dps=args.dps,
                         constraint_type=args.constraint_type,
                         num_sectors=args.num_sectors or 10)
            m["label"] = f"uncon_ib{ib:g}"
            results.append(m)
            print(f"  feas={m['feasibility_rate']:.3f}  ret={m['expected_return']:.4f}  "
                  f"var={m['portfolio_var']:.5f}  Neff={m['effective_N']:.1f}")

    # Constrained grid
    lam0s = [float(x) for x in args.lam0_grid.split(",")]
    decays = [float(x) for x in args.decay_grid.split(",")] if args.decay_grid else [args.dual_lambda_decay]
    for ib in ibs:
        for ds in dual_steps:
            for l0 in lam0s:
                for decay in decays:
                    tag = f"ib{ib:g}_ds{ds:g}"
                    if l0 != 0:
                        tag += f"_l{l0:g}"
                    if len(decays) > 1:
                        tag += f"_dec{decay:g}"
                    print(f"[constrained] ib={ib:g} ds={ds:g} lam0={l0:g} decay={decay:g}")
                    m = _run_one(ib, ds, True, mu, Sigma, scenarios, budgets, alpha,
                                 T=args.T, K=args.K, B=args.B, N=N,
                                 device=device, seed=args.seed,
                                 shared_lambda=args.shared_lambda,
                                 normalize_constraints=not args.no_normalize,
                                 beta_schedule=args.beta_schedule,
                                 lam0=l0,
                                 use_dps=args.dps,
                                 constraint_type=args.constraint_type,
                                 num_sectors=args.num_sectors or 10,
                                 dual_lambda_decay=decay)
                    m["label"] = tag
                    m["lam0"] = l0
                    m["decay"] = decay
                    results.append(m)
                    print(f"  feas={m['feasibility_rate']:.3f}  ret={m['expected_return']:.4f}  "
                          f"var={m['portfolio_var']:.5f}  Neff={m['effective_N']:.1f}")

    gamma_tag = f"_g{args.gamma:g}" if args.gamma is not None else ""
    out_dir = experiment_dir("diagnostics", f"mc_probe_{args.size}_{args.structure}_{args.constraint_type}{gamma_tag}_K{args.K}_T{args.T}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Pretty print
    lines = ["", "=== MC probe summary ===", ""]
    hdr = (f"{'label':<28s} {'ib':>5s} {'ds':>6s} {'l0':>4s} {'decay':>7s} "
           f"{'O2mean':>7s} {'O2max':>7s} {'ret':>7s} {'Neff':>5s} "
           f"{'strict':>6s} {'1%':>5s} {'2%':>5s} {'5%':>5s} {'10%':>5s} {'20%':>5s}")
    lines.append(hdr)
    for r in results:
        l0 = r.get("lam0", 0.0)
        dec = r.get("decay", 0.0)
        lines.append(
            f"{r['label']:<28s} {r['ib']:>5.2g} {r['dual_step']:>6.2g} {l0:>4.3g} {dec:>7.4f} "
            f"{r.get('opt2_mean_rel_vio',0):>7.3f} {r['opt2_max_rel_vio']:>7.3f} {r['expected_return']:>7.4f} "
            f"{r['effective_N']:>5.1f} "
            f"{r.get('O2csat_strict', r['opt2_csat']):>6.2f} "
            f"{r.get('O2csat_1pct', 0):>5.2f} "
            f"{r.get('O2csat_2pct', 0):>5.2f} "
            f"{r.get('O2csat_5pct', 0):>5.2f} "
            f"{r.get('O2csat_10pct', 0):>5.2f} "
            f"{r.get('O2csat_20pct', 0):>5.2f}"
        )
    msg = "\n".join(lines)
    print(msg)
    (out_dir / "summary.txt").write_text(msg + "\n")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
