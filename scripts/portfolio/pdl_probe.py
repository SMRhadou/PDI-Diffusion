#!/usr/bin/env python3
"""PD-Langevin probe: sweep primal_lr, dual_lr, num_iters on the portfolio problem."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import experiment_dir, _project_root  # noqa: E402

_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (  # noqa: E402
    make_portfolio_problem, shortfall_contributions_torch,
    variance_contributions_torch,
)
from baselines import pd_langevin  # noqa: E402


def _eval(w, scenarios_t, budgets_t, alpha, constraint_type="shortfall", Sigma_t=None):
    with torch.no_grad():
        if constraint_type in ("variance", "variance_sector"):
            risk = variance_contributions_torch(w, Sigma_t)
        else:
            risk = shortfall_contributions_torch(w, scenarios_t, alpha)
        bud = budgets_t.unsqueeze(0)
        c_mean = risk.mean(dim=0)
        vio = (c_mean - budgets_t).clamp_min(0.0)
        rel_vio = vio / budgets_t.clamp_min(1e-12)
        ret = torch.matmul(scenarios_t, w.t()).mean(dim=0)
        ent = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=1)
        neff = torch.exp(ent)

        tols = {"strict": 1e-8}
        for pct in [1, 2, 5, 10, 20]:
            tols[f"{pct}pct"] = pct / 100.0 * budgets_t
        by_tol = {}
        for tname, tval in tols.items():
            sat = (c_mean <= budgets_t + tval).float()
            by_tol[f"O2csat_{tname}"] = float(sat.mean().item())
            by_tol[f"O2feas_{tname}"] = float(sat.all().item())

    return {
        "ret": float(ret.mean().item()),
        "Neff": float(neff.mean().item()),
        "O2mxrv": float(rel_vio.max().item()),
        **by_tol,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--K-factors", type=int, default=None,
                    help="Number of latent factors (default: N//10)")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--plr-grid", type=str, default="1e-3,3e-3,1e-2,3e-2,0.1")
    ap.add_argument("--dlr-grid", type=str, default="1,3,10,30,100")
    ap.add_argument("--iters-grid", type=str, default="500,1000,2000")
    ap.add_argument("--noise-grid", type=str, default="0.01,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--structure", type=str, default="homogeneous",
                    choices=["homogeneous", "sectors", "sectors_fat"])
    ap.add_argument("--num-sectors", type=int, default=None)
    ap.add_argument("--constraint-type", type=str, default="shortfall",
                    choices=["variance", "variance_sector", "shortfall"])
    ap.add_argument("--budget-type", type=str, default="proportional",
                    choices=["proportional", "uniform"])
    ap.add_argument("--high-risk-budget-scale", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PDL PROBE — full parameter dump")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print("=" * 60)

    K_factors = args.K_factors if args.K_factors is not None else max(args.N // 10, 5)
    extra = {}
    if args.high_risk_budget_scale is not None:
        extra["high_risk_budget_scale"] = args.high_risk_budget_scale
    mu, Sigma, scenarios, budgets, alpha = make_portfolio_problem(
        N=args.N, K_factors=K_factors, R_scenarios=args.R, gamma=args.gamma, seed=0,
        structure=args.structure, num_sectors=args.num_sectors,
        constraint_type=args.constraint_type,
        budget_type=args.budget_type, **extra)

    scenarios_t = torch.tensor(scenarios, dtype=torch.float32, device=device)
    budgets_t = torch.tensor(budgets, dtype=torch.float32, device=device)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    Sigma_t = torch.tensor(Sigma, dtype=torch.float32, device=device)

    plrs = [float(x) for x in args.plr_grid.split(",")]
    dlrs = [float(x) for x in args.dlr_grid.split(",")]
    iters_list = [int(x) for x in args.iters_grid.split(",")]
    noises = [float(x) for x in args.noise_grid.split(",")]

    results = []
    hdr = (f"{'label':<30s} {'plr':>7s} {'dlr':>6s} {'iters':>5s} {'ns':>5s} "
           f"{'O2mxrv':>7s} {'ret':>7s} {'Neff':>5s} "
           f"{'strict':>6s} {'1%':>5s} {'5%':>5s} {'10%':>5s} {'20%':>5s} {'wall':>5s}")
    print(hdr)

    for ni in iters_list:
        for ns in noises:
            for plr in plrs:
                for dlr in dlrs:
                    t0 = time.time()
                    w, info = pd_langevin(
                        mu_t, Sigma_t, scenarios_t, budgets_t, alpha,
                        B=args.B, num_iters=ni, primal_lr=plr,
                        dual_lr=dlr, noise_scale=ns,
                        device=device, seed=args.seed,
                        constraint_type=args.constraint_type)
                    wall = time.time() - t0
                    m = _eval(w, scenarios_t, budgets_t, alpha,
                              constraint_type=args.constraint_type, Sigma_t=Sigma_t)
                    m["wall"] = wall
                    m["plr"] = plr
                    m["dlr"] = dlr
                    m["iters"] = ni
                    m["noise"] = ns
                    m["lam_mean"] = info["final_lambda_mean"]
                    m["lam_max"] = info["final_lambda_max"]
                    label = f"plr{plr:g}_dlr{dlr:g}_n{ni}_ns{ns:g}"
                    m["label"] = label
                    results.append(m)
                    print(f"{label:<30s} {plr:>7.4f} {dlr:>6.3g} {ni:>5d} {ns:>5.3g} "
                          f"{m['O2mxrv']:>7.3f} {m['ret']:>7.4f} {m['Neff']:>5.1f} "
                          f"{m.get('O2csat_strict',0):>6.2f} "
                          f"{m.get('O2csat_1pct',0):>5.2f} "
                          f"{m.get('O2csat_5pct',0):>5.2f} "
                          f"{m.get('O2csat_10pct',0):>5.2f} "
                          f"{m.get('O2csat_20pct',0):>5.2f} "
                          f"{wall:>5.1f}s")

    out_dir = experiment_dir("diagnostics", f"pdl_probe_N{args.N}_{args.structure}_{args.constraint_type}_g{args.gamma:g}_B{args.B}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
