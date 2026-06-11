#!/usr/bin/env python3
"""Probe a portfolio problem: check that constraints actually bind.

Implements HANDOFF §5: before training a constrained model, verify that
the constraint is doing work. Run unconstrained vs constrained and compare.
If similar, tighten gamma until constraints bind.

Usage:
    python scripts/portfolio/probe_problem.py --size small
    python scripts/portfolio/probe_problem.py --size small --gamma 1.0
    python scripts/portfolio/probe_problem.py --size small --scan-gamma
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import _project_root  # noqa: E402

_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (  # noqa: E402
    make_portfolio_problem,
)


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=200, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, seed=0),
    "large": dict(N=200, K_factors=20, R_scenarios=1000, seed=0),
}


def _evaluate_weights(weights_np: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                      scenarios: np.ndarray, budgets: np.ndarray,
                      alpha: float = 0.0) -> dict:
    """Compute metrics for a [B, N] array of weights."""
    if weights_np.ndim == 1:
        weights_np = weights_np[None, :]
    B, N = weights_np.shape
    # Expected-shortfall constraint: c_j(x) = x_j * E[(alpha - r^T x)_+]
    port_ret = scenarios @ weights_np.T  # [R, B]
    shortfall = np.maximum(alpha - port_ret, 0.0).mean(axis=0)  # [B]
    c = weights_np * shortfall[:, None]  # [B, N]
    violation = np.maximum(c - budgets[None, :], 0.0)
    ret = port_ret.mean(axis=0)  # [B]
    port_var = np.einsum('bi,ij,bj->b', weights_np, Sigma, weights_np)
    feas = (c <= budgets[None, :] + 1e-10).all(axis=1).astype(float)
    hhi = (weights_np ** 2).sum(axis=1)
    entropy = -(weights_np * np.log(np.clip(weights_np, 1e-12, None))).sum(axis=1)
    return {
        "expected_return": float(ret.mean()),
        "portfolio_var": float(port_var.mean()),
        "mean_violation": float(violation.mean()),
        "max_violation": float(violation.max()),
        "feasibility_rate": float(feas.mean()),
        "hhi": float(hhi.mean()),
        "effective_N": float(np.exp(entropy).mean()),
    }


def _markowitz_unconstrained(mu: np.ndarray, Sigma: np.ndarray,
                              risk_aversion: float = 1.0) -> np.ndarray:
    """Closed-form long-only mean-variance optimum on the simplex.

    Solves max_x (mu^T x - risk_aversion/2 * x^T Sigma x) s.t. 1^T x = 1, x >= 0
    via a cvxpy QP if available, else falls back to Sigma^-1 mu normalised.
    """
    try:
        import cvxpy as cp
        N = len(mu)
        x = cp.Variable(N)
        objective = cp.Maximize(mu @ x - 0.5 * risk_aversion * cp.quad_form(x, cp.psd_wrap(Sigma)))
        constraints = [cp.sum(x) == 1.0, x >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return np.asarray(x.value).flatten()
    except Exception:
        # Fallback: tangency-like
        w = np.linalg.solve(Sigma + 1e-6 * np.eye(len(mu)), mu)
        w = np.maximum(w, 0.0)
        return w / w.sum()


def _markowitz_constrained(mu: np.ndarray, Sigma: np.ndarray,
                             budgets: np.ndarray,
                             risk_aversion: float = 5.0) -> np.ndarray:
    """Mean-variance with per-asset risk-budget constraints via SLSQP.

    Solves the non-convex NLP
        max  mu^T x - 0.5 * risk_aversion * x^T Sigma x
        s.t. sum(x) == 1, x >= 0,  x_j * (Sigma x)_j <= b_j  forall j.
    """
    from scipy.optimize import minimize
    N = len(mu)

    def obj(x):
        return 0.5 * risk_aversion * x @ Sigma @ x - mu @ x

    def obj_grad(x):
        return risk_aversion * Sigma @ x - mu

    def ineq(x):
        return budgets - x * (Sigma @ x)

    def ineq_jac(x):
        Sx = Sigma @ x
        return -np.diag(Sx) - np.diag(x) @ Sigma

    eq = {"type": "eq", "fun": lambda x: x.sum() - 1.0,
          "jac": lambda x: np.ones(N)}
    ineq_c = {"type": "ineq", "fun": ineq, "jac": ineq_jac}
    bounds = [(0.0, 1.0)] * N

    x0 = np.ones(N) / N
    res = minimize(obj, x0, jac=obj_grad, method="SLSQP",
                   bounds=bounds, constraints=[eq, ineq_c],
                   options={"ftol": 1e-10, "maxiter": 500})
    if not res.success or np.any(np.isnan(res.x)):
        # SLSQP failed (likely infeasible x0). Return equal-weight as fallback.
        return x0
    return res.x


def probe_one(size: str, gamma: float, verbose: bool = True) -> dict:
    pcfg = PROBLEM_CONFIGS[size]
    mu, Sigma, scenarios, budgets, alpha = make_portfolio_problem(gamma=gamma, **pcfg)
    N = len(mu)

    # Baselines
    w_eq = np.ones(N) / N
    sigma_diag = np.sqrt(np.diag(Sigma))
    w_rp = (1.0 / sigma_diag) / (1.0 / sigma_diag).sum()
    w_mv_u = _markowitz_unconstrained(mu, Sigma, risk_aversion=5.0)
    w_mv_c = _markowitz_constrained(mu, Sigma, budgets, risk_aversion=5.0)

    results = {
        "equal_weight": _evaluate_weights(w_eq, mu, Sigma, scenarios, budgets, alpha),
        "risk_parity": _evaluate_weights(w_rp, mu, Sigma, scenarios, budgets, alpha),
        "markowitz_unconstrained": _evaluate_weights(w_mv_u, mu, Sigma, scenarios, budgets, alpha),
        "markowitz_constrained": _evaluate_weights(w_mv_c, mu, Sigma, scenarios, budgets, alpha),
    }

    # Is the constraint binding? Markowitz-unconstrained gives the optimal
    # return; if it is already feasible, our constraint is inactive.
    mv_u_infeas = 1.0 - results["markowitz_unconstrained"]["feasibility_rate"]

    # "Slack" = fraction of shortfall constraints that are active for
    # markowitz_constrained
    port_ret_c = scenarios @ w_mv_c  # [R]
    sf_c = np.maximum(alpha - port_ret_c, 0.0).mean()
    c_mvc = w_mv_c * sf_c  # [N]
    slack_frac = float((c_mvc < budgets - 1e-6).mean())
    binding_frac = 1.0 - slack_frac

    summary = {
        "size": size,
        "N": N,
        "gamma": gamma,
        "mv_unconstrained_infeas_rate": float(mv_u_infeas),
        "mv_constrained_binding_frac": float(binding_frac),
        "mv_u_return": results["markowitz_unconstrained"]["expected_return"],
        "mv_c_return": results["markowitz_constrained"]["expected_return"],
        "return_gap": results["markowitz_unconstrained"]["expected_return"] -
                      results["markowitz_constrained"]["expected_return"],
        "baselines": results,
    }
    if verbose:
        print(f"\n=== size={size}  N={N}  gamma={gamma} ===")
        print(f"Markowitz (unconstrained) infeasibility: {mv_u_infeas:.3f} "
              f"[>0 means constraint cuts off the unconstrained optimum]")
        print(f"Markowitz (constrained)  binding fraction: {binding_frac:.3f} "
              f"[fraction of constraints active at opt]")
        print(f"return gap (uncon - con): {summary['return_gap']:.5f}")
        print("")
        for label, m in results.items():
            print(f"  {label:<28s}  ret={m['expected_return']:+.4f}  "
                  f"var={m['portfolio_var']:.5f}  "
                  f"vio_mean={m['mean_violation']:.5f}  "
                  f"vio_max={m['max_violation']:.5f}  "
                  f"feas={m['feasibility_rate']:.3f}  "
                  f"HHI={m['hhi']:.4f}  N_eff={m['effective_N']:.1f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(PROBLEM_CONFIGS.keys()), default="small")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override gamma; default uses PROBLEM_CONFIGS")
    parser.add_argument("--scan-gamma", action="store_true",
                        help="Scan gamma in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}")
    args = parser.parse_args()

    if args.scan_gamma:
        gammas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        results = []
        for g in gammas:
            r = probe_one(args.size, g)
            results.append(r)
        print("\n=== gamma scan summary ===")
        print(f"{'gamma':>6s} {'uncon_infeas':>14s} {'binding':>10s} {'ret_gap':>10s}")
        for r in results:
            print(f"{r['gamma']:>6.2f} {r['mv_unconstrained_infeas_rate']:>14.3f} "
                  f"{r['mv_constrained_binding_frac']:>10.3f} "
                  f"{r['return_gap']:>10.5f}")
        out = Path(_project_root()) / "outputs" / "portfolio" / "diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        outfile = out / f"probe_gamma_scan_{args.size}.json"
        outfile.write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {outfile}")
    else:
        gamma = args.gamma if args.gamma is not None else {"small": 2.0, "medium": 1.5, "large": 1.0}[args.size]
        r = probe_one(args.size, gamma)
        out = Path(_project_root()) / "outputs" / "portfolio" / "diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        outfile = out / f"probe_{args.size}_gamma{gamma:g}.json"
        outfile.write_text(json.dumps(r, indent=2))
        print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
