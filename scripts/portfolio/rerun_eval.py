#!/usr/bin/env python3
"""Decoupled evaluation of CED and all baselines on the portfolio problem.

Loads a trained checkpoint (or runs MC-only) and evaluates all methods at
MATCHED hyperparameters (HANDOFF §14.4: no silent baseline inheritance).

Usage:
    # Pure MC + analytical + trainable baselines (no checkpoint):
    python scripts/portfolio/rerun_eval.py --size small --ib 100 --dual-step 1000

    # With a trained CED checkpoint:
    python scripts/portfolio/rerun_eval.py --size small --ib 100 --dual-step 1000 \\
        --ced-ckpt <train_dir>/score_net_best_pareto.pt
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
)
from pdi.models.portfolio_backbone import (  # noqa: E402
    PortfolioScoreBackbone,
)
from pdi.trainers.energy_score import (  # noqa: E402
    ScoreNetWithLambda,
)

import baselines as bl  # noqa: E402


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=200, gamma=1.0, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5, seed=0),
    "large": dict(N=200, K_factors=20, R_scenarios=1000, gamma=1.0, seed=0),
}


class _NoOp(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


def _make_batch(B, N):
    graphs = [Data(x=None, y=torch.zeros(N, 1, 1),
                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                   num_nodes=N) for _ in range(B)]
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def compute_metrics(w: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor,
                     scenarios: torch.Tensor, budgets: torch.Tensor,
                     alpha: torch.Tensor | float,
                     eps_feas: float = 0.1, strict_tol: float = 1e-8) -> dict:
    """Expectation-constraint metrics for a batch of portfolios [B, N]."""
    with torch.no_grad():
        c = shortfall_contributions_torch(w, scenarios, alpha)
        violation = (c - budgets.unsqueeze(0)).clamp_min(0.0)
        rel_violation = violation / budgets.unsqueeze(0).clamp_min(1e-12)
        ret = torch.matmul(scenarios, w.t()).mean(dim=0)
        Sx = torch.matmul(w, Sigma)
        var = (Sx * w).sum(dim=1)
        sharpe = ret / torch.sqrt(var.clamp_min(1e-12))

        feas_strict = (c <= budgets.unsqueeze(0) + strict_tol).all(dim=1).float()
        feas_relaxed = (rel_violation.amax(dim=1) <= eps_feas).float()
        csat = (c <= budgets.unsqueeze(0) + strict_tol).float().mean(dim=1)

        ent = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=1)
        hhi = (w ** 2).sum(dim=1)
        total_c = c.sum(dim=1)
        total_budget = budgets.sum()
        budget_util = total_c / total_budget.clamp_min(1e-12)

        # Option 2: E_x[c_j(x)] <= b_j (batch-mean feasibility)
        c_mean = c.mean(dim=0)  # [N]
        opt2_vio = (c_mean - budgets).clamp_min(0.0)
        opt2_rel = opt2_vio / budgets.clamp_min(1e-12)
        opt2_feas = float((c_mean <= budgets + strict_tol).all().item())
        opt2_csat = float((c_mean <= budgets + strict_tol).float().mean().item())
        opt2_max_rel = float(opt2_rel.max().item())
        opt2_util = float((c_mean / budgets.clamp_min(1e-12)).mean().item())

    return {
        "expected_return_mean": float(ret.mean().item()),
        "expected_return_std": float(ret.std().item()),
        "portfolio_var_mean": float(var.mean().item()),
        "portfolio_var_std": float(var.std().item()),
        "sharpe_mean": float(sharpe.mean().item()),
        "mean_violation": float(violation.mean().item()),
        "max_violation": float(violation.max().item()),
        "mean_rel_violation": float(rel_violation.mean().item()),
        "max_rel_violation": float(rel_violation.max().item()),
        "feas_strict": float(feas_strict.mean().item()),
        f"feas_relaxed_{eps_feas}": float(feas_relaxed.mean().item()),
        "constraint_sat_rate": float(csat.mean().item()),
        "entropy_mean": float(ent.mean().item()),
        "effective_N": float(torch.exp(ent).mean().item()),
        "hhi_mean": float(hhi.mean().item()),
        "budget_utilization": float(budget_util.mean().item()),
        "opt2_feas": opt2_feas,
        "opt2_csat": opt2_csat,
        "opt2_max_rel_vio": opt2_max_rel,
        "opt2_util": opt2_util,
    }


def print_row(label: str, m: dict, wall: float, eps_feas: float):
    feas_r_key = f"feas_relaxed_{eps_feas}"
    print(f"  {label:<26s} ret={m['expected_return_mean']:+.4f}±{m['expected_return_std']:.4f}  "
          f"var={m['portfolio_var_mean']:.5f}  "
          f"fs={m['feas_strict']:.3f}  f_{eps_feas}={m[feas_r_key]:.3f}  "
          f"vio={m['mean_violation']:.4f}/{m['max_violation']:.4f}  "
          f"Neff={m['effective_N']:5.1f}  util={m['budget_utilization']:.3f}  "
          f"wall={wall:6.2f}s")


# ---------------------------------------------------------------
# CED score-net loading helpers
# ---------------------------------------------------------------

def _make_trained_score_estimator(score_net, alphas_cumprod, N):
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    def _fn(self, x_t, t, context, dual_lambda, inverse_beta_override=None):
        del context, inverse_beta_override
        ei = edge_index.to(x_t.device)
        with torch.no_grad():
            eps_pred = score_net(
                x=x_t, timesteps=t, dual_lambda=dual_lambda,
                edge_index=ei, edge_weight=None,
                cond=None, return_intermediates=False,
            )
        ab = alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))
    return _fn


def load_ced_scorenet(ckpt_path: Path, N: int, T: int, hidden: int, num_layers: int,
                       device) -> nn.Module:
    backbone = PortfolioScoreBackbone(d=N, hidden=hidden, num_layers=num_layers,
                                       num_timesteps=T, cond_channels=1)
    score_net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    score_net.load_state_dict(state)
    score_net.to(device).eval()
    return score_net


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(PROBLEM_CONFIGS.keys()), default="small")
    parser.add_argument("--ib", type=float, default=100.0)
    parser.add_argument("--dual-step", type=float, default=1000.0)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--beta-schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--K", type=int, default=32,
                        help="MC samples for CED/MC inference")
    parser.add_argument("--B", type=int, default=256,
                        help="evaluation batch size (HANDOFF §8: >= 512 for papers)")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ced-ckpt", type=str, default=None,
                        help="path to trained score_net checkpoint (.pt)")
    parser.add_argument("--eps-feas", type=float, default=0.1,
                        help="tolerance for relaxed feasibility metric")
    parser.add_argument("--label", type=str, default="rerun_eval")
    # baseline hyperparameters (pin explicitly per HANDOFF §14.4)
    parser.add_argument("--pdl-primal-lr", type=float, default=1e-2)
    parser.add_argument("--pdl-dual-lr", type=float, default=1000.0)
    parser.add_argument("--pdl-iters", type=int, default=500)
    parser.add_argument("--pdl-noise", type=float, default=0.1)
    parser.add_argument("--policy-iters", type=int, default=2000)
    parser.add_argument("--policy-primal-lr", type=float, default=1e-3)
    parser.add_argument("--policy-dual-lr", type=float, default=100.0)
    parser.add_argument("--policy-noise", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    pcfg = PROBLEM_CONFIGS[args.size]
    N = pcfg["N"]
    mu_np, Sigma_np, scen_np, bud_np, alpha = make_portfolio_problem(**pcfg)

    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float32, device=device)
    scenarios = torch.tensor(scen_np, dtype=torch.float32, device=device)
    budgets = torch.tensor(bud_np, dtype=torch.float32, device=device)

    out_dir = experiment_dir("score_net_eval",
                              f"{args.size}_ib{args.ib:g}_ds{args.dual_step:g}_{args.label}")
    print(f"output dir: {out_dir}")
    print(f"problem: {args.size} N={N} ib={args.ib} ds={args.dual_step} T={args.T} B={args.B}")

    results = {}

    # ------------------------------------------------------------------
    # Analytical baselines
    # ------------------------------------------------------------------
    print("\n=== Analytical baselines ===")

    # Equal weight
    t0 = time.time()
    w = bl.equal_weight(N, args.B, device)
    wall = time.time() - t0
    m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["equal_weight"] = {"metrics": m, "wall_sec": wall}
    print_row("equal_weight", m, wall, args.eps_feas)

    # Risk parity
    t0 = time.time()
    w = bl.risk_parity(Sigma, args.B)
    wall = time.time() - t0
    m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["risk_parity"] = {"metrics": m, "wall_sec": wall}
    print_row("risk_parity", m, wall, args.eps_feas)

    # Markowitz unconstrained (single solution, broadcast)
    t0 = time.time()
    w_np = bl.markowitz_unconstrained(mu_np, Sigma_np, risk_aversion=5.0)
    w = torch.tensor(w_np, dtype=torch.float32, device=device).unsqueeze(0).expand(args.B, -1)
    wall = time.time() - t0
    m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["markowitz_unconstrained"] = {"metrics": m, "wall_sec": wall}
    print_row("markowitz_uncon", m, wall, args.eps_feas)

    # Markowitz constrained (oracle upper bound on feasibility)
    t0 = time.time()
    w_np = bl.markowitz_constrained(mu_np, Sigma_np, bud_np, risk_aversion=5.0)
    w = torch.tensor(w_np, dtype=torch.float32, device=device).unsqueeze(0).expand(args.B, -1)
    wall = time.time() - t0
    m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["markowitz_constrained"] = {"metrics": m, "wall_sec": wall}
    print_row("markowitz_con", m, wall, args.eps_feas)

    # ------------------------------------------------------------------
    # Diffusion baselines
    # ------------------------------------------------------------------
    print("\n=== Diffusion baselines ===")

    def _make_ced_sampler(ib, dual_step, mc_samples):
        return PortfolioEnergyDDPM(
            model=_NoOp(),
            num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
            portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_np, portfolio_alpha=alpha,
            energy_mc_samples=mc_samples,
            inverse_beta=ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=dual_step,
            dual_num_outer_iterations=1,
            dual_lambda_init=0.0,
            dual_lambda_max=1e6,
            shared_lambda=True,
        ).to(device)

    def _make_ced_sampler_unconstrained(ib, mc_samples):
        """No dual update: lambda stays at zero."""
        return PortfolioEnergyDDPM(
            model=_NoOp(),
            num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
            portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_np, portfolio_alpha=alpha,
            energy_mc_samples=mc_samples,
            inverse_beta=ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=1e-12,  # effectively zero
            dual_num_outer_iterations=1,
            dual_lambda_init=0.0,
            dual_lambda_max=1e6,
            shared_lambda=True,
        ).to(device)

    shape = (args.B, 1, N, 1)
    data = _make_batch(args.B, N).to(device)

    # Unconstrained MC (lambda = 0 forever)
    torch.manual_seed(args.seed)
    t0 = time.time()
    sampler_un = _make_ced_sampler_unconstrained(args.ib, args.K)
    z = sampler_un.sample(shape=shape, device=device, data=data)
    w = sampler_un.z_to_portfolio_weights(z)
    wall = time.time() - t0
    m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["unconstrained_mc"] = {"metrics": m, "wall_sec": wall}
    print_row("unconstrained_mc", m, wall, args.eps_feas)

    # MC teacher (energy-guided with dual ascent)
    torch.manual_seed(args.seed)
    t0 = time.time()
    sampler_mc = _make_ced_sampler(args.ib, args.dual_step, args.K)
    z = sampler_mc.sample(shape=shape, device=device, data=data)
    w = sampler_mc.z_to_portfolio_weights(z)
    wall = time.time() - t0
    m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["mc_teacher"] = {"metrics": m, "wall_sec": wall}
    print_row("mc_teacher", m, wall, args.eps_feas)

    # PDM (unconstrained sampler + projection)
    torch.manual_seed(args.seed)
    t0 = time.time()
    sampler_pdm = _make_ced_sampler_unconstrained(args.ib, args.K)
    z_un = sampler_pdm.sample(shape=shape, device=device, data=data)
    w_un = sampler_pdm.z_to_portfolio_weights(z_un)
    w_proj = bl.blend_to_feasible(w_un, Sigma, scenarios, float(alpha), budgets, num_iters=40, tol=1e-8)
    wall = time.time() - t0
    m = compute_metrics(w_proj, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    results["pdm"] = {"metrics": m, "wall_sec": wall}
    print_row("pdm", m, wall, args.eps_feas)

    # Rejection sampling (unconstrained + reject infeasible)
    torch.manual_seed(args.seed)
    t0 = time.time()
    sampler_rej = _make_ced_sampler_unconstrained(args.ib, args.K)
    w_rej, rej_info = bl.rejection_sampling(
        sampler=sampler_rej, shape=shape, data=data, device=device,
        scenarios=scenarios, alpha=float(alpha), budgets=budgets,
        max_retries=3, tol=1e-6,
    )
    wall = time.time() - t0
    m = compute_metrics(w_rej, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    m.update(rej_info)
    results["rejection"] = {"metrics": m, "wall_sec": wall}
    print_row("rejection", m, wall, args.eps_feas)

    # CED trained score net (if checkpoint provided)
    if args.ced_ckpt is not None:
        ckpt = Path(args.ced_ckpt)
        if not ckpt.exists():
            print(f"[WARN] ced checkpoint not found: {ckpt}")
        else:
            print(f"\n=== CED trained net: {ckpt.name} ===")
            score_net = load_ced_scorenet(ckpt, N=N, T=args.T,
                                            hidden=args.hidden,
                                            num_layers=args.num_layers,
                                            device=device)
            torch.manual_seed(args.seed)
            t0 = time.time()
            sampler_ced = _make_ced_sampler(args.ib, args.dual_step, args.K)
            sampler_ced._estimate_score = types.MethodType(
                _make_trained_score_estimator(score_net, sampler_ced.alphas_cumprod, N),
                sampler_ced,
            )
            z = sampler_ced.sample(shape=shape, device=device, data=data)
            w = sampler_ced.z_to_portfolio_weights(z)
            wall = time.time() - t0
            m = compute_metrics(w, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
            results["ced_trained"] = {"metrics": m, "wall_sec": wall, "ckpt": str(ckpt)}
            print_row("ced_trained", m, wall, args.eps_feas)

    # ------------------------------------------------------------------
    # Primal-dual Langevin (no diffusion, classical baseline)
    # ------------------------------------------------------------------
    print("\n=== PD-Langevin ===")
    t0 = time.time()
    w_pdl, pdl_info = bl.pd_langevin(
        mu=mu, Sigma=Sigma, scenarios=scenarios, budgets=budgets,
        alpha=float(alpha), B=args.B,
        num_iters=args.pdl_iters, primal_lr=args.pdl_primal_lr,
        dual_lr=args.pdl_dual_lr, noise_scale=args.pdl_noise,
        device=device, seed=args.seed,
    )
    wall = time.time() - t0
    m = compute_metrics(w_pdl, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    m.update(pdl_info)
    results["pd_langevin"] = {"metrics": m, "wall_sec": wall}
    print_row("pd_langevin", m, wall, args.eps_feas)

    # ------------------------------------------------------------------
    # Trainable policy net
    # ------------------------------------------------------------------
    print("\n=== Policy net (trainable) ===")
    t0 = time.time()
    _, w_pol, pol_info = bl.train_policy_net(
        mu=mu, Sigma=Sigma, scenarios=scenarios, budgets=budgets,
        alpha=float(alpha),
        B=args.B, num_iters=args.policy_iters,
        primal_lr=args.policy_primal_lr, dual_lr=args.policy_dual_lr,
        inverse_beta=args.ib, noise_scale=args.policy_noise,
        shared_lambda=True, device=device, seed=args.seed,
        hidden=args.hidden, num_layers=args.num_layers,
        log_every=max(1, args.policy_iters // 4),
    )
    wall = time.time() - t0
    m = compute_metrics(w_pol, mu, Sigma, scenarios, budgets, alpha, args.eps_feas)
    m.update(pol_info)
    results["policy_net"] = {"metrics": m, "wall_sec": wall}
    print_row("policy_net", m, wall, args.eps_feas)

    # ------------------------------------------------------------------
    # Save and summarise
    # ------------------------------------------------------------------
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Also save the raw weight samples for figure generation
    all_weights = {}
    # Re-run to dump weights under the same rng seeds (deterministic) — or just
    # capture during the run above. Saving per-method summary for now.

    # Summary table
    lines = ["=== Portfolio experiment summary ===", ""]
    lines.append(f"problem: {args.size}  N={N}  ib={args.ib}  ds={args.dual_step}  T={args.T}  B={args.B}")
    lines.append("")
    feas_r_key = f"feas_relaxed_{args.eps_feas}"
    lines.append(f"{'method':<26s} {'ret':>10s} {'var':>10s} {'sharpe':>7s} "
                  f"{'feas_strict':>11s} {f'feas@{args.eps_feas}':>10s} "
                  f"{'vio_mean':>9s} {'vio_max':>9s} {'Neff':>6s} {'wall':>7s}")
    for label, r in results.items():
        m = r["metrics"]
        lines.append(
            f"{label:<26s} {m['expected_return_mean']:>+10.4f} "
            f"{m['portfolio_var_mean']:>10.5f} {m['sharpe_mean']:>7.3f} "
            f"{m['feas_strict']:>11.3f} {m[feas_r_key]:>10.3f} "
            f"{m['mean_violation']:>9.4f} {m['max_violation']:>9.4f} "
            f"{m['effective_N']:>6.1f} {r['wall_sec']:>6.2f}s"
        )
    msg = "\n".join(lines)
    print("\n" + msg)
    (out_dir / "summary.txt").write_text(msg + "\n")

    print(f"\nSaved to: {out_dir}")
    return results, out_dir


if __name__ == "__main__":
    main()
