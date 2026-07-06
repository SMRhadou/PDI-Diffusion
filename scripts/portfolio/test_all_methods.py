#!/usr/bin/env python3
"""Evaluate all methods on a held-out test set of problem instances."""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import experiment_dir, _project_root

_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (
    PortfolioEnergyDDPM, make_portfolio_problem,
    variance_contributions_torch, shortfall_contributions_torch,
    variance_contributions_np,
)
from pdi.models.portfolio_gnn_backbone import (
    PortfolioGNNBackbone, build_dense_adjacency,
)
from pdi.trainers.energy_score import ScoreNetWithLambda
from baselines import pd_langevin, _compute_constraints

PROBLEM_CONFIGS = {
    "crypto": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0,
                   structure="sectors", num_sectors=10,
                   constraint_type="variance", budget_type="uniform"),
}

class _NoOp(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        return torch.zeros_like(x), None


def metrics_per_instance(w, Sigma, scenarios, budgets, alpha, constraint_type):
    with torch.no_grad():
        B, N = w.shape
        if constraint_type in ("variance", "variance_sector"):
            c = variance_contributions_torch(w, Sigma)
        else:
            c = shortfall_contributions_torch(w, scenarios, alpha)
        ret = (scenarios @ w.t()).mean(dim=0)
        var = (w @ Sigma * w).sum(dim=1)
        sharpe = ret / var.sqrt().clamp_min(1e-12)
        c_mean = c.mean(dim=0)
        csat = float((c_mean <= budgets + 1e-8).float().mean())
        o2max = float((c_mean - budgets).clamp_min(0).max())
        ent = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
        neff = float(ent.exp().mean())
        top1 = int(w.argmax(dim=1).unique().numel())
    return dict(ret=float(ret.mean()), sharpe=float(sharpe.mean()),
                csat=csat, o2max=o2max, neff=neff, top1=top1)


def run_ced_trained(score_net, mu, Sig, scen, bud, alpha, A_dense, B, T, device,
                    ib, ds, decay, lam0, lam_max, constraint_type,
                    objective_scale=1.0, constraint_scale=1.0):
    bud_scaled = bud * constraint_scale if constraint_type in ("variance", "variance_band", "variance_sector") else bud
    sampler = PortfolioEnergyDDPM(
        model=_NoOp(), num_timesteps=T, beta_schedule="cosine",
        portfolio_mu=mu, portfolio_Sigma=Sig, portfolio_scenarios=scen,
        portfolio_risk_budgets=bud_scaled, portfolio_alpha=alpha,
        energy_mc_samples=512, inverse_beta=ib, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred", dual_step_size=ds,
        dual_lambda_init=lam0, dual_lambda_max=lam_max,
        dual_lambda_decay=decay, shared_lambda=True,
        normalize_constraints=False, constraint_type=constraint_type, num_sectors=10,
        objective_scale=objective_scale, constraint_scale=constraint_scale,
    ).to(device)
    acp = sampler.alphas_cumprod.to(device)
    pc1 = sampler.posterior_mean_coef1.to(device)
    pc2 = sampler.posterior_mean_coef2.to(device)
    pv = sampler.posterior_variance.to(device)
    budgets_t = torch.tensor(bud, dtype=torch.float32, device=device)
    Sigma_t = torch.tensor(Sig, dtype=torch.float32, device=device)
    A_batch = A_dense.unsqueeze(0).expand(B, -1, -1)
    n_c = len(bud)

    torch.manual_seed(42)
    x_t = torch.randn(B, 1, len(mu), 1, device=device)
    lam = torch.full((1, n_c), lam0, device=device)
    for t_int in reversed(range(T)):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        dl = lam.expand(B, -1)
        with torch.no_grad():
            eps = score_net(x=x_t, timesteps=t, dual_lambda=dl,
                           edge_index=A_batch, edge_weight=None, cond=None)
        ab = acp[t_int]; sab = ab.sqrt().clamp_min(1e-6); somb = (1-ab).sqrt().clamp_min(1e-6)
        x0 = (x_t - somb * eps) / sab
        z = x0[:, 0, :, 0]; w = torch.softmax(z, dim=-1)
        zp = torch.log(w.clamp_min(1e-12)); zp = zp - zp.mean(dim=-1, keepdim=True)
        x0 = zp.view(B, 1, len(mu), 1)
        c = variance_contributions_torch(w, Sigma_t) * constraint_scale
        budgets_scaled_t = torch.tensor(bud_scaled, dtype=torch.float32, device=device)
        viol = (c - budgets_scaled_t.unsqueeze(0)).mean(dim=0)
        lam = ((1-decay)*lam + ds*viol.unsqueeze(0)).clamp(0, lam_max)
        mean = pc1[t_int]*x0 + pc2[t_int]*x_t
        x_t = mean + pv[t_int].sqrt().clamp_min(0)*torch.randn_like(x_t) if t_int > 0 else mean
    return torch.softmax(x_t[:, 0, :, 0], dim=-1)


def run_mc_ceiling(mu, Sig, scen, bud, alpha, B, T, K, device,
                   ib, ds, decay, lam0, lam_max, constraint_type,
                   objective_scale=1.0, constraint_scale=1.0):
    bud_scaled = bud * constraint_scale if constraint_type in ("variance", "variance_band", "variance_sector") else bud
    sampler = PortfolioEnergyDDPM(
        model=_NoOp(), num_timesteps=T, beta_schedule="cosine",
        portfolio_mu=mu, portfolio_Sigma=Sig, portfolio_scenarios=scen,
        portfolio_risk_budgets=bud_scaled, portfolio_alpha=alpha,
        energy_mc_samples=K, inverse_beta=ib, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred", dual_step_size=ds,
        dual_lambda_init=lam0, dual_lambda_max=lam_max,
        dual_lambda_decay=decay, shared_lambda=True,
        normalize_constraints=False, constraint_type=constraint_type, num_sectors=10,
        objective_scale=objective_scale, constraint_scale=constraint_scale,
    ).to(device)
    from torch_geometric.data import Data, Batch
    N = len(mu)
    graphs = [Data(x=None, y=torch.zeros(N, 1, 1),
                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                   num_nodes=N) for _ in range(B)]
    data = Batch.from_data_list(graphs).to(device)
    torch.manual_seed(42)
    z = sampler.sample(shape=(B, 1, N, 1), device=device, data=data)
    return sampler.z_to_portfolio_weights(z)


def run_unconstrained_mc(mu, Sig, scen, bud, alpha, B, T, K, device, constraint_type,
                         objective_scale=1.0, constraint_scale=1.0):
    return run_mc_ceiling(mu, Sig, scen, bud, alpha, B, T, K, device,
                          ib=2000, ds=1e-10, decay=0, lam0=0, lam_max=1e-10,
                          constraint_type=constraint_type,
                          objective_scale=objective_scale, constraint_scale=constraint_scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--size", type=str, default="crypto")
    p.add_argument("--num-test", type=int, default=20)
    p.add_argument("--test-seed-start", type=int, default=3000)
    p.add_argument("--B", type=int, default=256)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--tagconv-K", type=int, default=2)
    p.add_argument("--ib", type=float, default=2000)
    p.add_argument("--ds", type=float, default=500)
    p.add_argument("--decay", type=float, default=0.001)
    p.add_argument("--lam0", type=float, default=0.1)
    p.add_argument("--lam-max", type=float, default=10000)
    p.add_argument("--pdl-plr", type=float, default=0.01)
    p.add_argument("--pdl-dlr", type=float, default=100.0)
    p.add_argument("--pdl-iters", type=int, default=500)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcfg = dict(PROBLEM_CONFIGS[args.size])
    N = pcfg["N"]
    constraint_type = pcfg.pop("constraint_type", "shortfall")

    # O(1) energy scaling for variance-based constraints
    if constraint_type in ("variance", "variance_band", "variance_sector"):
        # Compute from seed=0 instance as reference
        _ref_mu, _ref_Sig, _ref_scen, _ref_bud, _ = make_portfolio_problem(
            seed=0, constraint_type=constraint_type, **pcfg)
        _c_eq_mean = variance_contributions_np(
            np.ones(pcfg["N"]) / pcfg["N"], _ref_Sig).mean()
        _ret_mean = abs(_ref_scen.mean(axis=0)).mean()
        _obj_scale = 1.0 / max(_ret_mean, 1e-12)
        _constraint_scale = 1.0 / max(_c_eq_mean, 1e-12)
    else:
        _obj_scale = 1.0
        _constraint_scale = 1.0

    # Load trained score net
    backbone = PortfolioGNNBackbone(d=N, hidden=args.hidden, num_layers=args.num_layers,
                                     num_timesteps=args.T, K=args.tagconv_K)
    score_net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    score_net.load_state_dict(state)
    score_net.to(device).eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Score net params: {sum(p.numel() for p in score_net.parameters())/1e6:.2f}M")

    methods = ["ced_trained", "mc_ceiling", "pd_langevin", "unconstrained_mc", "equal_weight"]
    all_results = {m: [] for m in methods}

    for i in range(args.num_test):
        seed = args.test_seed_start + i
        mu, Sig, scen, bud, alpha = make_portfolio_problem(
            seed=seed, constraint_type=constraint_type, **pcfg)
        Sigma_t = torch.tensor(Sig, dtype=torch.float32, device=device)
        scen_t = torch.tensor(scen, dtype=torch.float32, device=device)
        bud_t = torch.tensor(bud, dtype=torch.float32, device=device)
        A_dense = torch.tensor(build_dense_adjacency(Sig, top_k=20),
                               dtype=torch.float32, device=device)

        print(f"\n--- Instance {i+1}/{args.num_test} (seed={seed}) ---")

        # CED trained
        t0 = time.time()
        w_ced = run_ced_trained(score_net, mu, Sig, scen, bud, alpha, A_dense,
                                args.B, args.T, device, args.ib, args.ds, args.decay,
                                args.lam0, args.lam_max, constraint_type,
                                objective_scale=_obj_scale, constraint_scale=_constraint_scale)
        m = metrics_per_instance(w_ced, Sigma_t, scen_t, bud_t, alpha, constraint_type)
        all_results["ced_trained"].append(m)
        print(f"  CED-trained: ret={m['ret']:.4f} sharpe={m['sharpe']:.3f} csat={m['csat']:.3f} O2max={m['o2max']:.6f} Neff={m['neff']:.1f} |top1|={m['top1']} ({time.time()-t0:.1f}s)")

        # MC ceiling
        t0 = time.time()
        w_mc = run_mc_ceiling(mu, Sig, scen, bud, alpha, args.B, args.T, args.K, device,
                              args.ib, args.ds, args.decay, args.lam0, args.lam_max,
                              constraint_type,
                              objective_scale=_obj_scale, constraint_scale=_constraint_scale)
        m = metrics_per_instance(w_mc, Sigma_t, scen_t, bud_t, alpha, constraint_type)
        all_results["mc_ceiling"].append(m)
        print(f"  MC-ceiling:  ret={m['ret']:.4f} sharpe={m['sharpe']:.3f} csat={m['csat']:.3f} O2max={m['o2max']:.6f} Neff={m['neff']:.1f} |top1|={m['top1']} ({time.time()-t0:.1f}s)")

        # PDL
        t0 = time.time()
        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
        bud_scaled_t = torch.tensor(bud * _constraint_scale, dtype=torch.float32, device=device) if constraint_type in ("variance", "variance_band", "variance_sector") else bud_t
        w_pdl, _ = pd_langevin(mu_t, Sigma_t, scen_t, bud_scaled_t, alpha=float(alpha),
                                B=args.B, num_iters=args.pdl_iters,
                                primal_lr=args.pdl_plr, dual_lr=args.pdl_dlr,
                                device=device, seed=42,
                                constraint_type=constraint_type,
                                objective_scale=_obj_scale,
                                constraint_scale=_constraint_scale)
        m = metrics_per_instance(w_pdl, Sigma_t, scen_t, bud_t, alpha, constraint_type)
        all_results["pd_langevin"].append(m)
        print(f"  PDL:         ret={m['ret']:.4f} sharpe={m['sharpe']:.3f} csat={m['csat']:.3f} O2max={m['o2max']:.6f} Neff={m['neff']:.1f} |top1|={m['top1']} ({time.time()-t0:.1f}s)")

        # Unconstrained MC
        t0 = time.time()
        w_unc = run_unconstrained_mc(mu, Sig, scen, bud, alpha, args.B, args.T, args.K,
                                     device, constraint_type,
                                     objective_scale=_obj_scale, constraint_scale=_constraint_scale)
        m = metrics_per_instance(w_unc, Sigma_t, scen_t, bud_t, alpha, constraint_type)
        all_results["unconstrained_mc"].append(m)
        print(f"  Uncon-MC:    ret={m['ret']:.4f} sharpe={m['sharpe']:.3f} csat={m['csat']:.3f} O2max={m['o2max']:.6f} Neff={m['neff']:.1f} |top1|={m['top1']} ({time.time()-t0:.1f}s)")

        # Equal weight
        w_eq = torch.ones(args.B, N, device=device) / N
        m = metrics_per_instance(w_eq, Sigma_t, scen_t, bud_t, alpha, constraint_type)
        all_results["equal_weight"].append(m)
        print(f"  Equal-wt:    ret={m['ret']:.4f} sharpe={m['sharpe']:.3f} csat={m['csat']:.3f} O2max={m['o2max']:.6f} Neff={m['neff']:.1f} |top1|={m['top1']}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Method':>18s} {'Return':>14s} {'Sharpe':>14s} {'Csat':>14s} {'O2max':>14s} {'Neff':>10s} {'|top1|':>10s}")
    print("-" * 90)
    for mname in methods:
        vals = all_results[mname]
        for k in ["ret", "sharpe", "csat", "o2max", "neff", "top1"]:
            v = [r[k] for r in vals]
        line = f"{mname:>18s}"
        for k in ["ret", "sharpe", "csat", "o2max"]:
            v = [r[k] for r in vals]
            line += f"  {np.mean(v):.4f}±{np.std(v):.4f}"
        for k in ["neff", "top1"]:
            v = [r[k] for r in vals]
            line += f"  {np.mean(v):>6.1f}±{np.std(v):.1f}"
        print(line)
    print("=" * 90)


if __name__ == "__main__":
    main()
