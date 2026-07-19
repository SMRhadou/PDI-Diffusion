#!/usr/bin/env python3
"""Train a portfolio score net with best-model selection and calibrated prior.

Implements HANDOFF best practices:
- Calibrated exponential lambda prior (run calibrate_lambda.py first)
- Best-model selection with lexicographic tuple (feas_ok, -var, return)
- Multiple best checkpoints: best_sharpe, best_feas, best_pareto (HANDOFF §14.1)
- Eval batch >= 512 for stable mode estimates (HANDOFF §6)
- eps_norm = 0.1 default (HANDOFF §11)
- Logged lambda trajectory, loss curves, eval metrics every K iters.

Usage:
    python scripts/portfolio/train_portfolio.py --size small --ib 100 --dual-step 1000 \\
        --mu-min 0.5 --mu-max 10 --rho-max 0.9 --num-outer 400 --label mumax10_rho09
"""
from __future__ import annotations

import argparse
import json
import logging
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
    PortfolioEnergyDDPM, make_portfolio_problem, make_enriched_portfolio_problem,
    shortfall_contributions_torch, variance_contributions_torch,
)
from baselines import _compute_constraints  # noqa: E402
from pdi.models.portfolio_backbone import (  # noqa: E402
    PortfolioScoreBackbone,
)
from pdi.models.portfolio_gnn_backbone import (  # noqa: E402
    PortfolioGNNBackbone, build_dense_adjacency,
)
from pdi.trainers.energy_score import (  # noqa: E402
    EnergyScoreTrainConfig, EnergyScoreTrainer, ScoreNetWithLambda,
    ExponentialLambdaPrior,
)


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=200, gamma=1.0, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5, seed=0),
    "crypto": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                   structure="sectors", num_sectors=10,
                   constraint_type="variance", budget_type="uniform"),
    "sectors_shortfall_g3": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                                 structure="sectors", num_sectors=10,
                                 constraint_type="shortfall"),
    "crypto_band": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                        structure="sectors", num_sectors=10,
                        constraint_type="variance_band"),
}


def expand_enriched_lambda(dual_lambda, num_nodes, num_sectors=10, log1p=True):
    """Map [B, n_constraints] → [B, N, C] for enriched portfolio constraints.

    Sector upper/lower duals are separate channels (they demand opposite
    responses; summing them made the direction unidentifiable).
    C=5 with stress (var, sec_upper, sec_lower, stress, Neff),
    C=4 without (var, sec_upper, sec_lower, Neff).
    """
    B, C_total = dual_lambda.shape
    N = num_nodes
    S = num_sectors
    dl = torch.log1p(dual_lambda) if log1p else dual_lambda
    var_lam = dl[:, :N]
    sector_ids = torch.arange(N, device=dl.device) % S
    sec_up_node = dl[:, N:N + S][:, sector_ids]
    sec_lo_node = dl[:, N + S:N + 2 * S][:, sector_ids]
    M = C_total - N - 2 * S - 1
    neff_per_node = dl[:, -1:].expand(B, N)
    if M > 0:
        stress_mean = dl[:, N + 2 * S:N + 2 * S + M].mean(dim=1, keepdim=True).expand(B, N)
        return torch.stack([var_lam, sec_up_node, sec_lo_node, stress_mean,
                            neff_per_node], dim=-1)
    return torch.stack([var_lam, sec_up_node, sec_lo_node, neff_per_node], dim=-1)


class _NoOpModel(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


def _make_batch(B, N):
    graphs = [Data(x=None, y=torch.zeros(N, 1, 1),
                   edge_index=torch.zeros(2, 0, dtype=torch.long),
                   num_nodes=N) for _ in range(B)]
    return Batch.from_data_list(graphs)


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


def _eval_metrics_per_instance(w, Sigma, scenarios, budgets, alpha,
                               constraint_type="shortfall", extra=None):
    """Compute metrics for ONE problem instance. Matches figures.py table metrics."""
    with torch.no_grad():
        if constraint_type == "enriched":
            c = _compute_constraints(w, Sigma, scenarios, alpha, "enriched", extra=extra)
        elif constraint_type == "variance_band":
            c_var = variance_contributions_torch(w, Sigma)
            c = torch.cat([c_var, -c_var], dim=1)  # [B, 2N]
        elif constraint_type in ("variance", "variance_sector"):
            c = variance_contributions_torch(w, Sigma)
        else:
            c = shortfall_contributions_torch(w, scenarios, alpha)
        B, N = w.shape
        ret = torch.matmul(scenarios, w.t()).mean(dim=0)  # [B]
        Sx = torch.matmul(w, Sigma)
        var = (Sx * w).sum(dim=1)  # [B]
        sharpe = ret / torch.sqrt(var.clamp_min(1e-12))  # [B]
        # avg-constraint feasibility: average c over samples, then check per constraint
        c_mean = c.mean(dim=0)  # [N]
        opt2_csat = float((c_mean <= budgets + 1e-8).float().mean().item())
        opt2_max_vio = float((c_mean - budgets).clamp_min(0.0).max().item())
        # N_eff
        log_w = torch.log(w.clamp_min(1e-12))
        ent = -(w * log_w).sum(dim=1)  # [B]
        neff = float(ent.exp().mean().item())
        # top-1 diversity
        top1 = w.argmax(dim=1)  # [B]
        n_top1 = int(top1.unique().numel())
        counts = torch.bincount(top1, minlength=N).float()
        p = counts / counts.sum()
        p_pos = p[p > 0]
        top1_entropy = float(-(p_pos * p_pos.log()).sum().item())
        # mean violation
        violation = (c - budgets.unsqueeze(0)).clamp_min(0.0)
    return {
        "ret": float(ret.mean().item()),
        "sharpe": float(sharpe.mean().item()),
        "opt2_csat": opt2_csat,
        "opt2_max_vio": opt2_max_vio,
        "neff": neff,
        "n_top1": n_top1,
        "top1_entropy": top1_entropy,
        "vio_mean": float(violation.mean().item()),
    }


def _eval_checkpoint(score_net, sampler_factory, shape, data, device, mu, Sigma,
                     scenarios, budgets, alpha, B=256, seed=0, eps_feas=0.1,
                     constraint_type="shortfall"):
    """Run the score net through the sampler and compute metrics."""
    torch.manual_seed(seed)
    score_net.eval()
    sampler = sampler_factory()
    N = sampler._port_mu.numel()
    sampler._estimate_score = types.MethodType(
        _make_trained_score_estimator(score_net, sampler.alphas_cumprod, N),
        sampler,
    )
    z = sampler.sample(shape=shape, device=device, data=data)
    w = sampler.z_to_portfolio_weights(z)
    return _eval_metrics(w, mu, Sigma, scenarios, budgets, alpha, eps_feas=eps_feas,
                        constraint_type=constraint_type)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(PROBLEM_CONFIGS.keys()), default="small")
    p.add_argument("--ib", type=float, default=10.0)
    p.add_argument("--dual-step", type=float, default=0.05)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--beta-schedule", type=str, default="cosine",
                   choices=["linear", "cosine", "sigmoid"])
    p.add_argument("--num-outer", type=int, default=400)
    p.add_argument("--inner-steps", type=int, default=10)
    p.add_argument("--inner-steps-max", type=int, default=100,
                   help="Ramp inner steps to this value")
    p.add_argument("--inner-steps-ramp-iter", type=int, default=3000,
                   help="SGD iter at which inner steps reach max")
    p.add_argument("--lr-min", type=float, default=0.0,
                   help="Min LR for cosine decay (0=no decay)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-B", type=int, default=256)
    p.add_argument("--mc-samples-train", type=int, default=512)
    p.add_argument("--mc-samples-eval", type=int, default=512)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mu-min", type=float, default=0.5, help="exponential lambda prior mu_min")
    p.add_argument("--mu-max", type=float, default=10.0, help="exponential lambda prior mu_max")
    p.add_argument("--rho-max", type=float, default=0.7)
    p.add_argument("--shared-lambda", action="store_true", default=True)
    p.add_argument("--no-normalize", action="store_true", default=False)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--buffer-cap", type=int, default=8192)
    p.add_argument("--eval-every", type=int, default=20,
                   help="Evaluate every N outer iterations")
    p.add_argument("--eval-start", type=int, default=0,
                   help="Skip eval until this outer iteration")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--target-clip-norm", type=float, default=20.0)
    p.add_argument("--perturb-fraction", type=float, default=0.5)
    p.add_argument("--perturb-x-std", type=float, default=0.5)
    p.add_argument("--perturb-lambda-std", type=float, default=0.5)
    p.add_argument("--perturb-lambda-add", type=float, default=0.0,
                   help="Additive lambda perturbation scale (x prior mean mu(t)); "
                        "resurrects zeroed lambdas. 0=off (portfolio only)")
    p.add_argument("--perturb-lambda-add-frac", type=float, default=0.25,
                   help="Fraction of lambda entries receiving the additive kick")
    p.add_argument("--num-rollouts-per-outer", type=int, default=4)
    p.add_argument("--dual-lambda-max", type=float, default=50.0)
    p.add_argument("--lam0", type=float, default=0.1)
    p.add_argument("--backbone", type=str, default="gnn", choices=["mlp", "gnn"])
    p.add_argument("--tagconv-K", type=int, default=2, help="TAGConv hops")
    p.add_argument("--dual-lambda-decay", type=float, default=0.0)
    p.add_argument("--constraint-type", type=str, default=None,
                   help="Override constraint type from problem config")
    p.add_argument("--sector-gamma", type=float, default=1.5)
    p.add_argument("--skip-stress", action="store_true", default=False,
                   help="Enriched without stress constraints")
    p.add_argument("--no-log1p-lambda", action="store_true", default=False,
                   help="Disable log1p normalization of lambda conditioning")
    p.add_argument("--lambda-dropout", type=float, default=0.0,
                   help="Fraction of minibatch samples with lambda zeroed out (CFG-style)")
    p.add_argument("--problem-seed", type=int, default=None,
                   help="Override problem seed (for multi-instance training)")
    p.add_argument("--rotate-every", type=int, default=0,
                   help="Rotate problem instance every N outer iters (0=disabled)")
    p.add_argument("--num-instances", type=int, default=20,
                   help="Number of problem instances to rotate through")
    p.add_argument("--num-val-instances", type=int, default=10,
                   help="Number of held-out val instances (seeds start at 1000)")
    p.add_argument("--best-min-csat", type=float, default=0.8,
                   help="best_* checkpoints require val csat >= this floor")
    p.add_argument("--best-min-neff", type=float, default=5.0,
                   help="best_* checkpoints require val Neff >= this floor")
    p.add_argument("--best-gate-csat", type=float, default=0.9,
                   help="csat gate for best_return / best_neff checkpoints")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)
    log = logging.getLogger("train_portfolio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    pcfg = dict(PROBLEM_CONFIGS[args.size])
    if args.problem_seed is not None:
        pcfg["seed"] = args.problem_seed
    N = pcfg["N"]
    constraint_type = pcfg.pop("constraint_type", "shortfall")
    if args.constraint_type is not None:
        constraint_type = args.constraint_type
    if constraint_type == "enriched":
        mu_np, Sigma_np, scen_np, bud_np, alpha, _extra_constraints = \
            make_enriched_portfolio_problem(sector_gamma=args.sector_gamma, skip_stress=args.skip_stress, **pcfg)
        ret_mean = abs(scen_np.mean(axis=0)).mean()
        obj_scale = 1.0 / max(ret_mean, 1e-12)
        constraint_scale = 1.0
        bud_scaled = bud_np
        log.info("enriched problem: %d constraints, obj_scale=%.4g",
                 len(bud_np), obj_scale)
    else:
        mu_np, Sigma_np, scen_np, bud_np, alpha = make_portfolio_problem(
            constraint_type=constraint_type, **pcfg)
        _extra_constraints = {}
        if constraint_type in ("variance", "variance_band", "variance_sector"):
            from pdi.diffusion.portfolio_energy_ddpm import variance_contributions_np
            c_eq_mean = variance_contributions_np(np.ones(N) / N, Sigma_np).mean()
            ret_mean = abs(scen_np.mean(axis=0)).mean()
            obj_scale = 1.0 / max(ret_mean, 1e-12)
            constraint_scale = 1.0 / max(c_eq_mean, 1e-12)
            bud_scaled = bud_np * constraint_scale
            log.info("O(1) energy scaling: obj_scale=%.4g constraint_scale=%.4g "
                     "c_eq_mean=%.4e ret_mean=%.4e", obj_scale, constraint_scale,
                     c_eq_mean, ret_mean)
        else:
            obj_scale = 1.0
            constraint_scale = 1.0
            bud_scaled = bud_np

    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float32, device=device)
    scenarios = torch.tensor(scen_np, dtype=torch.float32, device=device)
    budgets = torch.tensor(bud_np, dtype=torch.float32, device=device)

    out_dir = experiment_dir(
        "score_net_train",
        f"{args.size}_ib{args.ib:g}_ds{args.dual_step:g}_mumax{args.mu_max:g}_rho{args.rho_max:g}_{args.label}",
    )

    # File logging + args dump (like WRA)
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("output dir: %s", out_dir)
    log.info("config: %s", vars(args))
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Score net
    if constraint_type == "enriched":
        _lambda_channels = 4 if args.skip_stress else 5
    else:
        _lambda_channels = 1
    _cond_ch = _lambda_channels  # no dataset cond, so total cond = lambda channels
    if args.backbone == "gnn":
        backbone = PortfolioGNNBackbone(
            d=N, hidden=args.hidden, num_layers=args.num_layers,
            num_timesteps=args.T, K=args.tagconv_K, cond_channels=_cond_ch,
        )
    else:
        backbone = PortfolioScoreBackbone(
            d=N, hidden=args.hidden, num_layers=args.num_layers,
            num_timesteps=args.T, cond_channels=_cond_ch,
        )
    score_net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0,
                                   lambda_channels=_lambda_channels)
    log.info("score_net params: %.2fM", sum(p.numel() for p in score_net.parameters()) / 1e6)

    # MC sampler for training
    def _mk_training_sampler():
        return PortfolioEnergyDDPM(
            model=_NoOpModel(),
            num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
            portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_scaled,
            portfolio_alpha=alpha,
            energy_mc_samples=args.mc_samples_train,
            inverse_beta=args.ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=args.dual_step,
            dual_lambda_init=args.lam0,
            dual_lambda_max=args.dual_lambda_max,
            dual_lambda_decay=args.dual_lambda_decay,
            shared_lambda=args.shared_lambda,
            normalize_constraints=not args.no_normalize,
            objective_scale=obj_scale,
            constraint_type=constraint_type,
            num_sectors=pcfg.get("num_sectors", 10),
            constraint_scale=constraint_scale,
            extra_constraints=_extra_constraints,
        )
    mc_sampler = _mk_training_sampler()

    cfg = EnergyScoreTrainConfig(
        lr=args.lr,
        k_train=args.mc_samples_train,
        inner_steps_per_outer=args.inner_steps,
        minibatch_size=args.minibatch_size,
        rho_max=args.rho_max,
        warmup_iters=args.warmup,
        buffer_capacity=args.buffer_cap,
        lambda_prior_mu_min=args.mu_min,
        lambda_prior_mu_max=args.mu_max,
        target_normalize=True,
        target_normalize_eps=0.1,
        target_clip_norm=args.target_clip_norm,
        perturb_fraction=args.perturb_fraction,
        perturb_x_std=args.perturb_x_std,
        perturb_lambda_std=args.perturb_lambda_std,
        num_rollouts_per_outer=args.num_rollouts_per_outer,
    )

    trainer = EnergyScoreTrainer(
        score_net=score_net, mc_sampler=mc_sampler,
        config=cfg, device=device, rng_seed=args.seed,
    )

    # Override buffer push: count T per record (not T×B) so capacity
    # limits the number of (instance, timestep) pairs, not samples.
    from pdi.trainers.energy_score.trajectory_buffer import _RolloutRecord
    def _portfolio_push(data, x_by_t, lambda_by_t, t_values,
                        sample_weights=None, persistent=False):
        buf = trainer.buffer
        T = x_by_t.shape[0]
        sw = None
        if sample_weights is not None:
            sw = sample_weights.detach().to("cpu", torch.float32).contiguous()
        record = _RolloutRecord(
            data=data.detach().cpu() if hasattr(data, "detach") else data.cpu(),
            x_by_t=x_by_t.detach().to("cpu", torch.float32).contiguous(),
            lambda_by_t=lambda_by_t.detach().to("cpu", torch.float32).contiguous(),
            t_values=t_values.detach().to("cpu", torch.long).contiguous(),
            sample_weights=sw,
            persistent=bool(persistent),
        )
        buf._records.append(record)
        buf._entry_count += T
        while buf._entry_count > buf.capacity:
            evict_idx = next(
                (i for i, r in enumerate(buf._records) if not r.persistent), None)
            if evict_idx is None or len(buf._records) <= 1:
                break
            oldest = buf._records.pop(evict_idx)
            buf._entry_count -= oldest.t_values.shape[0]
    trainer.buffer.push_trajectory = _portfolio_push

    # Monkey-patch _build_micro_context to read instance data from PyG batch
    _orig_build_ctx = trainer._build_micro_context
    def _patched_build_ctx(data):
        if hasattr(data, "_port_Sigma"):
            from pdi.diffusion.portfolio_energy_ddpm import PortfolioContext
            batch_size = int(data.num_graphs)
            dev = trainer.device
            Sigma = data._port_Sigma.to(dev)
            scenarios = data._port_scenarios.to(dev)
            risk_budgets = data._port_risk_budgets.to(dev)
            alpha_val = data._port_alpha.to(dev)
            mu = mc_sampler._port_mu.to(dev)
            r_min = (-risk_budgets).unsqueeze(0).expand(batch_size, -1)
            ctx_kwargs = {}
            if hasattr(data, "_port_extra") and data._port_extra:
                ex = data._port_extra
                if "sector_ids" in ex:
                    ctx_kwargs["sector_ids"] = torch.as_tensor(ex["sector_ids"], dtype=torch.long, device=dev)
                    ctx_kwargs["sector_upper"] = torch.as_tensor(ex["sector_upper"], dtype=torch.float32, device=dev)
                    ctx_kwargs["sector_lower"] = torch.as_tensor(ex["sector_lower"], dtype=torch.float32, device=dev)
                    ctx_kwargs["num_sectors"] = len(ex["sector_upper"])
                    ctx_kwargs["n_sector_constraints"] = 2 * len(ex["sector_upper"])
                if "stress_returns" in ex:
                    ctx_kwargs["stress_returns"] = torch.as_tensor(ex["stress_returns"], dtype=torch.float32, device=dev)
                    ctx_kwargs["stress_limits"] = torch.as_tensor(ex["stress_limits"], dtype=torch.float32, device=dev)
                    ctx_kwargs["stress_eps"] = torch.as_tensor(ex["stress_eps"], dtype=torch.float32, device=dev)
                    ctx_kwargs["n_stress_constraints"] = len(ex["stress_limits"])
                if "neff_target" in ex:
                    ctx_kwargs["neff_target"] = float(ex["neff_target"])
                if "constraint_scales" in ex:
                    cs = torch.as_tensor(ex["constraint_scales"], dtype=torch.float32, device=dev)
                    ctx_kwargs["constraint_scales"] = cs
                    risk_budgets = risk_budgets * cs
                    r_min = (-risk_budgets).unsqueeze(0).expand(batch_size, -1)
            return PortfolioContext(
                mu=mu, Sigma=Sigma, scenarios=scenarios,
                risk_budgets=risk_budgets, alpha=alpha_val, r_min=r_min,
                **ctx_kwargs,
            )
        return _orig_build_ctx(data)
    trainer._build_micro_context = _patched_build_ctx

    # Monkey-patch _gather_minibatch: sample individual (x_t, t, λ) points,
    # grouped by record (=instance) for batched GPU calls.
    def _patched_gather_minibatch(entries):
        from collections import defaultdict
        groups = defaultdict(list)
        for e in entries:
            groups[id(e.record)].append(e)

        micro_batches = []
        _dev = trainer.device
        for _, group in groups.items():
            record = group[0].record
            data = record.data
            num_nodes = int(data.y.size(0) // int(data.num_graphs))
            n = len(group)
            x_parts, lam_parts, t_parts = [], [], []
            for e in group:
                x_parts.append(record.x_by_t[e.step_index, e.sample_index:e.sample_index+1])
                lam_parts.append(record.lambda_by_t[e.step_index, e.sample_index:e.sample_index+1])
                t_parts.append(int(record.t_values[e.step_index].item()))
            x_t_mb = torch.cat(x_parts, dim=0).to(_dev)
            lam_mb = torch.cat(lam_parts, dim=0).to(_dev)
            t_mb = torch.tensor(t_parts, device=_dev, dtype=torch.long)
            group_data = _make_batch(n, N)
            if hasattr(data, "_port_Sigma"):
                group_data._port_Sigma = data._port_Sigma
                group_data._port_scenarios = data._port_scenarios
                group_data._port_risk_budgets = data._port_risk_budgets
                group_data._port_alpha = data._port_alpha
                if hasattr(data, "_port_extra"):
                    group_data._port_extra = data._port_extra
            group_data = group_data.to(_dev)
            if hasattr(data, "_port_A"):
                A_inst = data._port_A.to(_dev)
                A_batch = A_inst.unsqueeze(0).expand(n, -1, -1)
            else:
                A_batch = torch.zeros(n, N, N, device=_dev)
            micro_batches.append({
                "data": group_data,
                "x_t": x_t_mb,
                "lambda_on_policy": lam_mb,
                "t": t_mb,
                "cond_full": None,
                "edge_index": A_batch,
                "edge_weight": None,
                "num_nodes": lam_mb.shape[1],
                "batch_size": n,
            })
        return {"micro_batches": micro_batches}
    trainer._gather_minibatch = _patched_gather_minibatch

    # Monkey-patch train_step: loop over micro-batches (one per instance),
    # accumulate gradients, then do a single optimizer step.
    _orig_train_step = trainer.train_step

    def _patched_train_step() -> dict:
        if trainer.buffer.num_records == 0:
            raise RuntimeError("train_step called before any rollout populated the buffer.")

        entries = trainer.buffer.sample(trainer.cfg.minibatch_size)
        mb = trainer._gather_minibatch(entries)
        micro_batches = mb["micro_batches"]
        rho = trainer._current_rho()
        alphas_cumprod = trainer.mc_sampler.alphas_cumprod

        trainer.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_M = 0
        all_cos = []
        all_rms_ratio = []
        all_t = []

        for ub in micro_batches:
            M = ub["x_t"].shape[0]
            lam = trainer._resolve_lambda(ub, rho).detach()

            if trainer.cfg.perturb_fraction > 0:
                mask = torch.rand(M, device=trainer.device) < trainer.cfg.perturb_fraction
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    ub["x_t"][idx] += trainer.cfg.perturb_x_std * torch.randn_like(ub["x_t"][idx])
                    lam[idx] = (lam[idx] * (1.0 + trainer.cfg.perturb_lambda_std * torch.randn_like(lam[idx]))).clamp_min(0.0)
                    if args.perturb_lambda_add > 0:
                        # Additive kick on a sparse mask: reaches lambda patterns
                        # multiplicative jitter cannot (zeros stay zero under x).
                        mu_t = trainer.lambda_prior.mean_at_t(ub["t"][idx]).to(lam.dtype).view(-1, 1)
                        add_mask = (torch.rand_like(lam[idx]) < args.perturb_lambda_add_frac).float()
                        lam[idx] = lam[idx] + add_mask * args.perturb_lambda_add * mu_t * torch.randn_like(lam[idx]).abs()

            if args.lambda_dropout > 0:
                drop_mask = torch.rand(M, device=trainer.device) < args.lambda_dropout
                if drop_mask.any():
                    lam[drop_mask] = 0.0

            with torch.no_grad():
                context = trainer._build_micro_context(ub["data"])
                mc_chunk = max(1, trainer.cfg.mc_score_chunk)
                eps_parts = []
                from dataclasses import fields as _dc_fields, replace as _dc_replace
                ctx_fields = _dc_fields(context)
                _batch_ctx_fields = {"r_min", "risk_budgets"}
                for c0 in range(0, M, mc_chunk):
                    c1 = min(c0 + mc_chunk, M)
                    ctx_updates = {}
                    for f in ctx_fields:
                        if f.name not in _batch_ctx_fields:
                            continue
                        v = getattr(context, f.name)
                        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == M:
                            ctx_updates[f.name] = v[c0:c1]
                    ctx_chunk = _dc_replace(context, **ctx_updates) if ctx_updates else context
                    mc_score_c = trainer.mc_sampler._estimate_score(
                        x_t=ub["x_t"][c0:c1], t=ub["t"][c0:c1],
                        context=ctx_chunk, dual_lambda=lam[c0:c1],
                    )
                    ab_c = alphas_cumprod[ub["t"][c0:c1]].view(-1, 1, 1, 1)
                    sqrt_omb_c = torch.sqrt((1.0 - ab_c).clamp_min(1e-12))
                    eps_c = -mc_score_c * sqrt_omb_c
                    if trainer.cfg.target_clip_norm > 0:
                        flat_c = eps_c.reshape(c1 - c0, -1)
                        norms_c = flat_c.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        scale_c = (trainer.cfg.target_clip_norm / norms_c).clamp_max(1.0)
                        eps_c = (flat_c * scale_c).view_as(eps_c)
                    eps_parts.append(eps_c)
                eps_target = torch.cat(eps_parts, dim=0)

            lam_net = expand_enriched_lambda(lam, N, pcfg.get("num_sectors", 10), log1p=not args.no_log1p_lambda) if constraint_type == "enriched" else lam
            eps_pred = trainer.score_net(
                x=ub["x_t"], timesteps=ub["t"], dual_lambda=lam_net,
                edge_index=ub["edge_index"], edge_weight=ub["edge_weight"],
                cond=ub["cond_full"], return_intermediates=False,
            )

            w = trainer._loss_weight(ub["t"]).view(-1, 1, 1, 1)
            if trainer.cfg.target_normalize:
                tgt_rms = eps_target.detach().reshape(M, -1).pow(2).mean(dim=1).clamp_min(
                    float(trainer.cfg.target_normalize_eps) ** 2
                ).sqrt().view(M, 1, 1, 1)
                resid = (eps_pred - eps_target) / tgt_rms
            else:
                resid = eps_pred - eps_target
            loss = (resid ** 2 * w).sum() / trainer.cfg.minibatch_size
            loss.backward()

            total_loss += float(loss.detach().item()) * M
            total_M += M

            with torch.no_grad():
                p = eps_pred.detach().reshape(M, -1)
                tgt = eps_target.detach().reshape(M, -1)
                p_n = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
                t_n = tgt / tgt.norm(dim=1, keepdim=True).clamp_min(1e-12)
                all_cos.append((p_n * t_n).sum(dim=1))
                rms_floor = float(trainer.cfg.target_normalize_eps) if trainer.cfg.target_normalize else 1e-12
                all_rms_ratio.append(p.pow(2).mean(dim=1).sqrt() / tgt.pow(2).mean(dim=1).sqrt().clamp_min(rms_floor))
                all_t.append(ub["t"].float())

        if trainer.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                trainer.score_net.parameters(), float(trainer.cfg.grad_clip_norm))
        trainer.optimizer.step()
        trainer._iteration += 1

        cos_vals = torch.cat(all_cos)
        rms_ratio = torch.cat(all_rms_ratio)
        t_vals = torch.cat(all_t)
        return {
            "loss": total_loss / max(total_M, 1),
            "rho": rho,
            "iter": trainer._iteration,
            "cos_mean": float(cos_vals.mean()),
            "cos_median": float(cos_vals.median()),
            "pred_rms_ratio_mean": float(rms_ratio.mean()),
            "t_mean": float(t_vals.mean()),
        }

    trainer.train_step = _patched_train_step

    # Training loader: dummy PyG batches
    def _infinite_loader():
        class _L:
            def __init__(self, batch, n_batches=6):
                self._b = batch; self._n = n_batches
            def __iter__(self):
                for _ in range(self._n):
                    yield self._b
            def __len__(self):
                return self._n
        batch = _make_batch(args.batch_size, N)
        return _L(batch, n_batches=6)
    train_loader = _infinite_loader()

    # Validation dataset: fixed held-out problem instances (seeds 1000+)
    eval_data = _make_batch(args.eval_B, N).to(device)
    eval_shape = (args.eval_B, 1, N, 1)
    val_instances = []
    for vi in range(args.num_val_instances):
        _vpcfg = dict(PROBLEM_CONFIGS[args.size])
        _vpcfg["seed"] = 1000 + vi
        _vct = _vpcfg.pop("constraint_type", "shortfall")
        if constraint_type == "enriched":
            v_mu, v_Sig, v_scen, v_bud, v_alpha, v_extra = \
                make_enriched_portfolio_problem(
                    sector_gamma=args.sector_gamma, skip_stress=args.skip_stress, **_vpcfg)
        else:
            v_mu, v_Sig, v_scen, v_bud, v_alpha = make_portfolio_problem(
                constraint_type=_vct, **_vpcfg)
            v_extra = {}
        v_A_np = build_dense_adjacency(v_Sig, top_k=20)
        v_bud_scaled = v_bud * constraint_scale if constraint_type in ("variance", "variance_band", "variance_sector") else v_bud
        inst_dict = {
            "mu_np": v_mu, "Sigma_np": v_Sig, "scen_np": v_scen,
            "bud_np": v_bud, "alpha": v_alpha,
            "mu": torch.tensor(v_mu, dtype=torch.float32, device=device),
            "Sigma": torch.tensor(v_Sig, dtype=torch.float32, device=device),
            "scenarios": torch.tensor(v_scen, dtype=torch.float32, device=device),
            "budgets": torch.tensor(v_bud, dtype=torch.float32, device=device),
            "budgets_scaled": torch.tensor(v_bud_scaled, dtype=torch.float32, device=device),
            "A": torch.tensor(v_A_np, dtype=torch.float32, device=device),
            "extra": v_extra,
        }
        if constraint_type == "enriched" and "constraint_scales" in v_extra:
            inst_dict["constraint_scales"] = torch.tensor(
                v_extra["constraint_scales"], dtype=torch.float32, device=device)
            S = pcfg.get("num_sectors", 10)
            sid = torch.as_tensor(v_extra["sector_ids"], dtype=torch.long, device=device)
            _egpu = {
                "S": S,
                "sector_ids_exp": sid.unsqueeze(0).expand(args.eval_B, -1),
                "sector_upper": torch.as_tensor(v_extra["sector_upper"], dtype=torch.float32, device=device).unsqueeze(0),
                "sector_lower": torch.as_tensor(v_extra["sector_lower"], dtype=torch.float32, device=device).unsqueeze(0),
                "neff_target": float(v_extra["neff_target"]),
                "constraint_scales": inst_dict["constraint_scales"].unsqueeze(0),
            }
            if "stress_returns" in v_extra:
                _egpu["stress_returns_t"] = torch.as_tensor(v_extra["stress_returns"], dtype=torch.float32, device=device).t()
                _egpu["stress_limits"] = torch.as_tensor(v_extra["stress_limits"], dtype=torch.float32, device=device).unsqueeze(0)
                _egpu["stress_eps"] = torch.as_tensor(v_extra["stress_eps"], dtype=torch.float32, device=device).unsqueeze(0)
            inst_dict["_enriched_gpu"] = _egpu
        val_instances.append(inst_dict)
    log.info("val dataset: %d instances (seeds 1000-%d)", len(val_instances), 1000 + len(val_instances) - 1)

    # Batched val eval: one score-net forward pass per timestep for all instances
    _ref_sampler = PortfolioEnergyDDPM(
        model=_NoOpModel(),
        num_timesteps=args.T, beta_schedule=args.beta_schedule,
        portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
        portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_scaled,
        portfolio_alpha=alpha,
        energy_mc_samples=args.mc_samples_eval,
        inverse_beta=args.ib, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=args.dual_step,
        dual_lambda_init=args.lam0,
        dual_lambda_max=args.dual_lambda_max,
        dual_lambda_decay=args.dual_lambda_decay,
        shared_lambda=args.shared_lambda,
        normalize_constraints=not args.no_normalize,
        objective_scale=obj_scale,
        constraint_type=constraint_type,
        num_sectors=pcfg.get("num_sectors", 10),
        constraint_scale=constraint_scale,
        extra_constraints=_extra_constraints,
    ).to(device)
    _alphas_cumprod = _ref_sampler.alphas_cumprod.to(device)
    _post_var = _ref_sampler.posterior_variance.to(device)
    _post_coef1 = _ref_sampler.posterior_mean_coef1.to(device)
    _post_coef2 = _ref_sampler.posterior_mean_coef2.to(device)
    _ei_empty = torch.zeros((2, 0), dtype=torch.long, device=device)
    _dual_step_val = args.dual_step
    _dual_decay_val = args.dual_lambda_decay
    _dual_max_val = args.dual_lambda_max
    _normalize_val = not args.no_normalize
    _n_constraints = len(val_instances[0]["bud_np"])
    # Build batched A: [B_total, N, N] by repeating each instance's A for its B samples
    _val_A_list = []
    for vi in val_instances:
        _val_A_list.append(vi["A"].unsqueeze(0).expand(args.eval_B, -1, -1))
    _val_A_batched = torch.cat(_val_A_list, dim=0)  # [B_total, N, N]
    log.info("val batched A: %s", _val_A_batched.shape)

    def _eval_on_val(score_net_to_eval):
        score_net_to_eval.eval()
        torch.manual_seed(args.seed)
        n_inst = len(val_instances)
        B = args.eval_B
        B_total = B * n_inst
        T = args.T

        x_t = torch.randn(B_total, 1, N, 1, device=device)
        dual_lambda = torch.full((n_inst, _n_constraints), args.lam0,
                                 device=device)

        for t_int in reversed(range(T)):
            t = torch.full((B_total,), t_int, device=device, dtype=torch.long)
            dl_full = dual_lambda.unsqueeze(1).expand(n_inst, B, _n_constraints).reshape(B_total, _n_constraints)

            with torch.no_grad():
                dl_net = expand_enriched_lambda(dl_full, N, pcfg.get("num_sectors", 10), log1p=not args.no_log1p_lambda) if constraint_type == "enriched" else dl_full
                eps_pred = score_net_to_eval(
                    x=x_t, timesteps=t, dual_lambda=dl_net,
                    edge_index=_val_A_batched, edge_weight=None,
                    cond=None, return_intermediates=False,
                )

            # x0_pred + simplex projection
            ab = _alphas_cumprod[t_int]
            sqrt_ab = torch.sqrt(ab.clamp_min(1e-12))
            sqrt_omb = torch.sqrt((1.0 - ab).clamp_min(1e-12))
            x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab
            z_flat = x0_pred[:, 0, :, 0]
            w = torch.softmax(z_flat, dim=-1)
            z_proj = torch.log(w.clamp_min(1e-12))
            z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
            x0_pred = z_proj.view(B_total, 1, N, 1)

            # Per-instance dual ascent (cheap)
            for i in range(n_inst):
                lo, hi = i * B, (i + 1) * B
                w_i = torch.softmax(x0_pred[lo:hi, 0, :, 0], dim=-1)
                if constraint_type == "enriched":
                    c_var = (w_i * torch.matmul(w_i, val_instances[i]["Sigma"]))
                    ex = val_instances[i]["_enriched_gpu"]
                    B_val = w_i.shape[0]
                    exposure = torch.zeros(B_val, ex["S"], device=device, dtype=w_i.dtype)
                    exposure.scatter_add_(1, ex["sector_ids_exp"][:B_val], w_i)
                    upper_res = exposure - ex["sector_upper"]
                    lower_res = ex["sector_lower"] - exposure
                    _parts = [c_var, upper_res, lower_res]
                    if "stress_returns_t" in ex:
                        stress_port = torch.matmul(w_i, ex["stress_returns_t"])
                        excess_loss = torch.clamp(-stress_port - ex["stress_limits"], min=0.0)
                        _parts.append(excess_loss - ex["stress_eps"])
                    ent = -(w_i * torch.log(w_i.clamp_min(1e-12))).sum(dim=1, keepdim=True)
                    _parts.append(torch.exp(ent) - ex["neff_target"])
                    c_i = torch.cat(_parts, dim=1)
                    if ex.get("constraint_scales") is not None:
                        c_i = c_i * ex["constraint_scales"]
                        violation_i = c_i - (val_instances[i]["budgets"] * val_instances[i]["constraint_scales"]).unsqueeze(0)
                    else:
                        violation_i = c_i - val_instances[i]["budgets"].unsqueeze(0)
                elif constraint_type == "variance_band":
                    c_var = variance_contributions_torch(w_i, val_instances[i]["Sigma"]) * constraint_scale
                    negated_c = torch.cat([-c_var, c_var], dim=1)
                    bud_s = val_instances[i]["budgets_scaled"]
                    violation_i = (-bud_s).unsqueeze(0) - negated_c
                elif constraint_type in ("variance", "variance_sector"):
                    c_i = variance_contributions_torch(w_i, val_instances[i]["Sigma"]) * constraint_scale
                    violation_i = c_i - val_instances[i]["budgets_scaled"].unsqueeze(0)
                else:
                    c_i = shortfall_contributions_torch(
                        w_i, val_instances[i]["scenarios"], val_instances[i]["alpha"])
                    violation_i = c_i - val_instances[i]["budgets"].unsqueeze(0)
                if _normalize_val:
                    norm_bud = val_instances[i]["budgets_scaled"] if constraint_type in ("variance", "variance_band", "variance_sector") else val_instances[i]["budgets"]
                    violation_i = violation_i / norm_bud.unsqueeze(0).abs().clamp_min(1e-12)
                mean_viol = violation_i.mean(dim=0)
                step_t = _dual_step_val
                lam_decayed = (1.0 - _dual_decay_val) * dual_lambda[i]
                dual_lambda[i] = (lam_decayed + step_t * mean_viol).clamp(0.0, _dual_max_val)

            # Posterior mean
            coef1 = _post_coef1[t_int]
            coef2 = _post_coef2[t_int]
            mean = coef1 * x0_pred + coef2 * x_t
            if t_int > 0:
                var = _post_var[t_int]
                x_t = mean + torch.sqrt(var.clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        # Compute per-instance metrics (matches figures.py table)
        all_metrics = []
        for i in range(n_inst):
            lo, hi = i * B, (i + 1) * B
            w_i = torch.softmax(x_t[lo:hi, 0, :, 0], dim=-1)
            m = _eval_metrics_per_instance(
                w_i, val_instances[i]["Sigma"],
                val_instances[i]["scenarios"], val_instances[i]["budgets"],
                val_instances[i]["alpha"], constraint_type=constraint_type,
                extra=val_instances[i].get("extra"),
            )
            all_metrics.append(m)
        result = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            result[key] = float(np.mean(vals))
            result[key + "_std"] = float(np.std(vals))
        return result

    # Best-model tracking
    best_sharpe = {"sharpe": -float("inf"), "iter": -1, "state": None, "metrics": None}
    best_feas = {"feas_relaxed": -1.0, "iter": -1, "state": None, "metrics": None}
    best_pareto = {"pareto": -float("inf"), "iter": -1, "state": None, "metrics": None}
    best_vio = {"vio": float("inf"), "iter": -1, "state": None, "metrics": None}
    best_return = {"ret": -float("inf"), "iter": -1, "state": None, "metrics": None}
    best_neff = {"neff": -float("inf"), "iter": -1, "state": None, "metrics": None}

    history = []
    eval_history = []

    def _hook(info):
        history.append(info)
        if info["iter"] % 25 == 0 or info["iter"] <= 5:
            log.info("iter=%d loss=%.4f cos=%.3f pred_rms=%.3f t=%.0f rho=%.2f",
                     info["iter"], info["loss"], info.get("cos_mean", 0),
                     info.get("pred_rms_ratio_mean", 0), info.get("t_mean", 0),
                     info.get("rho", 0))

    def _eval_and_checkpoint(outer_idx, sgd_count):
        metrics = _eval_on_val(score_net)
        metrics["outer"] = outer_idx
        metrics["sgd_iter"] = sgd_count
        eval_history.append(metrics)
        with open(out_dir / "eval_history.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        log.info("[eval outer=%d sgd=%d] val(%d) ret=%.4f±%.4f sharpe=%.3f±%.3f "
                 "csat=%.3f±%.3f O2max=%.6f vio_mean=%.2e Neff=%.1f |top1|=%.0f",
                 outer_idx, sgd_count, len(val_instances),
                 metrics["ret"], metrics["ret_std"],
                 metrics["sharpe"], metrics["sharpe_std"],
                 metrics["opt2_csat"], metrics["opt2_csat_std"],
                 metrics["opt2_max_vio"], metrics["vio_mean"],
                 metrics["neff"], metrics["n_top1"])

        _sd = {k: v.detach().cpu().clone() for k, v in score_net.state_dict().items()}
        # Degenerate (collapsed) nets can win single metrics — e.g. a one-hot
        # net minimizes batch-mean max-violation — so best_* checkpoints are
        # only eligible above these floors.
        eligible = (metrics["opt2_csat"] >= args.best_min_csat
                    and metrics["neff"] >= args.best_min_neff)
        if not eligible:
            log.info("  [checkpoint] ineligible for best_* (csat=%.3f < %g or "
                     "Neff=%.1f < %g)", metrics["opt2_csat"], args.best_min_csat,
                     metrics["neff"], args.best_min_neff)
        if eligible and metrics["sharpe"] > best_sharpe["sharpe"]:
            best_sharpe.update(sharpe=metrics["sharpe"], iter=sgd_count,
                               state=_sd, metrics=metrics.copy())
            torch.save(_sd, out_dir / "score_net_best_sharpe.pt")
        if eligible and metrics["opt2_csat"] > best_feas["feas_relaxed"]:
            best_feas.update(feas_relaxed=metrics["opt2_csat"], iter=sgd_count,
                             state=_sd, metrics=metrics.copy())
            torch.save(_sd, out_dir / "score_net_best_feas.pt")
        pareto_score = metrics["opt2_csat"] + 0.1 * metrics["sharpe"]
        if eligible and pareto_score > best_pareto["pareto"]:
            best_pareto.update(pareto=pareto_score, iter=sgd_count,
                               state=_sd, metrics=metrics.copy())
            torch.save(_sd, out_dir / "score_net_best_pareto.pt")
        if eligible and metrics["vio_mean"] < best_vio["vio"]:
            best_vio.update(vio=metrics["vio_mean"], iter=sgd_count,
                            state=_sd, metrics=metrics.copy())
            torch.save(_sd, out_dir / "score_net_best_vio.pt")
        # Constrained selections: best return / best Neff subject to csat gate.
        gated = eligible and metrics["opt2_csat"] >= args.best_gate_csat
        if gated and metrics["ret"] > best_return["ret"]:
            best_return.update(ret=metrics["ret"], iter=sgd_count,
                               state=_sd, metrics=metrics.copy())
            torch.save(_sd, out_dir / "score_net_best_return.pt")
        if gated and metrics["neff"] > best_neff["neff"]:
            best_neff.update(neff=metrics["neff"], iter=sgd_count,
                             state=_sd, metrics=metrics.copy())
            torch.save(_sd, out_dir / "score_net_best_neff.pt")
        torch.save(_sd, out_dir / "score_net_last.pt")
        score_net.train()

    # Pre-generate all training instances
    log.info("pre-generating %d training instances...", args.num_instances)
    _t_gen = time.time()
    _all_train_instances = []
    for si in range(args.num_instances):
        if (si + 1) % 50 == 0 or si == 0:
            log.info("  generating instance %d/%d...", si + 1, args.num_instances)
        _tpcfg = dict(PROBLEM_CONFIGS[args.size])
        _tpcfg["seed"] = si
        _tct = _tpcfg.pop("constraint_type", "shortfall")
        if constraint_type == "enriched":
            t_mu, t_Sig, t_scen, t_bud, t_alpha, t_extra = \
                make_enriched_portfolio_problem(
                    sector_gamma=args.sector_gamma, skip_stress=args.skip_stress, **_tpcfg)
        else:
            t_mu, t_Sig, t_scen, t_bud, t_alpha = make_portfolio_problem(
                constraint_type=_tct, **_tpcfg)
            t_extra = {}
        A_np = build_dense_adjacency(t_Sig, top_k=20)
        t_bud_scaled = t_bud * constraint_scale if constraint_type in ("variance", "variance_band", "variance_sector") else t_bud
        inst = {
            "Sigma": torch.tensor(t_Sig, dtype=torch.float32, device=device),
            "scenarios": torch.tensor(t_scen, dtype=torch.float32, device=device),
            "budgets": torch.tensor(t_bud, dtype=torch.float32, device=device),
            "budgets_scaled": torch.tensor(t_bud_scaled, dtype=torch.float32, device=device),
            "alpha": t_alpha,
            "A": torch.tensor(A_np, dtype=torch.float32, device=device),
            "extra": t_extra,
        }
        if constraint_type == "enriched" and "constraint_scales" in t_extra:
            inst["constraint_scales"] = torch.tensor(
                t_extra["constraint_scales"], dtype=torch.float32, device=device)
            S = pcfg.get("num_sectors", 10)
            sid = torch.as_tensor(t_extra["sector_ids"], dtype=torch.long, device=device)
            _tegpu = {
                "S": S,
                "sector_ids_exp": sid.unsqueeze(0).expand(args.batch_size, -1),
                "sector_upper": torch.as_tensor(t_extra["sector_upper"], dtype=torch.float32, device=device).unsqueeze(0),
                "sector_lower": torch.as_tensor(t_extra["sector_lower"], dtype=torch.float32, device=device).unsqueeze(0),
                "neff_target": float(t_extra["neff_target"]),
                "constraint_scales": inst["constraint_scales"].unsqueeze(0),
            }
            if "stress_returns" in t_extra:
                _tegpu["stress_returns_t"] = torch.as_tensor(t_extra["stress_returns"], dtype=torch.float32, device=device).t()
                _tegpu["stress_limits"] = torch.as_tensor(t_extra["stress_limits"], dtype=torch.float32, device=device).unsqueeze(0)
                _tegpu["stress_eps"] = torch.as_tensor(t_extra["stress_eps"], dtype=torch.float32, device=device).unsqueeze(0)
            inst["_enriched_gpu"] = _tegpu
        _all_train_instances.append(inst)
    log.info("pre-generated %d training instances (with correlation graphs)", len(_all_train_instances))

    _instances_per_rollout = args.num_rollouts_per_outer  # batch this many instances per rollout

    def _batched_rollout(outer_idx):
        """Run one batched rollout: N_inst instances in one score-net forward pass."""
        n_inst = _instances_per_rollout
        B = args.batch_size
        B_total = B * n_inst
        T = args.T

        # Pick instances round-robin
        base = (outer_idx * n_inst) % len(_all_train_instances)
        inst_indices = [(base + i) % len(_all_train_instances) for i in range(n_inst)]
        instances = [_all_train_instances[j] for j in inst_indices]
        rollout_A = torch.cat(
            [inst["A"].unsqueeze(0).expand(B, -1, -1) for inst in instances], dim=0)

        x_t = torch.randn(B_total, 1, N, 1, device=device)
        dual_lambda = torch.full((n_inst, _n_constraints), args.lam0, device=device)

        x_by_t = []
        lambda_by_t = []
        t_values = []

        for t_int in reversed(range(T)):
            t = torch.full((B_total,), t_int, device=device, dtype=torch.long)
            dl_full = dual_lambda.unsqueeze(1).expand(n_inst, B, _n_constraints).reshape(B_total, _n_constraints)

            # Record BEFORE update
            x_by_t.append(x_t.detach().cpu())
            lambda_by_t.append(dl_full.detach().cpu())
            t_values.append(t_int)

            with torch.no_grad():
                dl_net = expand_enriched_lambda(dl_full, N, pcfg.get("num_sectors", 10), log1p=not args.no_log1p_lambda) if constraint_type == "enriched" else dl_full
                eps_pred = score_net(
                    x=x_t, timesteps=t, dual_lambda=dl_net,
                    edge_index=rollout_A, edge_weight=None,
                    cond=None, return_intermediates=False,
                )

            # x0_pred + simplex projection
            ab = _alphas_cumprod[t_int]
            sqrt_ab = torch.sqrt(ab.clamp_min(1e-12))
            sqrt_omb = torch.sqrt((1.0 - ab).clamp_min(1e-12))
            x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab
            z_flat = x0_pred[:, 0, :, 0]
            w = torch.softmax(z_flat, dim=-1)
            z_proj = torch.log(w.clamp_min(1e-12))
            z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
            x0_pred = z_proj.view(B_total, 1, N, 1)

            # Per-instance dual ascent
            for i in range(n_inst):
                lo, hi = i * B, (i + 1) * B
                w_i = torch.softmax(x0_pred[lo:hi, 0, :, 0], dim=-1)
                if constraint_type == "enriched":
                    c_var = (w_i * torch.matmul(w_i, instances[i]["Sigma"]))
                    ex = instances[i]["_enriched_gpu"]
                    exposure = torch.zeros(B, ex["S"], device=device, dtype=w_i.dtype)
                    exposure.scatter_add_(1, ex["sector_ids_exp"][:B], w_i)
                    upper_res = exposure - ex["sector_upper"]
                    lower_res = ex["sector_lower"] - exposure
                    _parts_r = [c_var, upper_res, lower_res]
                    if "stress_returns_t" in ex:
                        stress_port = torch.matmul(w_i, ex["stress_returns_t"])
                        excess_loss = torch.clamp(-stress_port - ex["stress_limits"], min=0.0)
                        _parts_r.append(excess_loss - ex["stress_eps"])
                    ent = -(w_i * torch.log(w_i.clamp_min(1e-12))).sum(dim=1, keepdim=True)
                    _parts_r.append(torch.exp(ent) - ex["neff_target"])
                    c_i = torch.cat(_parts_r, dim=1)
                    if ex.get("constraint_scales") is not None:
                        c_i = c_i * ex["constraint_scales"]
                        violation_i = c_i - (instances[i]["budgets"] * instances[i]["constraint_scales"]).unsqueeze(0)
                    else:
                        violation_i = c_i - instances[i]["budgets"].unsqueeze(0)
                elif constraint_type == "variance_band":
                    c_var = variance_contributions_torch(w_i, instances[i]["Sigma"]) * constraint_scale
                    negated_c = torch.cat([-c_var, c_var], dim=1)
                    bud_s = instances[i]["budgets_scaled"]
                    violation_i = (-bud_s).unsqueeze(0) - negated_c
                elif constraint_type in ("variance", "variance_sector"):
                    c_i = variance_contributions_torch(w_i, instances[i]["Sigma"]) * constraint_scale
                    violation_i = c_i - instances[i]["budgets_scaled"].unsqueeze(0)
                else:
                    c_i = shortfall_contributions_torch(
                        w_i, instances[i]["scenarios"], instances[i]["alpha"])
                    violation_i = c_i - instances[i]["budgets"].unsqueeze(0)
                if _normalize_val:
                    norm_bud = instances[i]["budgets_scaled"] if constraint_type in ("variance", "variance_band", "variance_sector") else instances[i]["budgets"]
                    violation_i = violation_i / norm_bud.unsqueeze(0).abs().clamp_min(1e-12)
                mean_viol = violation_i.mean(dim=0)
                lam_decayed = (1.0 - _dual_decay_val) * dual_lambda[i]
                dual_lambda[i] = (lam_decayed + _dual_step_val * mean_viol).clamp(0.0, _dual_max_val)

            # Posterior mean
            coef1 = _post_coef1[t_int]
            coef2 = _post_coef2[t_int]
            mean = coef1 * x0_pred + coef2 * x_t
            if t_int > 0:
                var = _post_var[t_int]
                x_t = mean + torch.sqrt(var.clamp_min(0.0)) * torch.randn_like(x_t)
            else:
                x_t = mean

        # Push per-instance trajectories to buffer (attach instance data to PyG batch)
        x_stack = torch.stack(x_by_t, dim=0)      # [T, B_total, 1, N, 1]
        l_stack = torch.stack(lambda_by_t, dim=0)  # [T, B_total, C]

        _lflat = l_stack.reshape(-1)
        _lidx = torch.randperm(len(_lflat))[:min(100000, len(_lflat))]
        _lsample = _lflat[_lidx]
        log.info("  lambda stats (all t): max=%.2f p96=%.2f mean=%.2f",
                 float(_lflat.max()), float(_lsample.quantile(0.96)),
                 float(_lflat.mean()))
        t_tensor = torch.tensor(t_values, dtype=torch.long)
        for i in range(n_inst):
            lo, hi = i * B, (i + 1) * B
            inst_data = _make_batch(B, N)
            inst_data._port_Sigma = instances[i]["Sigma"].cpu()
            inst_data._port_scenarios = instances[i]["scenarios"].cpu()
            inst_data._port_risk_budgets = instances[i]["budgets"].cpu()
            inst_data._port_alpha = torch.tensor(float(instances[i]["alpha"]))
            inst_data._port_A = instances[i]["A"].cpu()
            inst_data._port_extra = instances[i].get("extra", {})
            trainer.buffer.push_trajectory(
                data=inst_data,
                x_by_t=x_stack[:, lo:hi],
                lambda_by_t=l_stack[:, lo:hi],
                t_values=t_tensor,
            )

    # Cosine LR scheduler
    if args.lr_min > 0:
        import math as _math
        total_sgd = args.num_outer * args.inner_steps_max  # upper bound
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=total_sgd, eta_min=args.lr_min)
    else:
        scheduler = None

    # Custom training loop (replaces trainer.fit)
    log.info("training %d outer, inner steps %d->%d (ramp at iter %d)",
             args.num_outer, args.inner_steps, args.inner_steps_max,
             args.inner_steps_ramp_iter)
    log.info("batched rollout: %d instances per forward pass", _instances_per_rollout)
    if scheduler:
        log.info("LR cosine decay: %.2e -> %.2e", args.lr, args.lr_min)
    t0 = time.time()
    score_net.train()
    _sgd_count = 0
    log.info("starting training loop...")
    for outer in range(args.num_outer):
        t_rollout = time.time()
        _batched_rollout(outer)
        log.info("[outer %d/%d] rollout done (%.1fs), buffer=%d records",
                 outer + 1, args.num_outer, time.time() - t_rollout,
                 trainer.buffer.num_records)
        # Ramp inner steps: linear from inner_steps to inner_steps_max
        frac = min(_sgd_count / max(args.inner_steps_ramp_iter, 1), 1.0)
        n_inner = int(args.inner_steps + frac * (args.inner_steps_max - args.inner_steps))
        for _ in range(n_inner):
            step_info = trainer.train_step()
            _hook(step_info)
            if scheduler:
                scheduler.step()
            _sgd_count += 1
        if (outer + 1) % args.eval_every == 0 and (outer + 1) >= args.eval_start:
            _eval_and_checkpoint(outer, _sgd_count)
    train_wall = time.time() - t0
    log.info("training done in %.1f s (%.1f min)", train_wall, train_wall / 60)

    # Save final + best checkpoints
    torch.save(score_net.state_dict(), out_dir / "score_net_last.pt")
    if best_sharpe["state"] is not None:
        torch.save(best_sharpe["state"], out_dir / "score_net_best_sharpe.pt")
    if best_feas["state"] is not None:
        torch.save(best_feas["state"], out_dir / "score_net_best_feas.pt")
    if best_pareto["state"] is not None:
        torch.save(best_pareto["state"], out_dir / "score_net_best_pareto.pt")

    summary = {
        "config": vars(args),
        "training_wall_sec": train_wall,
        "num_history": len(history),
        "best_sharpe": {"iter": best_sharpe["iter"], "metrics": best_sharpe["metrics"]},
        "best_feas": {"iter": best_feas["iter"], "metrics": best_feas["metrics"]},
        "best_pareto": {"iter": best_pareto["iter"], "metrics": best_pareto["metrics"]},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Save history
    np.savez(out_dir / "history.npz",
             iter=np.array([h["iter"] for h in history]),
             loss=np.array([h["loss"] for h in history]),
             cos_mean=np.array([h.get("cos_mean", 0) for h in history]),
             pred_rms=np.array([h.get("pred_rms_ratio_mean", 0) for h in history]),
             t_mean=np.array([h.get("t_mean", 0) for h in history]),
             rho=np.array([h.get("rho", 0) for h in history]))
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f)
    with open(out_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f)

    log.info("=" * 70)
    log.info("best_sharpe  (iter %d): sharpe=%.3f csat=%.3f ret=%.4f",
             best_sharpe["iter"],
             best_sharpe["metrics"]["sharpe"] if best_sharpe["metrics"] else 0,
             best_sharpe["metrics"]["opt2_csat"] if best_sharpe["metrics"] else 0,
             best_sharpe["metrics"]["ret"] if best_sharpe["metrics"] else 0)
    log.info("best_feas    (iter %d): csat=%.3f sharpe=%.3f ret=%.4f",
             best_feas["iter"],
             best_feas["metrics"]["opt2_csat"] if best_feas["metrics"] else 0,
             best_feas["metrics"]["sharpe"] if best_feas["metrics"] else 0,
             best_feas["metrics"]["ret"] if best_feas["metrics"] else 0)
    log.info("best_pareto  (iter %d): score=%.3f csat=%.3f sharpe=%.3f",
             best_pareto["iter"],
             best_pareto["pareto"] if best_pareto["pareto"] > -1 else 0,
             best_pareto["metrics"]["opt2_csat"] if best_pareto["metrics"] else 0,
             best_pareto["metrics"]["sharpe"] if best_pareto["metrics"] else 0)
    log.info("output: %s", out_dir)


if __name__ == "__main__":
    main()
