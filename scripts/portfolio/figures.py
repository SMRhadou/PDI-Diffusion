#!/usr/bin/env python3
"""Generate paper figures from a saved eval run.

Takes a rerun_eval output directory (with results.json and optionally saved
weights.npz) and produces:
    - fig_comparison_bars.{png,pdf}   -- headline feas / return / var bars
    - fig_risk_cdf.{png,pdf}          -- per-asset risk-contribution CDF
    - fig_return_vs_violation.{png,pdf}
    - fig_weight_heatmap.{png,pdf}
    - fig_dual_evolution.{png,pdf}    -- lambda trajectory (from calibrate_lambda)
    - fig_feasibility_violation_scatter.{png,pdf}

Usage:
    python scripts/portfolio/figures.py \\
        --eval-dir <outputs/portfolio/score_net_eval/...> \\
        --calib-dir <outputs/portfolio/diagnostics/lambda_calib_...>
"""

from __future__ import annotations

import argparse
import json
import math
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
    shortfall_contributions_np, shortfall_contributions_torch,
    variance_contributions_np, variance_contributions_torch,
)
from pdi.models.portfolio_backbone import (  # noqa: E402
    PortfolioScoreBackbone,
)
from pdi.models.portfolio_gnn_backbone import (  # noqa: E402
    PortfolioGNNBackbone, build_dense_adjacency,
)
from pdi.trainers.energy_score import (  # noqa: E402
    ScoreNetWithLambda,
)
import baselines as bl  # noqa: E402


PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=1000, gamma=2.0, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5, seed=0),
    "large": dict(N=500, K_factors=50, R_scenarios=1000, gamma=2.0, seed=0),
    "large_g1.25": dict(N=500, K_factors=50, R_scenarios=1000, gamma=1.25, seed=0),
    "large_sectors": dict(N=500, K_factors=50, R_scenarios=1000, gamma=1.25, seed=0,
                          structure="sectors", num_sectors=10),
    "large_sectors_fat": dict(N=500, K_factors=50, R_scenarios=1000, gamma=1.25, seed=0,
                              structure="sectors_fat", num_sectors=10, df=5.0),
    "large_sectfat_g10": dict(N=500, K_factors=50, R_scenarios=1000, gamma=10.0, seed=0,
                              structure="sectors_fat", num_sectors=10,
                              constraint_type="shortfall"),
    "crypto": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                   structure="sectors", num_sectors=10,
                   constraint_type="variance", budget_type="uniform"),
    "large_varsector_g1.5": dict(N=500, K_factors=50, R_scenarios=1000, gamma=1.5, seed=0,
                                 structure="sectors", num_sectors=10,
                                 constraint_type="variance_sector"),
    "sectors_shortfall_g3": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                                 structure="sectors", num_sectors=10,
                                 constraint_type="shortfall"),
    "crypto_dual": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                        structure="sectors", num_sectors=10,
                        constraint_type="dual", budget_type="uniform"),
    "crypto_band": dict(N=500, K_factors=50, R_scenarios=1000, gamma=3.0, seed=0,
                        structure="sectors", num_sectors=10,
                        constraint_type="variance_band"),
}


# Methods to include in the paper (ordered to match synthetic convention)
# "net" variants use the trained CED score network with lambda=0.
# "mc" variants use the MC IS score with lambda=0.
PAPER_METHODS = [
    "pd_langevin",
    "mc_teacher",
    "ced_trained",
    "pdm_net",
    "pdm_mc",
    "dps_net",
    "dps_mc",
    "unconstrained_net",
    "unconstrained_mc",
    "equal_weight",
    "markowitz_unconstrained",
    "markowitz_constrained",
]

METHOD_LABELS = {
    "equal_weight": "Equal wt",
    "risk_parity": "Risk par",
    "markowitz_unconstrained": "Markowitz\n(uncon)",
    "markowitz_constrained": "Markowitz\n(con)",
    "unconstrained_mc": "Uncon (MC)",
    "unconstrained_net": "Uncon (net)",
    "mc_teacher": "PDI-MC",
    "pdm_mc": "PDM (MC)",
    "pdm_net": "PDM (net)",
    "dps_mc": "DPS (MC)",
    "dps_net": "DPS (net)",
    "rejection": "DDPM\n+reject",
    "pd_langevin": "PDL",
    "ced_trained": "PDI-Net",
    "ced_ceiling_fixlam": r"PDI-MC ($\lambda^*$)",
    "ced_trained_fixlam": r"PDI-Net ($\lambda^*$)",
    "pdm_net_lam1": r"PDM ($\lambda\!=\!1$)",
    "pdm_net_lam300": r"PDM ($\lambda\!=\!300$)",
    "mc_fix_lamfinal1": r"MC fix $\lambda^*_1$",
    "mc_warm_lamfinal1": r"MC warm $\lambda^*_1$+PD",
    "mc_fix_lamfinal3": r"MC fix $\lambda^*_3$",
    "mc_fix_lamavg1": r"MC fix $\bar\lambda_1$",
    "mc_warm_lamavg1": r"MC warm $\bar\lambda_1$+PD",
}


# ---------------------------------------------------------------
# Sample collection: re-run each method to get the weights
# ---------------------------------------------------------------

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


def _install_lambda_trace(sampler):
    """Monkey-patch sampler to record shared-λ at each reverse step."""
    sampler._lambda_trace = []
    _orig_dual = sampler._dual_ascent_step

    def _traced_dual(dual_lambda, ergodic_rates, context, t=None):
        updated = _orig_dual(dual_lambda, ergodic_rates, context, t=t)
        sampler._lambda_trace.append(updated.detach().cpu().mean(dim=0).clone())
        return updated

    sampler._dual_ascent_step = _traced_dual
    return sampler


def _make_trained_score_estimator(score_net, alphas_cumprod, dense_A=None):
    _empty_ei = torch.zeros((2, 0), dtype=torch.long)

    def _fn(self, x_t, t, context, dual_lambda, inverse_beta_override=None):
        del context, inverse_beta_override
        B = x_t.shape[0]
        if dense_A is not None:
            ei = dense_A.to(x_t.device).unsqueeze(0).expand(B, -1, -1)
        else:
            ei = _empty_ei.to(x_t.device)
        with torch.no_grad():
            eps_pred = score_net(
                x=x_t, timesteps=t, dual_lambda=dual_lambda,
                edge_index=ei, edge_weight=None,
                cond=None, return_intermediates=False,
            )
        ab = alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))
    return _fn


def _pdm_sample(sampler, shape, device, data, Sigma, budgets, constraint_type,
                extra=None, pdm_proj_lr=0.1):
    """PDM: run reverse process, project x_t onto feasible set after each step."""
    B, _, N, _ = shape
    x_t = torch.randn(shape, device=device, dtype=torch.float32)
    context = sampler._build_energy_context(
        data=data, batch_size=B, num_nodes=N, device=device, dtype=x_t.dtype)
    dual_lambda = sampler._init_dual_lambda(
        batch_size=B, num_nodes=N, device=device, dtype=x_t.dtype,
        init_value=sampler.dual_lambda_init)

    for t_int in reversed(range(sampler.num_timesteps)):
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
        with torch.no_grad():
            score = sampler._estimate_score(
                x_t=x_t, t=t_tensor, context=context, dual_lambda=dual_lambda)
            x0_pred, eps_pred = sampler._score_to_x0_eps(x_t=x_t, t=t_tensor, score=score)
            mean = sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t_tensor)
            if t_int > 0:
                var = sampler._gather(sampler.posterior_variance, t_tensor).view(-1, 1, 1, 1)
                x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
            else:
                x_t = mean
        with torch.enable_grad():
            w = sampler.z_to_portfolio_weights(x_t.detach())
            _proj_iters = 100 if constraint_type == "enriched" else 1
            w_proj = bl.project_to_feasible(w, Sigma, None, 0.0, budgets,
                                             constraint_type=constraint_type,
                                             num_iters=_proj_iters, lr=pdm_proj_lr,
                                             extra=extra)
            z_proj = torch.log(w_proj.clamp_min(1e-12))
            z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
            x_t = z_proj.view(B, 1, N, 1)
    return x_t


def collect_all_weights(size: str, ib: float, dual_step: float, T: int,
                         B: int, K: int, seed: int, device,
                         ced_ckpt: Path | None = None,
                         hidden: int = 256, num_layers: int = 4,
                         problem_seed: int | None = None,
                         include_mc_variants: bool = False,
                         mc_only: bool = False,
                         beta_schedule: str = "cosine",
                         pdl_primal_lr: float = 0.01,
                         pdl_dual_lr: float = 100.0,
                         pdl_noise_scale: float = 0.01,
                         lam0: float = 0.0,
                         sub_batch: int = 0,
                         dual_lambda_decay: float = 0.0,
                         normalize_constraints: bool = True,
                         backbone: str = "mlp",
                         tagconv_K: int = 2,
                         mc_lambda_study: bool = False,
                         dps_scale: float = 1.0,
                         dps_sweep: str = None,
                         pdm_proj_lr: float = 0.1) -> dict:
    """Run all methods and return a dict of {method_name: weights_tensor [B, N]}.

    Args:
        problem_seed: override the problem-generation seed (for multi-instance sweeps).
    """
    pcfg = dict(PROBLEM_CONFIGS[size])
    if problem_seed is not None:
        pcfg["seed"] = int(problem_seed)
    N = pcfg["N"]
    constraint_type = pcfg.pop("constraint_type", "shortfall")
    _extra_constraints = {}
    if constraint_type == "enriched":
        sector_gamma = pcfg.pop("sector_gamma", 1.5)
        mu_np, Sigma_np, scen_np, bud_np, alpha_np, _extra_constraints = \
            make_enriched_portfolio_problem(
                N=pcfg["N"], K_factors=pcfg.get("K_factors", 50),
                R_scenarios=pcfg.get("R_scenarios", 1000),
                gamma=pcfg.get("gamma", 3.0), sector_gamma=sector_gamma,
                seed=pcfg.get("seed", 0), structure=pcfg.get("structure", "sectors"),
                num_sectors=pcfg.get("num_sectors", 10),
                budget_type=pcfg.get("budget_type", "uniform"),
            )
        ret_mean = abs(scen_np.mean(axis=0)).mean()
        _obj_scale = 1.0 / max(ret_mean, 1e-12)
        _constraint_scale = 1.0
        bud_scaled = bud_np
    else:
        mu_np, Sigma_np, scen_np, bud_np, alpha_np = make_portfolio_problem(
            constraint_type=constraint_type, **pcfg)
        if constraint_type in ("variance", "variance_band", "variance_sector"):
            c_eq_mean = variance_contributions_np(np.ones(N) / N, Sigma_np).mean()
            ret_mean = abs(scen_np.mean(axis=0)).mean()
            _obj_scale = 1.0 / max(ret_mean, 1e-12)
            _constraint_scale = 1.0 / max(c_eq_mean, 1e-12)
            bud_scaled = bud_np * _constraint_scale
        else:
            _obj_scale = 1.0
            _constraint_scale = 1.0
            bud_scaled = bud_np

    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float32, device=device)
    scenarios = torch.tensor(scen_np, dtype=torch.float32, device=device)
    budgets = torch.tensor(bud_np, dtype=torch.float32, device=device)

    def _mk_sampler(ib_, ds_, lam0_override=None):
        return PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=T, beta_schedule=beta_schedule,
            portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
            portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_scaled,
            portfolio_alpha=alpha_np,
            energy_mc_samples=K,
            inverse_beta=ib_, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=ds_,
            dual_lambda_init=lam0 if lam0_override is None else lam0_override,
            dual_lambda_max=1e6,
            dual_lambda_decay=dual_lambda_decay,
            shared_lambda=True,
            normalize_constraints=normalize_constraints,
            objective_scale=_obj_scale,
            constraint_type=constraint_type,
            num_sectors=pcfg.get("num_sectors", 10),
            constraint_scale=_constraint_scale,
            extra_constraints=_extra_constraints,
        ).to(device)

    shape = (B, 1, N, 1)
    data = _make_batch(B, N).to(device)
    weights = {}

    # Analytical
    weights["equal_weight"] = (torch.ones(B, N, device=device) / N).detach()
    weights["risk_parity"] = bl.risk_parity(Sigma, B).detach()
    mu_u = bl.markowitz_unconstrained(mu_np, Sigma_np, risk_aversion=5.0)
    weights["markowitz_unconstrained"] = torch.tensor(mu_u, dtype=torch.float32,
                                                        device=device).unsqueeze(0).expand(B, -1).contiguous()
    if constraint_type == "enriched":
        bud_for_mkz = bud_np[:N]
        mu_c = bl.markowitz_constrained(mu_np, Sigma_np, bud_for_mkz, risk_aversion=5.0,
                                         constraint_type="variance")
    else:
        bud_for_mkz = bud_np[:N] if constraint_type == "dual" else bud_np
        mu_c = bl.markowitz_constrained(mu_np, Sigma_np, bud_for_mkz, risk_aversion=5.0,
                                         constraint_type=constraint_type)
    weights["markowitz_constrained"] = torch.tensor(mu_c, dtype=torch.float32,
                                                      device=device).unsqueeze(0).expand(B, -1).contiguous()

    # MC teacher (always run — primary MC sampler)
    lambda_traces = {}
    if sub_batch > 0 and B >= sub_batch:
        n_sub = B // sub_batch
        torch.manual_seed(seed)
        s = _mk_sampler(ib, dual_step)
        s._lambda_trace_subs = []
        _lam_max = s.dual_lambda_max
        _norm_c = s.normalize_constraints

        def _sub_batch_dual(dual_lambda, ergodic_rates, context, t=None,
                            _nsub=n_sub, _sb=sub_batch, _sampler=s,
                            _lmax=_lam_max, _nc=_norm_c):
            if t is None:
                step = _sampler.dual_step_size_by_t[0].to(
                    device=dual_lambda.device, dtype=dual_lambda.dtype)
            else:
                t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(
                    0, _sampler.num_timesteps - 1)
                step = _sampler._gather(
                    _sampler.dual_step_size_by_t.to(device=dual_lambda.device), t_safe
                ).to(dtype=dual_lambda.dtype)[0]
            violation = context.r_min - ergodic_rates  # [B, N]
            if _nc:
                violation = violation / context.risk_budgets.unsqueeze(0).clamp_min(1e-12)
            _decay = _sampler.dual_lambda_decay
            parts = []
            for i in range(_nsub):
                lo, hi = i * _sb, (i + 1) * _sb
                mean_viol = violation[lo:hi].mean(dim=0, keepdim=True)  # [1, N]
                lam_decayed = (1.0 - _decay) * dual_lambda[lo:lo+1]
                lam_i = (lam_decayed + step * mean_viol).clamp_min(0.0)
                if _lmax is not None:
                    lam_i = lam_i.clamp_max(float(_lmax))
                parts.append(lam_i.expand(_sb, -1))
            result = torch.cat(parts, dim=0)
            _sampler._lambda_trace_subs.append(
                torch.stack([result[i*_sb].detach().cpu() for i in range(_nsub)])
            )
            return result

        s._dual_ascent_step = _sub_batch_dual
        z = s.sample(shape=shape, device=device, data=data)
        weights["mc_teacher"] = s.z_to_portfolio_weights(z).detach()
        # Build per-sub-batch traces [T, N] for each sub-batch
        traces_tensor = torch.stack(s._lambda_trace_subs)  # [T, n_sub, N]
        for sb_i in range(n_sub):
            lambda_traces[f"mc_teacher_sub{sb_i}"] = traces_tensor[:, sb_i, :]
    else:
        torch.manual_seed(seed)
        s = _install_lambda_trace(_mk_sampler(ib, dual_step))
        z = s.sample(shape=shape, device=device, data=data)
        weights["mc_teacher"] = s.z_to_portfolio_weights(z).detach()
        lambda_traces["mc_teacher"] = torch.stack(s._lambda_trace)

    # --- MC ceiling lambda study (opt-in) ---
    mc_lam_trace = lambda_traces.get("mc_teacher")
    if mc_lambda_study and mc_lam_trace is not None:
        lam_final_1 = mc_lam_trace[-1].to(device)   # [N]
        lam_avg_1 = mc_lam_trace.mean(dim=0).to(device)  # [N]

        def _noop_dual(dual_lambda, ergodic_rates, context, t=None):
            return dual_lambda

        def _set_fixed_lam(sampler, lam_vec):
            sampler._dual_ascent_step = _noop_dual
            _lv = lam_vec.clone()
            def _fi(batch_size, num_nodes, device, dtype, init_value, _l=_lv):
                return _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
            sampler._init_dual_lambda = _fi

        def _set_init_lam(sampler, lam_vec):
            _lv = lam_vec.clone()
            def _fi(batch_size, num_nodes, device, dtype, init_value, _l=_lv):
                return _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
            sampler._init_dual_lambda = _fi

        # 2. Fixed lam_final from #1
        torch.manual_seed(seed)
        s2 = _mk_sampler(ib, dual_step)
        _set_fixed_lam(s2, lam_final_1)
        z2 = s2.sample(shape=shape, device=device, data=data)
        weights["mc_fix_lamfinal1"] = s2.z_to_portfolio_weights(z2).detach()

        # 3. Warm-start lam_final#1 + dual ascent
        torch.manual_seed(seed)
        s3 = _install_lambda_trace(_mk_sampler(ib, dual_step))
        _set_init_lam(s3, lam_final_1)
        z3 = s3.sample(shape=shape, device=device, data=data)
        weights["mc_warm_lamfinal1"] = s3.z_to_portfolio_weights(z3).detach()
        lam_final_3 = s3._lambda_trace[-1].to(device)
        lambda_traces["mc_warm_lamfinal1"] = torch.stack(s3._lambda_trace)

        # 4. Fixed lam_final from #3
        torch.manual_seed(seed)
        s4 = _mk_sampler(ib, dual_step)
        _set_fixed_lam(s4, lam_final_3)
        z4 = s4.sample(shape=shape, device=device, data=data)
        weights["mc_fix_lamfinal3"] = s4.z_to_portfolio_weights(z4).detach()

        # 5. Fixed time-avg lambda from #1
        torch.manual_seed(seed)
        s5 = _mk_sampler(ib, dual_step)
        _set_fixed_lam(s5, lam_avg_1)
        z5 = s5.sample(shape=shape, device=device, data=data)
        weights["mc_fix_lamavg1"] = s5.z_to_portfolio_weights(z5).detach()

        # 6. Warm-start lam_avg#1 + dual ascent
        torch.manual_seed(seed)
        s6 = _install_lambda_trace(_mk_sampler(ib, dual_step))
        _set_init_lam(s6, lam_avg_1)
        z6 = s6.sample(shape=shape, device=device, data=data)
        weights["mc_warm_lamavg1"] = s6.z_to_portfolio_weights(z6).detach()
        lambda_traces["mc_warm_lamavg1"] = torch.stack(s6._lambda_trace)

        print("[mc_lam_study] lam_final#1: mean=%.2f med=%.2f max=%.1f" % (
            lam_final_1.mean(), lam_final_1.median(), lam_final_1.max()))
        print("[mc_lam_study] lam_avg#1:   mean=%.2f med=%.2f max=%.1f" % (
            lam_avg_1.mean(), lam_avg_1.median(), lam_avg_1.max()))
        print("[mc_lam_study] lam_final#3: mean=%.2f med=%.2f max=%.1f" % (
            lam_final_3.mean(), lam_final_3.median(), lam_final_3.max()))

    if include_mc_variants:
        import types as _types
        B_var = B
        shape_var = (B_var, 1, N, 1)
        data_var = _make_batch(B_var, N).to(device)

        # --- MC-score variants (IS score, no trained net) ---

        # Unconstrained MC (lambda~0)
        torch.manual_seed(seed)
        s = _mk_sampler(ib, 1e-12, lam0_override=0.0)
        z = s.sample(shape=shape_var, device=device, data=data_var)
        weights["unconstrained_mc"] = s.z_to_portfolio_weights(z).detach()

        # PDM (MC) = unconstrained MC + per-step x_t projection
        if constraint_type != "dual":
            torch.manual_seed(seed)
            s_pdm = _mk_sampler(ib, 1e-12, lam0_override=0.0)
            _pdm_bud = torch.tensor(bud_np, dtype=torch.float32, device=device) if constraint_type == "enriched" else budgets
            z_pdm = _pdm_sample(s_pdm, shape_var, device, data_var,
                                 Sigma, _pdm_bud, constraint_type,
                                 extra=_extra_constraints if constraint_type == "enriched" else None,
                                 pdm_proj_lr=pdm_proj_lr)
            weights["pdm_mc"] = s_pdm.z_to_portfolio_weights(z_pdm).detach()

        # DPS (MC) = unconstrained MC + gradient guidance through x_t
        torch.manual_seed(seed)
        s_dps = _mk_sampler(ib, 1e-12, lam0_override=0.0)
        _N = N
        _Sig = Sigma
        _scen = scenarios
        if constraint_type == "enriched":
            _dps_scales = torch.tensor(_extra_constraints["constraint_scales"], dtype=torch.float32, device=device)
            _bud = torch.tensor(bud_np * _extra_constraints["constraint_scales"], dtype=torch.float32, device=device)
        else:
            _dps_scales = None
            _bud = torch.tensor(bud_scaled, dtype=torch.float32, device=device)
        _alpha_val = float(alpha_np)
        _cs_dps = _constraint_scale
        _ct = constraint_type
        _extra_dps = _extra_constraints
        _dps_alphas_cumprod = s_dps.alphas_cumprod
        def _dps_correct(self, x_t, t, x0_pred, eps_pred,
                         _s=dps_scale, _sc=_scen, _b=_bud, _a=_alpha_val,
                         _n=_N, _sig=_Sig, _ctype=_ct, _ac=_dps_alphas_cumprod,
                         _cscale=_cs_dps, _ext=_extra_dps, _dscales=_dps_scales):
            B_loc = x_t.shape[0]
            x_t_g = x_t.detach().requires_grad_(True)
            alpha_bar = _ac[t].view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
            sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
            x0_tw = (x_t_g - sqrt_1mab * eps_pred.detach()) / sqrt_ab
            w_g = torch.softmax(x0_tw[:, 0, :, 0], dim=-1)
            c_g = bl._compute_constraints(w_g, _sig, _sc, _a, _ctype, extra=_ext)
            if _dscales is not None:
                c_g = c_g * _dscales.unsqueeze(0)
            elif _cscale != 1.0:
                c_g = c_g * _cscale
            loss = ((c_g - _b.unsqueeze(0)).clamp_min(0.0) ** 2).sum()
            grad_xt = torch.autograd.grad(loss, x_t_g)[0]
            x_t_corrected = (x_t - _s * grad_xt).detach()
            x0_raw = (x_t_corrected - sqrt_1mab * eps_pred) / sqrt_ab
            w_proj = torch.softmax(x0_raw[:, 0, :, 0], dim=-1)
            z_proj = torch.log(w_proj.clamp_min(1e-12))
            z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
            x0_new = z_proj.view_as(x0_raw)
            eps_new = (x_t - sqrt_ab * x0_new) / sqrt_1mab
            return x0_new, eps_new
        orig_score_to_x0_dps = s_dps._score_to_x0_eps
        def _dps_guide(self, x_t, t, score,
                       _orig=orig_score_to_x0_dps):
            x0_pred, eps_pred = _orig(x_t, t, score)
            return _dps_correct(self, x_t, t, x0_pred, eps_pred)
        s_dps._score_to_x0_eps = _types.MethodType(_dps_guide, s_dps)
        z_dps = s_dps.sample(shape=shape_var, device=device, data=data_var)
        weights["dps_mc"] = s_dps.z_to_portfolio_weights(z_dps).detach()

        # --- Net-score variants (trained CED net, lambda=0) ---
        # These require a trained score net; skipped if ced_ckpt is None.
        # Will be populated in the CED section below.

    if not mc_only:
        # PD-Langevin (per-sample lambda; ib removed)
        _pdl_plr = pdl_primal_lr
        _pdl_dlr = pdl_dual_lr
        if constraint_type == "enriched":
            _bud_pdl = torch.tensor(bud_np, dtype=torch.float32, device=device)
            _pdl_cs = 1.0
            _pdl_extra = _extra_constraints
        else:
            _bud_pdl = torch.tensor(bud_scaled, dtype=torch.float32, device=device)
            _pdl_cs = _constraint_scale
            _pdl_extra = None
        w_pdl, _ = bl.pd_langevin(mu, Sigma, scenarios, _bud_pdl, alpha=float(alpha_np), B=B,
                                    num_iters=500, primal_lr=_pdl_plr,
                                    dual_lr=_pdl_dlr,
                                    device=device, seed=seed,
                                    constraint_type=constraint_type,
                                    objective_scale=_obj_scale,
                                    constraint_scale=_pdl_cs,
                                    extra=_pdl_extra)
        weights["pd_langevin"] = w_pdl.detach()

    # CED trained + net-score variants (all require trained score net)
    if ced_ckpt is not None and Path(ced_ckpt).exists():
        import types as _types_net
        _use_gnn = backbone == "gnn"
        _tagconv_K = tagconv_K
        if _use_gnn:
            backbone = PortfolioGNNBackbone(d=N, hidden=hidden,
                                            num_layers=num_layers,
                                            num_timesteps=T, K=_tagconv_K)
            _dense_A = torch.tensor(build_dense_adjacency(Sigma_np, top_k=20),
                                    dtype=torch.float32)
        else:
            backbone = PortfolioScoreBackbone(d=N, hidden=hidden,
                                                num_layers=num_layers,
                                                num_timesteps=T, cond_channels=1)
            _dense_A = None
        score_net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0)
        state = torch.load(ced_ckpt, map_location="cpu", weights_only=False)
        score_net.load_state_dict(state)
        score_net.to(device).eval()
        trained_score_fn = _make_trained_score_estimator(
            score_net, _mk_sampler(ib, dual_step).alphas_cumprod, dense_A=_dense_A)

        # CED (ours) = trained net + dual ascent
        torch.manual_seed(seed)
        s = _install_lambda_trace(_mk_sampler(ib, dual_step))
        s._estimate_score = _types_net.MethodType(trained_score_fn, s)
        z = s.sample(shape=shape, device=device, data=data)
        weights["ced_trained"] = s.z_to_portfolio_weights(z).detach()
        ced_final_lam = s._lambda_trace[-1].to(device) if s._lambda_trace else None

        # CED-ceiling with fixed final λ from MC ceiling run (opt-in)
        mc_final_lam = lambda_traces.get("mc_teacher")
        if mc_lambda_study and mc_final_lam is not None:
            lam_final_mc = mc_final_lam[-1].to(device)  # [N]
            torch.manual_seed(seed)
            s_mc_fix = _mk_sampler(ib, dual_step)
            def _noop_dual_mc(dual_lambda, ergodic_rates, context, t=None):
                return dual_lambda
            s_mc_fix._dual_ascent_step = _noop_dual_mc
            s_mc_fix.dual_lambda_init = 0.0
            # Override init to use the converged λ
            _orig_init = s_mc_fix._init_dual_lambda
            def _fixed_init(batch_size, num_nodes, device, dtype, init_value,
                            _lam=lam_final_mc):
                return _lam.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
            s_mc_fix._init_dual_lambda = _fixed_init
            z_mc_fix = s_mc_fix.sample(shape=shape, device=device, data=data)
            weights["ced_ceiling_fixlam"] = s_mc_fix.z_to_portfolio_weights(z_mc_fix).detach()

        # CED-trained with fixed final λ from CED run (opt-in)
        if mc_lambda_study and ced_final_lam is not None:
            torch.manual_seed(seed)
            s_ced_fix = _mk_sampler(ib, dual_step)
            s_ced_fix._estimate_score = _types_net.MethodType(trained_score_fn, s_ced_fix)
            def _noop_dual_ced(dual_lambda, ergodic_rates, context, t=None):
                return dual_lambda
            s_ced_fix._dual_ascent_step = _noop_dual_ced
            _orig_init2 = s_ced_fix._init_dual_lambda
            _ced_lam = ced_final_lam
            def _fixed_init2(batch_size, num_nodes, device, dtype, init_value,
                             _lam=_ced_lam):
                return _lam.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
            s_ced_fix._init_dual_lambda = _fixed_init2
            z_ced_fix = s_ced_fix.sample(shape=shape, device=device, data=data)
            weights["ced_trained_fixlam"] = s_ced_fix.z_to_portfolio_weights(z_ced_fix).detach()

        # Unconstrained (net) = trained net, lambda=0
        torch.manual_seed(seed)
        s_unc = _mk_sampler(ib, 1e-12, lam0_override=0.0)
        s_unc._estimate_score = _types_net.MethodType(trained_score_fn, s_unc)
        z_unc = s_unc.sample(shape=shape, device=device, data=data)
        weights["unconstrained_net"] = s_unc.z_to_portfolio_weights(z_unc).detach()

        # PDM (net) = trained net + per-step x_t projection, lambda=0
        if constraint_type != "dual":
            torch.manual_seed(seed)
            s_pdm_net = _mk_sampler(ib, 1e-12, lam0_override=0.0)
            s_pdm_net._estimate_score = _types_net.MethodType(trained_score_fn, s_pdm_net)
            z_pdm_net = _pdm_sample(s_pdm_net, shape, device, data,
                                     Sigma, budgets, constraint_type,
                                     pdm_proj_lr=pdm_proj_lr)
            weights["pdm_net"] = s_pdm_net.z_to_portfolio_weights(z_pdm_net).detach()

        # DPS (net) = trained net + gradient guidance, lambda=0
        def _run_dps_net(scale, label):
            torch.manual_seed(seed)
            s_d = _mk_sampler(ib, 1e-12, lam0_override=0.0)
            s_d._estimate_score = _types_net.MethodType(trained_score_fn, s_d)
            _ac_d = s_d.alphas_cumprod
            _s_d = scale
            _bud_d = torch.tensor(bud_scaled, dtype=torch.float32, device=device)
            _cs_d = _constraint_scale
            def _dps_correct_d(self, x_t, t, x0_pred, eps_pred,
                               _s=_s_d, _sc=scenarios, _b=_bud_d,
                               _a=float(alpha_np), _n=N, _sig=Sigma,
                               _ctype=constraint_type, _ac=_ac_d,
                               _cscale=_cs_d):
                B_loc = x_t.shape[0]
                x_t_g = x_t.detach().requires_grad_(True)
                alpha_bar = _ac[t].view(-1, 1, 1, 1)
                sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
                sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
                x0_tw = (x_t_g - sqrt_1mab * eps_pred.detach()) / sqrt_ab
                w_g = torch.softmax(x0_tw[:, 0, :, 0], dim=-1)
                if _ctype == "shortfall":
                    c_g = shortfall_contributions_torch(w_g, _sc, _a)
                elif _ctype == "variance_band":
                    c_var = variance_contributions_torch(w_g, _sig) * _cscale
                    c_g = torch.cat([c_var, -c_var], dim=-1)
                elif _ctype == "dual":
                    c_var = variance_contributions_torch(w_g, _sig)
                    c_short = shortfall_contributions_torch(w_g, _sc, _a)
                    c_g = torch.cat([c_var, c_short], dim=-1)
                else:
                    c_g = variance_contributions_torch(w_g, _sig) * _cscale
                loss = ((c_g - _b.unsqueeze(0)).clamp_min(0.0) ** 2).sum()
                grad_xt = torch.autograd.grad(loss, x_t_g)[0]
                x_t_corrected = (x_t - _s * grad_xt).detach()
                x0_raw = (x_t_corrected - sqrt_1mab * eps_pred) / sqrt_ab
                w_proj = torch.softmax(x0_raw[:, 0, :, 0], dim=-1)
                z_proj = torch.log(w_proj.clamp_min(1e-12))
                z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
                x0_new = z_proj.view_as(x0_raw)
                eps_new = (x_t - sqrt_ab * x0_new) / sqrt_1mab
                return x0_new, eps_new
            orig_s2x0_d = s_d._score_to_x0_eps
            def _guide_d(self, x_t, t, score, _orig=orig_s2x0_d):
                x0_pred, eps_pred = _orig(x_t, t, score)
                return _dps_correct_d(self, x_t, t, x0_pred, eps_pred)
            s_d._score_to_x0_eps = _types_net.MethodType(_guide_d, s_d)
            z_d = s_d.sample(shape=shape, device=device, data=data)
            weights[label] = s_d.z_to_portfolio_weights(z_d).detach()

        try:
            _run_dps_net(dps_scale, "dps_net")
        except Exception as e:
            print(f"  [dps_net] skipped: {e}")

        # DPS sweep (opt-in)
        if dps_sweep:
            for sv in [float(x) for x in dps_sweep.split(",")]:
                try:
                    _run_dps_net(sv, f"dps_net_s{sv:g}")
                except Exception as e:
                    print(f"  [dps_net_s{sv:g}] skipped: {e}")

        # PDM (net) with fixed λ=1 and λ=300 (opt-in)
        if mc_lambda_study:
          for _lam_pdm, _lbl in [(1.0, "pdm_net_lam1"), (300.0, "pdm_net_lam300")]:
            torch.manual_seed(seed)
            s_pl = _mk_sampler(ib, 1e-12, lam0_override=_lam_pdm)
            def _noop_dual_pdm(dual_lambda, ergodic_rates, context, t=None):
                return dual_lambda
            s_pl._dual_ascent_step = _noop_dual_pdm
            s_pl._estimate_score = _types_net.MethodType(trained_score_fn, s_pl)
            orig_s2x0_pl = s_pl._score_to_x0_eps
            _sig_pl, _sc_pl, _b_pl = Sigma, scenarios, budgets
            _a_pl, _n_pl, _ct_pl = float(alpha_np), N, constraint_type
            def _pdm_proj_lam(self, x_t, t, score,
                              _orig=orig_s2x0_pl, _sig=_sig_pl, _sc=_sc_pl,
                              _b=_b_pl, _a=_a_pl, _n=_n_pl, _ct=_ct_pl):
                x0_pred, eps_pred = _orig(x_t, t, score)
                B_loc = x0_pred.shape[0]
                w_pred = self.z_to_portfolio_weights(x0_pred)
                w_proj = bl.blend_to_feasible(w_pred, _sig, _sc, _a, _b,
                                              constraint_type=_ct,
                                              num_iters=40, tol=1e-8)
                z_proj = torch.log(w_proj.clamp_min(1e-12))
                z_proj = z_proj - z_proj.mean(dim=-1, keepdim=True)
                x0_new = z_proj.view(B_loc, 1, _n, 1)
                alpha_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
                sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1e-12))
                sqrt_1mab = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))
                eps_new = (x_t - sqrt_ab * x0_new) / sqrt_1mab
                return x0_new, eps_new
            s_pl._score_to_x0_eps = _types_net.MethodType(_pdm_proj_lam, s_pl)
            z_pl = s_pl.sample(shape=shape, device=device, data=data)
            weights[_lbl] = s_pl.z_to_portfolio_weights(z_pl).detach()

    problem = {
        "mu": mu_np, "Sigma": Sigma_np, "scenarios": scen_np,
        "budgets": bud_np, "alpha": float(alpha_np),
        "constraint_type": constraint_type,
    }
    if _extra_constraints:
        problem["extra"] = _extra_constraints
    return {
        "weights": weights,
        "problem": problem,
        "lambda_traces": lambda_traces,
    }


# ---------------------------------------------------------------
# Caching: skip recomputation when iterating on plot code
# ---------------------------------------------------------------

def _cache_key(args) -> str:
    return (f"{args.size}_K{args.num_instances}_ib{args.ib:g}_ds{args.dual_step:g}"
            f"_T{args.T}_B{args.B}_mc{args.K}_seed{args.seed}")


def _cache_path(args) -> Path:
    root = Path(args.cache_dir) if args.cache_dir else (
        _project_root() / "outputs" / "portfolio" / "figures_cache"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_cache_key(args)}.pt"


def _save_cache(path: Path, data_list: list) -> None:
    payload = []
    for d in data_list:
        payload.append({
            "weights": {m: w.detach().cpu() for m, w in d["weights"].items()},
            "problem": d["problem"],
            "problem_seed": d.get("problem_seed", None),
        })
    torch.save(payload, path)


def _load_cache(path: Path, device) -> list | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for d in payload:
        d["weights"] = {m: w.to(device) for m, w in d["weights"].items()}
    return payload


# ---------------------------------------------------------------
# Multi-instance collection
# ---------------------------------------------------------------

def collect_multi_instance(size: str, num_instances: int, ib: float,
                            dual_step: float, T: int, B: int, K: int,
                            seed: int, device,
                            ced_ckpt: Path | None = None,
                            hidden: int = 256, num_layers: int = 4,
                            include_mc_variants: bool = False,
                            mc_only: bool = False,
                            beta_schedule: str = "cosine",
                            pdl_primal_lr: float = 0.01,
                            pdl_dual_lr: float = 100.0,
                            pdl_noise_scale: float = 0.01,
                            lam0: float = 0.0,
                            sub_batch: int = 0,
                            dual_lambda_decay: float = 0.0,
                            normalize_constraints: bool = True,
                            mc_lambda_study: bool = False,
                            backbone: str = "mlp",
                            tagconv_K: int = 2,
                            dps_scale: float = 1.0,
                            dps_sweep: str = None,
                            pdm_proj_lr: float = 0.1) -> list:
    """Collect weights for K instances."""
    pcfg = dict(PROBLEM_CONFIGS[size])
    N = pcfg["N"]

    data_list = []
    for s in range(num_instances):
        ti = time.time()
        d = collect_all_weights(
            size=size, ib=ib, dual_step=dual_step, T=T, B=B, K=K,
            seed=seed, device=device, ced_ckpt=ced_ckpt,
            hidden=hidden, num_layers=num_layers,
            problem_seed=s,
            include_mc_variants=include_mc_variants,
            mc_only=mc_only,
            beta_schedule=beta_schedule,
            pdl_primal_lr=pdl_primal_lr, pdl_dual_lr=pdl_dual_lr,
            pdl_noise_scale=pdl_noise_scale, lam0=lam0, sub_batch=sub_batch,
            dual_lambda_decay=dual_lambda_decay,
            normalize_constraints=normalize_constraints,
            mc_lambda_study=mc_lambda_study,
            backbone=backbone, tagconv_K=tagconv_K,
            dps_scale=dps_scale, dps_sweep=dps_sweep,
            pdm_proj_lr=pdm_proj_lr,
        )
        d["problem_seed"] = s
        data_list.append(d)
        print(f"  [multi] instance {s+1}/{num_instances} ({time.time()-ti:.1f}s)")
    return data_list


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def _compute_contributions_np(w, problem):
    """Dispatch constraint evaluation by constraint_type."""
    ct = problem.get("constraint_type", "shortfall")
    if ct == "shortfall":
        return shortfall_contributions_np(w, problem["scenarios"], problem["alpha"])
    elif ct == "variance_band":
        c_var = variance_contributions_np(w, problem["Sigma"])
        return np.concatenate([c_var, -c_var], axis=-1)
    elif ct in ("variance", "variance_sector"):
        return variance_contributions_np(w, problem["Sigma"])
    elif ct == "dual":
        c_var = variance_contributions_np(w, problem["Sigma"])
        c_short = shortfall_contributions_np(w, problem["scenarios"], problem["alpha"])
        return np.concatenate([c_var, c_short], axis=-1)
    elif ct == "enriched":
        extra = problem["extra"]
        N = w.shape[1]
        S = len(extra["sector_upper"])
        c_var = variance_contributions_np(w, problem["Sigma"])
        B = w.shape[0]
        sector_ids = extra["sector_ids"]
        exposure = np.zeros((B, S))
        for s in range(S):
            exposure[:, s] = w[:, sector_ids == s].sum(axis=1)
        upper_res = exposure - extra["sector_upper"][None, :]
        lower_res = extra["sector_lower"][None, :] - exposure
        stress_port = w @ extra["stress_returns"].T
        excess_loss = np.maximum(-stress_port - extra["stress_limits"][None, :], 0.0)
        stress_res = excess_loss - extra["stress_eps"][None, :]
        ent = -(w * np.log(np.clip(w, 1e-12, None))).sum(axis=1, keepdims=True)
        neff_val = np.exp(ent)
        neff_res = neff_val - extra.get("neff_target", 500.0)
        return np.concatenate([c_var, upper_res, lower_res, stress_res, neff_res], axis=-1)
    else:
        raise ValueError(f"Unknown constraint_type: {ct}")


def _per_instance_metrics(weights_dict, problem, methods, eps_feas=0.1):
    """Compute per-method metrics for ONE instance. Returns dict[method] -> metrics dict."""
    Sigma = problem["Sigma"]; scen = problem["scenarios"]; bud = problem["budgets"]
    alpha = problem["alpha"]
    N = len(problem["mu"])
    metrics = {}
    for m in methods:
        if m not in weights_dict:
            continue
        w = weights_dict[m].cpu().numpy()
        c = _compute_contributions_np(w, problem)
        violation = np.maximum(c - bud[None, :], 0.0)
        rel_vio = violation / np.maximum(np.abs(bud[None, :]), 1e-12)
        ret = (scen @ w.T).mean(axis=0)
        var = np.einsum('bi,ij,bj->b', w, Sigma, w)
        sharpe = ret / np.sqrt(np.clip(var, 1e-12, None))
        feas_rel = (rel_vio.max(axis=1) <= eps_feas).astype(float)
        ent = -(w * np.log(np.clip(w, 1e-12, None))).sum(axis=1)
        Neff = np.exp(ent)
        c_mean = c.mean(axis=0)
        opt2_feas = float((c_mean <= bud + 1e-8).all())
        opt2_csat = float((c_mean <= bud + 1e-8).mean())
        opt2_vio = np.maximum(c_mean - bud, 0.0) / np.maximum(np.abs(bud), 1e-12)
        opt2_max_rel = float(opt2_vio.max()) if opt2_vio.size else 0.0
        # Cross-sample diversity: how many distinct assets win the top-1 spot,
        # and the entropy of that distribution.
        top1 = w.argmax(axis=1)                              # [B]
        counts = np.bincount(top1, minlength=N).astype(float)
        p_top1 = counts / counts.sum()
        top1_distinct = int((counts > 0).sum())
        top1_entropy = float(-(p_top1[p_top1 > 0] *
                                np.log(p_top1[p_top1 > 0])).sum())
        metrics[m] = dict(
            sharpe=float(sharpe.mean()), sharpe_std=float(sharpe.std()),
            feas=float(feas_rel.mean()),
            opt2_feas=opt2_feas, opt2_csat=opt2_csat,
            opt2_max_rel_vio=opt2_max_rel,
            vio_max=float(violation.max()),
            vio_mean=float(np.maximum(c_mean - bud, 0.0).mean()),
            ret=float(ret.mean()), var=float(var.mean()),
            Neff=float(Neff.mean()),
            top1_distinct=top1_distinct, top1_entropy=top1_entropy,
        )
        if problem.get("constraint_type") == "enriched":
            extra = problem["extra"]
            n_var = N
            S = len(extra["sector_upper"])
            M = len(extra["stress_limits"])
            var_c_mean = c_mean[:n_var]
            var_bud = bud[:n_var]
            metrics[m]["var_csat"] = float((var_c_mean <= var_bud + 1e-8).mean())
            metrics[m]["var_vio"] = float(np.maximum(var_c_mean - var_bud, 0.0).mean())
            var_bud_scale = max(abs(var_bud).mean(), 1e-12)
            metrics[m]["var_rvio"] = float(np.maximum(var_c_mean - var_bud, 0.0).mean() / var_bud_scale)
            sec_c_mean = c_mean[n_var:n_var + 2 * S]
            sec_bud = bud[n_var:n_var + 2 * S]
            metrics[m]["sector_csat"] = float((sec_c_mean <= sec_bud + 1e-8).mean())
            metrics[m]["sector_vio"] = float(np.maximum(sec_c_mean - sec_bud, 0.0).mean())
            sec_scale = max(abs(extra["sector_upper"]).mean(), 1e-12)
            metrics[m]["sector_rvio"] = float(np.maximum(sec_c_mean - sec_bud, 0.0).mean() / sec_scale)
            str_end = n_var + 2 * S + M
            str_c_mean = c_mean[n_var + 2 * S:str_end]
            str_bud = bud[n_var + 2 * S:str_end]
            metrics[m]["stress_csat"] = float((str_c_mean <= str_bud + 1e-8).mean())
            metrics[m]["stress_vio"] = float(np.maximum(str_c_mean - str_bud, 0.0).mean())
            str_scale = max(abs(extra["stress_eps"]).mean(), 1e-12)
            metrics[m]["stress_rvio"] = float(np.maximum(str_c_mean - str_bud, 0.0).mean() / str_scale)
            neff_c_mean = c_mean[str_end:]
            neff_bud = bud[str_end:]
            metrics[m]["neff_csat"] = float((neff_c_mean <= neff_bud + 1e-8).all())
            neff_target = extra.get("neff_target", 60.0)
            metrics[m]["neff_vio"] = float(np.maximum(neff_c_mean - neff_bud, 0.0).mean())
            metrics[m]["neff_rvio"] = float(np.maximum(neff_c_mean - neff_bud, 0.0).mean() / neff_target)
    return metrics


def plot_comparison_bars(data, out_dir, methods=PAPER_METHODS, eps_feas=0.1):
    """4-panel bar chart: Sharpe, O2csat, distinct top-1, top-1 entropy.

    Bars are means across instances (no std bars). `data` is either a single
    instance dict or a list of instance dicts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_list = data if isinstance(data, list) else [data]
    K = len(data_list)
    N = len(data_list[0]["problem"]["mu"])

    # Include all available weight keys so sweep entries get metrics computed
    all_weight_keys = set()
    for d in data_list:
        all_weight_keys.update(d["weights"].keys())
    methods_extended = list(methods) + [k for k in sorted(all_weight_keys) if k not in methods]

    per_inst = [_per_instance_metrics(d["weights"], d["problem"], methods_extended,
                                        eps_feas=eps_feas) for d in data_list]

    methods_found = [m for m in methods_extended if any(m in pi for pi in per_inst)]
    labels = [METHOD_LABELS.get(m, m) for m in methods_found]
    x = np.arange(len(methods_found))

    def _stack(key):
        return {m: np.array([pi[m][key] for pi in per_inst if m in pi])
                for m in methods_found}

    import matplotlib.cm as cm
    cmap = cm.get_cmap("viridis", len(methods_found))
    colors = [cmap(i) for i in range(len(methods_found))]

    ret = _stack("ret")
    o2csat = _stack("opt2_csat")
    neff = _stack("Neff")
    top1H = _stack("top1_entropy")
    vio_mean = _stack("vio_mean")
    top1_distinct = _stack("top1_distinct")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    def _bar(ax, stat_dict, title, fmt="{:.2f}", ylim=None):
        means = np.array([stat_dict[m].mean() for m in methods_found])
        bars = ax.bar(x, means, color=colors, edgecolor="black")
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)

    _bar(axes[0], ret, "Expected return", fmt="{:.3f}")
    _bar(axes[1], o2csat, "Avg-constraint feasibility", ylim=(0, 1.1))
    _bar(axes[2], neff, rf"$N_{{\mathrm{{eff}}}}$ (max $= {N}$)", fmt="{:.1f}")
    _bar(axes[3], top1H, rf"top-1 entropy (max $\log {N} \approx {np.log(N):.2f}$)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_comparison_bars.png", dpi=150)
    fig.savefig(out_dir / "fig_comparison_bars.pdf")
    plt.close(fig)

    agg = {}
    for m in methods_found:
        agg[m] = {
            k: {"mean": float(np.mean(vals[m]))}
            for k, vals in {
                "ret": ret, "opt2_csat": o2csat,
                "Neff": neff, "top1_entropy": top1H,
                "vio_mean": vio_mean, "top1_distinct": top1_distinct,
            }.items()
        }

    # Check if enriched
    is_enriched = data_list[0]["problem"].get("constraint_type") == "enriched"

    if is_enriched:
        var_csat = _stack("var_csat")
        sector_csat = _stack("sector_csat")
        stress_csat = _stack("stress_csat")
        neff_csat_vals = _stack("neff_csat")
        var_vio = _stack("var_vio")
        sector_vio = _stack("sector_vio")
        stress_vio = _stack("stress_vio")
        neff_vio_vals = _stack("neff_vio")
        var_rvio = _stack("var_rvio")
        sector_rvio = _stack("sector_rvio")
        stress_rvio = _stack("stress_rvio")
        neff_rvio = _stack("neff_rvio")

    # Write LaTeX table
    show_std = K > 1

    def _enriched_table(use_rvio=False):
        tag = "rvio" if use_rvio else "baselines"
        vio_label = "RVio" if use_rvio else "Vio"
        lines_e = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Enriched portfolio baseline comparison ($N=" + str(N) + r"$)}",
            r"\label{tab:portfolio_enriched_" + tag + r"}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l c c c c c c c c c c c c c c}",
            r"\toprule",
            r"Method & Return & Var\% & Sec\% & Str\% & Neff\% & Feas\%"
            r" & Var " + vio_label + r" & Sec " + vio_label + r" & Str " + vio_label + r" & Neff " + vio_label
            + r" & $N_{\mathrm{eff}}$ & Entropy & $|\mathrm{top1}|$ \\",
            r"\midrule",
        ]
        for m in methods_found:
            lbl = METHOD_LABELS.get(m, m).replace("\n", " ")
            r_m, r_s = float(ret[m].mean()), float(ret[m].std())
            f_m, f_s = float(o2csat[m].mean()), float(o2csat[m].std())
            ne_m, ne_s = float(neff[m].mean()), float(neff[m].std())
            h_m, h_s = float(top1H[m].mean()), float(top1H[m].std())
            td_m, td_s = float(top1_distinct[m].mean()), float(top1_distinct[m].std())
            vc_m, vc_s = float(var_csat[m].mean()), float(var_csat[m].std())
            sc_m, sc_s = float(sector_csat[m].mean()), float(sector_csat[m].std())
            st_m, st_s = float(stress_csat[m].mean()), float(stress_csat[m].std())
            nc_m, nc_s = float(neff_csat_vals[m].mean()), float(neff_csat_vals[m].std())
            if use_rvio:
                v1_m, v1_s = float(var_rvio[m].mean()), float(var_rvio[m].std())
                v2_m, v2_s = float(sector_rvio[m].mean()), float(sector_rvio[m].std())
                v3_m, v3_s = float(stress_rvio[m].mean()), float(stress_rvio[m].std())
                v4_m, v4_s = float(neff_rvio[m].mean()), float(neff_rvio[m].std())
                vfmt = lambda vm, vs: f"{vm:.3f}$\\pm${vs:.3f}" if show_std else f"{vm:.3f}"
            else:
                v1_m, v1_s = float(var_vio[m].mean()), float(var_vio[m].std())
                v2_m, v2_s = float(sector_vio[m].mean()), float(sector_vio[m].std())
                v3_m, v3_s = float(stress_vio[m].mean()), float(stress_vio[m].std())
                v4_m, v4_s = float(neff_vio_vals[m].mean()), float(neff_vio_vals[m].std())
                vfmt = lambda vm, vs: f"{vm:.2e}$\\pm${vs:.2e}" if show_std else f"{vm:.2e}"
            if show_std:
                lines_e.append(
                    f"  {lbl} & {r_m:.3f}$\\pm${r_s:.3f}"
                    f" & {vc_m:.2f}$\\pm${vc_s:.2f}"
                    f" & {sc_m:.2f}$\\pm${sc_s:.2f}"
                    f" & {st_m:.2f}$\\pm${st_s:.2f}"
                    f" & {nc_m:.2f}$\\pm${nc_s:.2f}"
                    f" & {f_m:.2f}$\\pm${f_s:.2f}"
                    f" & {vfmt(v1_m, v1_s)}"
                    f" & {vfmt(v2_m, v2_s)}"
                    f" & {vfmt(v3_m, v3_s)}"
                    f" & {vfmt(v4_m, v4_s)}"
                    f" & {ne_m:.1f}$\\pm${ne_s:.1f}"
                    f" & {h_m:.2f}$\\pm${h_s:.2f}"
                    f" & {td_m:.0f}$\\pm${td_s:.0f} \\\\")
            else:
                lines_e.append(
                    f"  {lbl} & {r_m:.3f}"
                    f" & {vc_m:.2f} & {sc_m:.2f} & {st_m:.2f} & {nc_m:.2f} & {f_m:.2f}"
                    f" & {vfmt(v1_m, v1_s)} & {vfmt(v2_m, v2_s)}"
                    f" & {vfmt(v3_m, v3_s)} & {vfmt(v4_m, v4_s)}"
                    f" & {ne_m:.1f} & {h_m:.2f} & {td_m:.0f} \\\\")
        lines_e += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        return lines_e

    if is_enriched:
        # Raw violation table
        lines = _enriched_table(use_rvio=False)
        tex_path = out_dir / "table_baselines.tex"
        tex_path.write_text("\n".join(lines) + "\n")
        print(f"saved {tex_path}")
        # Relative violation table
        lines_rv = _enriched_table(use_rvio=True)
        rv_path = out_dir / "table_rvio.tex"
        rv_path.write_text("\n".join(lines_rv) + "\n")
        print(f"saved {rv_path}")
    else:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Portfolio baseline comparison ($N=" + str(N) + r"$)}",
            r"\label{tab:portfolio_baselines}",
            r"\begin{tabular}{l c c c c c c}",
            r"\toprule",
            r"Method & Return & Feasibility & Mean Viol. & $N_{\mathrm{eff}}$ & Entropy & $|\mathrm{top1}|$ \\",
            r"\midrule",
        ]
        for m in methods_found:
            lbl = METHOD_LABELS.get(m, m).replace("\n", " ")
            r_m, r_s = float(ret[m].mean()), float(ret[m].std())
            f_m, f_s = float(o2csat[m].mean()), float(o2csat[m].std())
            ne_m, ne_s = float(neff[m].mean()), float(neff[m].std())
            h_m, h_s = float(top1H[m].mean()), float(top1H[m].std())
            td_m, td_s = float(top1_distinct[m].mean()), float(top1_distinct[m].std())
            v_m, v_s = float(vio_mean[m].mean()), float(vio_mean[m].std())
            lines.append(
                f"  {lbl} & {r_m:.3f}$\\pm${r_s:.3f}"
                f" & {f_m:.2f}$\\pm${f_s:.2f}"
                f" & {v_m:.2e}$\\pm${v_s:.2e}"
                f" & {ne_m:.1f}$\\pm${ne_s:.1f}"
                f" & {h_m:.2f}$\\pm${h_s:.2f}"
                f" & {td_m:.0f}$\\pm${td_s:.0f} \\\\")
    if not is_enriched:
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tex_path = out_dir / "table_baselines.tex"
        tex_path.write_text("\n".join(lines) + "\n")
        print(f"saved {tex_path}")

    return agg


def plot_return_vs_violation(data, out_dir, methods=PAPER_METHODS, instance_idx=0):
    """Scatter: expected return vs max avg-constraint relative violation per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    })

    data_list = data if isinstance(data, list) else [data]
    d = data_list[instance_idx]
    methods_found = [m for m in methods if m in d["weights"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    scen = d["problem"]["scenarios"]
    bud = d["problem"]["budgets"]
    alpha = d["problem"]["alpha"]

    for i, m in enumerate(methods_found):
        lbl = METHOD_LABELS.get(m, m).replace("\n", " ")
        w = d["weights"][m].cpu().numpy()
        ret_per_sample = (scen @ w.T).mean(axis=0)  # [B]
        c = _compute_contributions_np(w, d["problem"])  # [B, N]
        c_mean = c.mean(axis=0)  # [N] avg-constraint
        rel_vio = (c_mean - bud) / np.maximum(bud, 1e-12)
        max_rel_vio = rel_vio.max()
        mean_ret = ret_per_sample.mean()
        ax.scatter(max_rel_vio, mean_ret,
                   marker=markers[i % len(markers)], s=120, zorder=3,
                   edgecolors="black", linewidths=0.6, label=lbl)

    ax.axvline(0.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"Max relative constraint value $\max_j\; (\bar{c}_j - b_j) / b_j$", fontsize=16)
    ax.set_ylabel(r"Expected return $E_\xi[r^\top x]$", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_return_vs_violation.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_return_vs_violation.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weight_heatmap(data, out_dir, methods=PAPER_METHODS, instance_idx=0):
    """Grid of weight heatmaps, one subplot per method, ALL B samples on x-axis.

    Asset order is per-method (sorted by mean weight). One instance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_list = data if isinstance(data, list) else [data]
    d0 = data_list[instance_idx]
    weights_dict = d0["weights"]
    methods_found = [m for m in methods if m in weights_dict]
    n = len(methods_found)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                              squeeze=False)

    for idx, m in enumerate(methods_found):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        w = weights_dict[m].cpu().numpy()
        order = np.argsort(w.mean(axis=0))[::-1]
        im = ax.imshow(w[:, order].T, aspect="auto", cmap="magma",
                       vmin=0, vmax=min(0.5, float(w.max())))
        ax.set_title(METHOD_LABELS.get(m, m).replace("\n", " "), fontsize=10)
        if r == nrows - 1:
            ax.set_xlabel(f"Sample (B={w.shape[0]})")
        if c == 0:
            ax.set_ylabel("Asset (sorted by mean weight)")
        plt.colorbar(im, ax=ax, shrink=0.7)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(f"Portfolio weight heatmaps  (instance {instance_idx})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_weight_heatmap.png", dpi=150)
    fig.savefig(out_dir / "fig_weight_heatmap.pdf")
    plt.close(fig)


def plot_dual_trace(data, out_dir, instance_idx=0):
    """Plot λ trajectory over reverse diffusion steps."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_list = data if isinstance(data, list) else [data]
    traces = data_list[instance_idx].get("lambda_traces", {})
    if not traces:
        print("  [dual_trace] no lambda traces recorded, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, tr in traces.items():
        tr_np = tr.numpy()  # [T, N]
        T_steps, N_constraints = tr_np.shape
        t_axis = np.arange(T_steps)
        lam_mean = tr_np.mean(axis=1)
        lam_max = tr_np.max(axis=1)
        lam_max = tr_np.max(axis=1)
        label = name.replace("mc_teacher_sub", "sub-")
        alpha = 0.4 if "sub" in name else 1.0
        axes[0].plot(t_axis, lam_mean, alpha=alpha, label=label)
        axes[1].plot(t_axis, lam_max, alpha=alpha, label=label)

    axes[0].set_title(r"$\bar{\lambda}$ (mean over constraints)")
    axes[0].set_xlabel("reverse step")
    axes[0].set_ylabel(r"$\lambda$")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].set_title(r"$\max_j \lambda_j$")
    axes[1].set_xlabel("reverse step")
    axes[1].set_ylabel(r"$\lambda_{\max}$")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_dual_trace.png", dpi=150)
    fig.savefig(out_dir / "fig_dual_trace.pdf")
    plt.close(fig)


# ---------------------------------------------------------------
# Per-step traces (CED reverse SDE + PD-Langevin iterations)
# ---------------------------------------------------------------

def _sample_with_trace(sampler, shape, device, data, snap_timesteps):
    """Run reverse SDE step-by-step, recording lambda_mean and weight snaps.

    snap_timesteps are raw t_int values (0=data side, T-1=noise side).
    Returns (final_w, lambda_mean[T] indexed by step order, snaps[t_int]).
    """
    import tqdm
    B, _, N, _ = shape
    x_init = torch.randn(shape, device=device, dtype=torch.float32)
    context = sampler._build_energy_context(
        data=data, batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
    )
    dual_lambda = sampler._init_dual_lambda(
        batch_size=B, num_nodes=N, device=device, dtype=torch.float32,
        init_value=sampler.dual_lambda_init,
    )
    x_t = x_init.clone()

    snap_set = set(int(t) for t in snap_timesteps)
    snaps = {}
    lam_per_step = []   # one float per reverse step in order (noise -> data)
    lam_vectors = []    # [T, N] full lambda vector per step
    violation_traces = []  # [T, N] per-constraint violation per step
    constraint_vals = []   # [T, N] per-constraint c_j values (for CDF)
    ret_traces = []     # one float per step: mean return of x0_pred

    iterator = tqdm.tqdm(reversed(range(sampler.num_timesteps)),
                         total=sampler.num_timesteps,
                         desc="CED trace", unit="step")
    for t_int in iterator:
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
        score = sampler._estimate_score(
            x_t=x_t, t=t_tensor, context=context, dual_lambda=dual_lambda,
            inverse_beta_override=None,
        )
        x0_pred, _ = sampler._score_to_x0_eps(x_t=x_t, t=t_tensor, score=score)
        ergodic_rates, _ = sampler._ergodic_rates_from_samples(
            x=x0_pred, context=context,
        )
        dual_lambda = sampler._dual_ascent_step(
            dual_lambda=dual_lambda, ergodic_rates=ergodic_rates,
            context=context, t=t_tensor,
        )
        lam_per_step.append(float(dual_lambda.mean().item()))
        lam_vectors.append(dual_lambda[0].detach().cpu().numpy())

        with torch.no_grad():
            w_pred = sampler.z_to_portfolio_weights(x0_pred)
            if getattr(sampler, 'constraint_type', 'shortfall').startswith("variance"):
                c = variance_contributions_torch(w_pred, context.Sigma)
            else:
                c = shortfall_contributions_torch(w_pred, context.scenarios, context.alpha)
            c_mean = c.mean(dim=0).cpu().numpy()  # [N]
            bud = context.risk_budgets.cpu().numpy()
            violation_traces.append(np.maximum(c_mean - bud, 0.0))
            constraint_vals.append(c_mean)
            ret_step = float((context.scenarios @ w_pred.t()).mean().item())
            ret_traces.append(ret_step)

        mean = sampler._posterior_mean(x0_pred=x0_pred, x_t=x_t, t=t_tensor)
        if t_int > 0:
            var = sampler._gather(sampler.posterior_variance, t_tensor).view(-1, 1, 1, 1)
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = mean

        if int(t_int) in snap_set:
            with torch.no_grad():
                snaps[int(t_int)] = sampler.z_to_portfolio_weights(x_t).cpu().numpy()

    final_w = sampler.z_to_portfolio_weights(x_t).detach().cpu().numpy()
    traces = {
        "lam_mean": np.asarray(lam_per_step),
        "lam_vectors": np.stack(lam_vectors),       # [T, N]
        "violations": np.stack(violation_traces),    # [T, N]
        "constraint_vals": np.stack(constraint_vals),  # [T, N]
        "returns": np.asarray(ret_traces),             # [T]
    }
    return final_w, traces, snaps


def _pd_langevin_with_trace(mu, Sigma, scenarios, budgets, alpha, B, num_iters,
                              primal_lr, dual_lr, noise_scale,
                              snap_iters, device, seed=42,
                              constraint_type="variance"):
    """Run PD-Langevin with per-constraint λ tracing.

    Returns (final_w, traces_dict, snaps).
    traces_dict has 'lam_mean' [T], 'lam_vectors' [T, N].
    """
    import tqdm
    from baselines import _compute_constraints
    torch.manual_seed(seed)
    N = len(mu)
    n_constraints = len(budgets)
    z = torch.randn(B, N, device=device)
    lam = torch.zeros(B, n_constraints, device=device)

    snap_set = set(int(i) for i in snap_iters)
    snaps = {}
    lam_per_step = []
    lam_vectors = []

    for k in tqdm.trange(num_iters, desc="PDL trace"):
        z_req = z.detach().requires_grad_(True)
        x = torch.softmax(z_req, dim=-1)
        c = _compute_constraints(x, Sigma, scenarios, alpha, constraint_type)
        expected_return = torch.matmul(scenarios, x.t()).mean(dim=0)
        violation = c - budgets.unsqueeze(0)
        lag_pen = (lam * violation).sum(dim=1)
        energy = -expected_return + lag_pen

        grad = torch.autograd.grad(energy.sum(), z_req)[0]
        noise = math.sqrt(2 * primal_lr) * torch.randn_like(z) if k < num_iters - 1 else 0.0
        z = (z_req - primal_lr * grad + noise).detach()

        with torch.no_grad():
            x_now = torch.softmax(z, dim=-1)
            c_now = _compute_constraints(x_now, Sigma, scenarios, alpha, constraint_type)
            v = c_now - budgets.unsqueeze(0)
            lam = (lam + dual_lr * v).clamp_min(0.0)

        lam_per_step.append(float(lam.mean().item()))
        lam_vectors.append(lam.mean(dim=0).detach().cpu().numpy())

        if k in snap_set:
            with torch.no_grad():
                snaps[int(k)] = torch.softmax(z, dim=-1).cpu().numpy()

    final_w = torch.softmax(z, dim=-1).detach().cpu().numpy()
    traces = {
        "lam_mean": np.asarray(lam_per_step),
        "lam_vectors": np.stack(lam_vectors),  # [T, N]
    }
    return final_w, traces, snaps


def plot_sample_evolution_compare(ced_snaps, pdl_snaps, ced_t_to_disp, pdl_t_to_disp,
                                    out_dir):
    """2 rows (top=CED, bot=PDL) × 4 cols (4 timesteps).

    ced_snaps[t_int] -> [B, N] weights; pdl_snaps[iter_idx] -> [B, N] weights.
    `*_to_disp` map raw step keys -> display-t (0=noise, T=data) for column titles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ced_keys_sorted = sorted(ced_snaps.keys(), key=lambda k: ced_t_to_disp[k])
    pdl_keys_sorted = sorted(pdl_snaps.keys(), key=lambda k: pdl_t_to_disp[k])
    n = max(len(ced_keys_sorted), len(pdl_keys_sorted))

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        if i < len(ced_keys_sorted):
            k = ced_keys_sorted[i]
            w = ced_snaps[k]
            N = w.shape[1]
            ax = axes[0, i]
            ax.bar(np.arange(N), w.mean(axis=0), color="C0", alpha=0.85)
            ax.axhline(1.0 / N, color="k", lw=0.5, ls="--")
            ax.set_title(f"t = {ced_t_to_disp[k]}")
            if i == 0:
                ax.set_ylabel("CED  E_b[w_j]")

        if i < len(pdl_keys_sorted):
            k = pdl_keys_sorted[i]
            w = pdl_snaps[k]
            N = w.shape[1]
            ax = axes[1, i]
            ax.bar(np.arange(N), w.mean(axis=0), color="C3", alpha=0.85)
            ax.axhline(1.0 / N, color="k", lw=0.5, ls="--")
            ax.set_xlabel("asset")
            if i == 0:
                ax.set_ylabel("PDL  E_b[w_j]")

    fig.suptitle("Sample histogram evolution  (t=0: noise   t=T: data)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sample_evolution.png", dpi=150)
    fig.savefig(out_dir / "fig_sample_evolution.pdf")
    plt.close(fig)


def plot_dual_evolution_compare(ced_lams, pdl_lams, out_dir):
    """Two subplots (CED, PDL); each shows λ trajectory per instance.

    `*_lams` is a list of 1D arrays (one per instance), already in display-t order
    (index 0 = noise side, last = data side).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for traces, ax, title, color in [
        (ced_lams, axes[0], "CED reverse SDE", "C0"),
        (pdl_lams, axes[1], "PD-Langevin", "C3"),
    ]:
        for i, lam in enumerate(traces):
            xs = np.linspace(0, 1, len(lam))
            ax.plot(xs, lam, color=color, alpha=0.55, lw=1.2,
                    label=f"inst {i}" if i < 5 else None)
        ax.set_title(title)
        ax.set_xlabel("normalized t  (0=noise, 1=data)")
        ax.set_ylabel(r"$\bar\lambda$")
        ax.grid(alpha=0.3)
        if traces:
            ax.legend(fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_dual_evolution.png", dpi=150)
    fig.savefig(out_dir / "fig_dual_evolution.pdf")
    plt.close(fig)


def fig_dual_panel(mc_traces, ced_traces, pdl_traces, budgets, out_dir,
                   n_lines=8, hard_frac=0.15):
    """2×3 panel: rows = easy/hard constraints, cols = MC ceiling / CED / PDL.

    Easy/hard split is computed **per method** from each method's own final λ.
    PDL λ is averaged over samples (per-sample → mean).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        ("CED-ceiling (MC)", mc_traces, "#2c7fb8"),
        (r"CED (ours)", ced_traces, "#d95f02"),
        (r"PD-Langevin ($\bar{\lambda}$ over samples)", pdl_traces, "#7570b3"),
    ]

    rng = np.random.RandomState(0)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex="col")

    for col, (title, traces, color) in enumerate(methods):
        if traces is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=12)
                if row == 0:
                    axes[row, col].set_title(title, fontsize=9)
            continue

        lam = traces["lam_vectors"]  # [T, N]
        T_steps, N = lam.shape
        t_axis = np.arange(T_steps)

        bud = np.array(budgets).ravel()
        lam_norm = lam / bud[None, :].clip(min=1e-12)

        final_lam = lam_norm[-1]
        threshold = np.quantile(final_lam, 1.0 - hard_frac)
        hard_idx = np.where(final_lam >= threshold)[0]
        easy_idx = np.where(final_lam < threshold)[0]
        n_easy = min(n_lines, len(easy_idx))
        n_hard = min(n_lines, len(hard_idx))
        easy_sel = rng.choice(easy_idx, n_easy, replace=False) if n_easy > 0 else []
        hard_sel = rng.choice(hard_idx, n_hard, replace=False) if n_hard > 0 else []

        cmap = plt.cm.tab10
        for row, (sel, row_label) in enumerate([
            (easy_sel, "Easy"),
            (hard_sel, "Hard"),
        ]):
            ax = axes[row, col]
            for idx, j in enumerate(sel):
                c = cmap(idx % 10)
                ax.plot(t_axis, lam_norm[:, j], color=c, alpha=0.7, lw=0.9)
                lam_bar_j = float(lam_norm[:, j].mean())
                ax.axhline(lam_bar_j, color=c, ls=":", alpha=0.5, lw=0.7)

            ax.grid(alpha=0.2, linewidth=0.5)
            ax.tick_params(axis="both", labelsize=7)

            if row == 0:
                ax.set_title(title, fontsize=9, pad=4)
            if row == 1:
                ax.set_xlabel("step", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{row_label}" + r"  $\lambda_j / b_j$", fontsize=8)

    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    fig.savefig(out_dir / "fig_dual_panel.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_dual_panel.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_dual_panel.png'}")


def fig_lam_study_dual_panel(traces_dict, out_dir, n_lines=8, hard_frac=0.15):
    """2×3 dual panel for lambda study: original PD / warm λ*+PD / warm λ̄+PD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        ("PD from 0", traces_dict.get("original"), "#2c7fb8"),
        (r"Warm $\lambda_T$ + PD", traces_dict.get("warm_lamfinal"), "#d95f02"),
        (r"Warm $\bar\lambda$ + PD", traces_dict.get("warm_lamavg"), "#7570b3"),
    ]

    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex="col")

    for col, (title, traces, color) in enumerate(methods):
        if traces is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=12)
                if row == 0:
                    axes[row, col].set_title(title, fontsize=9)
            continue

        lam = traces["lam_vectors"]  # [T, N]
        T_steps, N = lam.shape
        t_axis = np.arange(T_steps)

        final_lam = lam[-1]
        threshold = np.quantile(final_lam, 1.0 - hard_frac)
        hard_idx = np.where(final_lam >= threshold)[0]
        easy_idx = np.where(final_lam < threshold)[0]
        n_easy = min(n_lines, len(easy_idx))
        n_hard = min(n_lines, len(hard_idx))
        easy_sel = rng.choice(easy_idx, n_easy, replace=False) if n_easy > 0 else []
        hard_sel = rng.choice(hard_idx, n_hard, replace=False) if n_hard > 0 else []

        for row, (sel, row_label) in enumerate([
            (easy_sel, "Easy"),
            (hard_sel, "Hard"),
        ]):
            ax = axes[row, col]
            cmap = plt.cm.tab10
            for idx, j in enumerate(sel):
                c = cmap(idx % 10)
                ax.plot(t_axis, lam[:, j], color=c, alpha=0.7, lw=0.9)
                lam_avg_j = float(lam[:, j].mean())
                ax.axhline(lam_avg_j, color=c, ls=":", alpha=0.5, lw=0.7)
            ax.grid(alpha=0.2, linewidth=0.5)
            ax.tick_params(axis="both", labelsize=7)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=4)
            if row == 1:
                ax.set_xlabel("step", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{row_label}" + r"  $\lambda_j$", fontsize=8)

    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    fig.savefig(out_dir / "fig_lam_study_dual_panel.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_lam_study_dual_panel.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_lam_study_dual_panel.png'}")


def fig_lam_study_violations(traces_dict, out_dir):
    """Mean violation over steps for 5 MC ceiling lambda variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        ("PD from 0", traces_dict.get("original"), "#2c7fb8", "-"),
        (r"Warm $\lambda_T$ + PD", traces_dict.get("warm_lamfinal"), "#d95f02", "-"),
        (r"Warm $\bar\lambda$ + PD", traces_dict.get("warm_lamavg"), "#7570b3", "-"),
        (r"Fixed $\lambda_T$", traces_dict.get("fix_lamfinal"), "#e41a1c", "--"),
        (r"Fixed $\bar\lambda$", traces_dict.get("fix_lamavg"), "#4daf4a", "--"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    for label, traces, color, ls in methods:
        if traces is None:
            continue
        viol = traces["violations"]  # [T, N]
        mean_viol = viol.mean(axis=1)  # [T]
        t_axis = np.arange(len(mean_viol))
        ax.plot(t_axis, mean_viol, color=color, ls=ls, lw=1.5, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean violation", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_lam_study_violations.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_lam_study_violations.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_lam_study_violations.png'}")


def fig_lam_study_returns(traces_dict, out_dir):
    """Mean return over steps for 5 MC ceiling lambda variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        ("PD from 0", traces_dict.get("original"), "#2c7fb8", "-"),
        (r"Warm $\lambda_T$ + PD", traces_dict.get("warm_lamfinal"), "#d95f02", "-"),
        (r"Warm $\bar\lambda$ + PD", traces_dict.get("warm_lamavg"), "#7570b3", "-"),
        (r"Fixed $\lambda_T$", traces_dict.get("fix_lamfinal"), "#e41a1c", "--"),
        (r"Fixed $\bar\lambda$", traces_dict.get("fix_lamavg"), "#4daf4a", "--"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    for label, traces, color, ls in methods:
        if traces is None or "returns" not in traces:
            continue
        ret = traces["returns"]  # [T]
        t_axis = np.arange(len(ret))
        ax.plot(t_axis, ret, color=color, ls=ls, lw=1.5, label=label)

    ax.set_ylim(None, 0.45)
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean return", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="best", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_lam_study_returns.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_lam_study_returns.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_lam_study_returns.png'}")


def mc_lambda_study_figures(args, data_list, out_dir, device):
    """Run MC ceiling lambda study traces and generate figures."""
    from pdi.diffusion.portfolio_energy_ddpm import PortfolioEnergyDDPM

    problem = data_list[0]["problem"]
    N = len(problem["mu"])
    normalize_c = not args.no_normalize
    ct = problem.get("constraint_type", "shortfall")
    pcfg_local = dict(PROBLEM_CONFIGS[args.size])

    def _mk(ds, lam0=0.0):
        return PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=problem["mu"], portfolio_Sigma=problem["Sigma"],
            portfolio_scenarios=problem["scenarios"],
            portfolio_risk_budgets=problem["budgets"],
            portfolio_alpha=float(problem["alpha"]),
            energy_mc_samples=args.K,
            inverse_beta=args.ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred", dual_step_size=ds,
            dual_lambda_init=lam0, dual_lambda_max=1e6,
            dual_lambda_decay=args.dual_lambda_decay,
            shared_lambda=True,
            normalize_constraints=normalize_c,
            constraint_type=ct,
            num_sectors=pcfg_local.get("num_sectors", 10),
        ).to(device)

    def _noop_dual(dual_lambda, ergodic_rates, context, t=None):
        return dual_lambda

    def _set_fixed_lam(s, lam_vec):
        s._dual_ascent_step = _noop_dual
        _lv = lam_vec.clone()
        def _fi(batch_size, num_nodes, device, dtype, init_value, _l=_lv):
            return _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        s._init_dual_lambda = _fi

    def _set_init_lam(s, lam_vec):
        _lv = lam_vec.clone()
        def _fi(batch_size, num_nodes, device, dtype, init_value, _l=_lv):
            return _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        s._init_dual_lambda = _fi

    data = _make_batch(args.B, N).to(device)
    shape = (args.B, 1, N, 1)

    # 1. Original PD from 0
    print("[lam_study] running original PD...")
    torch.manual_seed(args.seed)
    s1 = _mk(args.dual_step)
    _, tr1, _ = _sample_with_trace(s1, shape, device, data, [])
    lam_final = torch.tensor(tr1["lam_vectors"][-1], device=device)
    lam_avg = torch.tensor(tr1["lam_vectors"].mean(axis=0), device=device)
    print("  lam_final: mean=%.2f max=%.1f" % (lam_final.mean(), lam_final.max()))
    print("  lam_avg:   mean=%.2f max=%.1f" % (lam_avg.mean(), lam_avg.max()))

    # Tail-average experiment: fix λ at the time-average of the last X% of the trajectory
    tail_fracs = [0.10, 0.20, 0.30, 0.40, 0.50, 1.00]
    lam_all = tr1["lam_vectors"]  # [T, N] numpy
    T_total = lam_all.shape[0]

    from pdi.diffusion.portfolio_energy_ddpm import variance_contributions_torch
    scen_t = torch.tensor(problem["scenarios"], device=device, dtype=torch.float32)
    Sigma_t = torch.tensor(problem["Sigma"], device=device, dtype=torch.float32)
    bud_np = problem["budgets"]

    tail_results = {}  # frac -> (ret, feas, vio, neff)
    tail_traces = {}   # frac -> traces dict with violations, returns
    for frac in tail_fracs:
        start_idx = int(T_total * (1.0 - frac))
        lam_tail = torch.tensor(lam_all[start_idx:].mean(axis=0), device=device)
        print(f"[lam_study] tail {int(frac*100)}%: avg over steps {start_idx}-{T_total-1}, "
              f"mean={lam_tail.mean():.2f} max={lam_tail.max():.1f}")
        torch.manual_seed(args.seed)
        s_tail = _mk(args.dual_step)
        _set_fixed_lam(s_tail, lam_tail)
        w_tail, tr_tail, _ = _sample_with_trace(s_tail, shape, device, data, [])
        tail_traces[frac] = tr_tail
        w_t = torch.tensor(w_tail, device=device, dtype=torch.float32)
        c = variance_contributions_torch(w_t, Sigma_t)
        c_avg = c.mean(dim=0).cpu().numpy()
        ret_val = float((scen_t @ w_t.t()).mean().item())
        feas_val = float((c_avg <= bud_np).mean())
        vio_val = float(np.maximum(c_avg - bud_np, 0.0).mean())
        ent = -(w_tail * np.log(np.clip(w_tail, 1e-12, None))).sum(axis=1)
        neff_val = float(np.exp(ent).mean())
        tail_results[frac] = (ret_val, feas_val, vio_val, neff_val)
        print(f"  ret={ret_val:.3f} feas={feas_val:.2f} vio={vio_val:.2e} Neff={neff_val:.1f}")
        sys.stdout.flush()

    # Reference: original PD traces (already have tr1)
    ref_vio_final = float(np.maximum(
        tr1["constraint_vals"][-1].mean() - bud_np, 0.0).mean()) if "constraint_vals" in tr1 else 0
    w_ref_np = np.exp(tr1["returns"][-1]) if "returns" in tr1 else 0  # not needed
    ref_ret = float(tr1["returns"][-1]) if "returns" in tr1 else 0
    ref_vio = float(tr1["violations"][-1].mean()) if "violations" in tr1 else 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = sorted(tail_results.keys())
    rets = [tail_results[f][0] for f in fracs]
    vios = [tail_results[f][2] for f in fracs]
    pct = [int(f * 100) for f in fracs]

    # Figure 1: mean vio + return vs tail fraction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.1))

    ax1.plot(pct, vios, "o-", color="#2c7fb8", lw=1.5, markersize=4, label=r"Fixed $\bar\lambda_{\mathrm{tail}}$")
    ax1.axhline(ref_vio, color="gray", ls="--", lw=1.0, label="PD from 0")
    ax1.set_xlabel(r"Tail fraction (\%)", fontsize=8)
    ax1.set_ylabel("Mean violation", fontsize=8)
    ax1.set_yscale("log")
    ax1.tick_params(axis="both", labelsize=7)
    ax1.grid(alpha=0.2, linewidth=0.5)
    ax1.legend(fontsize=6, framealpha=0.8)

    ax2.plot(pct, rets, "s-", color="#d95f02", lw=1.5, markersize=4, label=r"Fixed $\bar\lambda_{\mathrm{tail}}$")
    ax2.axhline(ref_ret, color="gray", ls="--", lw=1.0, label="PD from 0")
    ax2.set_xlabel(r"Tail fraction (\%)", fontsize=8)
    ax2.set_ylabel("Return", fontsize=8)
    ax2.tick_params(axis="both", labelsize=7)
    ax2.grid(alpha=0.2, linewidth=0.5)
    ax2.legend(fontsize=6, framealpha=0.8)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_tail_avg_lambda.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_tail_avg_lambda.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_tail_avg_lambda.png'}")

    # Figure 2: mean violation over diffusion steps per tail fraction
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    # PD reference
    vio_pd = tr1["violations"].mean(axis=1)
    ax.plot(np.arange(len(vio_pd)), vio_pd, color="gray", ls="--", lw=1.2, label="PD from 0")
    for i, frac in enumerate(fracs):
        tr = tail_traces[frac]
        vio_t = tr["violations"].mean(axis=1)
        ax.plot(np.arange(len(vio_t)), vio_t, color=cmap(i / max(len(fracs)-1, 1)),
                lw=1.3, label=f"tail {int(frac*100)}\\%")
    ax.set_yscale("log")
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean violation", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_tail_vio_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_tail_vio_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_tail_vio_trajectory.png'}")

    # Figure 3: return over diffusion steps per tail fraction
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    ret_pd = tr1["returns"]
    ax.plot(np.arange(len(ret_pd)), ret_pd, color="gray", ls="--", lw=1.2, label="PD from 0")
    for i, frac in enumerate(fracs):
        tr = tail_traces[frac]
        ret_t = tr["returns"]
        ax.plot(np.arange(len(ret_t)), ret_t, color=cmap(i / max(len(fracs)-1, 1)),
                lw=1.3, label=f"tail {int(frac*100)}\\%")
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean return", fontsize=8)
    ax.set_ylim(None, 0.45)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="best", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_tail_ret_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_tail_ret_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_tail_ret_trajectory.png'}")


def mc_beta_study_figures(args, data_list, out_dir, device):
    """Sweep ib (inverse beta) with everything else fixed. Trace vio + return over steps."""
    from pdi.diffusion.portfolio_energy_ddpm import (
        PortfolioEnergyDDPM, variance_contributions_torch,
    )
    problem = data_list[0]["problem"]
    N = len(problem["mu"])
    normalize_c = not args.no_normalize
    ct = problem.get("constraint_type", "shortfall")
    pcfg_local = dict(PROBLEM_CONFIGS[args.size])
    shape = (args.B, 1, N, 1)

    def _mk(ib):
        return PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=problem["mu"], portfolio_Sigma=problem["Sigma"],
            portfolio_scenarios=problem["scenarios"],
            portfolio_risk_budgets=problem["budgets"],
            portfolio_alpha=float(problem["alpha"]),
            energy_mc_samples=args.K,
            inverse_beta=ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred", dual_step_size=args.dual_step,
            dual_lambda_init=0.0, dual_lambda_max=1e6,
            dual_lambda_decay=args.dual_lambda_decay,
            shared_lambda=True,
            normalize_constraints=normalize_c,
            constraint_type=ct,
            num_sectors=pcfg_local.get("num_sectors", 10),
        ).to(device)

    ib_values = [100, 200, 300, 400, 500, 600, 800]
    data = _make_batch(args.B, N).to(device)
    traces_by_ib = {}

    for ib in ib_values:
        print(f"[beta_study] ib={ib} ...")
        torch.manual_seed(args.seed)
        s = _mk(ib)
        _, tr, _ = _sample_with_trace(s, shape, device, data, [])
        traces_by_ib[ib] = tr
        vio_final = float(tr["violations"][-1].mean())
        ret_final = float(tr["returns"][-1])
        print(f"  vio={vio_final:.2e} ret={ret_final:.3f}")
        sys.stdout.flush()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis

    # Figure 1: violation trajectory
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    for i, ib in enumerate(ib_values):
        vio = traces_by_ib[ib]["violations"].mean(axis=1)
        ax.plot(np.arange(len(vio)), vio, color=cmap(i / max(len(ib_values)-1, 1)),
                lw=1.3, label=f"$\\beta^{{-1}}={ib}$")
    ax.set_yscale("log")
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean violation", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_beta_vio_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_beta_vio_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_beta_vio_trajectory.png'}")

    # Figure 2: return trajectory
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    for i, ib in enumerate(ib_values):
        ret = traces_by_ib[ib]["returns"]
        ax.plot(np.arange(len(ret)), ret, color=cmap(i / max(len(ib_values)-1, 1)),
                lw=1.3, label=f"$\\beta^{{-1}}={ib}$")
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean return", fontsize=8)
    ax.set_ylim(None, 0.45)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="best", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_beta_ret_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_beta_ret_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_beta_ret_trajectory.png'}")


def mc_schedule_study_figures(args, data_list, out_dir, device):
    """Schedule comparison: cosine, linear, exp, poly2 at fixed ds. Trace vio + return."""
    from pdi.diffusion.portfolio_energy_ddpm import (
        PortfolioEnergyDDPM, variance_contributions_torch,
    )
    problem = data_list[0]["problem"]
    N = len(problem["mu"])
    normalize_c = not args.no_normalize
    ct = problem.get("constraint_type", "shortfall")
    pcfg_local = dict(PROBLEM_CONFIGS[args.size])
    shape = (args.B, 1, N, 1)
    data = _make_batch(args.B, N).to(device)

    def _mk(ds):
        return PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=problem["mu"], portfolio_Sigma=problem["Sigma"],
            portfolio_scenarios=problem["scenarios"],
            portfolio_risk_budgets=problem["budgets"],
            portfolio_alpha=float(problem["alpha"]),
            energy_mc_samples=args.K,
            inverse_beta=args.ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred", dual_step_size=ds,
            dual_lambda_init=0.0, dual_lambda_max=1e6,
            dual_lambda_decay=args.dual_lambda_decay,
            shared_lambda=True,
            normalize_constraints=normalize_c,
            constraint_type=ct,
            num_sectors=pcfg_local.get("num_sectors", 10),
        ).to(device)

    schedules = ["cosine", "ddpm_linear", "linear", "poly2"]
    traces_by_sched = {}

    for sname in schedules:
        print(f"[schedule_study] {sname} ...")
        ac = _build_alphas_cumprod(sname, args.T)
        torch.manual_seed(args.seed)
        s = _mk(args.dual_step)
        _override_schedule(s, ac)
        _, tr, _ = _sample_with_trace(s, shape, device, data, [])
        traces_by_sched[sname] = tr
        vio_final = float(tr["violations"][-1].mean())
        ret_final = float(tr["returns"][-1])
        print(f"  vio={vio_final:.2e} ret={ret_final:.3f}")
        sys.stdout.flush()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sched_colors_list = [SCHED_COLORS.get(s, "gray") for s in schedules]

    # Figure 1: violation trajectory
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    for sname, color in zip(schedules, sched_colors_list):
        vio = traces_by_sched[sname]["violations"].mean(axis=1)
        ax.plot(np.arange(len(vio)), vio, color=color, lw=1.3,
                label=SCHED_NAMES.get(sname, sname))
    ax.set_yscale("log")
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean violation", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_schedule_vio_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_schedule_vio_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_schedule_vio_trajectory.png'}")

    # Figure 2: return trajectory
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.1))
    for sname, color in zip(schedules, sched_colors_list):
        ret = traces_by_sched[sname]["returns"]
        ax.plot(np.arange(len(ret)), ret, color=color, lw=1.3,
                label=SCHED_NAMES.get(sname, sname))
    ax.set_xlabel("Reverse step", fontsize=8)
    ax.set_ylabel("Mean return", fontsize=8)
    ax.set_ylim(None, 0.45)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.5, loc="best", framealpha=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_schedule_ret_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig_schedule_ret_trajectory.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_schedule_ret_trajectory.png'}")


# ---------------------------------------------------------------
# New paper figures (matching synthetic pipeline)
# ---------------------------------------------------------------

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
})

SCHED_NAMES = {
    "ddpm_linear": r"Linear $\beta$ (DDPM)",
    "linear": "Linear SNR",
    "poly2": r"Polynomial ($p=2$)",
    "exp": "Exponential",
    "cosine": "Cosine",
}
SCHED_COLORS = {
    "ddpm_linear": "#d62728",
    "linear": "#1f77b4",
    "poly2": "#2ca02c",
    "exp": "#ff7f0e",
    "cosine": "#9467bd",
}


def _build_alphas_cumprod(name: str, T: int,
                          snr_min: float = 1e-3, snr_max: float = 1e3):
    k = torch.arange(T, dtype=torch.float64)
    tau = (T - 1 - k) / (T - 1)
    if name == "linear":
        snr = snr_min + (snr_max - snr_min) * tau
    elif name == "poly2":
        snr = snr_min + (snr_max - snr_min) * tau.pow(2)
    elif name == "exp":
        log_snr = math.log(snr_min) + tau * (math.log(snr_max) - math.log(snr_min))
        snr = torch.exp(log_snr)
    elif name == "ddpm_linear":
        beta_min, beta_max = 1e-4, 0.02
        betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float64)
        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0).clamp(1e-8, 1.0 - 1e-8).float()
    elif name == "cosine":
        s = 0.008
        steps = T + 1
        u = torch.linspace(0, T, steps, dtype=torch.float64)
        f = torch.cos((u / T + s) / (1 + s) * math.pi / 2).pow(2)
        ab_raw = f / f[0]
        betas = (1.0 - ab_raw[1:] / ab_raw[:-1]).clamp(1e-8, 0.999)
        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0).clamp(1e-8, 1.0 - 1e-8).float()
    else:
        raise ValueError(f"unknown schedule: {name}")
    alpha_bar = (snr / (1.0 + snr)).clamp(1e-8, 1.0 - 1e-8)
    return alpha_bar.float()


def _override_schedule(sampler, alphas_cumprod):
    device = sampler.alphas_cumprod.device
    ac = alphas_cumprod.to(device=device, dtype=sampler.alphas_cumprod.dtype)
    alphas = torch.empty_like(ac)
    alphas[0] = ac[0]
    alphas[1:] = ac[1:] / ac[:-1].clamp_min(1e-12)
    alphas = alphas.clamp(1e-8, 1.0)
    betas = (1.0 - alphas).clamp(1e-8, 0.999)
    ac_prev = torch.cat([torch.ones_like(ac[:1]), ac[:-1]], dim=0)
    sampler.betas.copy_(betas)
    sampler.alphas.copy_(alphas)
    sampler.alphas_cumprod.copy_(ac)
    if hasattr(sampler, "alphas_cumprod_prev"):
        sampler.alphas_cumprod_prev.copy_(ac_prev)
    sampler.sqrt_alphas_cumprod.copy_(torch.sqrt(ac))
    if hasattr(sampler, "sqrt_one_minus_alphas_cumprod"):
        sampler.sqrt_one_minus_alphas_cumprod.copy_(torch.sqrt(1.0 - ac))
    if hasattr(sampler, "sqrt_recip_alphas"):
        sampler.sqrt_recip_alphas.copy_(torch.sqrt(1.0 / alphas))
    posterior_variance = betas * (1.0 - ac_prev) / (1.0 - ac).clamp_min(1e-12)
    if hasattr(sampler, "posterior_variance"):
        sampler.posterior_variance.copy_(posterior_variance)
    if hasattr(sampler, "posterior_log_variance_clipped"):
        sampler.posterior_log_variance_clipped.copy_(
            torch.log(posterior_variance.clamp_min(1e-20)))
    if hasattr(sampler, "posterior_mean_coef1"):
        sampler.posterior_mean_coef1.copy_(
            betas * torch.sqrt(ac_prev) / (1.0 - ac).clamp_min(1e-12))
    if hasattr(sampler, "posterior_mean_coef2"):
        sampler.posterior_mean_coef2.copy_(
            (1.0 - ac_prev) * torch.sqrt(alphas) / (1.0 - ac).clamp_min(1e-12))


def fig_dual_ced_vs_mc(mc_traces, ced_traces, out_dir):
    """Side-by-side lambda norm: CED-ceiling (MC) vs CED (trained)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for label, trc, color in [
        ("CED-ceiling", mc_traces, "#4a90d9"),
        ("CED (ours)", ced_traces, "#d9534f"),
    ]:
        if trc is None:
            continue
        lam = trc["lam_vectors"]  # [T, N]
        norms = np.linalg.norm(lam, axis=1)
        step = np.arange(len(norms))
        ax.plot(step, norms, color=color, lw=2.5, label=label)
    ax.set_xlabel(r"Diffusion step $t$", fontsize=22)
    ax.set_ylabel(r"$\|\boldsymbol{\lambda}\|_2$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=16, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_dual_ced_vs_mc.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_dual_ced_vs_mc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_dual_ced_vs_mc.png'}")


def fig_violation_ced_vs_mc(mc_traces, ced_traces, out_dir):
    """Max violation, dual norm, and complementary slackness (log scale)."""
    fig, (ax_max, ax_dual, ax_cs) = plt.subplots(1, 3, figsize=(21, 4))
    floor = 1e-6
    for label, trc, color in [
        ("CED-ceiling", mc_traces, "#4a90d9"),
        ("CED (ours)", ced_traces, "#d9534f"),
    ]:
        if trc is None:
            continue
        viol = trc["violations"]  # [T, N]
        lam = trc["lam_vectors"]  # [T, N]
        step = np.arange(viol.shape[0])
        cs = np.abs(lam * viol).mean(axis=1)
        lam_norm = np.linalg.norm(lam, axis=1)
        ax_max.plot(step, np.clip(viol.max(axis=1), floor, None),
                    color=color, lw=3.0, label=label)
        ax_dual.plot(step, lam_norm,
                     color=color, lw=3.0, label=label)
        ax_cs.plot(step, np.clip(cs, floor, None),
                   color=color, lw=3.0, label=label)
    for ax, ylabel, use_log in [
        (ax_max, r"$\max_j\; [v_j(t)]_+$", True),
        (ax_dual, r"$\|\boldsymbol{\lambda}\|_2$", False),
        (ax_cs, r"$\mathrm{mean}_j\; |\lambda_j \cdot v_j|$", True),
    ]:
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel(r"Diffusion step $t$", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=16, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_violation_ced_vs_mc.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_violation_ced_vs_mc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_violation_ced_vs_mc.png'}")


def fig_schedule_sweep(problem, B, T, K, ib, dual_step, device, seed, out_dir,
                       beta_schedule="cosine"):
    """Dual evolution under 4 noise schedules (MC ceiling only)."""
    mu_np, Sigma_np, scen_np, bud_np, alpha_np = problem
    N = len(mu_np)
    schedules = ["linear", "poly2", "exp", "cosine"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for sname in schedules:
        sampler = PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=T, beta_schedule="cosine",
            portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
            portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_np,
            portfolio_alpha=alpha_np,
            energy_mc_samples=K, inverse_beta=ib,
            inverse_beta_schedule="constant",
            dual_update_mode="x0_pred", dual_step_size=dual_step,
            dual_lambda_init=0.0, dual_lambda_max=1e6,
            shared_lambda=True,
        ).to(device)
        ac = _build_alphas_cumprod(sname, T)
        _override_schedule(sampler, ac)
        data = _make_batch(B, N).to(device)
        torch.manual_seed(seed)
        _, traces, _ = _sample_with_trace(sampler, (B, 1, N, 1), device, data, [])
        norms = np.linalg.norm(traces["lam_vectors"], axis=1)
        ax.plot(np.arange(len(norms)), norms, color=SCHED_COLORS[sname], lw=2.5,
                label=SCHED_NAMES[sname])
    ax.set_xlabel(r"Diffusion step $t$", fontsize=22)
    ax.set_ylabel(r"$\|\boldsymbol{\lambda}\|_2$", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=16, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_schedule_sweep.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_schedule_sweep.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_schedule_sweep.png'}")


def fig_t0_effect(problem, B, T, K, ib, dual_step, device, seed, out_dir,
                  t0_values=None):
    """Feasibility (O2csat) vs T0 warmup delay, one curve per noise schedule."""
    if t0_values is None:
        t0_values = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    mu_np, Sigma_np, scen_np, bud_np, alpha_np = problem
    N = len(mu_np)
    schedules = ["linear", "poly2", "exp", "cosine"]
    scen_t = torch.tensor(scen_np, device=device, dtype=torch.float32)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for sname in schedules:
        feas_by_t0 = []
        for t0 in t0_values:
            sampler = PortfolioEnergyDDPM(
                model=_NoOp(), num_timesteps=T, beta_schedule="cosine",
                portfolio_mu=mu_np, portfolio_Sigma=Sigma_np,
                portfolio_scenarios=scen_np, portfolio_risk_budgets=bud_np,
                portfolio_alpha=alpha_np,
                energy_mc_samples=K, inverse_beta=ib,
                inverse_beta_schedule="constant",
                dual_update_mode="x0_pred", dual_step_size=dual_step,
                dual_lambda_init=0.0, dual_lambda_max=1e6,
                shared_lambda=True,
            ).to(device)
            ac = _build_alphas_cumprod(sname, T)
            _override_schedule(sampler, ac)
            orig_dual = sampler._dual_ascent_step
            threshold_k = T - 1 - t0
            def _gated(self, dual_lambda, ergodic_rates, context, t=None,
                       _thresh=threshold_k, _orig=orig_dual):
                if t is not None and t.numel() > 0 and int(t[0].item()) > _thresh:
                    return dual_lambda
                return _orig(dual_lambda, ergodic_rates, context, t)
            sampler._dual_ascent_step = types.MethodType(_gated, sampler)

            data = _make_batch(B, N).to(device)
            torch.manual_seed(seed)
            z = sampler.sample(shape=(B, 1, N, 1), device=device, data=data)
            w = sampler.z_to_portfolio_weights(z)
            from pdi.diffusion.portfolio_energy_ddpm import shortfall_contributions_torch
            c = shortfall_contributions_torch(w, scen_t, alpha_np)
            c_mean = c.mean(dim=0).cpu().numpy()
            csat = float((c_mean <= bud_np + 0.02 * bud_np).mean())
            feas_by_t0.append(csat)
            print(f"[{sname:8s} T0={t0:3d}] O2csat(2%)={csat:.2f}")
        ax.plot(t0_values, feas_by_t0, "-o", color=SCHED_COLORS[sname],
                lw=2.5, markersize=6, label=SCHED_NAMES[sname])

    ax.axhline(1.0, ls="--", color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"warmup $T_0$", fontsize=22)
    ax.set_ylabel(r"avg-constraint feasibility (2\% tol)", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=14, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_t0_effect.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_t0_effect.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_t0_effect.png'}")


CDF_METHODS = ["pd_langevin", "mc_teacher", "ced_trained"]


def fig_constraint_cdf(data_list, out_dir, instance_idx=0,
                       methods=None):
    """CDF of per-constraint relative violations for PDL and CED only."""
    if methods is None:
        methods = CDF_METHODS
    d = data_list[instance_idx] if isinstance(data_list, list) else data_list
    scen = d["problem"]["scenarios"]
    bud = d["problem"]["budgets"]
    alpha = d["problem"]["alpha"]

    methods_found = [m for m in methods if m in d["weights"]]
    cdf_colors = {"pd_langevin": "#4a90d9", "mc_teacher": "#e67e22",
                  "ced_trained": "#d9534f"}

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for m in methods_found:
        lbl = METHOD_LABELS.get(m, m).replace("\n", " ")
        w = d["weights"][m].cpu().numpy()
        c = _compute_contributions_np(w, d["problem"])  # [B, N]
        c_mean = c.mean(axis=0)  # [N] avg-constraint
        vio = c_mean - bud
        sorted_v = np.sort(vio)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.step(sorted_v, cdf, where="post",
                color=cdf_colors.get(m, "gray"), lw=2.5, label=lbl)
    ax.axvline(0.0, color="k", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel(r"avg-constraint violation $\bar{c}_j - b_j$", fontsize=22)
    ax.set_ylabel("CDF", fontsize=24)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_constraint_cdf.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "fig_constraint_cdf.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'fig_constraint_cdf.png'}")


def trace_extra_figures(data_list, out_dir, args, device, n_lam_instances=5):
    """Trace BOTH CED reverse SDE and PD-Langevin to produce figs 4 & 5.

    Fig 4 (sample histogram evolution): instance 0 only.
    Fig 5 (dual evolution): up to `n_lam_instances` instances overlaid.
    """
    from pdi.diffusion.portfolio_energy_ddpm import (
        PortfolioEnergyDDPM,
    )

    snap_fracs = sorted(float(x) for x in args.snap_frac.split(","))
    # CED: t_int = (T-1)*(1-frac). frac=0 → noise side (t_int=T-1), frac=1 → data (t_int=0).
    ced_snap_t = sorted({int(round((1.0 - f) * (args.T - 1))) for f in snap_fracs},
                          reverse=True)
    ced_t_to_disp = {t: int(round((args.T - 1) - t)) for t in ced_snap_t}
    # PDL: iter_idx = (num_iters-1)*frac (already in display order).
    num_iters = args.T
    pdl_snap_t = sorted({int(round(f * (num_iters - 1))) for f in snap_fracs})
    pdl_t_to_disp = {t: t for t in pdl_snap_t}

    ced_lams, pdl_lams = [], []
    ced_snaps0, pdl_snaps0 = None, None
    K_lam = min(n_lam_instances, len(data_list))

    for i in range(K_lam):
        problem = data_list[i]["problem"]
        N = len(problem["mu"])
        mu_t = torch.tensor(problem["mu"], dtype=torch.float32, device=device)
        Sigma_t = torch.tensor(problem["Sigma"], dtype=torch.float32, device=device)
        scen_t = torch.tensor(problem["scenarios"], dtype=torch.float32, device=device)
        bud_t = torch.tensor(problem["budgets"], dtype=torch.float32, device=device)
        alpha = float(problem["alpha"])

        torch.manual_seed(args.seed)
        normalize_c = not args.no_normalize
        ct = problem.get("constraint_type", "shortfall")
        pcfg_local = dict(PROBLEM_CONFIGS[args.size])
        sampler = PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=problem["mu"], portfolio_Sigma=problem["Sigma"],
            portfolio_scenarios=problem["scenarios"],
            portfolio_risk_budgets=problem["budgets"],
            portfolio_alpha=alpha,
            energy_mc_samples=args.K,
            inverse_beta=args.ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=args.dual_step,
            dual_lambda_init=0.0, dual_lambda_max=1e6,
            dual_lambda_decay=args.dual_lambda_decay,
            shared_lambda=True,
            normalize_constraints=normalize_c,
            constraint_type=ct,
            num_sectors=pcfg_local.get("num_sectors", 10),
        ).to(device)
        data = _make_batch(args.B, N).to(device)
        shape = (args.B, 1, N, 1)
        _, traces_mc, snaps_ced = _sample_with_trace(
            sampler, shape, device, data, ced_snap_t,
        )
        ced_lams.append(traces_mc["lam_mean"])

        constraint_type = problem.get("constraint_type", "shortfall")
        _, traces_pdl, snaps_pdl = _pd_langevin_with_trace(
            mu_t, Sigma_t, scen_t, bud_t, alpha,
            B=args.B, num_iters=num_iters,
            primal_lr=args.pdl_primal_lr, dual_lr=args.pdl_dual_lr,
            noise_scale=args.pdl_noise_scale,
            snap_iters=pdl_snap_t, device=device, seed=args.seed,
            constraint_type=constraint_type,
        )
        pdl_lams.append(traces_pdl["lam_mean"])

        if i == 0:
            ced_snaps0 = snaps_ced
            pdl_snaps0 = snaps_pdl
            mc_traces_0 = traces_mc
            pdl_traces_0 = traces_pdl
            bud_np_0 = problem["budgets"]

    # CED-trained trace (instance 0 only, if checkpoint available)
    ced_trained_traces_0 = None
    if args.ced_ckpt is not None and Path(args.ced_ckpt).exists():
        import types as _types_tr
        problem0 = data_list[0]["problem"]
        N0 = len(problem0["mu"])
        ct0 = problem0.get("constraint_type", "shortfall")
        normalize_c = not args.no_normalize

        sampler_tr = PortfolioEnergyDDPM(
            model=_NoOp(), num_timesteps=args.T, beta_schedule=args.beta_schedule,
            portfolio_mu=problem0["mu"], portfolio_Sigma=problem0["Sigma"],
            portfolio_scenarios=problem0["scenarios"],
            portfolio_risk_budgets=problem0["budgets"],
            portfolio_alpha=float(problem0["alpha"]),
            energy_mc_samples=args.K,
            inverse_beta=args.ib, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=args.dual_step,
            dual_lambda_init=0.0, dual_lambda_max=1e6,
            dual_lambda_decay=args.dual_lambda_decay,
            shared_lambda=True,
            normalize_constraints=normalize_c,
            constraint_type=ct0,
        ).to(device)

        _use_gnn = args.backbone == "gnn"
        if _use_gnn:
            from pdi.models.portfolio_gnn_backbone import (
                PortfolioGNNBackbone, build_dense_adjacency,
            )
            bb = PortfolioGNNBackbone(d=N0, hidden=args.hidden,
                                       num_layers=args.num_layers,
                                       num_timesteps=args.T, K=args.tagconv_K)
            Sigma_np0 = problem0["Sigma"]
            _dense_A = torch.tensor(build_dense_adjacency(Sigma_np0, top_k=20),
                                     dtype=torch.float32)
        else:
            bb = PortfolioScoreBackbone(d=N0, hidden=args.hidden,
                                         num_layers=args.num_layers,
                                         num_timesteps=args.T, cond_channels=1)
            _dense_A = None

        score_net_tr = ScoreNetWithLambda(backbone=bb, expected_cond_feats=0)
        ckpt = torch.load(args.ced_ckpt, map_location=device)
        score_net_tr.load_state_dict(ckpt)
        score_net_tr.to(device).eval()

        _tr_score_fn = _make_trained_score_estimator(
            score_net_tr, sampler_tr.alphas_cumprod, dense_A=_dense_A)
        sampler_tr._estimate_score = _types_tr.MethodType(_tr_score_fn, sampler_tr)

        data_tr = _make_batch(args.B, N0).to(device)
        torch.manual_seed(args.seed)
        _, ced_trained_traces_0, _ = _sample_with_trace(
            sampler_tr, (args.B, 1, N0, 1), device, data_tr, [],
        )
        print("[trace] CED-trained trace done")

    plot_sample_evolution_compare(ced_snaps0, pdl_snaps0,
                                    ced_t_to_disp, pdl_t_to_disp, out_dir)
    plot_dual_evolution_compare(ced_lams, pdl_lams, out_dir)

    fig_constraint_cdf(data_list, out_dir)

    if mc_traces_0 is not None:
        fig_dual_panel(mc_traces_0, ced_trained_traces_0, pdl_traces_0,
                       bud_np_0, out_dir)

    if mc_traces_0 is not None:
        fig_dual_ced_vs_mc(mc_traces_0, None, out_dir)
        fig_violation_ced_vs_mc(mc_traces_0, None, out_dir)
        problem_tuple = (
            data_list[0]["problem"]["mu"],
            data_list[0]["problem"]["Sigma"],
            data_list[0]["problem"]["scenarios"],
            data_list[0]["problem"]["budgets"],
            data_list[0]["problem"]["alpha"],
        )
        if getattr(args, 'schedule_sweep', False):
            fig_schedule_sweep(problem_tuple, args.B, args.T, args.K,
                               args.ib, args.dual_step, device, args.seed, out_dir)
            fig_t0_effect(problem_tuple, args.B, args.T, args.K,
                          args.ib, args.dual_step, device, args.seed, out_dir)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(PROBLEM_CONFIGS.keys()), default="small")
    parser.add_argument("--ib", type=float, default=100.0)
    parser.add_argument("--dual-step", type=float, default=1000.0)
    parser.add_argument("--lam0", type=float, default=0.0)
    parser.add_argument("--sub-batch", type=int, default=0,
                        help="Split MC teacher into sub-batches with independent shared-lambda. 0=disabled.")
    parser.add_argument("--dual-lambda-decay", type=float, default=0.0,
                        help="Weight decay on lambda each step: lambda *= (1-decay) before update.")
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="Disable constraint normalization (use raw violations).")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--beta-schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--B", type=int, default=256)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "gnn"])
    parser.add_argument("--tagconv-K", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ced-ckpt", type=str, default=None)
    parser.add_argument("--calib-dir", type=str, default=None)
    parser.add_argument("--label", type=str, default="paper_figures")
    parser.add_argument("--eps-feas", type=float, default=0.1)
    parser.add_argument("--num-instances", type=int, default=1,
                        help="Number of problem instances (seeds 0..K-1).")
    parser.add_argument("--trace", action="store_true",
                        help="Trace CED reverse-SDE + PD-Langevin to produce sample/dual "
                              "evolution figures (figs 4 & 5).")
    parser.add_argument("--snap-frac", type=str, default="0.0,0.33,0.66,1.0",
                        help="Fractions in [0,1] (0=noise, 1=data) at which to snap weight "
                              "distributions for fig 4. Default = 4 timesteps.")
    parser.add_argument("--instance-idx", type=int, default=0,
                        help="Instance index to show for the per-instance figs (frontier, heatmap, "
                              "fig 4 sample evolution).")
    parser.add_argument("--trace-lam-instances", type=int, default=5,
                        help="Number of instances to overlay in fig 5 dual evolution.")
    parser.add_argument("--pdl-primal-lr", type=float, default=0.01,
                        help="PD-Langevin tuned primal lr (per-sample λ winner).")
    parser.add_argument("--pdl-dual-lr", type=float, default=100.0,
                        help="PD-Langevin tuned dual lr.")
    parser.add_argument("--pdl-noise-scale", type=float, default=0.01,
                        help="PD-Langevin tuned noise scale.")
    parser.add_argument("--dps-scale", type=float, default=1.0,
                        help="DPS guidance scale.")
    parser.add_argument("--dps-sweep", type=str, default=None,
                        help="Comma-separated DPS scale values for sweep.")
    parser.add_argument("--pdm-proj-lr", type=float, default=0.1,
                        help="PDM projection learning rate.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for cached data_list (default: outputs/portfolio/figures_cache/). "
                              "Cache key is derived from (size, num_instances, ib, dual_step, T, B, K, "
                              "seed). Figures always regenerate.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore any cached weights and recompute from scratch.")
    parser.add_argument("--include-mc-variants", action="store_true",
                        help="Also run unconstrained_mc, pdm, rejection (3 extra MC sampler "
                              "calls per instance, ~3x slower). Default off.")
    parser.add_argument("--ib-grid", type=str, default=None,
                        help="Comma-separated ib values for sweep mode (e.g. '0.03,0.1,0.3'). "
                              "If set together with --ds-grid, runs each (ib, ds) pair, "
                              "writing per-config caches and figures.")
    parser.add_argument("--ds-grid", type=str, default=None,
                        help="Comma-separated dual_step values for sweep mode.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="In sweep mode, run this many (ib, ds) configs concurrently as "
                              "subprocesses (each gets its own CUDA context).")
    parser.add_argument("--mc-only", action="store_true",
                        help="Run only mc_teacher + analytical baselines. Skip pd_langevin. "
                              "Useful for fast (ib, ds) sweeps focused on the MC sampler.")
    parser.add_argument("--mc-lambda-study", action="store_true",
                        help="Run MC ceiling lambda study (fixed/warm-start variants + uniform sweep).")
    parser.add_argument("--schedule-sweep", action="store_true",
                        help="Run schedule sweep and T0 effect figures (only with --trace).")
    parser.add_argument("--beta-study", action="store_true",
                        help="Sweep inverse_beta with fixed ds, trace vio+return over steps.")
    parser.add_argument("--study-only", action="store_true",
                        help="Skip all baselines, only run study flags.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override budget tightness gamma for the chosen --size preset.")
    parser.add_argument("--budget-type", type=str, default=None,
                        choices=["proportional", "uniform"],
                        help="Override budget type for the chosen --size preset.")
    parser.add_argument("--constraint-type", type=str, default=None,
                        help="Override constraint type (e.g. 'enriched').")
    parser.add_argument("--sector-gamma", type=float, default=1.5,
                        help="Sector exposure tightness for enriched constraints.")
    args = parser.parse_args()

    if args.gamma is not None:
        PROBLEM_CONFIGS[args.size]["gamma"] = args.gamma
    if args.budget_type is not None:
        PROBLEM_CONFIGS[args.size]["budget_type"] = args.budget_type
    if args.constraint_type is not None:
        PROBLEM_CONFIGS[args.size]["constraint_type"] = args.constraint_type
    if args.constraint_type == "enriched":
        PROBLEM_CONFIGS[args.size]["sector_gamma"] = args.sector_gamma

    # Auto-detect architecture params from checkpoint's summary.json
    if args.ced_ckpt is not None:
        summary_path = Path(args.ced_ckpt).parent / "summary.json"
        if summary_path.exists():
            import json as _json
            with open(summary_path) as _f:
                _cfg = _json.load(_f).get("config", {})
            _arch_keys = {"backbone": str, "hidden": int, "num_layers": int, "tagconv_K": int}
            _overrides = []
            for key, typ in _arch_keys.items():
                if key in _cfg:
                    old_val = getattr(args, key)
                    new_val = typ(_cfg[key])
                    if old_val != new_val:
                        setattr(args, key, new_val)
                        _overrides.append(f"{key}: {old_val} -> {new_val}")
            if _overrides:
                print(f"[ced-ckpt] auto-detected arch from {summary_path.name}: "
                      + ", ".join(_overrides))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("FIGURES — full parameter dump")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    pcfg_display = dict(PROBLEM_CONFIGS[args.size])
    print(f"  problem_config ({args.size}): {pcfg_display}")
    print("=" * 60)

    out_dir = experiment_dir(
        "score_net_eval",
        f"figures_{args.size}_ib{args.ib:g}_K{args.num_instances}_{args.label}",
    )
    print(f"figures out_dir: {out_dir}")

    # ---- Sweep mode: --ib-grid + --ds-grid -------------------------------
    if args.ib_grid is not None or args.ds_grid is not None:
        ibs = [float(x) for x in (args.ib_grid or str(args.ib)).split(",")]
        dss = [float(x) for x in (args.ds_grid or str(args.dual_step)).split(",")]
        configs = [(ib, ds) for ib in ibs for ds in dss]
        print(f"Sweep mode: {len(configs)} (ib, ds) configs over "
              f"{args.num_instances} instances each.  parallel={args.parallel}")

        cache_root = Path(args.cache_dir) if args.cache_dir else (
            _project_root() / "outputs" / "portfolio" / "figures_cache"
        )
        cache_root.mkdir(parents=True, exist_ok=True)

        # Parallel sweep: launch each config as a subprocess
        if args.parallel > 1:
            import subprocess
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _launch(ib, ds):
                cmd = [
                    "micromamba", "run", "-n", "gdiff",
                    "python", str(_THIS_DIR / "figures.py"),
                    "--size", args.size,
                    "--num-instances", str(args.num_instances),
                    "--ib", str(ib), "--dual-step", str(ds),
                    "--T", str(args.T), "--B", str(args.B), "--K", str(args.K),
                    "--label", f"{args.label}_ib{ib:g}_ds{ds:g}",
                    "--seed", str(args.seed),
                    "--eps-feas", str(args.eps_feas),
                ]
                if args.mc_only:
                    cmd.append("--mc-only")
                if args.include_mc_variants:
                    cmd.append("--include-mc-variants")
                if args.cache_dir:
                    cmd += ["--cache-dir", args.cache_dir]
                if args.ced_ckpt:
                    cmd += ["--ced-ckpt", args.ced_ckpt]
                print(f"[sweep] launching ib={ib:g} ds={ds:g}")
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                wall = time.time() - t0
                if proc.returncode != 0:
                    print(f"[sweep] FAILED ib={ib:g} ds={ds:g} ({wall:.1f}s)\n"
                          f"  stderr tail:\n{proc.stderr[-500:]}")
                else:
                    print(f"[sweep] done ib={ib:g} ds={ds:g} ({wall:.1f}s)")
                return ib, ds, proc.returncode, wall

            print(f"[sweep] running {len(configs)} configs with up to {args.parallel} concurrent...")
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futs = [ex.submit(_launch, ib, ds) for ib, ds in configs]
                for f in as_completed(futs):
                    f.result()

            # Aggregate sweep_summary by reading per-config cache files.
            sweep_summary = []
            for ib, ds in configs:
                args_cfg = argparse.Namespace(**vars(args))
                args_cfg.ib = ib; args_cfg.dual_step = ds
                cache_path_cfg = _cache_path(args_cfg)
                if not cache_path_cfg.exists():
                    sweep_summary.append({"ib": ib, "ds": ds, "missing": True})
                    continue
                dlist = _load_cache(cache_path_cfg, device)
                metrics = _per_instance_metrics(dlist[0]["weights"], dlist[0]["problem"],
                                                  PAPER_METHODS, eps_feas=args.eps_feas)
                # Aggregate Sharpe/O2csat/O2mxrv across instances for mc_teacher
                shr, csat, mxrv = [], [], []
                for d in dlist:
                    pim = _per_instance_metrics(d["weights"], d["problem"],
                                                  ["mc_teacher"], eps_feas=args.eps_feas)
                    if "mc_teacher" in pim:
                        shr.append(pim["mc_teacher"]["sharpe"])
                        csat.append(pim["mc_teacher"]["opt2_csat"])
                        mxrv.append(pim["mc_teacher"]["opt2_max_rel_vio"])
                if shr:
                    sweep_summary.append({
                        "ib": ib, "ds": ds, "cache": str(cache_path_cfg),
                        "mc_sharpe_mean": float(np.mean(shr)),
                        "mc_sharpe_std": float(np.std(shr)),
                        "mc_o2_csat_mean": float(np.mean(csat)),
                        "mc_o2_mxrv_mean": float(np.mean(mxrv)),
                    })
            sweep_path = cache_root / f"sweep_summary_{args.size}_K{args.num_instances}_{args.label}.json"
            sweep_path.write_text(json.dumps(sweep_summary, indent=2))
            print(f"\nParallel sweep summary: {sweep_path}")
            print(f"{'ib':>8s} {'ds':>8s} {'Sharpe':>8s} {'O2csat':>8s} {'O2mxrv':>8s}")
            for r in sweep_summary:
                if r.get("missing"):
                    print(f"{r['ib']:>8.4g} {r['ds']:>8.4g}     MISSING")
                    continue
                print(f"{r['ib']:>8.4g} {r['ds']:>8.4g} "
                      f"{r['mc_sharpe_mean']:>8.3f} {r['mc_o2_csat_mean']:>8.3f} "
                      f"{r['mc_o2_mxrv_mean']:>8.3f}")
            return

        sweep_summary = []
        for ib, ds in configs:
            args.ib = ib; args.dual_step = ds
            cfg_label = f"ib{ib:g}_ds{ds:g}"
            cfg_out = experiment_dir(
                "score_net_eval",
                f"figures_{args.size}_K{args.num_instances}_{cfg_label}_{args.label}",
            )
            cache_path_cfg = _cache_path(args)
            print(f"\n[sweep] === ({ib:g}, {ds:g}) -> {cfg_out.name}")
            data_list = None
            if not args.force_recompute:
                data_list = _load_cache(cache_path_cfg, device)
                if data_list is not None:
                    print(f"  loaded cache {cache_path_cfg}")
            if data_list is None:
                data_list = collect_multi_instance(
                    size=args.size, num_instances=args.num_instances,
                    ib=ib, dual_step=ds, T=args.T, B=args.B, K=args.K,
                    seed=args.seed, device=device,
                    ced_ckpt=Path(args.ced_ckpt) if args.ced_ckpt else None,
                    hidden=args.hidden, num_layers=args.num_layers,
                    include_mc_variants=args.include_mc_variants,
                )
                _save_cache(cache_path_cfg, data_list)
                print(f"  saved cache {cache_path_cfg}")
            metrics = plot_comparison_bars(data_list, cfg_out, eps_feas=args.eps_feas)
            (cfg_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_return_vs_violation(data_list, cfg_out, instance_idx=args.instance_idx)
            plot_weight_heatmap(data_list, cfg_out, instance_idx=args.instance_idx)
            mc = metrics.get("mc_teacher", {})
            sweep_summary.append({
                "ib": ib, "ds": ds, "out_dir": str(cfg_out),
                "mc_sharpe_mean": mc.get("sharpe", {}).get("mean"),
                "mc_o2_csat_mean": mc.get("opt2_csat", {}).get("mean"),
                "mc_o2_mxrv_mean": mc.get("opt2_max_rel_vio", {}).get("mean"),
            })
            _ss = sweep_summary[-1]
            _parts = []
            if _ss['mc_sharpe_mean'] is not None:
                _parts.append(f"Sharpe={_ss['mc_sharpe_mean']:.3f}")
            if _ss['mc_o2_csat_mean'] is not None:
                _parts.append(f"O2csat={_ss['mc_o2_csat_mean']:.3f}")
            if _ss['mc_o2_mxrv_mean'] is not None:
                _parts.append(f"O2mxrv={_ss['mc_o2_mxrv_mean']:.3f}")
            print(f"  mc_teacher: {'  '.join(_parts) if _parts else 'no metrics'}")

        sweep_path = (Path(args.cache_dir) if args.cache_dir
                       else _project_root() / "outputs" / "portfolio" / "figures_cache"
                       ) / f"sweep_summary_{args.size}_K{args.num_instances}_{args.label}.json"
        sweep_path.write_text(json.dumps(sweep_summary, indent=2))
        print(f"\nSweep summary saved to {sweep_path}")
        return

    # --study-only: skip baselines, build minimal data_list with problem info only
    if args.study_only:
        pcfg = dict(PROBLEM_CONFIGS[args.size])
        ct = pcfg.pop("constraint_type", "shortfall")
        mu_np, Sigma_np, scen_np, bud_np, alpha_np = make_portfolio_problem(
            constraint_type=ct, **pcfg)
        data_list = [{"weights": {}, "problem": {
            "mu": mu_np, "Sigma": Sigma_np, "scenarios": scen_np,
            "budgets": bud_np, "alpha": float(alpha_np),
            "constraint_type": ct,
        }, "lambda_traces": {}}]
        if args.mc_lambda_study:
            print("Running MC lambda study figures (study-only)...")
            mc_lambda_study_figures(args, data_list, out_dir, device)
        if args.schedule_sweep:
            print("Running schedule comparison (study-only)...")
            mc_schedule_study_figures(args, data_list, out_dir, device)
        if args.beta_study:
            print("Running beta study (study-only)...")
            mc_beta_study_figures(args, data_list, out_dir, device)
        print(f"\nAll figures saved to: {out_dir}")
        for f in sorted(out_dir.glob("*.png")):
            print(f"  {f.name}")
        return

    cache_path = _cache_path(args)
    data_list = None
    if not args.force_recompute:
        data_list = _load_cache(cache_path, device)
        if data_list is not None:
            print(f"Loaded cached weights from {cache_path}  (use --force-recompute to rebuild)")

    t0 = time.time()
    if data_list is None:
        if args.num_instances <= 1:
            data = collect_all_weights(
                size=args.size, ib=args.ib, dual_step=args.dual_step, T=args.T,
                B=args.B, K=args.K, seed=args.seed, device=device,
                ced_ckpt=Path(args.ced_ckpt) if args.ced_ckpt else None,
                hidden=args.hidden, num_layers=args.num_layers,
                include_mc_variants=args.include_mc_variants,
                mc_only=args.mc_only,
                beta_schedule=args.beta_schedule,
                pdl_primal_lr=args.pdl_primal_lr, pdl_dual_lr=args.pdl_dual_lr,
                pdl_noise_scale=args.pdl_noise_scale, lam0=args.lam0,
                sub_batch=args.sub_batch, dual_lambda_decay=args.dual_lambda_decay,
                normalize_constraints=not args.no_normalize,
                backbone=args.backbone, tagconv_K=args.tagconv_K,
                mc_lambda_study=args.mc_lambda_study,
                dps_scale=args.dps_scale, dps_sweep=args.dps_sweep,
                pdm_proj_lr=args.pdm_proj_lr,
            )
            data_list = [data]
        else:
            data_list = collect_multi_instance(
                size=args.size, num_instances=args.num_instances,
                ib=args.ib, dual_step=args.dual_step, T=args.T, B=args.B, K=args.K,
                seed=args.seed, device=device,
                ced_ckpt=Path(args.ced_ckpt) if args.ced_ckpt else None,
                hidden=args.hidden, num_layers=args.num_layers,
                include_mc_variants=args.include_mc_variants,
                mc_only=args.mc_only,
                beta_schedule=args.beta_schedule,
                pdl_primal_lr=args.pdl_primal_lr, pdl_dual_lr=args.pdl_dual_lr,
                pdl_noise_scale=args.pdl_noise_scale, lam0=args.lam0,
                sub_batch=args.sub_batch, dual_lambda_decay=args.dual_lambda_decay,
                normalize_constraints=not args.no_normalize,
                mc_lambda_study=args.mc_lambda_study,
                dps_scale=args.dps_scale, dps_sweep=args.dps_sweep,
                backbone=args.backbone, tagconv_K=args.tagconv_K,
                pdm_proj_lr=args.pdm_proj_lr,
            )
        _save_cache(cache_path, data_list)
        print(f"Saved weights cache to {cache_path}")
    wall = time.time() - t0
    method_names = list(data_list[0]["weights"].keys())
    print(f"collected weights in {wall:.1f}s over {len(data_list)} instance(s); "
          f"methods: {method_names}")

    # Save raw weights for reproducibility (first instance + optionally pooled)
    weight_arrays = {m: w.cpu().numpy() for m, w in data_list[0]["weights"].items()}
    np.savez(out_dir / "weights_seed0.npz", **weight_arrays)
    if len(data_list) > 1:
        pooled = {m: np.concatenate([d["weights"][m].cpu().numpy() for d in data_list
                                       if m in d["weights"]], axis=0)
                   for m in method_names}
        np.savez(out_dir / "weights_all_instances.npz", **pooled)

    print("\nGenerating figures...")
    metrics = plot_comparison_bars(data_list, out_dir, eps_feas=args.eps_feas)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_return_vs_violation(data_list, out_dir, instance_idx=args.instance_idx)
    plot_weight_heatmap(data_list, out_dir, instance_idx=args.instance_idx)
    fig_constraint_cdf(data_list, out_dir, instance_idx=args.instance_idx)
    plot_dual_trace(data_list, out_dir, instance_idx=args.instance_idx)

    if args.trace:
        print("\nTracing CED + PD-Langevin for evolution figures...")
        trace_extra_figures(data_list, out_dir, args, device,
                              n_lam_instances=args.trace_lam_instances)

    if args.mc_lambda_study:
        print("\nRunning MC lambda study figures...")
        mc_lambda_study_figures(args, data_list, out_dir, device)

    if args.schedule_sweep:
        print("\nRunning schedule comparison sweep...")
        mc_schedule_study_figures(args, data_list, out_dir, device)

    print(f"\nAll figures saved to: {out_dir}")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
