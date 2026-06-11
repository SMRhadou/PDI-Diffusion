#!/usr/bin/env python3
"""End-to-end portfolio experiment: MC baseline, train score net, evaluate.

Mirrors ``scripts/wra/diffusion/probe_option_a.py`` for the constrained
portfolio optimisation problem.  All data is synthetic (no external datasets).

Usage:
    python scripts/portfolio/run_portfolio_experiment.py --size small
    python scripts/portfolio/run_portfolio_experiment.py --size medium --ib 1.0
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

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _experiment_paths import experiment_dir, _project_root  # noqa: E402

# Ensure src/ is on the path
_SRC = str(_project_root() / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pdi.diffusion.portfolio_energy_ddpm import (  # noqa: E402
    PortfolioEnergyDDPM,
    make_portfolio_problem,
)
from pdi.models.portfolio_backbone import (  # noqa: E402
    PortfolioScoreBackbone,
)
from pdi.trainers.energy_score import (  # noqa: E402
    EnergyScoreTrainConfig,
    EnergyScoreTrainer,
    ScoreNetWithLambda,
)

# ---------------------------------------------------------------
# Problem instances
# ---------------------------------------------------------------

PROBLEM_CONFIGS = {
    "small": dict(N=50, K_factors=5, R_scenarios=200, gamma=1.0, seed=0),
    "medium": dict(N=100, K_factors=10, R_scenarios=500, gamma=1.5, seed=0),
    "large": dict(N=200, K_factors=20, R_scenarios=1000, gamma=1.0, seed=0),
}

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


class _NoOpModel(nn.Module):
    """Dummy backbone for MC-only sampler (model output is never used)."""
    def forward(self, x, timesteps, edge_index, edge_weight=None, cond=None,
                return_intermediates=False):
        del timesteps, edge_index, edge_weight, cond, return_intermediates
        return torch.zeros_like(x), None


def _make_trained_score_estimator(score_net, alphas_cumprod, N):
    """Return a bound method that replaces MC score estimation with the trained net."""
    # For portfolio there is no graph; use dummy edge_index
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


def _evaluate(z0, sampler, context, label, log):
    """Compute portfolio metrics from z-space samples."""
    with torch.no_grad():
        weights = sampler.z_to_portfolio_weights(z0)  # [B, N]
        risk = sampler._risk_contributions(weights, context.Sigma)  # [B, N]
        budgets = context.risk_budgets.unsqueeze(0)  # [1, N]
        violation = (risk - budgets).clamp_min(0.0)

        # Expected return (average over scenarios)
        ret_per_scenario = torch.matmul(context.scenarios, weights.t())  # [R, B]
        expected_return = ret_per_scenario.mean(dim=0)  # [B]

        # Portfolio variance: x^T Sigma x
        Sigma_x = torch.matmul(weights, context.Sigma)  # [B, N]
        port_var = (Sigma_x * weights).sum(dim=1)  # [B]

        # Feasibility: all risk contributions <= budget
        feas_per_sample = (risk <= budgets + 1e-8).all(dim=1).float()

        # Diversification: entropy of weights
        log_w = torch.log(weights.clamp_min(1e-12))
        entropy = -(weights * log_w).sum(dim=1)  # [B]

    metrics = {
        "label": label,
        "expected_return_mean": float(expected_return.mean().item()),
        "expected_return_std": float(expected_return.std().item()),
        "portfolio_var_mean": float(port_var.mean().item()),
        "mean_violation": float(violation.mean().item()),
        "max_violation": float(violation.max().item()),
        "feasibility_rate": float(feas_per_sample.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
    }
    log.info(
        "[%s]  ret=%.4f+/-%.4f  var=%.6f  vio=%.4f/%.4f  feas=%.3f  H=%.2f",
        label,
        metrics["expected_return_mean"], metrics["expected_return_std"],
        metrics["portfolio_var_mean"],
        metrics["mean_violation"], metrics["max_violation"],
        metrics["feasibility_rate"],
        metrics["entropy_mean"],
    )
    return metrics


def _baselines(context, N, device, log):
    """Compute analytical baselines."""
    results = []

    # 1. Equal weight (1/N)
    w_eq = torch.ones(1, N, device=device) / N
    metrics_eq = _evaluate_weights(w_eq, context, "equal_weight", log)
    results.append(metrics_eq)

    # 2. Risk parity: x_j proportional to 1/sigma_j
    sigma_diag = torch.sqrt(torch.diag(context.Sigma)).unsqueeze(0)  # [1, N]
    inv_sigma = 1.0 / sigma_diag.clamp_min(1e-12)
    w_rp = inv_sigma / inv_sigma.sum(dim=1, keepdim=True)
    metrics_rp = _evaluate_weights(w_rp, context, "risk_parity", log)
    results.append(metrics_rp)

    return results


def _evaluate_weights(weights, context, label, log):
    """Evaluate given portfolio weights [1, N] or [B, N]."""
    with torch.no_grad():
        risk = (torch.matmul(weights, context.Sigma) * weights)
        budgets = context.risk_budgets.unsqueeze(0)
        violation = (risk - budgets).clamp_min(0.0)
        ret_per = torch.matmul(context.scenarios, weights.t()).mean(dim=0)
        Sigma_x = torch.matmul(weights, context.Sigma)
        port_var = (Sigma_x * weights).sum(dim=1)
        feas = (risk <= budgets + 1e-8).all(dim=1).float()
        log_w = torch.log(weights.clamp_min(1e-12))
        entropy = -(weights * log_w).sum(dim=1)
    metrics = {
        "label": label,
        "expected_return_mean": float(ret_per.mean().item()),
        "expected_return_std": float(ret_per.std().item()),
        "portfolio_var_mean": float(port_var.mean().item()),
        "mean_violation": float(violation.mean().item()),
        "max_violation": float(violation.max().item()),
        "feasibility_rate": float(feas.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
    }
    log.info(
        "[%s]  ret=%.4f+/-%.4f  var=%.6f  vio=%.4f/%.4f  feas=%.3f  H=%.2f",
        label, metrics["expected_return_mean"], metrics["expected_return_std"],
        metrics["portfolio_var_mean"],
        metrics["mean_violation"], metrics["max_violation"],
        metrics["feasibility_rate"], metrics["entropy_mean"],
    )
    return metrics


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Portfolio experiment")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--ib", type=float, default=1.0, help="inverse_beta for train+infer")
    parser.add_argument("--T", type=int, default=500, help="diffusion timesteps")
    parser.add_argument("--beta-schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "sigmoid"],
                        help="Noise schedule (default: cosine)")
    parser.add_argument("--num-outer", type=int, default=200, help="training outer iterations")
    parser.add_argument("--batch-size", type=int, default=64, help="sampling batch size")
    parser.add_argument("--mc-samples", type=int, default=32, help="MC samples for training")
    parser.add_argument("--mc-samples-infer", type=int, default=64, help="MC samples for inference")
    parser.add_argument("--hidden", type=int, default=256, help="MLP hidden width")
    parser.add_argument("--num-layers", type=int, default=4, help="MLP depth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("portfolio.main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Generate problem
    pcfg = PROBLEM_CONFIGS[args.size]
    N = pcfg["N"]
    log.info("generating %s problem (N=%d, K=%d, R=%d, gamma=%.1f)",
             args.size, N, pcfg["K_factors"], pcfg["R_scenarios"], pcfg["gamma"])
    mu, Sigma, scenarios, risk_budgets = make_portfolio_problem(**pcfg)

    out_dir = experiment_dir(
        "score_net_train",
        f"{args.size}_ib{args.ib:g}_T{args.T}",
    )
    log.info("output dir: %s", out_dir)

    T = args.T
    IB = args.ib

    # ------------------------------------------------------------------
    # Build score net (MLP backbone)
    # ------------------------------------------------------------------
    # ScoreNetWithLambda adds 1 cond channel for lambda;
    # portfolio has no dataset conditioning, so expected_cond_feats=0
    backbone = PortfolioScoreBackbone(
        d=N, hidden=args.hidden, num_layers=args.num_layers,
        num_timesteps=T, cond_channels=1,  # lambda only
    )
    score_net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0)
    log.info("score_net params: %.2fM",
             sum(p.numel() for p in score_net.parameters()) / 1e6)

    # ------------------------------------------------------------------
    # MC sampler for training
    # ------------------------------------------------------------------
    mc_for_training = PortfolioEnergyDDPM(
        model=_NoOpModel(),
        num_timesteps=T, beta_schedule=args.beta_schedule,
        portfolio_mu=mu, portfolio_Sigma=Sigma,
        portfolio_scenarios=scenarios,
        portfolio_risk_budgets=risk_budgets,
        energy_mc_samples=args.mc_samples,
        inverse_beta=IB, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=0.5,
        dual_num_outer_iterations=1,
        dual_lambda_init=1.0,
        dual_lambda_max=100.0,
        min_energy_sigma=1e-4,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    cfg = EnergyScoreTrainConfig(
        lr=1e-4, k_train=args.mc_samples,
        inner_steps_per_outer=10,
        minibatch_size=4, rho_max=0.0, warmup_iters=600,
        buffer_capacity=8192,
        lambda_prior_mu_min=0.5, lambda_prior_mu_max=20.0,
    )
    trainer = EnergyScoreTrainer(
        score_net=score_net, mc_sampler=mc_for_training,
        config=cfg, device=device, rng_seed=args.seed,
    )

    # Build a fake DataLoader that yields Data objects with dummy graph data.
    # The trainer expects a PyG Data object with .y, .num_graphs, .edge_index etc.
    from torch_geometric.data import Data as PygData, Batch  # noqa: E402

    def _make_batch(batch_size):
        """Create a dummy PyG batch for the portfolio problem."""
        graphs = []
        for _ in range(batch_size):
            g = PygData(
                x=None,
                y=torch.zeros(N, 1, 1),  # dummy [N, T=1, F=1]
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
            graphs.append(g)
        return Batch.from_data_list(graphs)

    class _PortfolioLoader:
        """Infinite loader yielding identical dummy PyG batches."""
        def __init__(self, batch_size, n_batches=6):
            self._batch = _make_batch(batch_size)
            self._n = n_batches

        def __iter__(self):
            for _ in range(self._n):
                yield self._batch

        def __len__(self):
            return self._n

    train_loader = _PortfolioLoader(batch_size=args.batch_size, n_batches=6)

    num_outer = args.num_outer
    log.info("training for %d outer x %d inner = %d SGD steps ...",
             num_outer, cfg.inner_steps_per_outer, num_outer * cfg.inner_steps_per_outer)
    t0 = time.time()
    history_log = []

    def _hook(info):
        history_log.append(info)
        if info["iter"] % 25 == 0 or info["iter"] <= 5:
            log.info("iter=%d loss=%.4f cos=%.3f pred_rms=%.3f t=%.0f",
                     info["iter"], info["loss"], info.get("cos_mean", 0),
                     info.get("pred_rms_ratio_mean", 0), info.get("t_mean", 0))

    hist = trainer.fit(data_loader=train_loader, num_outer_iters=num_outer, log_fn=_hook)
    train_wall = time.time() - t0

    last = hist[-50:]
    cos_tail = float(np.mean([h.get("cos_mean", 0) for h in last]))
    loss_tail = float(np.mean([h["loss"] for h in last]))
    pred_rms_tail = float(np.mean([h.get("pred_rms_ratio_mean", 0) for h in last]))
    log.info("training done in %.1fs (%.1fmin); last-50 cos=%.3f loss=%.4f pred_rms=%.3f",
             train_wall, train_wall / 60, cos_tail, loss_tail, pred_rms_tail)
    torch.save(score_net.state_dict(), out_dir / "score_net.pt")

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array([h["iter"] for h in hist])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in [
        (axes[0], "loss", "normalized MSE"),
        (axes[1], "cos_mean", r"cos($\epsilon_{pred}$, $\epsilon_{target}$)"),
        (axes[2], "pred_rms_ratio_mean", "||pred|| / ||target||"),
    ]:
        ys = np.array([h.get(key, 0) for h in hist])
        roll = np.array([np.mean(ys[max(0, i - 25):i + 1]) for i in range(len(ys))])
        ax.plot(xs, ys, alpha=0.35)
        ax.plot(xs, roll, lw=2, color="darkred")
        ax.set_title(title)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)
    log.info("training curves saved to %s", out_dir / "training_curves.png")

    # ------------------------------------------------------------------
    # Inference comparison
    # ------------------------------------------------------------------
    shape = (args.batch_size, 1, N, 1)
    dummy_data = _make_batch(args.batch_size)

    def _make_sampler():
        return PortfolioEnergyDDPM(
            model=_NoOpModel(),
            num_timesteps=T, beta_schedule=args.beta_schedule,
            portfolio_mu=mu, portfolio_Sigma=Sigma,
            portfolio_scenarios=scenarios,
            portfolio_risk_budgets=risk_budgets,
            energy_mc_samples=args.mc_samples_infer,
            inverse_beta=IB, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=0.5,
            dual_num_outer_iterations=1,
            dual_lambda_init=1.0,
            dual_lambda_max=100.0,
        ).to(device)

    # Build context once for evaluation
    sampler_ref = _make_sampler()
    context = sampler_ref._build_energy_context(
        data=dummy_data, batch_size=args.batch_size, num_nodes=N,
        device=device, dtype=torch.float32,
    )

    # (1) MC baseline
    log.info("=== (1) MC baseline @ ib=%g ===", IB)
    torch.manual_seed(args.seed)
    sampler_mc = _make_sampler()
    t_mc = time.time()
    z0_mc = sampler_mc.sample(shape=shape, device=device, data=dummy_data)
    wall_mc = time.time() - t_mc
    m_mc = _evaluate(z0_mc, sampler_mc, context, f"MC_ib={IB}", log)
    m_mc["wall_time_sec"] = wall_mc

    # (2) Trained score net
    log.info("=== (2) Trained-net @ ib=%g ===", IB)
    torch.manual_seed(args.seed)
    sampler_tn = _make_sampler()
    sampler_tn._estimate_score = types.MethodType(
        _make_trained_score_estimator(score_net, sampler_tn.alphas_cumprod, N),
        sampler_tn,
    )
    score_net.eval()
    t_tn = time.time()
    z0_tn = sampler_tn.sample(shape=shape, device=device, data=dummy_data)
    wall_tn = time.time() - t_tn
    m_tn = _evaluate(z0_tn, sampler_tn, context, "trained_net", log)
    m_tn["wall_time_sec"] = wall_tn

    # (3) Unconstrained diffusion (lambda=0)
    log.info("=== (3) Unconstrained (lambda=0) ===")
    torch.manual_seed(args.seed)
    sampler_uc = PortfolioEnergyDDPM(
        model=_NoOpModel(),
        num_timesteps=T, beta_schedule=args.beta_schedule,
        portfolio_mu=mu, portfolio_Sigma=Sigma,
        portfolio_scenarios=scenarios,
        portfolio_risk_budgets=risk_budgets,
        energy_mc_samples=args.mc_samples_infer,
        inverse_beta=IB, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=0.0,  # no dual update
        dual_num_outer_iterations=1,
        dual_lambda_init=0.0,  # lambda stays zero
    ).to(device)
    t_uc = time.time()
    z0_uc = sampler_uc.sample(shape=shape, device=device, data=dummy_data)
    wall_uc = time.time() - t_uc
    m_uc = _evaluate(z0_uc, sampler_uc, context, "unconstrained", log)
    m_uc["wall_time_sec"] = wall_uc

    # (4) Analytical baselines
    log.info("=== (4) Analytical baselines ===")
    baseline_results = _baselines(context, N, device, log)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_results = [m_mc, m_tn, m_uc] + baseline_results
    summary = {
        "problem_size": args.size,
        "N": N,
        "ib": IB,
        "T": T,
        "training_wall_sec": train_wall,
        "training_iters": len(hist),
        "training_last50_cos_mean": cos_tail,
        "training_last50_loss_mean": loss_tail,
        "training_last50_pred_rms_mean": pred_rms_tail,
        "results": all_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Pretty print
    lines = [
        f"=== Portfolio {args.size}: N={N}, ib={IB}, T={T} ===",
        f"training: {len(hist)} iters in {train_wall:.0f}s; "
        f"last-50 cos={cos_tail:.3f}, loss={loss_tail:.4f}",
        "",
        f"{'method':<20s} {'E[ret]':>10s} {'var':>10s} {'vio_mean':>9s} "
        f"{'vio_max':>8s} {'feas':>6s} {'H':>6s} {'wall':>7s}",
    ]
    for m in all_results:
        wall_str = f"{m.get('wall_time_sec', 0):6.1f}s"
        lines.append(
            f"{m['label']:<20s} {m['expected_return_mean']:10.4f} "
            f"{m['portfolio_var_mean']:10.6f} {m['mean_violation']:9.5f} "
            f"{m['max_violation']:8.5f} {m['feasibility_rate']:6.3f} "
            f"{m['entropy_mean']:6.2f} {wall_str}"
        )
    msg = "\n".join(lines)
    log.info("\n%s", msg)
    (out_dir / "summary.txt").write_text(msg + "\n")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _plot_figures(out_dir, all_results, z0_mc, z0_tn, z0_uc, sampler_mc, context, log)
    log.info("done. outputs in %s", out_dir)


def _plot_figures(out_dir, all_results, z0_mc, z0_tn, z0_uc, sampler, context, log):
    """Generate paper-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = z0_mc.device

    # Convert z -> weights
    w_mc = sampler.z_to_portfolio_weights(z0_mc).detach().cpu().numpy()
    w_tn = sampler.z_to_portfolio_weights(z0_tn).detach().cpu().numpy()
    w_uc = sampler.z_to_portfolio_weights(z0_uc).detach().cpu().numpy()

    Sigma_np = context.Sigma.cpu().numpy()
    budgets_np = context.risk_budgets.cpu().numpy()
    N = Sigma_np.shape[0]

    def _risk_np(w):
        return (w @ Sigma_np) * w  # [B, N]

    # ------------------------------------------------------------------
    # 1. CDF of per-asset risk contributions
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for w, label, color in [
        (w_mc, "MC constrained", "tab:blue"),
        (w_tn, "Trained net", "tab:orange"),
        (w_uc, "Unconstrained", "tab:red"),
    ]:
        risk = _risk_np(w).flatten()
        sorted_r = np.sort(risk)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        ax.plot(sorted_r, cdf, label=label, color=color, lw=2)
    # Budget threshold line (use mean budget for reference)
    mean_budget = budgets_np.mean()
    ax.axvline(mean_budget, color="black", ls="--", lw=1.5, label=f"mean budget = {mean_budget:.5f}")
    ax.set_xlabel("Per-asset risk contribution $(\\Sigma x)_j \\cdot x_j$")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of per-asset risk contributions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_cdf.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Efficient frontier: expected return vs portfolio variance
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for w, label, color, marker in [
        (w_mc, "MC constrained", "tab:blue", "o"),
        (w_tn, "Trained net", "tab:orange", "s"),
        (w_uc, "Unconstrained", "tab:red", "^"),
    ]:
        scenarios_np = context.scenarios.cpu().numpy()
        ret = (scenarios_np @ w.T).mean(axis=0)  # [B]
        var = np.array([(wi @ Sigma_np @ wi) for wi in w])  # [B]
        ax.scatter(var, ret, label=label, color=color, marker=marker, alpha=0.5, s=20)

    # Equal weight and risk parity points
    w_eq = np.ones(N) / N
    ret_eq = (context.scenarios.cpu().numpy() @ w_eq).mean()
    var_eq = w_eq @ Sigma_np @ w_eq
    ax.scatter([var_eq], [ret_eq], marker="D", color="green", s=100,
               zorder=5, label="Equal weight", edgecolors="black")

    ax.set_xlabel("Portfolio variance $x^T \\Sigma x$")
    ax.set_ylabel("Expected return $E[r^T x]$")
    ax.set_title("Efficient frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "efficient_frontier.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Weight heatmap (top 20 assets for readability)
    # ------------------------------------------------------------------
    top_k = min(20, N)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, w, title in [
        (axes[0], w_mc, "MC constrained"),
        (axes[1], w_tn, "Trained net"),
        (axes[2], w_uc, "Unconstrained"),
    ]:
        # Show first 8 samples, top-k assets by mean weight
        mean_w = w.mean(axis=0)
        top_idx = np.argsort(mean_w)[-top_k:][::-1]
        im = ax.imshow(w[:8, top_idx].T, aspect="auto", cmap="viridis")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Asset (top 20)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Portfolio weight heatmaps")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_heatmap.png", dpi=130)
    plt.close(fig)

    log.info("figures saved: risk_cdf.png, efficient_frontier.png, weight_heatmap.png")


if __name__ == "__main__":
    main()
