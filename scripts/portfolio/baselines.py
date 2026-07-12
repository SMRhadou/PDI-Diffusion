"""Baselines for constrained portfolio optimisation.

Implements all comparison methods used in the paper:
- ``equal_weight``            : x_j = 1/N   (analytical)
- ``risk_parity``             : x_j prop 1/sigma_j  (analytical heuristic)
- ``markowitz_unconstrained`` : long-only mean-variance optimum (QP)
- ``markowitz_constrained``   : long-only MV + per-asset risk budgets (NLP/SLSQP)
- ``pd_langevin``             : primal-dual Langevin in z-space  (iterative)
- ``pdm``                     : unconstrained DDPM + per-step Cimmino projection
- ``rejection``               : unconstrained DDPM + rejection (resample infeasible)
- ``policy_net``              : trainable MLP policy, trained by primal-dual SGD
  with the same Lagrangian.

All methods return a tensor [B, N] of portfolio weights on the simplex, plus
a small dict of diagnostics.

References for trainable baselines:
- Paternain et al. 2019, "Constrained policy optimization" (NeurIPS).
- Chow et al. 2017, "Risk-constrained RL via Lagrangian approaches."
- Lagrangian policy networks are a standard constrained-learning baseline.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------
# Analytical baselines
# ---------------------------------------------------------------

def equal_weight(N: int, B: int, device=None) -> torch.Tensor:
    w = torch.ones(B, N, device=device) / N
    return w


def risk_parity(Sigma: torch.Tensor, B: int) -> torch.Tensor:
    """Inverse-volatility weights, x_j prop 1/sigma_j."""
    sigma_diag = torch.sqrt(torch.diag(Sigma)).clamp_min(1e-12)
    inv = 1.0 / sigma_diag
    w = inv / inv.sum()
    return w.unsqueeze(0).expand(B, -1).contiguous()


def markowitz_unconstrained(mu: np.ndarray, Sigma: np.ndarray,
                             risk_aversion: float = 5.0) -> np.ndarray:
    """Long-only mean-variance via cvxpy QP."""
    import cvxpy as cp
    N = len(mu)
    x = cp.Variable(N)
    objective = cp.Maximize(mu @ x - 0.5 * risk_aversion * cp.quad_form(x, cp.psd_wrap(Sigma)))
    constraints = [cp.sum(x) == 1.0, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return np.asarray(x.value).flatten()


def markowitz_constrained(mu: np.ndarray, Sigma: np.ndarray, budgets: np.ndarray,
                           risk_aversion: float = 5.0,
                           constraint_type: str = "variance") -> np.ndarray:
    """Long-only mean-variance with per-asset risk budgets via SLSQP."""
    from scipy.optimize import minimize
    N = len(mu)

    def obj(x):
        return 0.5 * risk_aversion * x @ Sigma @ x - mu @ x

    def obj_grad(x):
        return risk_aversion * Sigma @ x - mu

    constraints = [
        {"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(N)},
    ]

    if constraint_type == "variance_band":
        bud_upper = budgets[:N]
        bud_lower = -budgets[N:]

        def ineq_upper(x):
            return bud_upper - x * (Sigma @ x)

        def ineq_upper_jac(x):
            Sx = Sigma @ x
            return -np.diag(Sx) - np.diag(x) @ Sigma

        def ineq_lower(x):
            return x * (Sigma @ x) - bud_lower

        def ineq_lower_jac(x):
            Sx = Sigma @ x
            return np.diag(Sx) + np.diag(x) @ Sigma

        constraints.append({"type": "ineq", "fun": ineq_upper, "jac": ineq_upper_jac})
        constraints.append({"type": "ineq", "fun": ineq_lower, "jac": ineq_lower_jac})
    else:
        def ineq(x):
            return budgets - x * (Sigma @ x)

        def ineq_jac(x):
            Sx = Sigma @ x
            return -np.diag(Sx) - np.diag(x) @ Sigma

        constraints.append({"type": "ineq", "fun": ineq, "jac": ineq_jac})

    bounds = [(0.0, 1.0)] * N
    x0 = np.ones(N) / N
    res = minimize(obj, x0, jac=obj_grad, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-10, "maxiter": 500})
    if not res.success or np.any(np.isnan(res.x)):
        return x0  # fallback
    return res.x


# ---------------------------------------------------------------
# Primal-dual Langevin (iterative, no diffusion)
# ---------------------------------------------------------------

def _compute_constraints(x, Sigma, scenarios, alpha, constraint_type, num_sectors=10, **kwargs):
    """Compute per-asset constraint values c_j(x). Returns [B, N]."""
    if constraint_type == "variance":
        Sigma_x = torch.matmul(x, Sigma)  # [B, N]
        return x * Sigma_x
    elif constraint_type == "variance_sector":
        Sigma_x = torch.matmul(x, Sigma)  # [B, N]
        c_per_asset = x * Sigma_x  # [B, N]
        B, N = x.shape
        sector_ids = torch.arange(N, device=x.device) % num_sectors
        sector_totals = torch.zeros(B, num_sectors, device=x.device, dtype=x.dtype)
        sector_totals.scatter_add_(1, sector_ids.unsqueeze(0).expand(B, -1), c_per_asset)
        c = sector_totals.gather(1, sector_ids.unsqueeze(0).expand(B, -1))
        return c
    elif constraint_type == "variance_band":
        Sigma_x = torch.matmul(x, Sigma)
        c_var = x * Sigma_x
        return torch.cat([c_var, -c_var], dim=1)
    elif constraint_type == "dual":
        Sigma_x = torch.matmul(x, Sigma)
        c_var = x * Sigma_x
        port_ret = torch.matmul(scenarios, x.t())
        S = torch.clamp_min(alpha - port_ret, 0.0).mean(dim=0)
        c_short = x * S.unsqueeze(-1)
        return torch.cat([c_var, c_short], dim=1)
    elif constraint_type == "enriched":
        extra = kwargs.get("extra")
        B_sz, N = x.shape
        c_var = x * torch.matmul(x, Sigma)
        S = num_sectors
        sector_ids = torch.as_tensor(extra["sector_ids"], device=x.device, dtype=torch.long)
        exposure = torch.zeros(B_sz, S, device=x.device, dtype=x.dtype)
        exposure.scatter_add_(1, sector_ids.unsqueeze(0).expand(B_sz, -1), x)
        sector_upper = torch.as_tensor(extra["sector_upper"], device=x.device, dtype=x.dtype)
        sector_lower = torch.as_tensor(extra["sector_lower"], device=x.device, dtype=x.dtype)
        upper_res = exposure - sector_upper.unsqueeze(0)
        lower_res = sector_lower.unsqueeze(0) - exposure
        stress_ret_t = torch.as_tensor(extra["stress_returns"], device=x.device, dtype=x.dtype)
        stress_lim_t = torch.as_tensor(extra["stress_limits"], device=x.device, dtype=x.dtype)
        stress_eps_t = torch.as_tensor(extra["stress_eps"], device=x.device, dtype=x.dtype)
        stress_port = torch.matmul(x, stress_ret_t.t())
        excess_loss = torch.clamp(-stress_port - stress_lim_t.unsqueeze(0), min=0.0)
        stress_res = excess_loss - stress_eps_t.unsqueeze(0)
        ent = -(x * torch.log(x.clamp_min(1e-12))).sum(dim=1, keepdim=True)
        neff_val = torch.exp(ent)
        neff_target = extra.get("neff_target", 500.0)
        neff_res = neff_val - neff_target
        return torch.cat([c_var, upper_res, lower_res, stress_res, neff_res], dim=1)
    else:
        port_ret = torch.matmul(scenarios, x.t())  # [R, B]
        S = torch.clamp_min(alpha - port_ret, 0.0).mean(dim=0)  # [B]
        return x * S.unsqueeze(-1)


def pd_langevin(mu: torch.Tensor, Sigma: torch.Tensor, scenarios: torch.Tensor,
                budgets: torch.Tensor, alpha: float, B: int,
                num_iters: int = 500, primal_lr: float = 1e-2,
                dual_lr: float = 10.0,
                device=None, seed: int = 42,
                constraint_type: str = "variance",
                shared_lambda: bool = False,
                objective_scale: float = 1.0,
                constraint_scale: float = 1.0,
                extra: dict = None) -> Tuple[torch.Tensor, Dict]:
    """Primal-dual Langevin on z-space.

    constraint_type: "variance" → c_j = x_j (Σx)_j, "shortfall" → c_j = x_j E[(α−rᵀx)₊].
    shared_lambda: if True, dual updates use batch-mean violation (avg-constraint).
    """
    torch.manual_seed(seed)
    N = len(mu)
    n_constraints = len(budgets)
    z = torch.randn(B, N, device=device)
    if shared_lambda:
        lam = torch.zeros(1, n_constraints, device=device)
    else:
        lam = torch.zeros(B, n_constraints, device=device)

    if extra is not None and "constraint_scales" in extra:
        _cs_vec = torch.as_tensor(extra["constraint_scales"], dtype=torch.float32, device=device)
        budgets = budgets * _cs_vec
    else:
        _cs_vec = None

    for k in range(num_iters):
        z_req = z.detach().requires_grad_(True)
        x = torch.softmax(z_req, dim=-1)
        c = _compute_constraints(x, Sigma, scenarios, alpha, constraint_type, extra=extra)
        if _cs_vec is not None:
            c = c * _cs_vec.unsqueeze(0)
        else:
            c = c * constraint_scale
        expected_return = torch.matmul(scenarios, x.t()).mean(dim=0)  # [B]
        violation = c - budgets.unsqueeze(0)
        lag_pen = (lam * violation).sum(dim=1)
        energy = -objective_scale * expected_return + lag_pen

        grad = torch.autograd.grad(energy.sum(), z_req)[0]
        noise = math.sqrt(2 * primal_lr) * torch.randn_like(z) if k < num_iters - 1 else 0.0
        z = (z_req - primal_lr * grad + noise).detach()

        with torch.no_grad():
            x_now = torch.softmax(z, dim=-1)
            c_now = _compute_constraints(x_now, Sigma, scenarios, alpha, constraint_type, extra=extra)
            if _cs_vec is not None:
                c_now = c_now * _cs_vec.unsqueeze(0)
            else:
                c_now = c_now * constraint_scale
            v = c_now - budgets.unsqueeze(0)
            if shared_lambda:
                v_mean = v.mean(dim=0, keepdim=True)
                lam = (lam + dual_lr * v_mean).clamp_min(0.0)
            else:
                lam = (lam + dual_lr * v).clamp_min(0.0)

    w = torch.softmax(z, dim=-1).detach()
    return w, {"final_lambda_mean": float(lam.mean().item()),
                "final_lambda_max": float(lam.max().item())}


# ---------------------------------------------------------------
# Projected Diffusion Model (PDM)
# ---------------------------------------------------------------

def _max_violation(x: torch.Tensor, Sigma: torch.Tensor,
                    scenarios: torch.Tensor, alpha: float,
                    budgets: torch.Tensor,
                    constraint_type: str = "shortfall") -> torch.Tensor:
    """Per-sample max constraint violation. Returns [B]."""
    c = _compute_constraints(x, Sigma, scenarios, alpha, constraint_type)
    return (c - budgets.unsqueeze(0)).amax(dim=1)


def project_to_feasible(x: torch.Tensor, Sigma: torch.Tensor,
                         scenarios: torch.Tensor, alpha: float,
                         budgets: torch.Tensor,
                         constraint_type: str = "shortfall",
                         num_iters: int = 50, lr: float = 0.1,
                         extra: dict = None) -> torch.Tensor:
    """Project simplex weights to feasible set via linearised Cimmino projection.

    At each iteration, linearise the violated constraints at the current point
    and project onto the resulting half-spaces (parallel Cimmino).  The result
    is re-normalised to the simplex after each step.
    """
    w = x.clone()
    B, N = w.shape
    if constraint_type == "enriched":
        bud_t = budgets.unsqueeze(0)
        if extra is not None and "constraint_scales" in extra:
            cs = torch.as_tensor(extra["constraint_scales"], device=x.device, dtype=x.dtype).unsqueeze(0)
        else:
            cs = 1.0
        z = torch.log(w.clamp_min(1e-12))
        z = z - z.mean(dim=-1, keepdim=True)
        for _proj_i in range(num_iters):
            z_req = z.detach().requires_grad_(True)
            w_cur = torch.softmax(z_req, dim=-1)
            c = _compute_constraints(w_cur, Sigma, scenarios, alpha,
                                     constraint_type, extra=extra)
            viol = ((c - bud_t) * cs).clamp_min(0.0)
            loss = (viol ** 2).sum()
            if loss.item() < 1e-16:
                break
            grad = torch.autograd.grad(loss, z_req)[0]
            z = (z_req - lr * grad).detach()
        w_out = torch.softmax(z, dim=-1).detach()
        raw_viol = (c - bud_t).clamp_min(0.0)
        w_out._proj_max_raw_vio = float(raw_viol.max())
        w_out._proj_mean_raw_vio = float(raw_viol.mean())
        w_out._proj_frac_satisfied = float((raw_viol.max(dim=1).values < 1e-8).float().mean())
        w_out._proj_iters_used = _proj_i + 1
        return w_out
    Sig_diag = torch.diag(Sigma).unsqueeze(0)  # [1, N]
    if constraint_type == "variance_band":
        bud_upper = budgets[:N].unsqueeze(0)       # [1, N] = b_max
        bud_lower = (-budgets[N:]).unsqueeze(0)     # [1, N] = b_min
        for _ in range(num_iters):
            Sw = torch.matmul(w, Sigma)
            c = w * Sw
            upper_viol = (c - bud_upper).clamp_min(0.0)
            lower_viol = (bud_lower - c).clamp_min(0.0)
            if upper_viol.max().item() < 1e-12 and lower_viol.max().item() < 1e-12:
                break
            dc_diag = Sig_diag * w + Sw
            step = (upper_viol - lower_viol) / dc_diag.abs().clamp_min(1e-12)
            w = w - step
            w = w.clamp_min(0.0)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return w.detach()
    bud = budgets.unsqueeze(0)  # [1, N]
    for _ in range(num_iters):
        Sw = torch.matmul(w, Sigma)  # [B, N]
        c = w * Sw                    # [B, N]
        viol = (c - bud).clamp_min(0.0)  # [B, N]
        if viol.max().item() < 1e-12:
            break
        # ∂c_j/∂x_j = Σ_{jj} x_j + (Σx)_j
        # Cimmino step along coordinate j: Δx_j = -v_j / (∂c_j/∂x_j)
        dc_diag = Sig_diag * w + Sw  # [B, N]
        step = viol / dc_diag.abs().clamp_min(1e-12)  # [B, N]
        w = w - step
        w = w.clamp_min(0.0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w.detach()


def blend_to_feasible(x: torch.Tensor, Sigma: torch.Tensor,
                       scenarios: torch.Tensor, alpha: float,
                       budgets: torch.Tensor,
                       constraint_type: str = "shortfall",
                       num_iters: int = 50, tol: float = 1e-8) -> torch.Tensor:
    """Project to feasible via gradient descent (replaces old bisection blend)."""
    return project_to_feasible(x, Sigma, scenarios, alpha, budgets,
                                constraint_type=constraint_type,
                                num_iters=num_iters)


cimmino_project = blend_to_feasible


def projected_diffusion(sampler, shape, data, device, mu, Sigma, budgets,
                         poc_iters: int = 30) -> Tuple[torch.Tensor, Dict]:
    """Run unconstrained DDPM sampler, then project final simplex output.

    The sampler should already be configured with lambda_init=0 and no dual
    update (dual_step ~ 0). This is the "post-hoc projection" variant of PDM
    for portfolio; per-step projection would require projecting in z-space
    at every reverse step, which is more invasive.
    """
    with torch.no_grad():
        z = sampler.sample(shape=shape, device=device, data=data)
        x = sampler.z_to_portfolio_weights(z)  # [B, N]
        x_proj = cimmino_project(x, Sigma, budgets, num_iters=poc_iters)
    return x_proj, {"poc_iters": poc_iters}


def rejection_sampling(sampler, shape, data, device, scenarios, alpha, budgets,
                        max_retries: int = 5, tol: float = 1e-6) -> Tuple[torch.Tensor, Dict]:
    """Unconstrained DDPM + rejection against expectation-shortfall constraint.
    """
    B = shape[0]
    N = shape[2]
    feasible = []
    n_tried = 0
    n_accepted = 0
    for _ in range(max_retries):
        n_tried += B
        with torch.no_grad():
            z = sampler.sample(shape=shape, device=device, data=data)
            x = sampler.z_to_portfolio_weights(z)
            pr = torch.matmul(scenarios, x.t())
            S = torch.clamp_min(alpha - pr, 0.0).mean(dim=0)
            c = x * S.unsqueeze(-1)
            ok = (c <= budgets.unsqueeze(0) + tol).all(dim=1)
            if ok.any():
                feasible.append(x[ok])
                n_accepted += int(ok.sum().item())
                if n_accepted >= B:
                    break
    if len(feasible) == 0:
        print("[rejection] zero feasible samples; returning equal-weight")
        x_out = torch.ones(B, N, device=device) / N
    else:
        x_cat = torch.cat(feasible, dim=0)
        if x_cat.shape[0] >= B:
            x_out = x_cat[:B]
        else:
            # Pad with equal-weight
            pad = torch.ones(B - x_cat.shape[0], N, device=device) / N
            x_out = torch.cat([x_cat, pad], dim=0)
    return x_out, {"acceptance_rate": n_accepted / max(n_tried, 1),
                    "n_accepted": n_accepted, "n_tried": n_tried}


# ---------------------------------------------------------------
# Trainable policy network
# ---------------------------------------------------------------

class PortfolioPolicyNet(nn.Module):
    """MLP policy that outputs portfolio weights (via softmax).

    Input: problem parameters (mu, diag(Sigma), budgets), flattened.
    Output: z in R^N; portfolio weights = softmax(z).

    Problem parameters are constant in our experiments (single problem),
    so the net effectively learns a fixed allocation. The trainable baseline
    is useful as a non-diffusion reference: same Lagrangian, same dual-ascent
    discipline, different sampler.
    """

    def __init__(self, N: int, hidden: int = 256, num_layers: int = 4):
        super().__init__()
        in_dim = 3 * N  # mu, diag(Sigma), budgets
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, N))
        self.mlp = nn.Sequential(*layers)

    def forward(self, mu: torch.Tensor, Sigma_diag: torch.Tensor,
                budgets: torch.Tensor) -> torch.Tensor:
        """mu, Sigma_diag, budgets: each [N]. Returns z [1, N]."""
        inp = torch.cat([mu, Sigma_diag, budgets], dim=-1).unsqueeze(0)
        return self.mlp(inp)


def train_policy_net(mu: torch.Tensor, Sigma: torch.Tensor,
                      scenarios: torch.Tensor, budgets: torch.Tensor,
                      alpha: float,
                      B: int = 64, num_iters: int = 2000,
                      primal_lr: float = 1e-3, dual_lr: float = 100.0,
                      inverse_beta: float = 100.0,
                      noise_scale: float = 0.3,
                      shared_lambda: bool = False,
                      device=None, seed: int = 42,
                      hidden: int = 256, num_layers: int = 4,
                      log_every: int = 200) -> Tuple[PortfolioPolicyNet, torch.Tensor, Dict]:
    """Primal-dual training of a portfolio policy net.

    Objective (same Lagrangian as CED):
        min_theta max_lambda  E_eps [inverse_beta * (-E[r^T x] + lambda^T (rho(x) - b))]
        x = softmax(policy_net(problem) + noise_scale * eps)

    Noise ensures the policy outputs a distribution over x (not a single point),
    so the same diversity/feasibility metrics apply.
    """
    torch.manual_seed(seed)
    N = len(mu)
    mu = mu.to(device)
    Sigma = Sigma.to(device)
    scenarios = scenarios.to(device)
    budgets = budgets.to(device)
    net = PortfolioPolicyNet(N=N, hidden=hidden, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=primal_lr)

    if shared_lambda:
        lam = torch.zeros(1, N, device=device)
    else:
        lam = torch.zeros(B, N, device=device)

    Sigma_diag = torch.diag(Sigma)

    for k in range(num_iters):
        z_mean = net(mu, Sigma_diag, budgets)
        eps = torch.randn(B, N, device=device)
        z = z_mean.expand(B, -1) + noise_scale * eps
        x = torch.softmax(z, dim=-1)
        port_ret = torch.matmul(scenarios, x.t())  # [R, B]
        S = torch.clamp_min(alpha - port_ret, 0.0).mean(dim=0)  # [B]
        c = x * S.unsqueeze(-1)
        expected_return = port_ret.mean(dim=0)
        violation = c - budgets.unsqueeze(0)
        lag_pen = (lam * violation).sum(dim=1)
        loss = inverse_beta * (-expected_return + lag_pen).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            x_now = torch.softmax(net(mu, Sigma_diag, budgets).expand(B, -1)
                                   + noise_scale * torch.randn(B, N, device=device), dim=-1)
            pr = torch.matmul(scenarios, x_now.t())
            S_now = torch.clamp_min(alpha - pr, 0.0).mean(dim=0)
            c_now = x_now * S_now.unsqueeze(-1)
            v = c_now - budgets.unsqueeze(0)
            if shared_lambda:
                lam = (lam + dual_lr * v.mean(dim=0, keepdim=True)).clamp_min(0.0)
            else:
                lam = (lam + dual_lr * v).clamp_min(0.0)

        if (k + 1) % log_every == 0:
            print(f"  [policy_net] iter={k+1:5d}  loss={loss.item():.4f}  "
                  f"lam_mean={lam.mean().item():.3g}  "
                  f"feas={(c <= budgets.unsqueeze(0) + 1e-8).all(dim=1).float().mean().item():.3f}")

    # Final sample with noise for diversity
    with torch.no_grad():
        z_mean = net(mu, Sigma_diag, budgets)
        eps = torch.randn(B, N, device=device)
        z_final = z_mean.expand(B, -1) + noise_scale * eps
        w = torch.softmax(z_final, dim=-1)

    return net, w, {"final_lambda_mean": float(lam.mean().item()),
                    "final_lambda_max": float(lam.max().item())}


def train_policy_net_amortized(
    instances,  # list of dicts: {mu, Sigma, scenarios, budgets, alpha}, each torch tensors on device
    B: int = 256, num_iters: int = 4000,
    primal_lr: float = 1e-3, dual_lr: float = 100.0,
    inverse_beta: float = 100.0,
    noise_scale: float = 0.3,
    shared_lambda: bool = True,
    device=None, seed: int = 42,
    hidden: int = 256, num_layers: int = 4,
    log_every: int = 500,
    instances_per_batch: int | None = None,
) -> Tuple[PortfolioPolicyNet, Dict]:
    """Amortized primal-dual training over a dataset of problem instances.

    Trains ONE network that maps (mu, diag(Sigma), budgets) -> z policy, across
    all K instances. Each iteration samples a minibatch of instances and sums
    their Lagrangian losses. Per-instance duals are maintained.
    """
    torch.manual_seed(seed)
    K = len(instances)
    assert K >= 1
    N = int(instances[0]["mu"].shape[0])
    net = PortfolioPolicyNet(N=N, hidden=hidden, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=primal_lr)

    if shared_lambda:
        lam = torch.zeros(K, 1, N, device=device)  # per-instance shared lambda
    else:
        lam = torch.zeros(K, B, N, device=device)

    Sigma_diags = [torch.diag(inst["Sigma"]) for inst in instances]
    ibatch = min(instances_per_batch or K, K)

    for k in range(num_iters):
        idx = torch.randperm(K, device=device)[:ibatch]
        losses = []
        for i in idx.tolist():
            mu_i = instances[i]["mu"]
            Sd_i = Sigma_diags[i]
            bud_i = instances[i]["budgets"]
            scen_i = instances[i]["scenarios"]
            alpha_i = float(instances[i]["alpha"])

            z_mean = net(mu_i, Sd_i, bud_i)  # [1, N]
            eps = torch.randn(B, N, device=device)
            z = z_mean.expand(B, -1) + noise_scale * eps
            x = torch.softmax(z, dim=-1)
            port_ret = torch.matmul(scen_i, x.t())
            S = torch.clamp_min(alpha_i - port_ret, 0.0).mean(dim=0)
            c = x * S.unsqueeze(-1)
            expected_return = port_ret.mean(dim=0)
            violation = c - bud_i.unsqueeze(0)
            lag_pen = (lam[i] * violation).sum(dim=1)
            losses.append(inverse_beta * (-expected_return + lag_pen).mean())

        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # Dual update for the sampled instances (fresh samples)
        with torch.no_grad():
            for i in idx.tolist():
                mu_i = instances[i]["mu"]
                Sd_i = Sigma_diags[i]
                bud_i = instances[i]["budgets"]
                scen_i = instances[i]["scenarios"]
                alpha_i = float(instances[i]["alpha"])

                z_mean = net(mu_i, Sd_i, bud_i)
                eps = torch.randn(B, N, device=device)
                x_now = torch.softmax(z_mean.expand(B, -1) + noise_scale * eps, dim=-1)
                pr = torch.matmul(scen_i, x_now.t())
                S_now = torch.clamp_min(alpha_i - pr, 0.0).mean(dim=0)
                c_now = x_now * S_now.unsqueeze(-1)
                v = c_now - bud_i.unsqueeze(0)
                if shared_lambda:
                    lam[i] = (lam[i] + dual_lr * v.mean(dim=0, keepdim=True)).clamp_min(0.0)
                else:
                    lam[i] = (lam[i] + dual_lr * v).clamp_min(0.0)

        if (k + 1) % log_every == 0:
            print(f"  [policy_net_amortized] iter={k+1:5d}/{num_iters}  "
                  f"loss={loss.item():.4f}  lam_mean={lam.mean().item():.3g}  "
                  f"lam_max={lam.max().item():.3g}")

    return net, {"final_lambda_mean": float(lam.mean().item()),
                 "final_lambda_max": float(lam.max().item())}


def policy_net_inference(net: PortfolioPolicyNet, mu: torch.Tensor,
                          Sigma_diag: torch.Tensor, budgets: torch.Tensor,
                          B: int = 256, noise_scale: float = 0.3,
                          seed: int = 42) -> torch.Tensor:
    """Run the trained (amortized) policy on a single instance."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z_mean = net(mu, Sigma_diag, budgets)
        eps = torch.randn(B, mu.shape[0], device=mu.device)
        z = z_mean.expand(B, -1) + noise_scale * eps
        w = torch.softmax(z, dim=-1)
    return w
