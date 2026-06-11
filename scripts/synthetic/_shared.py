"""Shared utilities for synthetic constrained-Boltzmann experiments.

Every script in ``scripts/synthetic/`` that builds a problem, a sampler,
or evaluates samples should import from here instead of reimplementing
the same logic.  This eliminates the "passthrough bug" class of errors
where a new parameter is added to ``make_synthetic_problem()`` but not
forwarded by 20+ callers.
"""
from __future__ import annotations

import math
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdi.diffusion.synthetic_energy_ddpm import (
    SyntheticEnergyDDPM,
)
from pdi.models.synthetic_backbone import (
    SyntheticScoreBackbone,
)
from pdi.trainers.energy_score import (
    ScoreNetWithLambda,
)

from _experiment_paths import experiment_dir  # noqa: F401

# Lazy import to avoid circular dependency with run_synthetic_experiment.py
# which also imports from _shared.
_PRESETS = None
_make_synthetic_problem = None

def _ensure_presets():
    global _PRESETS, _make_synthetic_problem
    if _PRESETS is None:
        from run_synthetic_experiment import make_synthetic_problem as _msp, PRESETS as _P
        _PRESETS = _P
        _make_synthetic_problem = _msp

@property
def _presets_property(self):
    _ensure_presets()
    return _PRESETS

class _PresetsProxy:
    """Dict-like proxy that lazy-loads PRESETS on first access."""
    def __getitem__(self, key):
        _ensure_presets()
        return _PRESETS[key]
    def __contains__(self, key):
        _ensure_presets()
        return key in _PRESETS
    def __iter__(self):
        _ensure_presets()
        return iter(_PRESETS)
    def keys(self):
        _ensure_presets()
        return _PRESETS.keys()
    def values(self):
        _ensure_presets()
        return _PRESETS.values()
    def items(self):
        _ensure_presets()
        return _PRESETS.items()
    def __repr__(self):
        _ensure_presets()
        return repr(_PRESETS)

PRESETS = _PresetsProxy()


# ===================================================================
# 1. Problem construction
# ===================================================================

def _build_infeasible_2d(cov: float = 1.0) -> dict:
    """Three infeasible modes with a wedge constraint (d=2, m=2).

    All mode centres violate the constraints, so the unconstrained MoG has
    ~zero feasible mass.  This forces λ to rise substantially during sampling.
    """
    import math
    means = np.array([
        [ 3.0, 10.0],   # c1 ok, c2 violated
        [-2.0, -8.0],   # c1 violated, c2 ok
        [ 5.0,  0.0],   # both violated
    ], dtype=np.float64)
    K = means.shape[0]
    cov_diag = np.full((K, 2), cov, dtype=np.float64)
    weights = np.ones(K, dtype=np.float64) / K
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    A = np.array([[-inv_sqrt2, inv_sqrt2], [-1.0, 0.0]], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    shift = (weights[:, None] * means).sum(axis=0)
    avg_intra = (weights[:, None] * cov_diag).sum(axis=0)
    inter = (weights[:, None] * (means - shift) ** 2).sum(axis=0)
    scale = np.sqrt(np.maximum(avg_intra + inter, 1e-8))
    means_norm = (means - shift) / scale
    # Clip range: cover modes + feasible region (wedge extends to x1 ~ -15)
    feas_extent_orig = np.array([-15.0, 20.0])
    feas_extent_norm = (feas_extent_orig - shift) / scale
    clip_range = float(max(np.abs(means_norm).max() + 4.0 * np.sqrt(cov / scale**2).max(),
                           np.abs(feas_extent_norm).max()))
    return {
        "d": 2, "K": K, "m": 2,
        "means": torch.from_numpy(means_norm).float(),
        "cov_diag": torch.from_numpy(cov_diag / scale ** 2).float(),
        "weights": torch.from_numpy(weights).float(),
        "constraint_A": torch.from_numpy(A * scale).float(),
        "constraint_b": torch.from_numpy(b - A @ shift).float(),
        "norm_shift": torch.from_numpy(shift).float(),
        "norm_scale": torch.from_numpy(scale).float(),
        "means_original": torch.from_numpy(means).float(),
        "cov_diag_original": torch.from_numpy(cov_diag).float(),
        "calibrated_feasibility": 0.0,
        "n_infeasible_modes": 3,
        "clip_range": clip_range,
    }


def build_problem(preset_name: str, seed: int = 0) -> dict:
    """Build a synthetic problem from a named preset.

    This is the **single call site** for ``make_synthetic_problem`` that
    handles all preset-specific passthrough arguments.  Every script
    should call this instead of ``make_synthetic_problem`` directly.
    """
    if preset_name == "infeasible":
        return _build_infeasible_2d()

    _ensure_presets()
    p = _PRESETS[preset_name]
    return _make_synthetic_problem(
        d=p["d"],
        K_modes=p["K"],
        m_constraints=p["m"],
        spread=p["spread"],
        cov_lo=p["cov_lo"],
        cov_hi=p["cov_hi"],
        target_feasibility=p["target_feas"],
        seed=seed,
        norm=p.get("norm", "global"),
        min_infeasible_modes=int(p.get("min_infeasible_modes", 0)),
        max_feasible_margin=p.get("max_feasible_margin"),
        random_constraint_frac=float(p.get("random_constraint_frac", 0.0)),
        centroid_target_violation=p.get("centroid_target_violation"),
        n_active_constraints=int(p.get("n_active_constraints", 1)),
        avg_constraint_margins=p.get("avg_constraint_margins"),
    )


# ===================================================================
# 2. Sampler construction
# ===================================================================

def build_sampler(
    problem: dict,
    *,
    num_timesteps: int = 500,
    beta_schedule: str = "cosine",
    inverse_beta: float = 1.0,
    dual_step_size: float = 0.1,
    dual_lambda_init: float = 0.0,
    dual_lambda_max: Optional[float] = None,
    energy_mc_samples: int = 64,
    constraint_term_clamp: Optional[float] = None,
    shared_lambda: bool = False,
    device: str | torch.device = "cuda",
    **overrides,
) -> SyntheticEnergyDDPM:
    """Build a ``SyntheticEnergyDDPM`` sampler from a problem dict.

    Handles shared-lambda monkey-patching when ``shared_lambda=True``.
    """
    kw = dict(
        means=problem["means"],
        cov_diag=problem["cov_diag"],
        weights=problem["weights"],
        constraint_A=problem["constraint_A"],
        constraint_b=problem["constraint_b"],
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        energy_mc_samples=energy_mc_samples,
        inverse_beta=inverse_beta,
        inverse_beta_schedule="constant",
        dual_update_mode="x0_pred",
        dual_step_size=dual_step_size,
        dual_step_size_schedule="constant",
        dual_num_outer_iterations=1,
        dual_lambda_init=dual_lambda_init,
        dual_lambda_max=dual_lambda_max,
        constraint_term_clamp=constraint_term_clamp,
    )
    if "clip_range" in problem:
        kw["clip_range"] = problem["clip_range"]
    kw.update(overrides)
    s = SyntheticEnergyDDPM(**kw).to(device)
    for p in s.parameters():
        p.requires_grad_(False)
    if shared_lambda:
        install_shared_lambda(s)
    return s


# ===================================================================
# 3. Shared-lambda dual ascent
# ===================================================================

def _shared_dual_ascent_step(self, dual_lambda, ergodic_rates, context, t=None):
    """Batch-averaged dual ascent — all samples share the same lambda."""
    if t is None:
        dss = self.dual_step_size_by_t[0].to(
            device=dual_lambda.device, dtype=dual_lambda.dtype,
        )
    else:
        t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(
            0, self.num_timesteps - 1,
        )
        dss = self._gather(
            self.dual_step_size_by_t.to(device=dual_lambda.device), t_safe,
        ).to(dtype=dual_lambda.dtype)
        dss = dss.mean()

    violation = (context.r_min.view(-1, 1) - ergodic_rates).mean(dim=0, keepdim=True)
    lam_scalar = (dual_lambda.mean(dim=0, keepdim=True) + dss * violation).clamp_min(0.0)
    if self.dual_lambda_max is not None:
        lam_scalar = lam_scalar.clamp_max(float(self.dual_lambda_max))
    return lam_scalar.expand_as(dual_lambda).contiguous()


def install_shared_lambda(sampler: SyntheticEnergyDDPM) -> None:
    """Monkey-patch sampler to use batch-mean dual ascent, preserving
    dual-trace recording if active."""
    def _stepped(self, dual_lambda, ergodic_rates, context, t=None):
        updated = _shared_dual_ascent_step(
            self, dual_lambda, ergodic_rates, context, t,
        )
        if getattr(self, "_record_dual", False):
            m = self._syn_m
            self._dual_traces.append(
                updated[self._record_indices, :m].detach().cpu()
            )
        return updated
    sampler._dual_ascent_step = types.MethodType(_stepped, sampler)


def _noop_dual(self, dual_lambda, ergodic_rates, context, t=None):
    """No-op dual ascent (for unconstrained evaluation)."""
    return dual_lambda


# ===================================================================
# 4. Trained score-net wiring
# ===================================================================

def make_trained_score_fn(score_net, alphas_cumprod, d: int):
    """Return a bound method that replaces ``_estimate_score`` with a
    trained net's eps prediction converted to score."""
    dummy_ei = torch.zeros(2, 0, dtype=torch.long)

    def _fn(self, x_t, t, context, dual_lambda=None, inverse_beta_override=None):
        del context, inverse_beta_override
        if dual_lambda is None:
            dual_lambda = torch.zeros(x_t.shape[0], d, device=x_t.device, dtype=x_t.dtype)
        ei = dummy_ei.to(x_t.device)
        with torch.no_grad():
            eps_pred = score_net(
                x=x_t, timesteps=t, dual_lambda=dual_lambda,
                edge_index=ei, edge_weight=None,
                cond=None, return_intermediates=False,
            )
        ab = alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
        return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))

    return _fn


def wire_trained_net(
    sampler: SyntheticEnergyDDPM,
    score_net: ScoreNetWithLambda,
    d: int,
) -> None:
    """Replace sampler's ``_estimate_score`` with the trained net."""
    sampler._estimate_score = types.MethodType(
        make_trained_score_fn(score_net, sampler.alphas_cumprod, d),
        sampler,
    )


# ===================================================================
# 5. Checkpoint loading
# ===================================================================

def load_score_net(
    train_dir: str | Path,
    d: int,
    *,
    hidden: int = 256,
    num_layers: int = 4,
    num_timesteps: int = 500,
    ckpt_name: str = "score_net_best.pt",
    device: str | torch.device = "cuda",
) -> ScoreNetWithLambda:
    """Load a trained ``ScoreNetWithLambda`` from a checkpoint directory."""
    backbone = SyntheticScoreBackbone(
        d=d, hidden=hidden, num_layers=num_layers,
        num_timesteps=num_timesteps, cond_channels=1,
    )
    net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=0)
    ckpt = torch.load(Path(train_dir) / ckpt_name, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state)
    return net.to(device).eval()


# ===================================================================
# 6. Evaluation metrics
# ===================================================================

def compute_metrics(
    x: torch.Tensor,
    problem: dict,
    device: str | torch.device = "cuda",
    feas_tol: float = 0.0,
) -> dict:
    """Compute standard evaluation metrics for samples ``x`` [B, d].

    Returns dict with: feas_avg, U, modes_used, modes_coverage,
    entropy, mode_counts.
    """
    if x.shape[0] == 0:
        K = int(problem["means"].shape[0])
        return dict(feas_avg=0.0, U=float("nan"),
                    modes_used=0, modes_coverage=0.0, entropy=0.0,
                    mode_counts=[0] * K)

    d = int(problem["d"])
    K = int(problem["means"].shape[0])
    means = problem["means"].to(device).float()
    cov = problem["cov_diag"].to(device).float()
    weights = problem["weights"].to(device).float()
    log_w = torch.log(weights)
    log_norm = -0.5 * d * math.log(2 * math.pi) - 0.5 * torch.log(cov).sum(dim=1)
    A_full = problem["constraint_A"].to(device).float()
    b_full = problem["constraint_b"].to(device).float()
    m_real = int((A_full.abs().sum(1) > 0).sum().item())
    A = A_full[:m_real]
    b = b_full[:m_real]

    x = x.to(device).float()

    # Avg-constraint feasibility
    ax = x @ A.T
    feas_avg = float(((ax.mean(0) - b) >= -feas_tol).float().mean().item())

    # U = -log p_MoG(x)
    diff = x.unsqueeze(1) - means.unsqueeze(0)
    mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)
    log_p_k = log_w + log_norm - 0.5 * mahal
    log_p = torch.logsumexp(log_p_k, dim=1)
    U = float((-log_p).mean().item())

    # Mode counts (argmax posterior, no Mahalanobis gate)
    mode = log_p_k.argmax(dim=1)
    counts = torch.bincount(mode, minlength=K).cpu().numpy()
    n_used = int((counts > 0).sum())
    p_mode = counts / max(counts.sum(), 1)
    H = float(-(p_mode[p_mode > 0] * np.log(p_mode[p_mode > 0])).sum())

    return dict(
        feas_avg=feas_avg,
        U=U,
        modes_used=n_used,
        modes_coverage=n_used / K,
        entropy=H,
        mode_counts=counts.tolist(),
    )


# ===================================================================
# 7. PD-Langevin sampler
# ===================================================================

def pd_langevin_sample(
    problem: dict,
    B: int,
    T: int,
    lr: float,
    dual_lr: float,
    device: str | torch.device,
    seed: int = 0,
    shared: bool = False,
) -> torch.Tensor:
    """Primal-dual Langevin MCMC sampler. Returns [B, 1, d, 1]."""
    g = torch.Generator(device=torch.device(device)).manual_seed(int(seed))
    means = problem["means"].to(device).float()
    cov = problem["cov_diag"].to(device).float()
    log_w = torch.log(problem["weights"].to(device).float())
    A = problem["constraint_A"].to(device).float()
    b = problem["constraint_b"].to(device).float()
    m_real = int((A.abs().sum(dim=1) > 0).sum().item())
    A_real = A[:m_real]
    b_real = b[:m_real]
    d = int(problem["d"])

    x = torch.randn((B, d), generator=g, device=torch.device(device))
    lam = torch.zeros((B, m_real), device=torch.device(device))
    log_norm = -0.5 * d * math.log(2 * math.pi) - 0.5 * torch.log(cov).sum(dim=1)

    for _ in range(T):
        diff = x.unsqueeze(1) - means.unsqueeze(0)
        mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)
        log_comp = log_w + log_norm - 0.5 * mahal
        resp = torch.softmax(log_comp, dim=1)
        grad_logp = (resp.unsqueeze(-1) * (-diff) / cov.unsqueeze(0)).sum(dim=1)
        grad_U = -grad_logp
        grad_L = -(lam @ A_real)
        drift = -(grad_U + grad_L)
        noise = torch.randn_like(x)
        x = x + lr * drift + math.sqrt(2 * lr) * noise
        h = b_real - (x @ A_real.T)
        if shared:
            lam_scalar = torch.clamp(
                lam.mean(0, keepdim=True) + dual_lr * h.mean(0, keepdim=True),
                min=0.0,
            )
            lam = lam_scalar.expand_as(lam).contiguous()
        else:
            lam = torch.clamp(lam + dual_lr * h, min=0.0)
    return x.view(B, 1, d, 1)


# ===================================================================
# 8. Polytope projection (Cimmino)
# ===================================================================

def project_polytope(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                     max_iters: int = 80) -> torch.Tensor:
    """Project x [B,d] onto {y : Ay >= b} via parallel Cimmino projection."""
    a_norm_sq = (A ** 2).sum(dim=1).clamp_min(1e-12)
    A_step = A / a_norm_sq.unsqueeze(1)
    inv_m = 1.0 / float(A.shape[0])
    for _ in range(max_iters):
        violations = (b - x @ A.T).clamp(min=0)
        x = x + inv_m * (violations @ A_step)
    return x
