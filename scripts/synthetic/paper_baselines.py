"""Tuned baseline hyperparameters for paper figures on each synthetic preset.

These are loaded by ``paper_figures.py``. Bars / dual / sample / vector-field
/ confusion plots use exactly these values for MC Ceiling and PD-Langevin so
that figures stay reproducible across re-runs and across different CED
checkpoints.

Selection criterion (per preset): largest mode coverage (entropy) achievable
while keeping ``feas_avg = 1`` (marginal constraint satisfied) at the same
T as MC Ceiling so the comparison is matched.
"""
from __future__ import annotations

# Schema:
#   PAPER_BASELINES[preset_name] = {
#       "mc":  {"inverse_beta": float, "dual_step_size": float},
#       "pdl": {"primal_lr": float, "dual_lr": float},
#       "T":   int,                           # diffusion / MCMC steps
#   }
PAPER_BASELINES: dict[str, dict] = {
    # All-identical-modes preset (uniform cov + uniform weights, well-spread
    # centres). MC Ceiling at (ib=10, ds=0.10) covers 10/12 modes with
    # feas_avg=1 (H≈1.54). PDL at (1e-3, 10) covers 11/12 modes (H≈1.86).
    "medium_uniform": {
        "mc":  {"inverse_beta": 10.0, "dual_step_size": 0.10},
        "pdl": {"primal_lr": 1e-3,    "dual_lr":         10.0},
        "T":   500,
    },
    # Tight-cov variant (cov=1) — shared λ stays > 0 at t=0.
    # MC at (ib=20, ds=0.20): feas_avg=1, 12/12 modes, H=2.05, λ_t0=0.50.
    # PDL at (1e-3, 30): feas_avg=1, 11/12 modes, H=1.40, U=9.60.
    "medium_hard": {
        "mc":  {"inverse_beta": 20.0, "dual_step_size": 0.20},
        "pdl": {"primal_lr": 1e-3,    "dual_lr":         30.0},
        "T":   500,
        "T_pdl": 1500,
    },
    # Heterogeneous-cov variant. Kept for reference — MC Ceiling collapses
    # to one mode at any (ib, ds) that meets feas_avg=1, so figures here
    # show the pathology rather than the recovered behaviour.
    "medium_active": {
        "mc":  {"inverse_beta": 10.0, "dual_step_size": 0.30},
        "pdl": {"primal_lr": 3e-3,    "dual_lr":         10.0},
        "T":   500,
    },
    # Higher-dim (d=30, m=25, cov=2) — sustained λ, learnable by score net.
    # Cosine schedule sweep (2026-04-26):
    # MC at (ib=50, ds=0.05): feas_avg=1, 9/12 modes, U=-0.53.
    # PDL at (1e-2, 1, lam0=5): feas_avg=1, 8/12 modes, U=14.57.
    "high_dim": {
        "mc":  {"inverse_beta": 50.0, "dual_step_size": 0.05},
        "pdl": {"primal_lr": 1e-2,    "dual_lr":         1.0, "lam0": 5.0},
        "T":   500,
        "beta_schedule": "cosine",
    },
    # Avg-constraint d=30 with margins (0.8, 2.0) — non-trivial λ*.
    # MC (cosine) at (ib=300, ds=1.0): feas_avg=1.0, 10/12 modes, U=39.
    # PDL at (1e-3, 10): feas_avg=1.0, U=67.
    "avg_hard": {
        "mc":  {"inverse_beta": 300.0, "dual_step_size": 1.0},
        "pdl": {"primal_lr": 1e-3,     "dual_lr":        10.0},
        "T":   500,
        "beta_schedule": "cosine",
    },
    # No constraint-violation forcing → λ* = 0 trivially (centroid feasible).
    "medium_tight": {
        "mc":  {"inverse_beta": 10.0, "dual_step_size": 0.10},
        "pdl": {"primal_lr": 1e-3,    "dual_lr":         10.0},
        "T":   500,
    },
    # 2D illustrative preset (d=2, K=5, m=2). Used for body figures
    # (IB sweep panels + vector field). ds=0.5 works across ib range.
    "easy": {
        "mc":  {"inverse_beta": 3.0, "dual_step_size": 0.5},
        "pdl": {"primal_lr": 1e-4,   "dual_lr":        3.0},
        "T":   500,
    },
    # A.s.-infeasible 2D (d=2, K=3, m=2). All modes outside the feasible
    # wedge, so λ must rise substantially. Used for body figures.
    "infeasible": {
        "mc":  {"inverse_beta": 3.0, "dual_step_size": 0.5},
        "pdl": {"primal_lr": 1e-3,   "dual_lr":        5.0},
        "T":   500,
    },
}
