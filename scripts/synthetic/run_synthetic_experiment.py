#!/usr/bin/env python3
"""End-to-end synthetic constrained Boltzmann experiment.

Workflow (mirrors scripts/wra/diffusion/probe_option_a.py):
  1. Generate a random MoG + halfspace-constraint problem instance.
  2. MC baseline: sample with the analytical energy-guided DDPM.
  3. Train a score net via iDEM-style distillation (EnergyScoreTrainer).
  4. Trained-net inference: swap in the learned score, run the sampler.
  5. Baselines: unconstrained DDPM, rejection sampling.
  6. Evaluate metrics and produce figures.

Usage::

    python scripts/synthetic/run_synthetic_experiment.py --difficulty easy
    python scripts/synthetic/run_synthetic_experiment.py --difficulty medium --ib 1.0
    python scripts/synthetic/run_synthetic_experiment.py --difficulty hard --ib 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup (so local imports work when run from repo root)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from _experiment_paths import experiment_dir  # noqa: E402
from _shared import _shared_dual_ascent_step  # noqa: E402

from pdi.diffusion.synthetic_energy_ddpm import (  # noqa: E402
    SyntheticEnergyDDPM,
)
from pdi.models.synthetic_backbone import (  # noqa: E402
    SyntheticScoreBackbone,
)
from pdi.trainers.energy_score import (  # noqa: E402
    EnergyScoreTrainConfig,
    EnergyScoreTrainer,
    ScoreNetWithLambda,
)


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------
PRESETS = {
    "easy":   dict(d=2,   K=5,  m=2,   spread=6.0, cov_lo=1.5, cov_hi=3.0, target_feas=0.50, norm="global"),
    "medium": dict(d=50,  K=5, m=10,  spread=3.0, cov_lo=5.0, cov_hi=10.0, target_feas=0.80, norm="global"),
    # K=8 modes, tighter constraints, ≥3 mode centres driven infeasible so
    # constraints actually bind on mass near several modes (not just tails).
    "medium_hard": dict(d=50, K=8, m=12, spread=3.0, cov_lo=5.0, cov_hi=10.0,
                        target_feas=0.60, norm="global", min_infeasible_modes=3,
                        # Tighten b so every feasible centre sits within 0.5
                        # (original-coord units) of some constraint. Forces
                        # constraints to bind near feasible modes => dual
                        # variables actually rise during sampling.
                        max_feasible_margin=0.5),
    # Moderate d, more modes, mix of pair-direction + random constraint normals
    # so halfspaces slice THROUGH mode basins, not just between modes.
    # Picked with (1)+(3) strategy: reduced d + random normals.
    "medium_tight": dict(d=20, K=12, m=15, spread=3.0, cov_lo=3.0, cov_hi=6.0,
                         target_feas=0.40, norm="global",
                         min_infeasible_modes=4, max_feasible_margin=0.5,
                         random_constraint_frac=0.5),
    # Same geometry as medium_tight but with the `n_active_constraints`
    # loosest constraints shifted so the unconstrained MoG centroid violates
    # each by `centroid_target_violation` (original-coord units). Forces the
    # shared-λ optimum λ* > 0 across multiple active constraints.
    "medium_active": dict(d=20, K=12, m=15, spread=3.0, cov_lo=3.0, cov_hi=6.0,
                          target_feas=0.40, norm="global",
                          min_infeasible_modes=4, max_feasible_margin=0.5,
                          random_constraint_frac=0.5,
                          centroid_target_violation=2.0,
                          n_active_constraints=3),
    # All modes identical (same scalar covariance), well-spread centers,
    # uniform weights. Designed so the shared-λ teacher (MC Ceiling) does NOT
    # collapse to a single dominant mode under high constraint pressure —
    # every feasible mode is energetically equivalent.
    "medium_uniform": dict(d=20, K=12, m=15, spread=4.0,
                           cov_lo=4.0, cov_hi=4.0,
                           target_feas=0.40, norm="global",
                           min_infeasible_modes=4, max_feasible_margin=0.5,
                           random_constraint_frac=0.5,
                           centroid_target_violation=2.0,
                           n_active_constraints=3),
    # Tight-covariance variant of medium_uniform. Smaller cov keeps modes
    # close to the constraint boundary so the shared-λ optimum stays > 0
    # at t=0 (constraint is genuinely active at the constrained Boltzmann
    # optimum). Probe (2026-04-25): rej_acc=0.022, λ peak=1.65, λ at t=0 ≈ 1.21.
    "medium_hard":    dict(d=20, K=12, m=15, spread=4.0,
                           cov_lo=1.0, cov_hi=1.0,
                           target_feas=0.40, norm="global",
                           min_infeasible_modes=4, max_feasible_margin=0.5,
                           random_constraint_frac=0.5,
                           centroid_target_violation=2.0,
                           n_active_constraints=3),
    "high_dim":      dict(d=30, K=12, m=25, spread=4.0,
                          cov_lo=2.0, cov_hi=2.0,
                          target_feas=0.40, norm="global",
                          min_infeasible_modes=4, max_feasible_margin=0.5,
                          random_constraint_frac=0.5,
                          centroid_target_violation=2.0,
                          n_active_constraints=5),
    # Avg-constraint formulation: constraints designed so the unconstrained
    # MoG violates in expectation (E[Ax-b] < 0), forcing λ* > 0.
    # Thresholds b are set in normalised space as centroid_proj + margin,
    # where centroid ≈ 0, so the violation equals the margin.
    "avg_hard":      dict(d=30, K=12, m=10, spread=4.0,
                          cov_lo=2.0, cov_hi=2.0,
                          target_feas=0.50, norm="global",
                          avg_constraint_margins=(0.8, 2.0)),
    "hard":   dict(d=100, K=20, m=50,  spread=0.8, cov_lo=2.0, cov_hi=5.0, target_feas=0.20, norm="global"),
}


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def make_synthetic_problem(
    d: int,
    K_modes: int,
    m_constraints: int,
    *,
    spread: float = 3.0,
    cov_lo: float = 0.5,
    cov_hi: float = 1.5,
    target_feasibility: float = 0.50,
    seed: int = 0,
    n_calibration: int = 50_000,
    norm: str = "global",
    min_infeasible_modes: int = 0,
    max_feasible_margin: float | None = None,
    random_constraint_frac: float = 0.0,
    centroid_target_violation: float | None = None,
    n_active_constraints: int = 1,
    avg_constraint_margins: tuple[float, float] | None = None,
    avg_constraint_seed: int = 100,
) -> dict:
    """Generate a random MoG + halfspace constraint problem.

    The returned problem is **normalised** so that x has approximately
    zero mean and unit variance under the unconstrained MoG.  This
    ensures the DDPM prior N(0, I) overlaps with the target, which is
    critical for the energy-guided reverse process.

    The unnormalization shift/scale are stored so samples can be mapped
    back to the original coordinate system for reporting.

    Returns a dict with torch tensors ready for ``SyntheticEnergyDDPM``.
    """
    rng = np.random.RandomState(seed)

    # MoG parameters (original space)
    means = rng.randn(K_modes, d).astype(np.float64) * spread
    cov_diag = np.stack(
        [np.ones(d) * (cov_lo + rng.rand() * (cov_hi - cov_lo)) for _ in range(K_modes)]
    )
    weights = np.ones(K_modes, dtype=np.float64) / K_modes

    # Constraint normals: mix of (a) directions between mode pairs, selected
    # greedily for angular diversity, and (b) random directions on the unit
    # sphere.  Pair-direction normals slice BETWEEN modes (naturally loose);
    # random normals slice THROUGH mode basins (make feasibility bind inside
    # the basin).  ``random_constraint_frac`` controls the mix.
    n_random = int(round(m_constraints * float(random_constraint_frac)))
    n_pair = m_constraints - n_random

    A = np.zeros((m_constraints, d), dtype=np.float64)
    if n_pair > 0:
        pairs = [(i, j) for i in range(K_modes) for j in range(i + 1, K_modes)]
        candidate_dirs = np.array([
            (means[j] - means[i]) / np.linalg.norm(means[j] - means[i])
            for i, j in pairs
        ])                                                          # [n_pairs, d]
        selected = [rng.randint(len(pairs))]
        for _ in range(n_pair - 1):
            best_idx, best_min_angle = -1, -1.0
            for c in range(len(pairs)):
                if c in selected:
                    continue
                cos_sims = np.abs(candidate_dirs[selected] @ candidate_dirs[c])
                min_angle = 1.0 - cos_sims.max()
                if min_angle > best_min_angle:
                    best_min_angle = min_angle
                    best_idx = c
            selected.append(best_idx)
        for idx, sel in enumerate(selected):
            A[idx] = candidate_dirs[sel]
    for idx in range(n_pair, m_constraints):
        r = rng.randn(d)
        A[idx] = r / np.linalg.norm(r)

    # --- Calibrate thresholds b so overall feasibility ≈ target -----------
    # Sample from unconstrained MoG
    ks = rng.choice(K_modes, size=n_calibration, p=weights)
    samples = np.stack([
        rng.multivariate_normal(means[k], np.diag(cov_diag[k]))
        for k in ks
    ])  # [n_cal, d]

    projections = (A @ samples.T)  # [m, n_cal]

    # Binary search over a shared quantile level q for all constraints
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        b_trial = np.percentile(projections, mid * 100, axis=1)  # [m]
        satisfied = (projections.T >= b_trial).all(axis=1)       # [n_cal]
        feas = satisfied.mean()
        if feas > target_feasibility:
            lo = mid   # tighten
        else:
            hi = mid   # loosen
    b = np.percentile(projections, ((lo + hi) / 2.0) * 100, axis=1)

    # Tighten b per-constraint until ≥min_infeasible_modes centres are infeasible.
    # Strategy: for each mode we want to drive infeasible, find the constraint
    # with the smallest margin (a_j^T mu_k - b_j > 0) and raise b_j just above
    # a_j^T mu_k.  This makes the centre infeasible w.r.t. that constraint.
    mode_proj = A @ means.T                                          # [m, K]
    slack = mode_proj - b[:, None]                                   # [m, K]
    infeas = (slack < 0).any(axis=0)                                 # [K]
    if infeas.sum() < min_infeasible_modes:
        feasible_centres = np.where(~infeas)[0]
        # Rank feasible centres by smallest minimum margin (easiest to cut)
        min_margins = slack[:, feasible_centres].min(axis=0)         # [n_feas]
        order = np.argsort(min_margins)                              # ascending
        need = int(min_infeasible_modes - infeas.sum())
        for k_local in order[:need]:
            k = feasible_centres[k_local]
            # Find constraint with smallest positive slack -> raise b to kill it
            j = int(np.argmin(slack[:, k]))
            b[j] = mode_proj[j, k] + 1e-3
            slack = mode_proj - b[:, None]
    # Optionally tighten b_j so each feasible centre's MIN margin to the
    # polytope is <= max_feasible_margin (in original-coordinate units).
    # Without this, feasible centres often sit far from every boundary and
    # the sampler's dual variables stay ~0 because constraints don't bind.
    if max_feasible_margin is not None and max_feasible_margin > 0:
        mode_proj2 = A @ means.T                                 # [m, K]
        slack2 = mode_proj2 - b[:, None]                         # [m, K]
        infeas_mask = (slack2 < 0).any(axis=0)                   # [K]
        feasible_centres = np.where(~infeas_mask)[0].tolist()
        # Iterate; each step picks the feasible centre with the largest
        # min-margin (furthest from polytope boundary) and tightens one
        # constraint.  We only raise b_j by an amount that keeps every
        # OTHER feasible centre non-negative (prevents flipping modes).
        for _iter in range(8 * K_modes):
            if not feasible_centres:
                break
            min_margins = slack2[:, feasible_centres].min(axis=0)
            worst = int(np.argmax(min_margins))
            if min_margins[worst] <= max_feasible_margin + 1e-6:
                break                                            # already tight enough
            k = feasible_centres[worst]
            # Try constraints in order of decreasing slack for centre k.
            # Pick the first one where tightening to target margin doesn't
            # flip any other feasible centre to infeasible.
            order = np.argsort(-slack2[:, k])
            done = False
            for j in order:
                other_feas = [kk for kk in feasible_centres if kk != k]
                desired_b_j = mode_proj2[j, k] - float(max_feasible_margin)
                if other_feas:
                    # Max raise without flipping: keep other centres' slacks >=0
                    max_safe_b_j = float(mode_proj2[j, other_feas].min())
                else:
                    max_safe_b_j = desired_b_j
                new_b_j = min(desired_b_j, max_safe_b_j)
                if new_b_j > b[j] + 1e-9:
                    b[j] = new_b_j
                    slack2 = mode_proj2 - b[:, None]
                    done = True
                    break
            if not done:
                # Can't tighten any constraint for k without flipping others.
                # Remove k from active list (accept its current margin).
                feasible_centres.remove(k)
    # Optional: shift the `n_active_constraints` loosest constraints so the
    # unconstrained MoG centroid violates each by `centroid_target_violation`
    # (original-coord units). This forces the shared-λ optimum λ* > 0 with
    # multiple active constraints.
    if centroid_target_violation is not None and centroid_target_violation > 0:
        mu_unc = (weights[:, None] * means).sum(axis=0)         # [d]
        margin = A @ mu_unc - b                                  # [m]
        n_active = int(max(1, n_active_constraints))
        n_active = min(n_active, m_constraints)
        j_order = np.argsort(margin)[:n_active]                  # smallest-margin first
        for j in j_order:
            b[int(j)] = float(A[int(j)] @ mu_unc + float(centroid_target_violation))

    final_feas = ((projections.T >= b).all(axis=1)).mean()
    final_infeas = int(((A @ means.T) < b[:, None]).any(axis=0).sum())

    # ------------------------------------------------------------------
    # Normalise:  x_norm = (x - shift) / scale
    # norm="intra": scale by average intra-mode std (modes stay wide but
    #   mode centres may end up beyond the N(0,I) prior's reach)
    # norm="global": scale by total-variance std so the overall MoG has
    #   unit variance and all mode centres fall within sqrt(d) of origin
    # ------------------------------------------------------------------
    shift = (weights[:, None] * means).sum(axis=0)               # [d]
    avg_intra_var = (weights[:, None] * cov_diag).sum(axis=0)    # [d]
    if norm == "intra":
        scale = np.sqrt(np.maximum(avg_intra_var, 1e-8))          # [d]
    elif norm == "global":
        inter_mode_var = (weights[:, None] * (means - shift) ** 2).sum(axis=0)
        global_var = avg_intra_var + inter_mode_var
        scale = np.sqrt(np.maximum(global_var, 1e-8))             # [d]
    else:
        raise ValueError(f"Unknown norm: {norm!r}")

    means_norm = (means - shift) / scale                          # [K, d]
    cov_diag_norm = cov_diag / (scale ** 2)                       # [K, d]

    # Constraint transform:  a^T x >= b
    #   ↔ a^T (scale * x_norm + shift) >= b
    #   ↔ (a ⊙ scale)^T x_norm >= b - a^T shift
    A_norm = A * scale                                            # [m, d]
    b_norm = b - A @ shift                                        # [m]

    # ------------------------------------------------------------------
    # Optional: replace constraints with ones designed for the
    # avg-constraint formulation  E_p[Ax - b] >= 0.
    #
    # In normalised space, E[x_norm] ≈ 0, so setting b_norm > 0 makes the
    # unconstrained distribution violate in expectation (E[rate] = -b < 0),
    # forcing lambda* > 0.  Margins are drawn uniformly in
    # [margin_lo, margin_hi].  Joint feasibility (exists a mode
    # re-weighting that satisfies all constraints simultaneously) is
    # verified via LP; infeasible constraints are relaxed automatically.
    # ------------------------------------------------------------------
    if avg_constraint_margins is not None:
        margin_lo, margin_hi = avg_constraint_margins
        rng_avg = np.random.RandomState(avg_constraint_seed)

        A_avg = rng_avg.randn(m_constraints, d).astype(np.float64)
        A_avg /= np.linalg.norm(A_avg, axis=1, keepdims=True)

        proj_norm = A_avg @ means_norm.T                         # [m, K]
        centroid_proj = proj_norm @ weights                      # [m]

        margins = np.linspace(margin_lo, margin_hi, m_constraints)
        rng_avg.shuffle(margins)
        b_avg = centroid_proj + margins                          # [m]

        # Note: no LP feasibility check — the Boltzmann mean-shift
        # mechanism (tilting by exp(λ^T A x)) can satisfy any finite
        # margin by shifting mode means, so the problem is always
        # feasible for sufficiently large λ*.  The sweep tests whether
        # the sampler can *find* that λ* within T steps.

        A_norm = A_avg
        b_norm = b_avg

    return {
        "d": d,
        "K": K_modes,
        "m": m_constraints,
        # Normalised params (feed to SyntheticEnergyDDPM)
        "means": torch.from_numpy(means_norm).float(),
        "cov_diag": torch.from_numpy(cov_diag_norm).float(),
        "weights": torch.from_numpy(weights).float(),
        "constraint_A": torch.from_numpy(A_norm).float(),
        "constraint_b": torch.from_numpy(b_norm).float(),
        "calibrated_feasibility": float(final_feas),
        "n_infeasible_modes": int(final_infeas),
        # Unnormalization: x_original = scale * x_norm + shift
        "norm_shift": torch.from_numpy(shift).float(),
        "norm_scale": torch.from_numpy(scale).float(),
        # Original-space params (for reporting / mode coverage)
        "means_original": torch.from_numpy(means).float(),
        "cov_diag_original": torch.from_numpy(cov_diag).float(),
    }


# ---------------------------------------------------------------------------
# Synthetic data loader (yields dummy PyG batches for the trainer)
# ---------------------------------------------------------------------------

class _SyntheticDataLoader:
    """Yields identical dummy PyG Batch objects for the trainer."""

    def __init__(self, d: int, batch_size: int, num_batches: int = 1):
        self._batch = SyntheticEnergyDDPM.make_dummy_data(batch_size, d)
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            yield self._batch

    def __len__(self):
        return self._num_batches


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(
    x0: torch.Tensor,
    sampler: SyntheticEnergyDDPM,
    label: str,
    log: logging.Logger,
) -> dict:
    """Compute synthetic metrics for a batch of samples x0 [B, 1, d, 1]."""
    ctx = sampler._build_energy_context(
        data=None, batch_size=x0.shape[0], num_nodes=x0.shape[2],
        device=x0.device, dtype=x0.dtype,
    )
    m = ctx.num_real_constraints

    with torch.no_grad():
        x_flat = x0[:, 0, :, 0]                                     # [B, d]
        U = sampler._mog_potential(x_flat, ctx)                      # [B]
        rates = sampler._constraint_values(x_flat, ctx)              # [B, d]
        real_rates = rates[:, :m]                                    # [B, m]

        # Per-sample: fraction of real constraints satisfied
        feas_per_sample = (real_rates >= -1e-6).float().mean(dim=1)  # [B]
        # Fully feasible = all m constraints satisfied
        fully_feas = (real_rates >= -1e-6).all(dim=1).float()        # [B]
        # Violation
        violation = (-real_rates).clamp_min(0.0)

    U_np = U.cpu().numpy()
    feas_np = feas_per_sample.cpu().numpy()
    full_feas_np = fully_feas.cpu().numpy()
    vio_np = violation.cpu().numpy()

    # Avg-constraint feasibility: fraction of constraints where batch-mean is satisfied
    with torch.no_grad():
        feas_avg = float(((real_rates.mean(0)) >= -1e-6).float().mean().item())

    metrics = {
        "label": label,
        "potential_mean": float(U_np.mean()),
        "potential_std": float(U_np.std()),
        "feas_avg": feas_avg,
        "full_feasibility": float(full_feas_np.mean()),
        "feasibility_per_constraint": float(feas_np.mean()),
        "mean_violation": float(vio_np.mean()),
        "max_violation": float(vio_np.max()),
    }
    log.info(
        "[%s]  U=%.2f +/- %.2f  feas_avg=%.3f  vio=%.4f/%.4f",
        label, metrics["potential_mean"], metrics["potential_std"],
        feas_avg,
        metrics["mean_violation"], metrics["max_violation"],
    )
    return metrics


def _mode_counts(
    x0: torch.Tensor,
    sampler: SyntheticEnergyDDPM,
    problem: dict,
    mahal_threshold: float | None = None,
) -> tuple[np.ndarray, int, np.ndarray, int, int]:
    """Return (counts, unique_modes, feas_counts, unique_feas_modes, n_unlabelled).

    Each sample is assigned to its Mahalanobis-closest MoG mode, but only
    if that Mahalanobis distance^2 is below ``mahal_threshold`` (by default
    the chi-squared 0.95 quantile for d degrees of freedom). Samples beyond
    every mode's ball are labelled -1 (unassigned) and excluded from
    ``counts`` / ``feas_counts``.

    This prevents the "label artefact" in which aggressive dual ascent
    pushes a sample far from every MoG centre but argmax-posterior still
    assigns it to the nearest-in-Mahalanobis centre (typically a wide
    infeasible mode like mode 6 in the medium_tight problem).

    ``feas_counts[k]`` further restricts to samples that are also
    geographically feasible (all m real constraints satisfied). An
    infeasible-centre mode can still contribute coverage via its feasible
    portion.
    """
    B = x0.shape[0]
    device = x0.device
    d = int(problem["d"])
    if mahal_threshold is None:
        # chi-squared 0.95 quantile for d dof; lazy import to avoid scipy dep
        # when unused. Fallback to closed-form approximation if scipy missing.
        try:
            from scipy.stats import chi2
            mahal_threshold = float(chi2.ppf(0.95, d))
        except Exception:
            # Wilson-Hilferty approximation: chi2_{d,p} ≈ d * (1 - 2/(9d) + z*sqrt(2/(9d)))^3
            z = 1.6449  # N(0,1) 0.95 quantile
            mahal_threshold = d * (1 - 2 / (9 * d) + z * (2 / (9 * d)) ** 0.5) ** 3
    ctx = sampler._build_energy_context(
        data=None, batch_size=B, num_nodes=d, device=device, dtype=torch.float32,
    )
    m_real = int(ctx.num_real_constraints)
    with torch.no_grad():
        xf = x0[:, 0, :, 0]
        diff = xf.unsqueeze(1) - ctx.means.unsqueeze(0)
        mahal = (diff.square() * ctx.precision_diag.unsqueeze(0)).sum(dim=2)
        log_comp = ctx.log_norm_const + ctx.log_weights - 0.5 * mahal
        assign = log_comp.argmax(dim=1)
        min_mahal = mahal.gather(1, assign.unsqueeze(1)).squeeze(1)
        unassigned = min_mahal > mahal_threshold
        assign[unassigned] = -1
        assign_np = assign.cpu().numpy()
        rates = sampler._constraint_values(xf, ctx)[:, :m_real]
        feas_mask = (rates >= -1e-6).all(dim=1).cpu().numpy()
    K_tot = int(problem["K"])
    labelled = assign_np >= 0
    counts = np.bincount(assign_np[labelled], minlength=K_tot)
    uniq = int((counts > 0).sum())
    feas_labelled = labelled & feas_mask
    feas_counts = np.bincount(assign_np[feas_labelled], minlength=K_tot)
    uniq_feas = int((feas_counts > 0).sum())
    n_unlabelled = int((~labelled).sum())
    return counts, uniq, feas_counts, uniq_feas, n_unlabelled


def _mode_coverage(
    x0: torch.Tensor,
    problem: dict,
    sigma_threshold: float = 2.0,
    min_frac: float = 0.05,
) -> float:
    """DEPRECATED: use ``_mode_counts`` instead.

    Mahalanobis-based mode coverage. Reported as 1.0 for essentially any
    batch in d>=50 because ``2*sqrt(d)`` is far too loose. Kept only so
    external callers don't break.

    A mode is "covered" if at least ``min_frac * B / K`` samples land
    within ``sigma_threshold * sqrt(d)`` Mahalanobis distance of its
    centre (i.e. at least 5% of the expected uniform share by default).

    Samples are unnormalised to the original coordinate system before
    comparing against the original mode centres.
    """
    x_flat = x0[:, 0, :, 0].cpu()                                    # [B, d]
    B = x_flat.shape[0]
    x_orig = x_flat * problem["norm_scale"].cpu() + problem["norm_shift"].cpu()
    means_orig = problem["means_original"].cpu()
    cov_orig = problem["cov_diag_original"].cpu()
    K = means_orig.shape[0]
    d = means_orig.shape[1]
    min_count = max(3, int(math.ceil(min_frac * B)))
    covered = 0
    for k in range(K):
        diff = x_orig - means_orig[k]                                # [B, d]
        dist = (diff.square() / cov_orig[k]).sum(dim=1).sqrt()      # [B]
        n_near = int((dist < sigma_threshold * math.sqrt(d)).sum().item())
        if n_near >= min_count:
            covered += 1
    return covered / K


def _pd_langevin_sample(
    problem: dict,
    B: int,
    T: int,
    lr: float,
    dual_lr: float,
    device: torch.device,
    seed: int = 0,
    shared: bool = False,
) -> torch.Tensor:
    """Canonical primal-dual Langevin MCMC sampler (per-sample λ) in normalised
    coords.

    Target: π(x) ∝ exp(-U(x))·1{Ax ≥ b}, enforced via per-sample Lagrangian
      L(x, λ) = U(x) - λᵀ(A x - b),
    primal: x ← x - lr·(∇U - Aᵀλ) + √(2·lr)·ξ,  ξ ~ N(0, I)
    dual:   λ ← [λ + dual_lr·(b - A x)]_+

    Returns x0 as a [B, 1, d, 1] tensor to match the DDPM sampler convention.
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    means = problem["means"].to(device).float()         # [K, d]
    cov = problem["cov_diag"].to(device).float()        # [K, d]
    log_w = torch.log(problem["weights"].to(device).float())  # [K]
    A = problem["constraint_A"].to(device).float()      # [m_pad, d]
    b = problem["constraint_b"].to(device).float()      # [m_pad]
    m_real = int((A.abs().sum(dim=1) > 0).sum().item()) # pad rows are zero
    A_real = A[:m_real]; b_real = b[:m_real]
    d = int(problem["d"])

    x = torch.randn((B, d), generator=g, device=device)  # prior N(0, I)
    lam = torch.zeros((B, m_real), device=device)
    log_norm = -0.5 * d * math.log(2 * math.pi) - 0.5 * torch.log(cov).sum(dim=1)  # [K]

    for _ in range(T):
        # MoG score (grad log p_MoG)
        diff = x.unsqueeze(1) - means.unsqueeze(0)             # [B, K, d]
        mahal = (diff.square() / cov.unsqueeze(0)).sum(dim=2)  # [B, K]
        log_comp = log_w + log_norm - 0.5 * mahal              # [B, K]
        resp = torch.softmax(log_comp, dim=1)                  # [B, K]
        grad_logp = (resp.unsqueeze(-1) * (-diff) / cov.unsqueeze(0)).sum(dim=1)  # [B, d]
        grad_U = -grad_logp                                    # U = -log p
        # Lagrangian term: grad_x sum_j lam_j h_j = grad_x lam^T (b - A x) = -A^T lam
        grad_L = -(lam @ A_real)                               # [B, d]
        drift = -(grad_U + grad_L)
        noise = torch.randn_like(x)
        x = x + lr * drift + math.sqrt(2 * lr) * noise
        # Dual ascent: per-sample by default; shared = single scalar broadcast.
        h = b_real - (x @ A_real.T)                            # [B, m]
        if shared:
            lam_scalar = torch.clamp(
                lam.mean(0, keepdim=True) + dual_lr * h.mean(0, keepdim=True),
                min=0.0,
            )
            lam = lam_scalar.expand_as(lam).contiguous()
        else:
            lam = torch.clamp(lam + dual_lr * h, min=0.0)
    return x.view(B, 1, d, 1)

def _rejection_sample(
    problem: dict,
    n_target: int,
    device: torch.device,
    max_attempts: int = 2_000_000,
) -> tuple[torch.Tensor, float]:
    """Sample from unconstrained MoG (normalised space), reject infeasible.

    Returns (x0_norm [n, 1, d, 1], accept_rate).
    """
    means = problem["means"].numpy()       # normalised
    cov_diag = problem["cov_diag"].numpy() # normalised
    weights = problem["weights"].numpy()
    A = problem["constraint_A"].numpy()    # normalised
    b = problem["constraint_b"].numpy()    # normalised
    d = problem["d"]
    K = problem["K"]

    accepted = []
    n_tried = 0
    rng = np.random.RandomState(99)
    while len(accepted) < n_target and n_tried < max_attempts:
        batch = min(10_000, max_attempts - n_tried)
        ks = rng.choice(K, size=batch, p=weights)
        xs = np.stack([
            rng.multivariate_normal(means[k], np.diag(cov_diag[k]))
            for k in ks
        ])
        n_tried += batch
        feas_mask = (xs @ A.T >= b).all(axis=1)
        accepted.extend(xs[feas_mask].tolist())

    accepted = np.array(accepted[:n_target])
    accept_rate = len(accepted) / max(n_tried, 1)
    x0 = torch.from_numpy(accepted).float().to(device)
    return x0.unsqueeze(1).unsqueeze(-1), accept_rate


# ---------------------------------------------------------------------------
# Score-net inference helper (same pattern as probe_option_a.py)
# ---------------------------------------------------------------------------

def _make_trained_score_fn(score_net, alphas_cumprod, d):
    """Return a bound method that replaces _estimate_score with the trained net."""
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


# ---------------------------------------------------------------------------
# ESS calibration (mirrors scripts/wra/diffusion/probe_ess_beta.py)
# ---------------------------------------------------------------------------

def _ess_over_k(raw_energies: torch.Tensor, ib: float) -> float:
    """Compute mean ESS/K for a batch of raw energies.

    Parameters
    ----------
    raw_energies : Tensor [K, B]
        Energies computed at inverse_beta=1.
    ib : float
        Candidate inverse_beta to evaluate.

    Returns
    -------
    float
        Mean ESS/K across the batch.  1.0 = uniform weights (healthy),
        1/K = collapsed to a single sample (broken).
    """
    E = raw_energies * ib                                             # [K, B]
    log_w = -E - torch.logsumexp(-E, dim=0, keepdim=True)            # [K, B]
    w = log_w.exp()
    ess = 1.0 / w.pow(2).sum(dim=0).clamp_min(1e-30)                # [B]
    K = raw_energies.shape[0]
    return float((ess / K).mean().item())


def _calibrate_inverse_beta(
    problem: dict,
    T: int,
    K: int,
    batch_size: int,
    device: torch.device,
    log: logging.Logger,
    target_ess_over_k: float = 0.5,
) -> float:
    """Find inverse_beta that keeps ESS/K near the target (default 0.5).

    Procedure (mirrors probe_ess_beta.py from the WRA tuning):
      1. Build a temporary sampler at inverse_beta=1.
      2. Draw random x_t and dual_lambda at a mid-range timestep.
      3. Compute K raw energies (at ib=1) via the MC perturbation loop.
      4. Binary-search over ib to find ESS/K ≈ target.

    The raw energies scale linearly with ib (E = ib * raw), so we only
    need one forward pass and can rescale analytically.
    """
    d = problem["d"]

    sampler = SyntheticEnergyDDPM(
        means=problem["means"], cov_diag=problem["cov_diag"],
        weights=problem["weights"],
        constraint_A=problem["constraint_A"], constraint_b=problem["constraint_b"],
        num_timesteps=T, energy_mc_samples=K,
        inverse_beta=1.0, inverse_beta_schedule="constant",
        dual_update_mode="x0_pred", dual_step_size=0.01,
        dual_lambda_init=0.0,
    ).to(device)
    for p in sampler.parameters():
        p.requires_grad_(False)

    ctx = sampler._build_energy_context(
        data=None, batch_size=batch_size, num_nodes=d,
        device=device, dtype=torch.float32,
    )

    # Probe at a mid-range timestep where sigma_t is moderate
    t_probe = T // 2
    t_tensor = torch.full((batch_size,), t_probe, device=device, dtype=torch.long)
    sigma_t = sampler._gather(
        sampler.sqrt_one_minus_alphas_cumprod, t_tensor,
    ).view(-1, 1, 1, 1).clamp_min(sampler.min_energy_sigma)

    torch.manual_seed(7777)
    x_t = torch.randn(batch_size, 1, d, 1, device=device)
    # Use moderate dual_lambda (not zero — we want to probe the regime
    # where constraints are active, matching training conditions)
    dual_lambda = torch.full((batch_size, d), 5.0, device=device)

    # Compute K raw energies at ib=1
    with torch.no_grad():
        raw_list = []
        for _ in range(K):
            x0_cand = x_t + torch.randn_like(x_t) * sigma_t
            e = sampler._energy_from_samples(
                x0_cand, ctx, dual_lambda=dual_lambda, t=t_tensor,
                inverse_beta_override=torch.tensor([1.0], device=device),
            )
            raw_list.append(e)
        raw_E = torch.stack(raw_list, dim=0)  # [K, B]

    raw_std = float(raw_E.std(dim=0).mean().item())
    raw_range = float((raw_E.max(0).values - raw_E.min(0).values).mean().item())
    log.info("ESS probe: t=%d  raw_E_std=%.2f  raw_E_range=%.2f",
             t_probe, raw_std, raw_range)

    # Binary search over log-scale ib
    lo, hi = -8.0, 4.0  # ib in [1e-8, 1e4]
    for _ in range(64):
        mid = (lo + hi) / 2.0
        ib_trial = 10.0 ** mid
        ess_k = _ess_over_k(raw_E, ib_trial)
        if ess_k > target_ess_over_k:
            lo = mid   # ESS too high → increase ib to sharpen softmax
        else:
            hi = mid   # ESS too low → decrease ib to flatten softmax

    ib_calibrated = 10.0 ** ((lo + hi) / 2.0)
    ess_final = _ess_over_k(raw_E, ib_calibrated)

    # Also report a grid for context
    ib_grid = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    log.info("ESS/K grid (K=%d, t=%d):", K, t_probe)
    for ib_g in ib_grid:
        log.info("  ib=%-10g  ESS/K=%.3f", ib_g, _ess_over_k(raw_E, ib_g))

    log.info("calibrated inverse_beta=%.4g  (ESS/K=%.3f, target=%.2f)",
             ib_calibrated, ess_final, target_ess_over_k)
    return ib_calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic constrained Boltzmann experiment")
    parser.add_argument("--difficulty", choices=list(PRESETS), default="easy")
    parser.add_argument("--ib", type=float, default=None,
                        help="inverse_beta (default: auto-calibrated from ESS probe)")
    parser.add_argument("--num-timesteps", type=int, default=500)
    parser.add_argument("--beta-schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "sigmoid"],
                        help="Noise schedule (default: cosine)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--energy-mc-samples", type=int, default=64)
    parser.add_argument("--train-mc-samples", type=int, default=64, help="k_train for MC target")
    parser.add_argument("--num-outer", type=int, default=200, help="outer training iterations")
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=256, help="MLP hidden dim")
    parser.add_argument("--num-layers", type=int, default=4, help="MLP residual blocks")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dual-step-size", type=float, default=0.01,
                        help="Dual ascent step size η. Default 0.01 avoids "
                             "the per-sample dual-runaway pathology; raise "
                             "only if you know the problem tolerates it.")
    parser.add_argument("--dual-lambda-init", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-ess-probe", action="store_true",
                        help="skip ESS calibration (use --ib directly)")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Run a small trained-net sample + compute U/feas every "
                             "N iters. 0 disables.")
    parser.add_argument("--eval-batch", type=int, default=512,
                        help="Batch size for the training-time eval samples "
                             "(used for best-model selection).  Keep >=512 "
                             "so unique_modes is a reliable signal.")
    parser.add_argument("--post-eval-batch", type=int, default=1024,
                        help="Batch size for the POST-TRAINING comparison "
                             "eval.  Decoupled from --batch-size so the "
                             "final report uses enough samples for stable "
                             "per-mode counts.")
    parser.add_argument("--prior-mu-min", type=float, default=1.0,
                        help="ExponentialLambdaPrior mu at t=0 (clean).")
    parser.add_argument("--prior-mu-max", type=float, default=20.0,
                        help="ExponentialLambdaPrior mu at t=T (noisy).")
    parser.add_argument("--importance-weight", action="store_true",
                        help="Reweight rollout samples in the buffer by "
                             "inverse mode frequency to counter distillation "
                             "mode collapse.")
    parser.add_argument("--run-label", type=str, default=None,
                        help="Optional extra suffix appended to the output "
                             "directory name so parallel runs do not collide.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint .pt file to initialize "
                             "the score net before training (two-stage training).")
    parser.add_argument("--target-normalize-eps", type=float, default=1e-3,
                        help="Floor for per-sample target RMS normalizer. "
                             "Raise (e.g. 0.1) to stop the floor amplifying "
                             "tiny-target samples into O(1/eps) residuals.")
    parser.add_argument("--constraint-term-clamp", type=float, default=None,
                        help="Cap each lambda_j * h_j(x) term (element-wise) "
                             "at this max value when evaluating the MC "
                             "energy during TRAINING only. Prevents the "
                             "constraint penalty from exploding at high "
                             "inverse_beta * lambda and underflowing the "
                             "Boltzmann weight. Inference samplers always "
                             "use the unclamped tempered posterior.")
    parser.add_argument("--rho-max", type=float, default=0.0,
                        help="Max fraction of minibatch rows that use the "
                             "coupled on-policy lambda (rest draw from the "
                             "decoupled prior). Linearly ramped from 0 over "
                             "warmup_iters SGD steps. Default 0 = fully "
                             "decoupled lambda training (historical default).")
    parser.add_argument("--perturb-fraction", type=float, default=0.0,
                        help="Fraction of train samples to perturb x_t and λ.")
    parser.add_argument("--perturb-x-std", type=float, default=0.1,
                        help="Std of Gaussian noise added to perturbed x_t.")
    parser.add_argument("--perturb-lambda-std", type=float, default=0.5,
                        help="Std of Gaussian noise added to perturbed λ (clamped ≥0).")
    parser.add_argument("--num-rollouts-per-outer", type=int, default=1,
                        help="Number of independent rollouts per outer iteration.")
    parser.add_argument("--dual-lambda-max", type=float, default=None,
                        help="Cap dual lambda during rollouts.")
    parser.add_argument("--loss-weighting", type=str, default="one_minus_alpha_bar",
                        choices=["unit", "one_minus_alpha_bar", "min_snr"],
                        help="Loss weighting across timesteps.")
    parser.add_argument("--target-clip-norm", type=float, default=0.0,
                        help="Clip MC score target to this max norm per sample. "
                             "iDEM uses 20 for d~30. 0 = disabled.")
    parser.add_argument("--mc-rollout-fraction", type=float, default=0.0,
                        help="Fraction of outer rollouts using MC ceiling "
                             "instead of the score net (teacher-mixing).")
    parser.add_argument("--ib-schedule-start", type=float, default=None,
                        help="If set together with --ib-schedule-end, anneal "
                             "the MC sampler's inverse_beta from START at "
                             "outer=0 to END at the final outer iter. Applied "
                             "before each outer rollout. Leave unset to hold "
                             "ib fixed at --ib.")
    parser.add_argument("--ib-schedule-end", type=float, default=None,
                        help="See --ib-schedule-start.")
    parser.add_argument("--ib-schedule-shape", type=str, default="linear",
                        choices=["linear", "log"],
                        help="Interpolation shape for the ib schedule.")
    parser.add_argument("--ib-schedule-warmup-iters", type=int, default=None,
                        help="If set (in SGD iters), ramp ib over the first "
                             "N SGD iters then hold at --ib-schedule-end. "
                             "Converted internally to outer iters (= N / inner_steps).")
    parser.add_argument("--best-u-weight", type=float, default=0.01,
                        help="Pareto weight for U in best-checkpoint score "
                             "(score = feas - w*U). Higher = prefer low U; "
                             "lower = prefer high feas. Default 0.01 trades "
                             "1 feas unit for ~100 U units.")
    parser.add_argument("--copy-mode-cov", type=str, default=None,
                        help="After problem generation, copy the covariance "
                             "diagonal of one mode onto another. Format "
                             "'src:dst' (1-origin mode indices), e.g. '10:11'.")
    parser.add_argument("--shift-mode", type=str, default=None,
                        help="After problem generation, shift one mode's "
                             "centre into the feasible interior by DELTA "
                             "(normalised units) along the interior-pointing "
                             "normal of its tightest-slack constraint. "
                             "Format 'idx:delta', e.g. '11:0.5'.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("synthetic.main")

    preset = PRESETS[args.difficulty]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.num_timesteps

    # ------------------------------------------------------------------
    # 1. Generate problem
    # ------------------------------------------------------------------
    log.info("generating problem instance (seed=%d) ...", args.seed)
    problem = make_synthetic_problem(
        d=preset["d"], K_modes=preset["K"], m_constraints=preset["m"],
        spread=preset["spread"], cov_lo=preset["cov_lo"], cov_hi=preset["cov_hi"],
        target_feasibility=preset["target_feas"], seed=args.seed,
        norm=preset.get("norm", "global"),
        min_infeasible_modes=int(preset.get("min_infeasible_modes", 0)),
        max_feasible_margin=preset.get("max_feasible_margin", None),
        random_constraint_frac=float(preset.get("random_constraint_frac", 0.0)),
        centroid_target_violation=preset.get("centroid_target_violation"),
        n_active_constraints=int(preset.get("n_active_constraints", 1)),
        avg_constraint_margins=preset.get("avg_constraint_margins"),
    )
    log.info("calibrated feasibility: %.3f (target %.3f)  infeasible_centres=%d/%d",
             problem["calibrated_feasibility"], preset["target_feas"],
             problem["n_infeasible_modes"], preset["K"])

    # Optional post-processing tweaks to individual modes.
    if args.copy_mode_cov is not None:
        src, dst = (int(x) for x in args.copy_mode_cov.split(":"))
        problem["cov_diag"][dst] = problem["cov_diag"][src].clone()
        problem["cov_diag_original"][dst] = problem["cov_diag_original"][src].clone()
        log.info("tweak: copied cov of mode %d onto mode %d", src, dst)
    if args.shift_mode is not None:
        idx, delta = args.shift_mode.split(":")
        idx, delta = int(idx), float(delta)
        A = problem["constraint_A"]             # [m, d] (normalised)
        b = problem["constraint_b"]             # [m]
        mu = problem["means"][idx]              # [d]
        slacks = A @ mu - b                     # [m]; >=0 means feasible
        tight_j = int(slacks.argmin().item())
        normal = A[tight_j]                     # interior points in +A direction
        normal = normal / normal.norm().clamp_min(1e-12)
        problem["means"][idx] = mu + delta * normal
        problem["means_original"][idx] = (
            problem["means"][idx] * problem["norm_scale"] + problem["norm_shift"]
        )
        log.info(
            "tweak: shifted mode %d by %g along normal of constraint %d "
            "(old min slack=%.3f, new min slack=%.3f)",
            idx, delta, tight_j, slacks[tight_j].item(),
            ((problem["constraint_A"] @ problem["means"][idx]) - problem["constraint_b"]).min().item(),
        )

    d = problem["d"]
    shape = (args.batch_size, 1, d, 1)

    # ------------------------------------------------------------------
    # 2. ESS probe — calibrate inverse_beta to the energy scale
    # ------------------------------------------------------------------
    def _make_sampler(ib_val, **overrides):
        kw = dict(
            means=problem["means"], cov_diag=problem["cov_diag"],
            weights=problem["weights"],
            constraint_A=problem["constraint_A"], constraint_b=problem["constraint_b"],
            num_timesteps=T,
            beta_schedule=args.beta_schedule,
            energy_mc_samples=args.energy_mc_samples,
            inverse_beta=ib_val, inverse_beta_schedule="constant",
            dual_update_mode="x0_pred",
            dual_step_size=args.dual_step_size,
            dual_num_outer_iterations=1,
            dual_lambda_init=args.dual_lambda_init,
            dual_lambda_max=args.dual_lambda_max,
        )
        kw.update(overrides)
        return SyntheticEnergyDDPM(**kw).to(device)

    if args.ib is not None:
        ib = args.ib
        log.info("using user-specified inverse_beta=%g", ib)
    elif args.skip_ess_probe:
        ib = 1.0
        log.info("ESS probe skipped; using default inverse_beta=%g", ib)
    else:
        ib = _calibrate_inverse_beta(
            problem=problem, T=T, K=args.energy_mc_samples,
            batch_size=min(args.batch_size, 16), device=device, log=log,
        )

    _label = f"{args.difficulty}_d{preset['d']}_ib{ib:g}"
    if args.run_label:
        _label = f"{_label}_{args.run_label}"
    out_dir = experiment_dir(
        "score_net_train",
        _label,
    )
    log.info("output dir: %s", out_dir)
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info("difficulty=%s  d=%d  K=%d  m=%d  ib=%g  T=%d  device=%s",
             args.difficulty, preset["d"], preset["K"], preset["m"], ib, T, device)

    # Save full CLI config for reproducibility
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    # Save problem definition
    prob_save = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                 for k, v in problem.items()}
    (out_dir / "problem.json").write_text(json.dumps(prob_save, indent=2))

    # Training-only clamp on lambda_j * h_j(x). Only passed to this (training)
    # sampler — all other samplers built via _make_sampler default to no clamp
    # so the true tempered posterior is used at inference.
    mc_sampler = _make_sampler(ib, constraint_term_clamp=args.constraint_term_clamp)

    if preset.get("avg_constraint_margins") is not None:
        mc_sampler._dual_ascent_step = types.MethodType(
            _shared_dual_ascent_step, mc_sampler,
        )
        log.info("avg-constraint preset → using shared-λ dual ascent for training rollouts")

    # ------------------------------------------------------------------
    # 3. Build score net
    # ------------------------------------------------------------------
    dataset_cond_channels = 0  # no dataset conditioning for synthetic
    backbone = SyntheticScoreBackbone(
        d=d, hidden=args.hidden, num_layers=args.num_layers,
        num_timesteps=T,
        cond_channels=dataset_cond_channels + 1,  # +1 for lambda channel
    )
    score_net = ScoreNetWithLambda(
        backbone=backbone, expected_cond_feats=dataset_cond_channels,
    )
    n_params = sum(p.numel() for p in score_net.parameters())
    log.info("score_net params: %.2fM", n_params / 1e6)

    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        score_net.load_state_dict(state)
        log.info("resumed weights from %s", args.resume_from)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    cfg = EnergyScoreTrainConfig(
        lr=args.lr,
        loss_weighting=args.loss_weighting,
        k_train=args.train_mc_samples,
        inner_steps_per_outer=args.inner_steps,
        minibatch_size=4,
        rho_max=args.rho_max,
        warmup_iters=min(200, args.num_outer * args.inner_steps // 4),
        buffer_capacity=8192,
        lambda_prior_mu_min=args.prior_mu_min,
        lambda_prior_mu_max=args.prior_mu_max,
        target_normalize_eps=args.target_normalize_eps,
        target_clip_norm=args.target_clip_norm,
        perturb_fraction=args.perturb_fraction,
        perturb_x_std=args.perturb_x_std,
        perturb_lambda_std=args.perturb_lambda_std,
        num_rollouts_per_outer=args.num_rollouts_per_outer,
        mc_rollout_fraction=args.mc_rollout_fraction,
        ib_schedule_enabled=(
            args.ib_schedule_start is not None
            and args.ib_schedule_end is not None
        ),
        ib_schedule_start=(
            args.ib_schedule_start if args.ib_schedule_start is not None else args.ib
        ),
        ib_schedule_end=(
            args.ib_schedule_end if args.ib_schedule_end is not None else args.ib
        ),
        ib_schedule_shape=args.ib_schedule_shape,
        ib_schedule_warmup_outer_iters=(
            args.ib_schedule_warmup_iters // args.inner_steps
            if args.ib_schedule_warmup_iters is not None else None
        ),
    )
    trainer = EnergyScoreTrainer(
        score_net=score_net,
        mc_sampler=mc_sampler,
        config=cfg,
        device=device,
        rng_seed=args.seed,
    )
    trainer.importance_weight_modes = bool(args.importance_weight)
    if trainer.importance_weight_modes:
        log.info("importance-weight rollouts: ON (inverse mode frequency)")

    data_loader = _SyntheticDataLoader(d=d, batch_size=args.batch_size, num_batches=1)

    log.info("training for %d outer x %d inner = %d SGD steps ...",
             args.num_outer, args.inner_steps, args.num_outer * args.inner_steps)

    t0 = time.time()

    # Pre-build a sampler that uses the current score_net (no MC) for periodic
    # training-time evaluation.  This is the same plumbing as the post-training
    # "trained_net" eval, but reused from a small batch every _eval_every iters.
    _eval_sampler = _make_sampler(ib)
    if preset.get("avg_constraint_margins") is not None:
        _eval_sampler._dual_ascent_step = types.MethodType(
            _shared_dual_ascent_step, _eval_sampler,
        )
    _eval_sampler._estimate_score = types.MethodType(
        _make_trained_score_fn(score_net, _eval_sampler.alphas_cumprod, d),
        _eval_sampler,
    )
    _eval_metric_sampler = _make_sampler(ib, energy_mc_samples=64)
    if preset.get("avg_constraint_margins") is not None:
        _eval_metric_sampler._dual_ascent_step = types.MethodType(
            _shared_dual_ascent_step, _eval_metric_sampler,
        )
    _eval_every = args.eval_every
    _eval_batch = args.eval_batch
    _eval_shape = (_eval_batch, 1, d, 1)

    # Best-model tracking: save three checkpoints per run —
    #   score_net_best_pareto.pt : max (feas - best_u_weight * U)
    #   score_net_best_u.pt      : min U
    #   score_net_best_feas.pt   : max feas (ties broken by min U)
    # Also keeps score_net_best.pt as alias for Pareto for backward compat.
    _best_u_weight = float(args.best_u_weight)
    _best_pareto = {"key": None, "iter": -1, "U": None, "feas": None, "modes": None}
    _best_u = {"key": None, "iter": -1, "U": None, "feas": None, "modes": None}
    _best_feas = {"key": None, "iter": -1, "U": None, "feas": None, "modes": None}
    _ckpt_pareto = out_dir / "score_net_best_pareto.pt"
    _ckpt_u = out_dir / "score_net_best_u.pt"
    _ckpt_feas = out_dir / "score_net_best_feas.pt"
    _best_ckpt_path = out_dir / "score_net_best.pt"  # alias → Pareto

    def _log_hook(info):
        it = info["iter"]
        if it % 25 == 0 or it <= 5:
            extra = ""
            if _eval_every > 0 and (it % _eval_every == 0 or it == 1):
                score_net.eval()
                with torch.no_grad():
                    torch.manual_seed(42 + it)
                    x0_eval = _eval_sampler.sample(
                        shape=_eval_shape, device=device,
                    )
                score_net.train()
                ctx = _eval_metric_sampler._build_energy_context(
                    data=None, batch_size=_eval_batch, num_nodes=d,
                    device=device, dtype=torch.float32,
                )
                m_n = ctx.num_real_constraints
                with torch.no_grad():
                    x_flat = x0_eval[:, 0, :, 0]
                    U_eval = _eval_metric_sampler._mog_potential(
                        x_flat, ctx,
                    ).mean().item()
                    rates = _eval_metric_sampler._constraint_values(
                        x_flat, ctx,
                    )[:, :m_n]
                    feas_full_eval = float(
                        (rates >= -1e-6).all(dim=1).float().mean()
                    )
                    feas_per_eval = float(
                        (rates >= -1e-6).float().mean()
                    )
                    # Mode assignment with Mahalanobis distance gating:
                    # samples whose Mahalanobis^2 to the nearest mode exceeds
                    # the chi-squared 0.95 quantile for d dof are unlabelled.
                    diff = x_flat.unsqueeze(1) - ctx.means.unsqueeze(0)
                    mahal = (diff.square() * ctx.precision_diag.unsqueeze(0)).sum(dim=2)
                    log_comp = ctx.log_norm_const + ctx.log_weights - 0.5 * mahal
                    assign = log_comp.argmax(dim=1)
                    min_mahal = mahal.gather(1, assign.unsqueeze(1)).squeeze(1)
                    # Wilson-Hilferty chi2_{d,0.95} without scipy dep:
                    _z = 1.6449
                    _chi2_th = d * (1 - 2 / (9 * d) + _z * (2 / (9 * d)) ** 0.5) ** 3
                    in_ball = min_mahal <= _chi2_th
                    assign_thr = assign.clone()
                    assign_thr[~in_ball] = problem["K"]  # sentinel "unassigned"
                    counts = torch.bincount(assign_thr, minlength=problem["K"] + 1)[: problem["K"]]
                    # Feasibility-aware mode counts: only count samples that
                    # are (labelled) AND (argmax-posterior k) AND (feasible).
                    feas_mask = (rates >= -1e-6).all(dim=1)
                    feas_label_mask = feas_mask & in_ball
                    feas_assign = assign[feas_label_mask]
                    feas_counts = torch.bincount(feas_assign, minlength=problem["K"])
                    # Require a mode to have >= 2% of samples (or >= 3) to count.
                    min_count = max(3, int(0.02 * _eval_batch))
                    unique_modes_eval = int((counts >= min_count).sum().item())
                    unique_feas_modes = int((feas_counts >= min_count).sum().item())
                    n_unlabelled_eval = int((~in_ball).sum().item())
                    A_full = problem["constraint_A"].to(device).float()
                    b_full = problem["constraint_b"].to(device).float()
                    ax_eval = x_flat @ A_full[:m_n].T
                    feas_avg_eval = float(
                        ((ax_eval.mean(0) - b_full[:m_n]) >= 0).float().mean().item()
                    )
                info["U_mean"] = float(U_eval)
                info["feas_full"] = feas_full_eval
                info["feas_avg"] = feas_avg_eval
                info["feas_per_c"] = feas_per_eval
                info["unique_modes"] = unique_modes_eval
                info["mode_counts"] = counts.cpu().tolist()
                info["unique_feas_modes"] = unique_feas_modes
                info["feas_mode_counts"] = feas_counts.cpu().tolist()
                info["n_unlabelled"] = n_unlabelled_eval
                extra = (f"  U={U_eval:.2f}"
                         f"  feas_avg={feas_avg_eval:.3f}"
                         f"  |M|={unique_modes_eval}/{problem['K']}"
                         f"  |M_feas|={unique_feas_modes}/{problem['K']}"
                         f"  unlab={n_unlabelled_eval}"
                         f"  counts={counts.cpu().tolist()}"
                         f"  feas_counts={feas_counts.cpu().tolist()}")

                # Best-model selection: use feas_avg for avg-constraint presets,
                # feas_full for pointwise-constraint presets.
                _feas_for_ckpt = feas_avg_eval if preset.get("avg_constraint_margins") else feas_full_eval
                pareto_key = float(_feas_for_ckpt) - _best_u_weight * float(U_eval)
                u_key = -float(U_eval)
                feas_key = (float(_feas_for_ckpt), -float(U_eval))
                cur_state = score_net.state_dict()
                if _best_pareto["key"] is None or pareto_key > _best_pareto["key"]:
                    _best_pareto.update(key=pareto_key, iter=it,
                                        U=float(U_eval), feas=float(_feas_for_ckpt),
                                        modes=int(unique_modes_eval))
                    torch.save(cur_state, _ckpt_pareto)
                    torch.save(cur_state, _best_ckpt_path)  # alias
                if _best_u["key"] is None or u_key > _best_u["key"]:
                    _best_u.update(key=u_key, iter=it,
                                   U=float(U_eval), feas=float(_feas_for_ckpt),
                                   modes=int(unique_modes_eval))
                    torch.save(cur_state, _ckpt_u)
                if _best_feas["key"] is None or feas_key > _best_feas["key"]:
                    _best_feas.update(key=feas_key, iter=it,
                                      U=float(U_eval), feas=float(_feas_for_ckpt),
                                      modes=int(unique_modes_eval))
                    torch.save(cur_state, _ckpt_feas)
            lam_stats = trainer.pop_lambda_stats()
            if lam_stats is not None:
                extra += (f"  λ(mean={lam_stats['mean']:.2f}"
                          f" q95={lam_stats['q95']:.2f}"
                          f" max={lam_stats['max']:.2f}"
                          f" n_ro={lam_stats['n_rollouts']})")
            log.info(
                "iter=%d  loss=%.4f  cos=%.3f  pred_rms=%.3f  t_mean=%.0f%s",
                it, info["loss"],
                info.get("cos_mean", 0), info.get("pred_rms_ratio_mean", 0),
                info.get("t_mean", 0), extra,
            )

    hist = trainer.fit(data_loader=data_loader, num_outer_iters=args.num_outer, log_fn=_log_hook)
    train_wall = time.time() - t0

    last = hist[-min(50, len(hist)):]
    cos_tail = float(np.mean([h.get("cos_mean", 0) for h in last]))
    loss_tail = float(np.mean([h["loss"] for h in last]))
    log.info("training done in %.1fs; last-50: cos=%.3f  loss=%.4f",
             train_wall, cos_tail, loss_tail)
    torch.save(score_net.state_dict(), out_dir / "score_net.pt")

    # Log all three best-criteria checkpoints.
    for tag, state in (("pareto", _best_pareto), ("u", _best_u), ("feas", _best_feas)):
        if state["iter"] >= 0:
            log.info("best %s checkpoint: iter=%d  U=%.2f  feas=%.3f  |M|=%d",
                     tag, state["iter"], state["U"], state["feas"], state["modes"])
    # Switch to the Pareto-best checkpoint for post-training evaluation (alias).
    if _best_pareto["iter"] >= 0 and _best_ckpt_path.exists():
        log.info("best eval checkpoint: iter=%d  U=%.2f  feas=%.3f  |M|=%d",
                 _best_pareto["iter"], _best_pareto["U"], _best_pareto["feas"], _best_pareto["modes"])
        score_net.load_state_dict(torch.load(_best_ckpt_path, map_location=device))
    else:
        log.info("no eval-best checkpoint saved; using final weights")

    # ------------------------------------------------------------------
    # 5. Plot training curves
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = np.array([h["iter"] for h in hist])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in [
        (axes[0], "loss", "normalised MSE"),
        (axes[1], "cos_mean", r"cos($\epsilon_{pred}$, $\epsilon_{target}$)"),
        (axes[2], "pred_rms_ratio_mean", r"$\|\epsilon_{pred}\| / \|\epsilon_{target}\|$"),
    ]:
        ys = np.array([h.get(key, 0) for h in hist])
        roll = np.array([np.mean(ys[max(0, i - 25):i + 1]) for i in range(len(ys))])
        ax.plot(iters, ys, alpha=0.35, label="raw")
        ax.plot(iters, roll, lw=2, color="darkred", label="MA-25")
        ax.set_title(title)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training — {args.difficulty} d={d} ib={ib:g}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Inference comparison (USE --post-eval-batch, not --batch-size)
    # ------------------------------------------------------------------
    results = []
    post_N = int(args.post_eval_batch)
    post_shape = (post_N, 1, d, 1)
    log.info("post-training eval batch: N=%d (decoupled from --batch-size=%d)",
             post_N, args.batch_size)

    # (a) MC baseline
    log.info("=== MC baseline (ib=%g) ===", ib)
    torch.manual_seed(42)
    sampler_mc = _make_sampler(ib, energy_mc_samples=64)
    if preset.get("avg_constraint_margins") is not None:
        sampler_mc._dual_ascent_step = types.MethodType(
            _shared_dual_ascent_step, sampler_mc,
        )
    t_mc = time.time()
    x0_mc = sampler_mc.sample(shape=post_shape, device=device)
    wall_mc = time.time() - t_mc
    m_mc = _evaluate(x0_mc, sampler_mc, f"MC_ib={ib:g}", log)
    m_mc["wall_sec"] = wall_mc
    mc_counts, mc_uniq, mc_feas_counts, mc_feas_uniq, mc_unlab = _mode_counts(x0_mc, sampler_mc, problem)
    m_mc["mode_counts"] = mc_counts.tolist()
    m_mc["unique_modes"] = mc_uniq
    m_mc["mode_coverage"] = mc_uniq / problem["K"]
    m_mc["feas_mode_counts"] = mc_feas_counts.tolist()
    m_mc["unique_feas_modes"] = mc_feas_uniq
    m_mc["feas_mode_coverage"] = mc_feas_uniq / problem["K"]
    m_mc["n_unlabelled"] = mc_unlab
    results.append(m_mc)

    # (b) Trained-net
    log.info("=== Trained net (ib=%g) ===", ib)
    torch.manual_seed(42)
    sampler_tn = _make_sampler(ib)
    if preset.get("avg_constraint_margins") is not None:
        sampler_tn._dual_ascent_step = types.MethodType(
            _shared_dual_ascent_step, sampler_tn,
        )
    sampler_tn._estimate_score = types.MethodType(
        _make_trained_score_fn(score_net, sampler_tn.alphas_cumprod, d),
        sampler_tn,
    )
    score_net.eval()
    t_tn = time.time()
    x0_tn = sampler_tn.sample(shape=post_shape, device=device)
    wall_tn = time.time() - t_tn
    m_tn = _evaluate(x0_tn, sampler_mc, "trained_net", log)
    m_tn["wall_sec"] = wall_tn
    tn_counts, tn_uniq, tn_feas_counts, tn_feas_uniq, tn_unlab = _mode_counts(x0_tn, sampler_mc, problem)
    m_tn["mode_counts"] = tn_counts.tolist()
    m_tn["unique_modes"] = tn_uniq
    m_tn["mode_coverage"] = tn_uniq / problem["K"]
    m_tn["feas_mode_counts"] = tn_feas_counts.tolist()
    m_tn["unique_feas_modes"] = tn_feas_uniq
    m_tn["feas_mode_coverage"] = tn_feas_uniq / problem["K"]
    m_tn["n_unlabelled"] = tn_unlab
    results.append(m_tn)

    # (c) Unconstrained (lambda=0, no dual ascent)
    log.info("=== Unconstrained DDPM ===")
    torch.manual_seed(42)
    sampler_uc = _make_sampler(ib, dual_lambda_init=0.0)

    def _noop_dual(self, dual_lambda, ergodic_rates, context, t=None):
        return dual_lambda
    sampler_uc._dual_ascent_step = types.MethodType(_noop_dual, sampler_uc)

    t_uc = time.time()
    x0_uc = sampler_uc.sample(shape=post_shape, device=device)
    wall_uc = time.time() - t_uc
    m_uc = _evaluate(x0_uc, sampler_mc, "unconstrained", log)
    m_uc["wall_sec"] = wall_uc
    uc_counts, uc_uniq, uc_feas_counts, uc_feas_uniq, uc_unlab = _mode_counts(x0_uc, sampler_mc, problem)
    m_uc["mode_counts"] = uc_counts.tolist()
    m_uc["unique_modes"] = uc_uniq
    m_uc["mode_coverage"] = uc_uniq / problem["K"]
    m_uc["feas_mode_counts"] = uc_feas_counts.tolist()
    m_uc["unique_feas_modes"] = uc_feas_uniq
    m_uc["feas_mode_coverage"] = uc_feas_uniq / problem["K"]
    m_uc["n_unlabelled"] = uc_unlab
    results.append(m_uc)

    # (d) PD Langevin baseline (primal-dual Langevin MCMC, no diffusion)
    log.info("=== PD Langevin ===")
    t_pd = time.time()
    x0_pd = _pd_langevin_sample(
        problem=problem, B=post_N, T=T,
        lr=1e-2, dual_lr=args.dual_step_size, device=device, seed=42,
    )
    wall_pd = time.time() - t_pd
    m_pd = _evaluate(x0_pd, sampler_mc, "pd_langevin", log)
    m_pd["wall_sec"] = wall_pd
    pd_counts, pd_uniq, pd_feas_counts, pd_feas_uniq, pd_unlab = _mode_counts(x0_pd, sampler_mc, problem)
    m_pd["mode_counts"] = pd_counts.tolist()
    m_pd["unique_modes"] = pd_uniq
    m_pd["mode_coverage"] = pd_uniq / problem["K"]
    m_pd["feas_mode_counts"] = pd_feas_counts.tolist()
    m_pd["unique_feas_modes"] = pd_feas_uniq
    m_pd["feas_mode_coverage"] = pd_feas_uniq / problem["K"]
    m_pd["n_unlabelled"] = pd_unlab
    results.append(m_pd)

    # (e) Rejection sampling
    log.info("=== Rejection sampling ===")
    t_rej = time.time()
    x0_rej, accept_rate = _rejection_sample(problem, n_target=post_N, device=device)
    wall_rej = time.time() - t_rej
    n_rej = x0_rej.shape[0]
    if n_rej < post_N:
        log.warning("rejection sampling only got %d / %d samples (accept_rate=%.4f)",
                     n_rej, post_N, accept_rate)
    if n_rej > 0:
        m_rej = _evaluate(x0_rej, sampler_mc, "rejection", log)
        rej_counts, rej_uniq, rej_feas_counts, rej_feas_uniq, rej_unlab = _mode_counts(x0_rej, sampler_mc, problem)
        m_rej["mode_counts"] = rej_counts.tolist()
        m_rej["unique_modes"] = rej_uniq
        m_rej["mode_coverage"] = rej_uniq / problem["K"]
        m_rej["feas_mode_counts"] = rej_feas_counts.tolist()
        m_rej["unique_feas_modes"] = rej_feas_uniq
        m_rej["feas_mode_coverage"] = rej_feas_uniq / problem["K"]
        m_rej["n_unlabelled"] = rej_unlab
    else:
        m_rej = {"label": "rejection", "potential_mean": float("nan"),
                 "feas_avg": 0.0, "full_feasibility": 0.0, "mode_counts": [0] * problem["K"],
                 "unique_modes": 0, "mode_coverage": 0.0}
    m_rej["wall_sec"] = wall_rej
    m_rej["accept_rate"] = accept_rate
    results.append(m_rej)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    summary = {
        "difficulty": args.difficulty,
        "d": d, "K": problem["K"], "m": problem["m"],
        "ib": ib, "T": T,
        "training_wall_sec": train_wall,
        "training_iters": len(hist),
        "cos_tail": cos_tail,
        "loss_tail": loss_tail,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    hdr = (f"{'method':<20s} {'U(x)':<14s} {'feas_avg':>10s} "
           f"{'vio_mean':>9s} {'vio_max':>8s} "
           f"{'|M|':>5s} {'counts':>32s} {'wall':>7s}")
    mode_rates = problem["means"] @ problem["constraint_A"].T - problem["constraint_b"]
    mode_feas = (mode_rates >= 0).all(dim=1).cpu().numpy()
    lines = [
        f"=== {args.difficulty} d={d} K={problem['K']} m={problem['m']} "
        f"ib={ib:g} (N_eval={post_N}) ===",
        hdr,
    ]
    for r in results:
        counts = r.get("mode_counts", [0] * problem["K"])
        cov_str = "[" + " ".join(
            f"{c}{'*' if mode_feas[k] else ''}" for k, c in enumerate(counts)
        ) + "]"
        lines.append(
            f"{r['label']:<20s} "
            f"{r.get('potential_mean', float('nan')):7.2f}+/-{r.get('potential_std', 0):5.2f} "
            f"{r.get('feas_avg', 0):10.3f} "
            f"{r.get('mean_violation', 0):9.4f} "
            f"{r.get('max_violation', 0):8.4f} "
            f"{r.get('unique_modes', 0):>3d}/{problem['K']:<1d} "
            f"{cov_str:>32s} "
            f"{r.get('wall_sec', 0):6.1f}s"
        )
    msg = "\n".join(lines)
    log.info("\n%s", msg)
    (out_dir / "summary.txt").write_text(msg + "\n")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    # (a) CDF of per-constraint values
    fig, ax = plt.subplots(figsize=(8, 5))
    for x0, label, color in [
        (x0_mc, f"MC (ib={ib:g})", "C0"),
        (x0_tn, "Trained net", "C1"),
        (x0_uc, "Unconstrained", "C2"),
    ]:
        ctx = sampler_mc._build_energy_context(
            None, x0.shape[0], d, x0.device, x0.dtype,
        )
        vals = sampler_mc._constraint_values(x0[:, 0, :, 0], ctx)[:, :problem["m"]]
        vals_np = vals.cpu().numpy().ravel()
        sorted_v = np.sort(vals_np)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, label=label, color=color)
    if n_rej > 0:
        ctx = sampler_mc._build_energy_context(
            None, x0_rej.shape[0], d, x0_rej.device, x0_rej.dtype,
        )
        vals = sampler_mc._constraint_values(x0_rej[:, 0, :, 0], ctx)[:, :problem["m"]]
        sorted_v = np.sort(vals.cpu().numpy().ravel())
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, label="Rejection", color="C3", ls="--")
    ax.axvline(0, color="k", ls=":", lw=1, label="feasibility boundary")
    ax.set_xlabel(r"$a_j^\top x - b_j$  (positive = satisfied)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Constraint satisfaction CDF — {args.difficulty}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cdf_constraint_values.png", dpi=130)
    plt.close(fig)

    # (b) 2D projection of samples (unnormalised to original space)
    scale2 = problem["norm_scale"][:2].numpy()
    shift2 = problem["norm_shift"][:2].numpy()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    for ax, x0, label in [
        (axes[0], x0_mc, "MC"),
        (axes[1], x0_tn, "Trained"),
        (axes[2], x0_uc, "Unconstrained"),
        (axes[3], x0_rej if n_rej > 0 else x0_mc[:0], "Rejection"),
    ]:
        if x0.shape[0] == 0:
            ax.set_title(f"{label} (no samples)")
            continue
        pts_norm = x0[:, 0, :2, 0].cpu().numpy()
        pts = pts_norm * scale2 + shift2   # unnormalise
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5)
        for k in range(problem["K"]):
            mu = problem["means_original"][k, :2].numpy()
            ax.plot(mu[0], mu[1], "rx", ms=8, mew=2)
        ax.set_title(label)
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.suptitle(f"2D projection (original space) — {args.difficulty} d={d}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "projection_2d.png", dpi=130)
    plt.close(fig)

    # (c) Bar chart: feasibility, objective, mode coverage, wall time
    labels_bar = [r["label"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].bar(labels_bar, [r.get("feas_avg", 0) for r in results])
    axes[0].set_ylabel("fraction")
    axes[0].set_title("Feasibility")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(labels_bar, [r.get("potential_mean", 0) for r in results])
    axes[1].set_ylabel(r"$\mathbb{E}[U(x)]$")
    axes[1].set_title("Objective (lower is better)")
    axes[2].bar(labels_bar, [r.get("feas_mode_coverage", 0) for r in results])
    axes[2].set_ylabel("fraction of K modes")
    axes[2].set_title("Feasible mode coverage")
    axes[2].set_ylim(0, 1.05)
    axes[3].bar(labels_bar, [r.get("wall_sec", 0) for r in results])
    axes[3].set_ylabel("seconds")
    axes[3].set_title("Wall time")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Comparison — {args.difficulty} d={d} ib={ib:g}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bars.png", dpi=130)
    plt.close(fig)

    log.info("all figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
