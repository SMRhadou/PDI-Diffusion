"""Convention for experiment output directories under outputs/.

Layout:
    outputs/wireless_resource_allocation-wra/<category>/<timestamp> - <label>/

Categories:
    energy_sampler   : MC inference / hyperparam sweeps over the analytical sampler
    score_net_train  : training runs for ScoreNetWithLambda (trainers.energy_score)
    score_net_eval   : inference using a trained score net (via test_energy_diffusion_medium.py
                       --score-net-path or equivalent)
    diagnostics      : one-off probes (ESS, cos, MC self-consistency, sweeps that
                       don't fit the above)

Use:
    from _experiment_paths import experiment_dir
    out_dir = experiment_dir("score_net_train", "option_a_ib5em4")
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

VALID_CATEGORIES = {
    "energy_sampler",
    "score_net_train",
    "score_net_eval",
    "diagnostics",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_dir(
    category: str,
    label: Optional[str] = None,
    *,
    timestamp: Optional[str] = None,
    project_root: Optional[Path] = None,
    create: bool = True,
) -> Path:
    """Return outputs/wireless_resource_allocation-wra/<category>/<ts> - <label>/.

    Parameters
    ----------
    category : one of VALID_CATEGORIES.
    label    : short human label (no slashes); appended after the timestamp
               separated by " - ". Optional but strongly recommended.
    timestamp: override the timestamp; default = now() formatted as %Y-%m-%d_%H-%M-%S.
    project_root : override the auto-detected project root.
    create   : if True, create the directory (parents=True, exist_ok=True).
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of {sorted(VALID_CATEGORIES)}; got {category!r}"
        )
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts} - {label}" if label else ts
    base = (
        (project_root or _project_root())
        / "outputs"
        / "wireless_resource_allocation-wra"
        / category
    )
    out = base / name
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out
