"""Convention for portfolio experiment output directories under outputs/.

Layout:
    outputs/portfolio/<category>/<timestamp> - <label>/

Categories:
    energy_sampler   : MC inference / hyperparam sweeps
    score_net_train  : training runs for ScoreNetWithLambda
    score_net_eval   : inference using a trained score net
    diagnostics      : one-off probes and sweeps

Use:
    from _experiment_paths import experiment_dir
    out_dir = experiment_dir("score_net_train", "small_gamma2")
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
    return Path(__file__).resolve().parents[2]


def experiment_dir(
    category: str,
    label: Optional[str] = None,
    *,
    timestamp: Optional[str] = None,
    project_root: Optional[Path] = None,
    create: bool = True,
) -> Path:
    """Return outputs/portfolio/<category>/<ts> - <label>/."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of {sorted(VALID_CATEGORIES)}; got {category!r}"
        )
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts} - {label}" if label else ts
    base = (
        (project_root or _project_root())
        / "outputs"
        / "portfolio"
        / category
    )
    out = base / name
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out
