"""Experiment directory helpers for synthetic Boltzmann experiments.

Mirrors ``scripts/wra/diffusion/_experiment_paths.py`` with a different
output stem (``synthetic_boltzmann/``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

VALID_CATEGORIES = frozenset({
    "mc_sampler",
    "score_net_train",
    "score_net_eval",
    "diagnostics",
})

_OUTPUT_STEM = "synthetic_boltzmann"


def _project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def experiment_dir(
    category: str,
    label: str | None = None,
    *,
    timestamp: str | None = None,
    project_root: Path | None = None,
    create: bool = True,
) -> Path:
    """Create and return an experiment output directory.

    Layout::

        outputs/synthetic_boltzmann/<category>/<timestamp> - <label>/
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}', must be one of {sorted(VALID_CATEGORIES)}"
        )
    if project_root is None:
        project_root = _project_root()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp} - {label}" if label else timestamp
    d = project_root / "outputs" / _OUTPUT_STEM / category / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
