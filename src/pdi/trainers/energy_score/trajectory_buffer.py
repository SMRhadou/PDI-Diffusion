"""Replay buffer storing full reverse-process trajectories.

Each buffer entry is one (rollout, timestep) pair, carrying the noisy sample
``x_t``, the integer timestep ``t``, the dual variable ``lambda_t`` at that
timestep, and a reference to the PyG batch that produced the rollout.  The
PyG batch is kept intact (on CPU) so that the channel context, edge graph,
and dataset conditioning can be re-assembled when computing the MC target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch_geometric.data import Data


@dataclass
class _RolloutRecord:
    """One rollout's full trajectory, shared across the buffer entries it produced."""

    data: Data  # PyG batch on CPU
    x_by_t: torch.Tensor  # [T, B, T_s, N, F] (CPU, float32)
    lambda_by_t: torch.Tensor  # [T, B, N] (CPU, float32)
    t_values: torch.Tensor  # [T] long (CPU) — integer timesteps corresponding to the first axis
    sample_weights: Optional[torch.Tensor] = None  # [B] float, optional per-sample importance weights
    persistent: bool = False  # if True, FIFO eviction skips this record


@dataclass
class TrajectoryEntry:
    """A lightweight handle yielded by ``TrajectoryBuffer.sample``."""

    record: _RolloutRecord
    step_index: int  # index into record.x_by_t / lambda_by_t / t_values
    sample_index: int  # index along the batch axis (0 <= i < B)


class TrajectoryBuffer:
    """FIFO buffer of full reverse-process trajectories.

    The buffer capacity is counted in *entries* (one per (rollout, sample, timestep)
    tuple), not rollouts.  Adding a new rollout may evict multiple older rollouts.
    """

    def __init__(self, capacity: int = 8192, rng: Optional[random.Random] = None):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self._records: List[_RolloutRecord] = []
        self._entry_count: int = 0
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def push_trajectory(
        self,
        data: Data,
        x_by_t: torch.Tensor,  # [T, B, T_s, N, F]
        lambda_by_t: torch.Tensor,  # [T, B, N]
        t_values: torch.Tensor,  # [T]
        sample_weights: Optional[torch.Tensor] = None,  # [B], default uniform
        persistent: bool = False,  # protect from FIFO eviction
    ) -> None:
        if x_by_t.ndim != 5:
            raise ValueError(f"x_by_t must be rank-5, got shape {tuple(x_by_t.shape)}")
        if lambda_by_t.ndim != 3:
            raise ValueError(
                f"lambda_by_t must be rank-3, got shape {tuple(lambda_by_t.shape)}"
            )
        if t_values.ndim != 1:
            raise ValueError(f"t_values must be rank-1, got shape {tuple(t_values.shape)}")
        T, B = x_by_t.shape[0], x_by_t.shape[1]
        if lambda_by_t.shape[:2] != (T, B):
            raise ValueError(
                f"lambda_by_t leading dims {tuple(lambda_by_t.shape[:2])} "
                f"do not match x_by_t {(T, B)}"
            )
        if t_values.shape[0] != T:
            raise ValueError(
                f"t_values length {t_values.shape[0]} != T={T} from x_by_t"
            )

        sw = None
        if sample_weights is not None:
            sw = sample_weights.detach().to("cpu", torch.float32).contiguous()
            if sw.ndim != 1 or sw.shape[0] != B:
                raise ValueError(
                    f"sample_weights must be rank-1 length B={B}, got {tuple(sw.shape)}"
                )
            if (sw < 0).any():
                raise ValueError("sample_weights must be non-negative")
        record = _RolloutRecord(
            data=data.detach().cpu() if hasattr(data, "detach") else data.cpu(),
            x_by_t=x_by_t.detach().to("cpu", torch.float32).contiguous(),
            lambda_by_t=lambda_by_t.detach().to("cpu", torch.float32).contiguous(),
            t_values=t_values.detach().to("cpu", torch.long).contiguous(),
            sample_weights=sw,
            persistent=bool(persistent),
        )
        self._records.append(record)
        self._entry_count += T * B

        # Evict oldest non-persistent records until we fit the capacity.
        while self._entry_count > self.capacity:
            evict_idx = next(
                (i for i, r in enumerate(self._records) if not r.persistent),
                None,
            )
            if evict_idx is None or len(self._records) <= 1:
                break
            oldest = self._records.pop(evict_idx)
            self._entry_count -= oldest.x_by_t.shape[0] * oldest.x_by_t.shape[1]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @property
    def num_entries(self) -> int:
        return self._entry_count

    @property
    def num_records(self) -> int:
        return len(self._records)

    def sample(self, m: int) -> List[TrajectoryEntry]:
        """Return ``m`` random (record, step, sample) triples.

        If records carry ``sample_weights``, the sample index within a record
        is drawn proportional to those weights (inverse-frequency mode
        reweighting).  Records themselves are weighted by the SUM of their
        sample weights so overall draw is proportional to per-sample weights.
        Records without weights contribute uniform mass.
        """
        if len(self._records) == 0:
            raise RuntimeError("Cannot sample from an empty TrajectoryBuffer.")
        # Build per-record mass for weighted record choice.
        rec_mass = []
        for r in self._records:
            if r.sample_weights is None:
                rec_mass.append(float(r.x_by_t.shape[1]))           # uniform weight 1 per sample
            else:
                rec_mass.append(float(r.sample_weights.sum().item()))
        total_mass = sum(rec_mass)
        out: List[TrajectoryEntry] = []
        for _ in range(m):
            # Record-level weighted choice
            if total_mass <= 0:
                record_idx = self._rng.randrange(len(self._records))
            else:
                u = self._rng.random() * total_mass
                acc = 0.0
                record_idx = 0
                for i, w in enumerate(rec_mass):
                    acc += w
                    if u <= acc:
                        record_idx = i
                        break
            record = self._records[record_idx]
            T = record.x_by_t.shape[0]
            B = record.x_by_t.shape[1]
            step_index = self._rng.randrange(T)
            # Sample-level weighted choice
            if record.sample_weights is None:
                sample_index = self._rng.randrange(B)
            else:
                w = record.sample_weights
                total = float(w.sum().item())
                if total <= 0:
                    sample_index = self._rng.randrange(B)
                else:
                    u = self._rng.random() * total
                    acc = 0.0
                    sample_index = 0
                    for i in range(B):
                        acc += float(w[i].item())
                        if u <= acc:
                            sample_index = i
                            break
            out.append(
                TrajectoryEntry(
                    record=record,
                    step_index=step_index,
                    sample_index=sample_index,
                )
            )
        return out
