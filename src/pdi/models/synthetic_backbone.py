"""MLP backbone for synthetic constrained Boltzmann diffusion.

Matches the UGNN forward signature ``(x, timesteps, edge_index, edge_weight,
cond, return_intermediates)`` so it plugs into ``ScoreNetWithLambda`` without
changes.  ``edge_index`` and ``edge_weight`` are accepted but ignored (no
graph structure in the synthetic problem).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -emb
        )
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]


class _ResBlock(nn.Module):
    """Pre-norm residual block with optional FiLM time conditioning.

    If ``time_dim`` is given, a learned affine (scale, shift) is produced from
    the time embedding and applied after the first linear layer.
    """

    def __init__(self, dim: int, time_dim: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.film = nn.Linear(time_dim, dim * 2) if time_dim > 0 else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.act(self.ln1(x))
        h = self.lin1(h)
        if self.film is not None and t_emb is not None:
            scale, shift = self.film(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale) + shift
        h = self.act(self.ln2(h))
        h = self.lin2(h)
        return x + h


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SyntheticScoreBackbone(nn.Module):
    """MLP backbone for eps-prediction in the synthetic Boltzmann problem.

    Input / output convention (matching the codebase ``[B, T, N, F]`` layout):
    - ``x``:    ``[B, 1, d, 1]``  (noisy sample)
    - ``cond``: ``[B, 1, d, C]``  (lambda conditioning from ScoreNetWithLambda)
    - output:   ``[B, 1, d, 1]``  (predicted noise)

    Internally everything is flattened to ``[B, ...]`` for the MLP.

    Parameters
    ----------
    d : int
        Dimensionality of the sample space (= number of "nodes").
    hidden : int
        Width of hidden layers.
    num_layers : int
        Number of residual blocks.
    num_timesteps : int
        Maximum diffusion timestep (used only for embedding range).
    cond_channels : int
        Number of conditioning feature channels per coordinate.
        ``ScoreNetWithLambda`` sets this to ``expected_cond_feats + 1``
        (dataset features + lambda).  For synthetic with no dataset
        conditioning, this is 1.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 256,
        num_layers: int = 4,
        num_timesteps: int = 500,
        cond_channels: int = 1,
    ):
        super().__init__()
        self.d = d
        self.cond_channels = cond_channels

        # Time embedding: scalar → hidden
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # Separate projections for x and cond (allows different input dims)
        self.x_proj = nn.Linear(d, hidden)
        self.cond_proj = nn.Linear(d * cond_channels, hidden)

        # Residual trunk — each block gets FiLM-conditioned on time embedding
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden, time_dim=hidden) for _ in range(num_layers)]
        )

        # Output head
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """Forward pass.

        Parameters match the UGNN signature so ``ScoreNetWithLambda`` works
        unchanged.  ``edge_index`` and ``edge_weight`` are ignored.
        """
        del edge_index, edge_weight  # no graph

        B = x.shape[0]
        x_flat = x.reshape(B, -1)  # [B, d]  (from [B, 1, d, 1])

        t_emb = self.time_mlp(timesteps)                       # [B, hidden]

        h = self.x_proj(x_flat)                                # [B, hidden]
        h = h + t_emb                                          # + time emb

        if cond is not None:
            cond_flat = cond.reshape(B, -1)                    # [B, d * C]
            h = h + self.cond_proj(cond_flat)                  # + cond emb

        for block in self.blocks:
            h = block(h, t_emb)

        out = self.out_proj(h)                                 # [B, d]
        out = out.view(B, 1, self.d, 1)                        # [B, 1, d, 1]

        if return_intermediates:
            return out, None
        return out
