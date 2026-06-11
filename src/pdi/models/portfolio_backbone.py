"""MLP backbone for portfolio score-net with UGNN-compatible forward signature.

This backbone replaces the graph-based UGNN for the portfolio experiment
where there is no graph structure. It matches the UGNN forward signature
``(x, timesteps, edge_index, edge_weight, cond, return_intermediates)``
so it plugs into ``ScoreNetWithLambda`` without changes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class PortfolioScoreBackbone(nn.Module):
    """MLP backbone matching the UGNN forward signature.

    Parameters
    ----------
    d : int
        Number of assets (= number of "nodes").
    hidden : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    num_timesteps : int
        Maximum diffusion timestep (for embedding dimension).
    cond_channels : int
        Number of conditioning feature channels (set by ScoreNetWithLambda
        to ``expected_cond_feats + 1`` for lambda).
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
        self.hidden = hidden
        self.num_timesteps = num_timesteps

        t_emb_dim = min(hidden, 128)
        self.time_embed = SinusoidalTimestepEmbedding(t_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(t_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Input: x[B, d] + cond[B, d * cond_channels]
        input_dim = d + d * cond_channels
        layers = [nn.Linear(input_dim + hidden, hidden), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, d))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """Forward pass matching UGNN signature.

        Parameters
        ----------
        x : [B, 1, N, 1]
            Input (z-space for portfolio).
        timesteps : [B] long
            Diffusion timestep indices.
        edge_index, edge_weight :
            Ignored (no graph).
        cond : [B, 1, N, cond_channels] or None
            Conditioning (lambda channel from ScoreNetWithLambda).
        return_intermediates : bool
            If True, return ``(output, None)`` tuple for compatibility.
        """
        del edge_index, edge_weight  # no graph

        B = x.shape[0]
        x_flat = x[:, 0, :, 0]  # [B, N]

        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, t_emb_dim]
        t_emb = self.time_proj(t_emb)       # [B, hidden]

        # Conditioning
        if cond is not None:
            # cond: [B, 1, N, C] -> [B, N*C]
            cond_flat = cond[:, 0].reshape(B, -1)
            inp = torch.cat([x_flat, cond_flat], dim=-1)
        else:
            inp = x_flat

        # Concatenate time embedding and run MLP
        h = torch.cat([inp, t_emb], dim=-1)
        out = self.mlp(h)  # [B, N]

        # Reshape to [B, 1, N, 1]
        out = out.view(B, 1, self.d, 1)

        if return_intermediates:
            return out, None
        return out
