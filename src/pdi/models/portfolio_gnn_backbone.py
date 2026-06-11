"""GNN backbone for portfolio score-net using dense graph convolution.

Since N=500 is small, we use dense batched matmul (A @ h) instead of sparse
TAGConv.  The adjacency A is pre-computed from the correlation matrix and
stored as a dense [N, N] tensor per instance.  For multi-instance batching
we stack adjacencies into [B, N, N] and do batched matmul — no block-diagonal
graphs, no sparse ops, no node replication.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def build_correlation_graph(
    Sigma: np.ndarray,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a k-NN graph from the correlation matrix, normalized by max eigenvalue.

    Returns (edge_index [2, E], edge_weight [E]) as numpy arrays.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    N = Sigma.shape[0]
    std = np.sqrt(np.diag(Sigma)).clip(1e-12)
    corr = Sigma / np.outer(std, std)
    np.fill_diagonal(corr, 0.0)
    abs_corr = np.abs(corr)

    rows, cols, weights = [], [], []
    for i in range(N):
        neighbors = np.argsort(abs_corr[i])[-top_k:]
        for j in neighbors:
            if abs_corr[i, j] > 0:
                rows.append(i)
                cols.append(j)
                weights.append(abs_corr[i, j])

    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)

    A = csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(N, N))
    try:
        lam_max = float(eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
    except Exception:
        lam_max = float(edge_weight.max()) if len(edge_weight) > 0 else 1.0
    if lam_max > 1e-12:
        edge_weight = edge_weight / lam_max

    return edge_index, edge_weight


def build_dense_adjacency(Sigma: np.ndarray, top_k: int = 20) -> np.ndarray:
    """Build dense [N, N] adjacency from correlation, eigenvalue-normalized."""
    ei, ew = build_correlation_graph(Sigma, top_k)
    N = Sigma.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    A[ei[0], ei[1]] = ew
    return A


class _DenseGraphConvBlock(nn.Module):
    """Dense graph conv + LayerNorm + residual + FiLM conditioning.

    Implements TAGConv-style polynomial filter using dense matmul:
        h_out = W_0 h + W_1 (A h) + W_2 (A^2 h) + ...
    """

    def __init__(self, hidden: int, K: int, t_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.K = K
        self.hop_projs = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=(k == 0)) for k in range(K + 1)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.film = nn.Linear(t_emb_dim, hidden * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h, A, t_emb):
        """
        h:     [B, N, hidden]
        A:     [B, N, N]  dense adjacency (eigenvalue-normalized)
        t_emb: [B, hidden]
        """
        residual = h
        out = self.hop_projs[0](h)           # k=0: identity
        Ak_h = h
        for k in range(1, self.K + 1):
            Ak_h = torch.bmm(A, Ak_h)        # A^k @ h
            out = out + self.hop_projs[k](Ak_h)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        scale, shift = self.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        out = out * (1.0 + scale) + shift
        return out + residual


class PortfolioGNNBackbone(nn.Module):
    """GNN backbone using dense graph convolution for portfolio score prediction.

    Parameters
    ----------
    d : int
        Number of assets (N).
    hidden : int
        Hidden dimension per node.
    num_layers : int
        Number of graph conv blocks.
    num_timesteps : int
        Maximum diffusion timestep.
    cond_channels : int
        Conditioning channels per node (1 for lambda).
    K : int
        Polynomial filter order (number of hops).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 128,
        num_layers: int = 6,
        num_timesteps: int = 500,
        cond_channels: int = 1,
        K: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.hidden = hidden

        t_emb_dim = min(hidden, 128)
        self.time_embed = SinusoidalTimestepEmbedding(t_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(t_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.input_proj = nn.Linear(1 + cond_channels, hidden)

        self.blocks = nn.ModuleList(
            [_DenseGraphConvBlock(hidden, K, hidden, dropout) for _ in range(num_layers)]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
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
        """
        x : [B, 1, N, 1]
        timesteps : [B]
        edge_index : [B, N, N]  dense adjacency (passed as edge_index for API compat)
        """
        B, _, N, _ = x.shape
        A = edge_index  # [B, N, N]

        t_emb = self.time_embed(timesteps)
        t_emb = self.time_proj(t_emb)           # [B, hidden]

        if cond is not None:
            node_in = torch.cat([x, cond], dim=-1)[:, 0, :, :]  # [B, N, 1+C]
        else:
            node_in = x[:, 0, :, :]
        h = self.input_proj(node_in)             # [B, N, hidden]

        for block in self.blocks:
            h = block(h, A, t_emb)

        h = self.output_proj(h)                  # [B, N, 1]
        out = h.unsqueeze(1)                     # [B, 1, N, 1]

        if return_intermediates:
            return out, None
        return out
