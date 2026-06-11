"""Score network with per-node lambda conditioning.

Wraps an existing GNN backbone (e.g. UGNN) that accepts
``(x, timesteps, edge_index, edge_weight, cond, return_intermediates)``.
The lambda vector is injected as an additional per-node conditioning channel:
given the dataset-provided ``cond`` tensor of shape ``[B, T_cond, N, F_cond]``,
we append a new feature dimension of ``lambda_j`` broadcast across ``T_cond``.
The wrapped backbone must therefore be instantiated with
``cond_channels = F_cond_dataset + 1``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScoreNetWithLambda(nn.Module):
    """
    Thin wrapper that adds a per-node lambda feature to the conditioning
    tensor, then delegates to an inner GNN backbone.

    Args:
        backbone: A ``nn.Module`` whose forward matches the UGNN signature:
            ``(x, timesteps, edge_index, edge_weight, cond, return_intermediates)``
            returning ``(output, intermediates)``.
        expected_cond_feats: The dataset-provided number of conditioning
            channels (e.g. ``dataset.model_cond_channels``). The wrapper will
            emit ``expected_cond_feats + 1`` channels. If no dataset cond
            exists, set to 0; the wrapper will produce a single-channel
            (lambda-only) cond tensor.
    """

    def __init__(self, backbone: nn.Module, expected_cond_feats: int = 0):
        super().__init__()
        if expected_cond_feats < 0:
            raise ValueError(f"expected_cond_feats must be >= 0, got {expected_cond_feats}")
        self.backbone = backbone
        self.expected_cond_feats = int(expected_cond_feats)

    @property
    def out_cond_feats(self) -> int:
        return self.expected_cond_feats + 1

    def _augment_cond(
        self,
        cond: Optional[torch.Tensor],  # [B, T_cond, N, F_cond] or None
        dual_lambda: torch.Tensor,  # [B, N]
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        lam = dual_lambda.to(device=device, dtype=dtype).view(batch_size, 1, num_nodes, 1)
        if cond is None:
            if self.expected_cond_feats != 0:
                raise ValueError(
                    f"cond is None but expected_cond_feats={self.expected_cond_feats}."
                )
            return lam  # [B, 1, N, 1]
        if cond.dim() != 4:
            raise ValueError(f"cond must be rank-4 [B,T,N,F], got shape {tuple(cond.shape)}")
        if cond.size(-1) != self.expected_cond_feats:
            raise ValueError(
                f"cond feature dim {cond.size(-1)} != expected_cond_feats "
                f"{self.expected_cond_feats}"
            )
        t_cond = cond.size(1)
        lam_expanded = lam.expand(batch_size, t_cond, num_nodes, 1)
        return torch.cat([cond, lam_expanded], dim=-1)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, N, F]
        timesteps: torch.Tensor,  # [B]
        dual_lambda: torch.Tensor,  # [B, N]
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        batch_size, _, num_nodes, _ = x.shape
        augmented_cond = self._augment_cond(
            cond=cond,
            dual_lambda=dual_lambda,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=x.device,
            dtype=x.dtype,
        )
        out = self.backbone(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=augmented_cond,
            return_intermediates=return_intermediates,
        )
        if isinstance(out, tuple):
            return out[0]
        return out
