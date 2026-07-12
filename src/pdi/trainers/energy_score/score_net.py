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
    Thin wrapper that adds per-node lambda features to the conditioning
    tensor, then delegates to an inner GNN backbone.

    For standard constraints (lambda dim == N), produces 1 lambda channel.
    For enriched constraints (lambda dim > N), expands to ``lambda_channels``
    per-node channels by broadcasting global lambdas to their nodes:
      ch 0: per-asset variance lambda
      ch 1: sector lambda (upper + lower summed, broadcast per sector)
      ch 2: mean stress lambda (broadcast to all)
      ch 3: Neff lambda (broadcast to all)
    """

    def __init__(self, backbone: nn.Module, expected_cond_feats: int = 0,
                 lambda_channels: int = 1, num_sectors: int = 10):
        super().__init__()
        if expected_cond_feats < 0:
            raise ValueError(f"expected_cond_feats must be >= 0, got {expected_cond_feats}")
        self.backbone = backbone
        self.expected_cond_feats = int(expected_cond_feats)
        self.lambda_channels = int(lambda_channels)
        self.num_sectors = int(num_sectors)

    @property
    def out_cond_feats(self) -> int:
        return self.expected_cond_feats + self.lambda_channels

    def _expand_enriched_lambda(
        self, dual_lambda: torch.Tensor, num_nodes: int,
    ) -> torch.Tensor:
        """Map [B, n_constraints] → [B, N, lambda_channels] for enriched."""
        B = dual_lambda.shape[0]
        N = num_nodes
        S = self.num_sectors
        var_lam = dual_lambda[:, :N]                          # [B, N]
        sec_lam = dual_lambda[:, N:N + 2 * S]                # [B, 2S]
        sec_upper = sec_lam[:, :S]                            # [B, S]
        sec_lower = sec_lam[:, S:2 * S]                       # [B, S]
        sec_sum = sec_upper + sec_lower                       # [B, S]
        sector_ids = torch.arange(N, device=dual_lambda.device) % S
        sec_per_node = sec_sum[:, sector_ids]                 # [B, N]
        M = dual_lambda.shape[1] - N - 2 * S - 1
        stress_lam = dual_lambda[:, N + 2 * S:N + 2 * S + M]  # [B, M]
        stress_mean = stress_lam.mean(dim=1, keepdim=True)    # [B, 1]
        stress_per_node = stress_mean.expand(B, N)            # [B, N]
        neff_lam = dual_lambda[:, -1:]                        # [B, 1]
        neff_per_node = neff_lam.expand(B, N)                 # [B, N]
        return torch.stack([var_lam, sec_per_node,
                            stress_per_node, neff_per_node], dim=-1)  # [B, N, 4]

    def _augment_cond(
        self,
        cond: Optional[torch.Tensor],
        dual_lambda: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dl = dual_lambda.to(device=device, dtype=dtype)
        if dl.shape[1] > num_nodes and self.lambda_channels > 1:
            lam = self._expand_enriched_lambda(dl, num_nodes)  # [B, N, C]
            lam = lam.unsqueeze(1)                             # [B, 1, N, C]
        else:
            lam = dl.view(batch_size, 1, num_nodes, 1)         # [B, 1, N, 1]
        if cond is None:
            if self.expected_cond_feats != 0:
                raise ValueError(
                    f"cond is None but expected_cond_feats={self.expected_cond_feats}."
                )
            return lam
        if cond.dim() != 4:
            raise ValueError(f"cond must be rank-4 [B,T,N,F], got shape {tuple(cond.shape)}")
        if cond.size(-1) != self.expected_cond_feats:
            raise ValueError(
                f"cond feature dim {cond.size(-1)} != expected_cond_feats "
                f"{self.expected_cond_feats}"
            )
        t_cond = cond.size(1)
        lam_expanded = lam.expand(batch_size, t_cond, num_nodes, self.lambda_channels)
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
