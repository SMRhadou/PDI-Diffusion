#!/usr/bin/env python3
"""Train an energy-guided score network for WRA.

Uses EnergyScoreTrainer to distill MC score targets into a UGNN backbone
wrapped with ScoreNetWithLambda (λ as extra conditioning channel).

Supports loading a pretrained vanilla DDIM checkpoint and finetuning
by expanding the conditioning from 3 to 4 channels (adding λ).

Usage:
    # Train from scratch
    micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score.py \
        --dataset wra_medium_outdoor_high_density \
        --inverse-beta 1.0 --dual-step-size 0.05 --dual-lambda-init 10.0 \
        --label energy_score_v1

    # Finetune from pretrained DDIM
    micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score.py \
        --dataset wra_medium_outdoor_high_density \
        --pretrained-checkpoint /path/to/cdd90515 \
        --pretrained-epoch 1850 \
        --label energy_score_ft_e1850
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra.utils import instantiate

from pdi.datasets.wra.datamodule import WRABuilder
from pdi.diffusion.energy_ddpm import EnergyDDPM, _EnergyScoreMixin, _safe_torch_load
from pdi.trainers.energy_score.trainer import (
    EnergyScoreTrainer,
    EnergyScoreTrainConfig,
)
from pdi.trainers.energy_score.score_net import ScoreNetWithLambda
from pdi.models.portfolio_gnn_backbone import PortfolioGNNBackbone
from _experiment_paths import experiment_dir
from wra_baselines import _evaluate_method, _load_ordered_channel_gains, _compute_full_ergodic_evolution

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _NoOpModel(nn.Module):
    def forward(self, x, timesteps, edge_index, edge_weight=None,
                cond=None, return_intermediates=False):
        return torch.zeros_like(x), None


def _load_dataset_cfg(project_root, batch_size, n_samples_per_input,
                      dataset_config="wra_medium_outdoor_high_density"):
    base = OmegaConf.load(project_root / "src/pdi/conf/dataset/wra.yaml")
    overlay = OmegaConf.load(
        project_root / f"src/pdi/conf/dataset/{dataset_config}.yaml")
    cfg = OmegaConf.merge(base, overlay)
    cfg.root = str(project_root / "data" / "wra")
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.persistent_workers = False
    cfg.batch_size = int(batch_size)
    cfg.batch_size_val = 1
    cfg.n_samples_per_input = int(n_samples_per_input)
    cfg.model_cond_channels = 3
    return cfg


def _compose_diffusion_cfg(conf_dir, name="energy_ddpm_wra"):
    p = conf_dir / f"{name}.yaml"
    if p.exists():
        return OmegaConf.load(p)
    return OmegaConf.load(conf_dir / "energy_ddpm.yaml")


def _build_mc_sampler(cfg, project_root, device, *,
                      dual_step_size=0.1, dual_lambda_init=10.0,
                      inverse_beta=1.0, dual_lambda_mode="shared_per_network",
                      energy_mc_samples=16, num_channel_realizations=10,
                      dual_num_channel_realizations=None,
                      dual_lambda_max=None):
    ddpm_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(ddpm_cfg, dict):
        raise ValueError("Config must be dict-like")
    dataset_root_raw = ddpm_cfg.get("dataset_root", "")
    if dataset_root_raw and "${" not in str(dataset_root_raw):
        dataset_root = str(Path(dataset_root_raw).expanduser().resolve())
    else:
        dataset_root = str(project_root / "data" / "wra")
    R = num_channel_realizations or int(ddpm_cfg.get("energy_num_channel_realizations", 200))
    sampler = EnergyDDPM(
        model=_NoOpModel(),
        num_timesteps=int(ddpm_cfg.get("num_timesteps", 500)),
        beta_schedule=str(ddpm_cfg.get("beta_schedule", "cosine")),
        sigmoid_range=float(ddpm_cfg.get("sigmoid_range", 6.0)),
        clip_denoised=bool(ddpm_cfg.get("clip_denoised", True)),
        energy_mc_samples=energy_mc_samples,
        energy_num_channel_realizations=R,
        dual_num_channel_realizations=dual_num_channel_realizations,
        inverse_beta=inverse_beta,
        inverse_beta_schedule=str(ddpm_cfg.get("inverse_beta_schedule", "linear")),
        dual_update_mode=str(ddpm_cfg.get("dual_update_mode", "x0_pred")),
        dual_step_size=dual_step_size,
        dual_step_size_schedule=str(ddpm_cfg.get("dual_step_size_schedule", "constant")),
        dual_num_outer_iterations=int(ddpm_cfg.get("dual_num_outer_iterations", 15)),
        dual_lambda_init=dual_lambda_init,
        dual_lambda_max=dual_lambda_max,
        dual_lambda_mode=dual_lambda_mode,
        langevin_step_size=float(ddpm_cfg.get("langevin_step_size", 1e-4)),
        langevin_noise_scale=float(ddpm_cfg.get("langevin_noise_scale", 1.0)),
        dataset_root=dataset_root,
        min_energy_sigma=float(ddpm_cfg.get("min_energy_sigma", 1e-4)),
        use_precomputed_channels=bool(ddpm_cfg.get("use_precomputed_channels", True)),
        allow_sparse_graph_fallback=False,
    ).to(device)
    sampler.eval()
    return sampler


def _parse_config_from_train_log(train_log_path):
    """Parse the resolved Hydra config from a train.log file."""
    cfg_lines = []
    capture = False
    with open(train_log_path) as f:
        for line in f:
            if "=== Resolved config ===" in line:
                capture = True
                continue
            if capture:
                m = re.match(r"\[\d{4}-\d{2}-\d{2}.*?\]\[.*?\]\[INFO\] - (.*)", line)
                if m:
                    cfg_lines.append(m.group(1))
                elif line.startswith("[") or line.startswith("Processing") or line.startswith("Skipping") or line.strip() == "":
                    break
                else:
                    cfg_lines.append(line.rstrip())
    return OmegaConf.create("\n".join(cfg_lines))


def _build_portfolio_backbone(num_nodes, device, model_cond_channels=3,
                              hidden=128, num_layers=6, K=2, num_timesteps=500):
    """Build PortfolioGNNBackbone for WRA (flat residual GNN, no U-Net)."""
    backbone = PortfolioGNNBackbone(
        d=num_nodes, hidden=hidden, num_layers=num_layers,
        num_timesteps=num_timesteps, cond_channels=model_cond_channels + 1, K=K,
    )
    net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=model_cond_channels)
    print(f"Portfolio-style backbone: {sum(p.numel() for p in net.parameters()):,} params")
    return net.to(device)


def _tile_batch(batch, K):
    """Repeat each graph in a PyG batch K times for independent trajectories."""
    from torch_geometric.data import Batch
    data_list = batch.to_data_list()
    tiled = [d.clone() for d in data_list for _ in range(K)]
    return Batch.from_data_list(tiled, follow_batch=["y"], exclude_keys=["info"])


def _sparse_to_dense_adj(edge_index, edge_weight, num_nodes, batch_size):
    """Convert PyG sparse edge_index to dense [B, N, N] adjacency."""
    A = torch.zeros(batch_size, num_nodes, num_nodes,
                    device=edge_index.device, dtype=torch.float32)
    row, col = edge_index
    graph_idx = row // num_nodes
    local_row = row % num_nodes
    local_col = col % num_nodes
    if edge_weight is not None:
        A[graph_idx, local_row, local_col] = edge_weight
    else:
        A[graph_idx, local_row, local_col] = 1.0
    return A


def _build_score_net_from_scratch(project_root, device, model_cond_channels=3,
                                   model_config="ugnn_wra_v3_ds4", base_channels=None):
    """Build UGNN backbone + ScoreNetWithLambda from config files."""
    conf_root = str(project_root / "src/pdi/conf")
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_root, version_base=None):
        from hydra import compose
        model_cfg = compose(config_name=f"model/{model_config}")
    model_sub = model_cfg.model if "model" in model_cfg else model_cfg
    if base_channels is not None:
        model_sub.config.base_channels = base_channels
    backbone = instantiate(model_sub)
    net = ScoreNetWithLambda(backbone=backbone, expected_cond_feats=model_cond_channels)
    return net.to(device)


def _build_score_net_from_pretrained(checkpoint_dir, device, epoch=None,
                                     model_cond_channels=3):
    """Load pretrained DDIM backbone, wrap with ScoreNetWithLambda.

    The pretrained model was trained with model_cond_channels conditioning.
    We create a new backbone with model_cond_channels+1 (adding λ channel),
    load pretrained weights with strict=False, and let the λ pathway
    initialize randomly.
    """
    ckpt_dir = Path(checkpoint_dir)
    train_log = ckpt_dir / "train.log"
    if not train_log.exists():
        raise FileNotFoundError(f"train.log not found in {ckpt_dir}")

    # Parse the exact config used during training
    cfg = _parse_config_from_train_log(train_log)

    # Build backbone with ORIGINAL cond_channels (for loading weights)
    backbone_orig = instantiate(cfg.model)

    # Load checkpoint weights into original backbone
    if epoch is not None:
        ckpt_path = ckpt_dir / "trainer_chkpts" / "best_models" / f"best_model_epoch_{epoch}.pt"
    else:
        ckpt_path = ckpt_dir / "trainer_chkpts" / "best_models" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt and ckpt["model"] is not None:
        backbone_orig.load_state_dict(ckpt["model"])
    print(f"Loaded pretrained backbone from {ckpt_path}")
    print(f"  Params: {sum(p.numel() for p in backbone_orig.parameters()):,}")

    # Wrap with ScoreNetWithLambda — λ gets appended as extra cond channel
    # ScoreNetWithLambda._augment_cond will concatenate λ to the existing cond
    # The backbone's cond_encoder expects model_cond_channels channels,
    # but now receives model_cond_channels+1. We need to expand the first
    # layer of the cond_encoder.
    net = ScoreNetWithLambda(
        backbone=backbone_orig,
        expected_cond_feats=model_cond_channels,
    )

    # Expand cond_encoder input projection to accept +1 channel
    _expand_cond_encoder(backbone_orig, model_cond_channels)

    return net.to(device)


def _expand_cond_encoder(backbone, orig_cond_channels):
    """Expand the conditioning encoder's first linear layer to accept +1 channel.

    Finds the first Linear layer in the cond_encoder path and expands its
    input dimension by 1, preserving pretrained weights for the original channels.
    """
    for attr_name in ["encoder", "decoder"]:
        module = getattr(backbone, attr_name, None)
        if module is None:
            continue
        cond_enc = getattr(module, "cond_encoder", None)
        if cond_enc is None:
            continue
        # Find the first linear/conv layer that takes cond_channels as input
        for name, layer in cond_enc.named_modules():
            if isinstance(layer, nn.Linear) and layer.in_features == orig_cond_channels:
                new_layer = nn.Linear(orig_cond_channels + 1, layer.out_features,
                                      bias=layer.bias is not None)
                with torch.no_grad():
                    new_layer.weight[:, :orig_cond_channels] = layer.weight
                    nn.init.xavier_uniform_(new_layer.weight[:, orig_cond_channels:])
                    if layer.bias is not None:
                        new_layer.bias.copy_(layer.bias)
                # Replace the layer
                parts = name.split(".")
                parent = cond_enc
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_layer)
                print(f"  Expanded {attr_name}.cond_encoder.{name}: "
                      f"{orig_cond_channels} → {orig_cond_channels + 1} inputs")
                break
            elif isinstance(layer, nn.Conv1d) and layer.in_channels == orig_cond_channels:
                new_layer = nn.Conv1d(orig_cond_channels + 1, layer.out_channels,
                                      kernel_size=layer.kernel_size, stride=layer.stride,
                                      padding=layer.padding, bias=layer.bias is not None)
                with torch.no_grad():
                    new_layer.weight[:, :orig_cond_channels] = layer.weight
                    nn.init.xavier_uniform_(new_layer.weight[:, orig_cond_channels:].unsqueeze(0))
                    if layer.bias is not None:
                        new_layer.bias.copy_(layer.bias)
                parts = name.split(".")
                parent = cond_enc
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_layer)
                print(f"  Expanded {attr_name}.cond_encoder.{name}: "
                      f"{orig_cond_channels} → {orig_cond_channels + 1} channels")
                break


# ---------------------------------------------------------------------------
# WRA evaluation — uses _evaluate_method from wra_baselines.py
# ---------------------------------------------------------------------------

def evaluate_on_test(score_net, mc_sampler, test_loader, device,
                     n_samples_per_input, max_batches=4, eval_timeslots=200,
                     num_trials=5, sub_batch=0, dual_step_size=None):
    """Run trained net inference on test set, return aggregated metrics.

    Uses the same _evaluate_method as wra_baselines.py for consistent metrics.
    """
    import types as _types

    score_net.eval()

    # Override dual step size if provided
    orig_ds_buf = None
    if dual_step_size is not None:
        orig_ds_buf = mc_sampler.dual_step_size_by_t.clone()
        mc_sampler.dual_step_size_by_t = torch.full_like(orig_ds_buf, dual_step_size)

    # Wire the trained net into mc_sampler for inference
    alphas_cumprod = mc_sampler.alphas_cumprod.to(device)

    def _trained_estimate_score(self, x_t, t, context, dual_lambda=None,
                                inverse_beta_override=None):
        B = x_t.shape[0]
        N = x_t.shape[2]
        if dual_lambda is None:
            dual_lambda = torch.zeros(B, N, device=x_t.device, dtype=x_t.dtype)
        data = self._current_eval_data
        cond = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
        edge_index = data.edge_index if hasattr(data, "edge_index") else None
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
        with torch.no_grad():
            eps_pred = score_net(
                x=x_t, timesteps=t, dual_lambda=dual_lambda,
                edge_index=edge_index, edge_weight=edge_weight,
                cond=cond, return_intermediates=False,
            )
        ab = alphas_cumprod[t].view(-1, 1, 1, 1)
        return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))

    orig_estimate = mc_sampler._estimate_score
    mc_sampler._estimate_score = _types.MethodType(_trained_estimate_score, mc_sampler)

    all_rates = []
    all_metrics = []
    r_min = None

    try:
        for bi, batch in enumerate(test_loader):
            if bi >= max_batches:
                break
            batch = batch.to(device)
            B = int(batch.num_graphs)
            N = int(batch.y.size(0) // B)
            shape = (B, int(batch.y.size(1)), N, int(batch.y.size(2)))
            K = n_samples_per_input

            mc_sampler._current_eval_data = batch

            context = mc_sampler._build_energy_context(
                data=batch, batch_size=B, num_nodes=N,
                device=device, dtype=torch.float32)
            if r_min is None:
                r_min = float(context.r_min[0])

            ds_name = str(getattr(batch, "dataset_name", ["unknown"])[0]) if isinstance(getattr(batch, "dataset_name", None), (list, tuple)) else "unknown"
            nid = int(getattr(batch, "network_id", [bi])[0]) if isinstance(getattr(batch, "network_id", None), (list, tuple, torch.Tensor)) else bi

            torch.manual_seed(42 + bi)
            _orig_R = mc_sampler.energy_num_channel_realizations
            _orig_dual_step_sb = None
            if sub_batch > 0 and B >= sub_batch:
                n_sub = B // sub_batch
                _orig_dual_step_sb = mc_sampler._dual_ascent_step
                _s_ref = mc_sampler

                def _sub_batch_dual(dual_lambda, ergodic_rates, context, t=None,
                                    _nsub=n_sub, _sb=sub_batch, _s=_s_ref):
                    if t is None:
                        step = _s.dual_step_size_by_t[0].to(
                            device=dual_lambda.device, dtype=dual_lambda.dtype)
                    else:
                        t_safe = t.to(device=dual_lambda.device, dtype=torch.long).clamp(
                            0, _s.num_timesteps - 1)
                        step = _s._gather(
                            _s.dual_step_size_by_t.to(device=dual_lambda.device), t_safe
                        ).to(dtype=dual_lambda.dtype)[0]
                    B_loc, N_loc = dual_lambda.shape
                    violation = context.r_min.view(-1, 1) - ergodic_rates
                    mean_viol = violation.view(_nsub, _sb, N_loc).mean(dim=1, keepdim=True)
                    lam_rep = dual_lambda.view(_nsub, _sb, N_loc)[:, :1, :]
                    updated = (lam_rep + step * mean_viol).clamp_min(0.0)
                    if _s.dual_lambda_max is not None:
                        updated = updated.clamp_max(float(_s.dual_lambda_max))
                    return updated.expand(_nsub, _sb, N_loc).reshape(B_loc, N_loc).contiguous()

                mc_sampler._dual_ascent_step = _sub_batch_dual
                mc_sampler.dual_lambda_mode = "per_sample"

            result = mc_sampler.sample(shape=shape, device=device, data=batch,
                                       n_samples_per_input=K)
            x = result[0] if isinstance(result, tuple) else result

            mc_sampler.energy_num_channel_realizations = _orig_R

            if _orig_dual_step_sb is not None:
                mc_sampler._dual_ascent_step = _orig_dual_step_sb
                mc_sampler.dual_lambda_mode = "shared_per_network"

            m = _evaluate_method(x, context, mc_sampler, ds_name, nid, N,
                                 K, device, num_trials,
                                 max_timeslots=eval_timeslots)
            all_rates.append(m["per_user_rates"])
            all_metrics.append(m)
    finally:
        mc_sampler._estimate_score = orig_estimate
        if hasattr(mc_sampler, "_current_eval_data"):
            del mc_sampler._current_eval_data
        if orig_ds_buf is not None:
            mc_sampler.dual_step_size_by_t = orig_ds_buf

    score_net.train()

    if not all_rates:
        return {}

    flat = np.concatenate(all_rates)
    mean_vio = float(np.maximum(r_min - flat, 0).mean())
    return {
        "mean_rate": float(np.mean(flat)),
        "p5_rate": float(np.percentile(flat, 5)),
        "p1_rate": float(np.percentile(flat, 1)),
        "mean_vio": mean_vio,
        "feasibility": float((flat >= r_min).mean()),
        "num_users": len(flat),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train energy-guided score net for WRA")
    parser.add_argument("--label", type=str, default="energy_score")
    parser.add_argument("--dataset", type=str, default="wra_medium_outdoor_high_density")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Pretrained model
    parser.add_argument("--pretrained-checkpoint", type=str, default=None,
                        help="Path to pretrained DDIM checkpoint dir for finetuning")
    parser.add_argument("--pretrained-epoch", type=int, default=None,
                        help="Specific epoch of pretrained model")
    parser.add_argument("--base-channels", type=int, default=None,
                        help="Override UGNN base_channels (default: from config, 64)")
    parser.add_argument("--backbone", type=str, default="ugnn",
                        choices=["ugnn", "portfolio_gnn"],
                        help="Score net backbone: ugnn (default) or portfolio_gnn (flat dense GNN)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dim for portfolio_gnn backbone")
    parser.add_argument("--num-layers", type=int, default=6, help="Num layers for portfolio_gnn backbone")
    parser.add_argument("--gnn-K", type=int, default=2, help="TAGConv hops for portfolio_gnn backbone")

    # MC sampler config
    parser.add_argument("--inverse-beta", type=float, default=1.0)
    parser.add_argument("--dual-step-size", type=float, default=0.05)
    parser.add_argument("--dual-lambda-init", type=float, default=10.0)
    parser.add_argument("--dual-lambda-max", type=float, default=None)
    parser.add_argument("--energy-mc-samples", type=int, default=16,
                        help="K for MC score estimation during inference/rollout")
    parser.add_argument("--num-channel-realizations", type=int, default=10)
    parser.add_argument("--dual-num-channel-realizations", type=int, default=None,
                        help="R for dual lambda updates (default: same as --num-channel-realizations)")

    # Training config
    parser.add_argument("--num-outer", type=int, default=400)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--inner-steps-max", type=int, default=100,
                        help="Inner steps ramp up to this value")
    parser.add_argument("--inner-steps-ramp-iter", type=int, default=3000,
                        help="SGD iter at which inner steps reach max")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Minimum LR for cosine decay")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of unique networks per rollout batch")
    parser.add_argument("--n-samples-per-network", type=int, default=10,
                        help="Independent trajectories per network per rollout")
    parser.add_argument("--train-mc-samples", type=int, default=256,
                        help="K for MC score target estimation during training")
    parser.add_argument("--rho-max", type=float, default=0.7)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--target-clip-norm", type=float, default=20.0)
    parser.add_argument("--target-normalize-eps", type=float, default=0.1)
    parser.add_argument("--buffer-capacity", type=int, default=4096)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--minibatch-size", type=int, default=128,
                        help="Number of (record, timestep) pairs sampled from buffer per SGD step")

    # Perturbation
    parser.add_argument("--perturb-fraction", type=float, default=0.5)
    parser.add_argument("--perturb-x-std", type=float, default=0.1)
    parser.add_argument("--perturb-lambda-std", type=float, default=0.5)
    parser.add_argument("--num-rollouts-per-outer", type=int, default=1)
    parser.add_argument("--sub-batch", type=int, default=0,
                        help="Sub-batch size for shared-λ groups during eval. 0 = no sub-batching.")

    # Lambda prior
    parser.add_argument("--prior-mu-min", type=float, default=1.0)
    parser.add_argument("--prior-mu-max", type=float, default=20.0)

    # ib schedule
    parser.add_argument("--ib-schedule", action="store_true")
    parser.add_argument("--ib-schedule-start", type=float, default=0.1)
    parser.add_argument("--ib-schedule-end", type=float, default=1.0)

    # Eval
    parser.add_argument("--eval-every", type=int, default=20,
                        help="Evaluate on test set every N outer iterations")
    parser.add_argument("--eval-after", type=int, default=400,
                        help="Skip eval until this outer iteration")
    parser.add_argument("--eval-samples", type=int, default=50,
                        help="K samples per network during eval")
    parser.add_argument("--eval-networks", type=int, default=4,
                        help="Number of test networks to evaluate on")
    parser.add_argument("--eval-timeslots", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume from (loads model weights only)")
    parser.add_argument("--ddim-warmup-checkpoint", type=str, default=None,
                        help="Path to trained DDIM checkpoint dir for warmup rollouts")
    parser.add_argument("--ddim-warmup-iters", type=int, default=100,
                        help="Use DDIM for rollouts during first N outer iterations")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    project_root = _project_root()

    output_dir = experiment_dir("score_net_train", args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    print(f"Output: {output_dir}")
    print(f"Args: {vars(args)}")

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------------------------------------------------------
    # Build MC sampler
    # ---------------------------------------------------------------
    conf_dir = project_root / "src/pdi/conf/diffusion"
    ddpm_cfg = _compose_diffusion_cfg(conf_dir, "energy_ddpm_wra")

    mc_sampler = _build_mc_sampler(
        ddpm_cfg, project_root, device,
        dual_step_size=args.dual_step_size,
        dual_lambda_init=args.dual_lambda_init,
        inverse_beta=args.inverse_beta,
        energy_mc_samples=args.energy_mc_samples,
        num_channel_realizations=args.num_channel_realizations,
        dual_num_channel_realizations=args.dual_num_channel_realizations,
        dual_lambda_max=args.dual_lambda_max,
    )

    # ---------------------------------------------------------------
    # Build score net
    # ---------------------------------------------------------------
    model_cond_channels = 3  # direct_signal, interference, r_min

    if args.backbone == "portfolio_gnn":
        print(f"\nUsing portfolio GNN backbone (hidden={args.hidden}, layers={args.num_layers}, K={args.gnn_K})")
        # Need num_nodes from dataset — peek at first batch
        ds_cfg_peek = _load_dataset_cfg(project_root, 1, 1, dataset_config=args.dataset)
        builder_peek = WRABuilder()
        ds_peek = builder_peek.build_datasets(ds_cfg_peek)
        sample = ds_peek["train"][0]
        num_nodes = sample.num_nodes
        print(f"  num_nodes={num_nodes}")
        score_net = _build_portfolio_backbone(
            num_nodes, device, model_cond_channels=model_cond_channels,
            hidden=args.hidden, num_layers=args.num_layers, K=args.gnn_K,
        )
        del ds_peek, builder_peek
    elif args.pretrained_checkpoint:
        print(f"\nFinetuning from pretrained: {args.pretrained_checkpoint}")
        score_net = _build_score_net_from_pretrained(
            args.pretrained_checkpoint, device,
            epoch=args.pretrained_epoch,
            model_cond_channels=model_cond_channels,
        )
    else:
        print("\nTraining from scratch")
        score_net = _build_score_net_from_scratch(
            project_root, device,
            model_cond_channels=model_cond_channels,
            base_channels=args.base_channels,
        )

    n_params = sum(p.numel() for p in score_net.parameters())
    n_trainable = sum(p.numel() for p in score_net.parameters() if p.requires_grad)
    print(f"Score net: {n_params:,} params ({n_trainable:,} trainable)")

    # For portfolio_gnn backbone, wrap forward to convert sparse edge_index → dense adj
    if args.backbone == "portfolio_gnn":
        import types as _types
        _orig_forward = score_net.forward
        def _dense_adj_forward(self, x, timesteps, dual_lambda, edge_index,
                               edge_weight=None, cond=None, return_intermediates=False):
            B, _, N, _ = x.shape
            if edge_index.dim() == 2:  # sparse [2, E] → dense [B, N, N]
                edge_index = _sparse_to_dense_adj(edge_index, edge_weight, N, B)
                edge_weight = None
            return _orig_forward(x=x, timesteps=timesteps, dual_lambda=dual_lambda,
                                 edge_index=edge_index, edge_weight=edge_weight,
                                 cond=cond, return_intermediates=return_intermediates)
        score_net.forward = _types.MethodType(_dense_adj_forward, score_net)

    # Resume from checkpoint if requested
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        score_net.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from {args.resume} (outer={ckpt.get('outer')}, iter={ckpt.get('iter')})")
        print(f"  metrics: {ckpt.get('metrics', {})}")

    # ---------------------------------------------------------------
    # Build datasets (one item per unique network, sample_id=0)
    # ---------------------------------------------------------------
    ds_cfg = _load_dataset_cfg(project_root, args.batch_size, 1,
                                dataset_config=args.dataset)
    builder = WRABuilder()
    datasets = builder.build_datasets(ds_cfg)

    # Deduplicate: keep only sample_id=0 for each network
    train_ds = datasets["train"]
    seen_nets = set()
    unique_indices = []
    for i, (ds_name, net_id, sample_id) in enumerate(train_ds.samples):
        key = (ds_name, net_id)
        if key not in seen_nets:
            seen_nets.add(key)
            unique_indices.append(i)
    train_ds.samples = [train_ds.samples[i] for i in unique_indices]
    print(f"Training on {len(train_ds.samples)} unique networks")

    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, follow_batch=["y"], exclude_keys=["info"],
    )

    # Val loader for model selection (1 net per batch)
    eval_cfg = _load_dataset_cfg(project_root, 1, args.eval_samples,
                                  dataset_config=args.dataset)
    eval_datasets = builder.build_datasets(eval_cfg)
    eval_loaders = builder.build_loaders(eval_cfg, eval_datasets)
    test_loader = eval_loaders["val"]

    # ---------------------------------------------------------------
    # Build trainer
    # ---------------------------------------------------------------
    train_config = EnergyScoreTrainConfig(
        lr=args.lr,
        weight_decay=1e-4,
        k_train=args.train_mc_samples,
        inner_steps_per_outer=args.inner_steps,
        minibatch_size=args.minibatch_size,
        rho_max=args.rho_max,
        warmup_iters=args.warmup_iters,
        buffer_capacity=args.buffer_capacity,
        lambda_prior_mu_min=args.prior_mu_min,
        lambda_prior_mu_max=args.prior_mu_max,
        target_normalize=False,
        target_normalize_eps=args.target_normalize_eps,
        target_clip_norm=args.target_clip_norm,
        grad_clip_norm=args.grad_clip_norm,
        log_every_n_iters=1,
        perturb_fraction=args.perturb_fraction,
        perturb_x_std=args.perturb_x_std,
        perturb_lambda_std=args.perturb_lambda_std,
        num_rollouts_per_outer=args.num_rollouts_per_outer,
        n_samples_per_network=args.n_samples_per_network,
        ib_schedule_enabled=args.ib_schedule,
        ib_schedule_start=args.ib_schedule_start,
        ib_schedule_end=args.ib_schedule_end,
    )

    trainer = EnergyScoreTrainer(
        score_net=score_net,
        mc_sampler=mc_sampler,
        config=train_config,
        device=device,
        rng_seed=args.seed,
    )

    # ---------------------------------------------------------------
    # Training loop with eval
    # ---------------------------------------------------------------
    print(f"\nStarting training: {args.num_outer} outer iters, "
          f"{args.inner_steps} inner steps each")
    print(f"Eval every {args.eval_every} outer iters on {args.eval_networks} val networks")
    print("=" * 70)

    best_metric = -float("inf")
    best_epoch = -1
    best_obj = -float("inf")
    best_obj_epoch = -1
    best_composite = -float("inf")
    best_composite_epoch = -1
    history = []
    eval_history = []
    train_it = iter(train_loader)
    sgd_count = 0

    # Eval-only mode: just run evaluate_on_test and exit
    if args.resume and args.num_outer == 0:
        print(f"\n--- Eval-only mode: {args.eval_networks} networks, K={args.eval_samples}, R={args.eval_timeslots} ---")
        metrics = evaluate_on_test(
            score_net, mc_sampler, test_loader, device,
            n_samples_per_input=args.eval_samples,
            max_batches=args.eval_networks,
            eval_timeslots=args.eval_timeslots,
            sub_batch=args.sub_batch,
            dual_step_size=args.dual_step_size,
        )
        if metrics:
            print(f"  Mean={metrics['mean_rate']:.4f}  p5={metrics['p5_rate']:.4f}  "
                  f"p1={metrics['p1_rate']:.4f}  Vio={metrics['mean_vio']:.4f}  "
                  f"Feas={metrics['feasibility']:.1%}")
            with open(output_dir / "eval_results.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved to {output_dir / 'eval_results.json'}")
        else:
            print("  No results.")
        return

    # Load DDIM for warmup rollouts if requested
    ddim_warmup = None
    if args.ddim_warmup_checkpoint:
        from wra_baselines import _build_vanilla_ddim
        ddim_warmup = _build_vanilla_ddim(args.ddim_warmup_checkpoint, device)
        print(f"DDIM warmup: {args.ddim_warmup_checkpoint}")
        print(f"  sampling_timesteps={ddim_warmup.sampling_timesteps}, eta={ddim_warmup.ddim_eta}")
        print(f"  Using DDIM rollouts for first {args.ddim_warmup_iters} outer iterations")

    # Cosine LR scheduler
    total_sgd = args.num_outer * args.inner_steps_max
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=total_sgd, eta_min=args.lr_min)

    for outer in range(args.num_outer):
        t0 = time.time()

        # Get training batch
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch = next(train_it)

        # IB schedule
        if args.ib_schedule:
            trainer._apply_ib_schedule(outer, args.num_outer)

        # Rollout (each rollout uses a fresh batch of unique networks,
        # tiled K times for independent trajectories)
        K = args.n_samples_per_network
        use_ddim = ddim_warmup is not None and outer < args.ddim_warmup_iters
        for _r in range(args.num_rollouts_per_outer):
            if _r > 0:
                try:
                    batch = next(train_it)
                except StopIteration:
                    train_it = iter(train_loader)
                    batch = next(train_it)
            tiled = _tile_batch(batch, K).to(device) if K > 1 else batch.to(device)
            if use_ddim:
                trainer.ddim_rollout(ddim_warmup, tiled)
            else:
                trainer.rollout(tiled)

        # Ramp inner steps: linear from inner_steps to inner_steps_max
        frac = min(sgd_count / max(args.inner_steps_ramp_iter, 1), 1.0)
        n_inner = int(args.inner_steps + frac * (args.inner_steps_max - args.inner_steps))
        for _inner in range(n_inner):
            t_inner = time.time()
            step_info = trainer.train_step()
            history.append(step_info)
            scheduler.step()
            sgd_count += 1
            dt_inner = time.time() - t_inner
            log.info(f"  [inner {_inner}/{n_inner}] loss={step_info['loss']:.4f} "
                     f"cos={step_info['cos_mean']:.3f} ({dt_inner:.2f}s)")

        dt = time.time() - t0
        it = step_info["iter"]

        # Log
        lam_stats = trainer.pop_lambda_stats()
        lam_str = f"λ_mean={lam_stats['mean']:.2f} λ_max={lam_stats['max']:.1f}" if lam_stats else ""
        lr_now = scheduler.get_last_lr()[0]
        log.info(f"[outer {outer:4d}] iter={it:5d} loss={step_info['loss']:.4f} "
                 f"cos={step_info['cos_mean']:.3f} rho={step_info['rho']:.2f} "
                 f"n_inner={n_inner} lr={lr_now:.1e} "
                 f"{lam_str} ({dt:.1f}s)")

        # Eval
        if ((outer > 0 and outer >= args.eval_after and outer % args.eval_every == 0)
                or outer == args.num_outer - 1):
            log.info(f"\n--- Eval at outer={outer} ---")
            metrics = evaluate_on_test(
                score_net, mc_sampler, test_loader, device,
                n_samples_per_input=args.eval_samples,
                max_batches=args.eval_networks,
                eval_timeslots=args.eval_timeslots,
            )
            if metrics:
                metrics["outer"] = outer
                metrics["iter"] = it
                eval_history.append(metrics)
                log.info(f"  Mean={metrics['mean_rate']:.4f}  p5={metrics['p5_rate']:.4f}  "
                         f"p1={metrics['p1_rate']:.4f}  Vio={metrics['mean_vio']:.4f}  "
                         f"Feas={metrics['feasibility']:.1%}")

                # Append to JSONL incrementally (survives crashes)
                with open(output_dir / "eval_history.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")

                # Best p5 (subject to feas > 0.8)
                score = metrics["p5_rate"] if metrics["feasibility"] > 0.8 else -1.0
                if score > best_metric:
                    best_metric = score
                    best_epoch = outer
                    torch.save({
                        "state_dict": score_net.state_dict(),
                        "outer": outer,
                        "iter": it,
                        "metrics": metrics,
                        "args": vars(args),
                    }, output_dir / "best_model.pt")
                    log.info(f"  ** New best p5 (p5={metrics['p5_rate']:.4f}, feas={metrics['feasibility']:.1%})")

                # Best mean rate (objective)
                if metrics["mean_rate"] > best_obj:
                    best_obj = metrics["mean_rate"]
                    best_obj_epoch = outer
                    torch.save({
                        "state_dict": score_net.state_dict(),
                        "outer": outer,
                        "iter": it,
                        "metrics": metrics,
                        "args": vars(args),
                    }, output_dir / "best_model_obj.pt")
                    log.info(f"  ** New best obj (mean={metrics['mean_rate']:.4f})")

                # Best composite: mean + 5*p5
                composite = metrics["mean_rate"] + 5 * metrics["p5_rate"]
                if composite > best_composite:
                    best_composite = composite
                    best_composite_epoch = outer
                    torch.save({
                        "state_dict": score_net.state_dict(),
                        "outer": outer,
                        "iter": it,
                        "metrics": metrics,
                        "args": vars(args),
                    }, output_dir / "best_model_composite.pt")
                    log.info(f"  ** New best composite (mean={metrics['mean_rate']:.4f}, p5={metrics['p5_rate']:.4f}, score={composite:.4f})")

            # Save latest checkpoint
            torch.save({
                "state_dict": score_net.state_dict(),
                "outer": outer,
                "iter": it,
                "metrics": metrics,
                "args": vars(args),
            }, output_dir / "latest_model.pt")
            ckpt_dir = output_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                "state_dict": score_net.state_dict(),
                "outer": outer,
                "iter": it,
                "metrics": metrics,
                "args": vars(args),
            }, ckpt_dir / f"outer_{outer:05d}.pt")
            log.info("")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    log.info("=" * 70)
    log.info(f"Training complete. Best p5 at outer={best_epoch} (p5={best_metric:.4f})")
    log.info(f"                  Best obj at outer={best_obj_epoch} (mean={best_obj:.4f})")
    log.info(f"                  Best composite at outer={best_composite_epoch} (score={best_composite:.4f})")
    print(f"Output: {output_dir}")

    # Save histories
    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f)
    with open(output_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f)

    # Print eval table
    if eval_history:
        print(f"\n{'Outer':>6} {'Iter':>6} {'Mean':>7} {'p5':>7} {'p1':>7} "
              f"{'Vio':>7} {'Feas':>7}")
        print("-" * 55)
        for e in eval_history:
            marker = " *" if e["outer"] == best_epoch else ""
            print(f"{e['outer']:6d} {e['iter']:6d} {e['mean_rate']:7.4f} "
                  f"{e['p5_rate']:7.4f} {e['p1_rate']:7.4f} "
                  f"{e['mean_vio']:7.4f} {e['feasibility']:6.1%}{marker}")

    # ---------------------------------------------------------------
    # Dual step size sweep on best model
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Dual step size sweep (best p5 model)")
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        score_net.load_state_dict(ckpt["state_dict"])
        score_net.to(device).eval()

        ds_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        ds_results = []
        print(f"\n{'ds':>6} {'Mean':>7} {'p5':>7} {'p1':>7} {'Vio':>7} {'Feas':>7}")
        print("-" * 50)
        for ds in ds_values:
            m = evaluate_on_test(
                score_net, mc_sampler, test_loader, device,
                n_samples_per_input=args.eval_samples,
                max_batches=args.eval_networks,
                eval_timeslots=args.eval_timeslots,
                dual_step_size=ds,
            )
            if m:
                m["dual_step_size"] = ds
                ds_results.append(m)
                print(f"{ds:6.3f} {m['mean_rate']:7.4f} {m['p5_rate']:7.4f} "
                      f"{m['p1_rate']:7.4f} {m['mean_vio']:7.4f} {m['feasibility']:6.1%}")

        with open(output_dir / "ds_sweep.json", "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"Saved ds sweep to {output_dir / 'ds_sweep.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()