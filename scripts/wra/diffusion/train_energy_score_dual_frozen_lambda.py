#!/usr/bin/env python3
"""Continue dual-update training with FROZEN lambda.

Resumes from a checkpoint, loads its lambda_state, and continues training
the score net WITHOUT updating lambda. This tests whether the score net
can learn to satisfy complementary slackness given enough time at a fixed
lambda.

Usage:
    micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score_dual_frozen_lambda.py \
        --resume outputs/wireless_resource_allocation-wra/score_net_train/2026-05-19_10-30-02\ -\ dual_update_v1/best_model.pt \
        --num-outer 200 --label frozen_lambda_from_outer80
"""
from __future__ import annotations

import argparse
import json
import logging
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

from pdi.datasets.wra.datamodule import WRABuilder
from pdi.diffusion.energy_ddpm import EnergyDDPM, _EnergyScoreMixin, _safe_torch_load
from pdi.models.portfolio_gnn_backbone import PortfolioGNNBackbone
from pdi.trainers.energy_score.trainer_dual import (
    DualUpdateTrainer,
    DualUpdateTrainConfig,
)
from _experiment_paths import experiment_dir
from wra_baselines import _evaluate_method, _load_ordered_channel_gains, _compute_full_ergodic_evolution

log = logging.getLogger(__name__)


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
                      inverse_beta=1.0, energy_mc_samples=16,
                      num_channel_realizations=10):
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
        inverse_beta=inverse_beta,
        inverse_beta_schedule=str(ddpm_cfg.get("inverse_beta_schedule", "linear")),
        dual_update_mode=str(ddpm_cfg.get("dual_update_mode", "x0_pred")),
        dual_step_size=dual_step_size,
        dual_step_size_schedule=str(ddpm_cfg.get("dual_step_size_schedule", "constant")),
        dual_num_outer_iterations=int(ddpm_cfg.get("dual_num_outer_iterations", 15)),
        dual_lambda_init=dual_lambda_init,
        dual_lambda_max=None,
        dual_lambda_mode="shared_per_network",
        langevin_step_size=float(ddpm_cfg.get("langevin_step_size", 1e-4)),
        langevin_noise_scale=float(ddpm_cfg.get("langevin_noise_scale", 1.0)),
        dataset_root=dataset_root,
        min_energy_sigma=float(ddpm_cfg.get("min_energy_sigma", 1e-4)),
        use_precomputed_channels=bool(ddpm_cfg.get("use_precomputed_channels", True)),
        allow_sparse_graph_fallback=False,
    ).to(device)
    sampler.eval()
    return sampler


def _build_backbone(num_nodes, device, model_cond_channels=3,
                    hidden=128, num_layers=6, K=2, num_timesteps=500):
    backbone = PortfolioGNNBackbone(
        d=num_nodes, hidden=hidden, num_layers=num_layers,
        num_timesteps=num_timesteps, cond_channels=model_cond_channels, K=K,
    )
    print(f"Portfolio backbone (no λ cond): {sum(p.numel() for p in backbone.parameters()):,} params")
    return backbone.to(device)


def _tile_batch(batch, K):
    from torch_geometric.data import Batch
    data_list = batch.to_data_list()
    tiled = [d.clone() for d in data_list for _ in range(K)]
    return Batch.from_data_list(tiled, follow_batch=["y"], exclude_keys=["info"])


def _sparse_to_dense_adj(edge_index, edge_weight, num_nodes, batch_size):
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


def evaluate_on_test(score_net, mc_sampler, test_loader, device,
                     n_samples_per_input, max_batches=4, eval_timeslots=200,
                     num_trials=5, lambda_state=None):
    import types as _types

    score_net.eval()
    alphas_cumprod = mc_sampler.alphas_cumprod.to(device)

    def _trained_estimate_score(self, x_t, t, context, dual_lambda=None,
                                inverse_beta_override=None):
        B = x_t.shape[0]
        N = x_t.shape[2]
        data = self._current_eval_data
        cond = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if hasattr(data, "x") and data.x is not None else None
        edge_index = data.edge_index if hasattr(data, "edge_index") else None
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
        with torch.no_grad():
            out = score_net(
                x=x_t, timesteps=t,
                edge_index=edge_index, edge_weight=edge_weight,
                cond=cond, return_intermediates=False,
            )
            eps_pred = out[0] if isinstance(out, tuple) else out
        ab = alphas_cumprod[t].view(-1, 1, 1, 1)
        return -eps_pred / torch.sqrt((1.0 - ab).clamp_min(1e-12))

    orig_estimate = mc_sampler._estimate_score
    mc_sampler._estimate_score = _types.MethodType(_trained_estimate_score, mc_sampler)

    all_rates = []
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
            net_key = (ds_name, nid)

            if lambda_state is not None and net_key in lambda_state:
                lam_vec = lambda_state[net_key].to(device)
            else:
                lam_vec = torch.full((N,), 10.0, device=device, dtype=torch.float32)

            orig_init = mc_sampler._init_dual_lambda
            orig_dual_step = mc_sampler._dual_ascent_step
            mc_sampler._init_dual_lambda = lambda batch_size, num_nodes, *, device, dtype, init_value, _l=lam_vec: _l.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
            mc_sampler._dual_ascent_step = lambda dual_lambda, ergodic_rates, context, t=None: dual_lambda

            torch.manual_seed(42 + bi)
            result = mc_sampler.sample(shape=shape, device=device, data=batch,
                                       n_samples_per_input=K)
            x = result[0] if isinstance(result, tuple) else result

            mc_sampler._init_dual_lambda = orig_init
            mc_sampler._dual_ascent_step = orig_dual_step

            m = _evaluate_method(x, context, mc_sampler, ds_name, nid, N,
                                 K, device, num_trials,
                                 max_timeslots=eval_timeslots)
            all_rates.append(m["per_user_rates"])
    finally:
        mc_sampler._estimate_score = orig_estimate
        if hasattr(mc_sampler, "_current_eval_data"):
            del mc_sampler._current_eval_data

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


def main():
    parser = argparse.ArgumentParser(description="Continue training with FROZEN lambda")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to checkpoint (e.g., best_model.pt)")
    parser.add_argument("--label", type=str, default="frozen_lambda")
    parser.add_argument("--dataset", type=str, default="wra_medium_outdoor_high_density")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-outer", type=int, default=200)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-samples-per-network", type=int, default=10)
    parser.add_argument("--train-mc-samples", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--eval-networks", type=int, default=4)
    parser.add_argument("--eval-timeslots", type=int, default=200)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    project_root = _project_root()

    output_dir = experiment_dir("score_net_train", args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    # Load checkpoint
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    orig_args = ckpt["args"]
    print(f"Resuming from: {args.resume}")
    print(f"  Original outer: {ckpt['outer']}, sgd_count: {ckpt['sgd_count']}")
    print(f"  Original metrics: {ckpt.get('metrics', {})}")
    print(f"Output: {output_dir}")

    with open(output_dir / "args.json", "w") as f:
        json.dump({**vars(args), "orig_args": orig_args, "resumed_from": args.resume}, f, indent=2)

    # Build MC sampler
    conf_dir = project_root / "src/pdi/conf/diffusion"
    ddpm_cfg = _compose_diffusion_cfg(conf_dir, "energy_ddpm_wra")

    mc_sampler = _build_mc_sampler(
        ddpm_cfg, project_root, device,
        dual_step_size=orig_args.get("dual_lr", 0.1),
        dual_lambda_init=orig_args.get("lambda_init", 10.0),
        inverse_beta=orig_args.get("inverse_beta", 1.0),
        energy_mc_samples=orig_args.get("energy_mc_samples", 16),
        num_channel_realizations=orig_args.get("num_channel_realizations", 10),
    )

    # Build backbone
    ds_cfg_peek = _load_dataset_cfg(project_root, 1, 1, dataset_config=args.dataset)
    builder_peek = WRABuilder()
    ds_peek = builder_peek.build_datasets(ds_cfg_peek)
    sample = ds_peek["train"][0]
    num_nodes = sample.num_nodes
    num_timesteps = mc_sampler.num_timesteps
    del ds_peek, builder_peek

    score_net = _build_backbone(
        num_nodes, device, model_cond_channels=3,
        hidden=orig_args.get("hidden", 128),
        num_layers=orig_args.get("num_layers", 6),
        K=orig_args.get("gnn_K", 2),
        num_timesteps=num_timesteps,
    )

    import types as _types
    _orig_forward = score_net.forward
    def _dense_adj_forward(self, x, timesteps, edge_index,
                           edge_weight=None, cond=None, return_intermediates=False):
        B, _, N, _ = x.shape
        if edge_index.dim() == 2:
            edge_index = _sparse_to_dense_adj(edge_index, edge_weight, N, B)
            edge_weight = None
        return _orig_forward(x=x, timesteps=timesteps,
                             edge_index=edge_index, edge_weight=edge_weight,
                             cond=cond, return_intermediates=return_intermediates)
    score_net.forward = _types.MethodType(_dense_adj_forward, score_net)

    score_net.load_state_dict(ckpt["state_dict"])
    print(f"Loaded score net weights from checkpoint")

    # Build datasets
    ds_cfg = _load_dataset_cfg(project_root, args.batch_size, 1,
                                dataset_config=args.dataset)
    builder = WRABuilder()
    datasets = builder.build_datasets(ds_cfg)

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

    eval_cfg = _load_dataset_cfg(project_root, 1, args.eval_samples,
                                  dataset_config=args.dataset)
    eval_datasets = builder.build_datasets(eval_cfg)
    eval_loaders = builder.build_loaders(eval_cfg, eval_datasets)
    test_loader = eval_loaders["val"]

    # Build trainer
    train_config = DualUpdateTrainConfig(
        lr=args.lr,
        weight_decay=1e-4,
        k_train=args.train_mc_samples,
        inner_steps_per_outer=args.inner_steps,
        minibatch_size=128,
        buffer_capacity=4096,
        target_clip_norm=20.0,
        grad_clip_norm=1.0,
        dual_lr=0.0,  # FROZEN — won't be used
        lambda_init=orig_args.get("lambda_init", 10.0),
        lambda_max=None,
        perturb_fraction=0.5,
        perturb_x_std=0.1,
        n_samples_per_network=args.n_samples_per_network,
    )

    trainer = DualUpdateTrainer(
        score_net=score_net,
        mc_sampler=mc_sampler,
        config=train_config,
        device=device,
        rng_seed=args.seed,
    )

    # Load frozen lambda state from checkpoint
    for k, v in ckpt["lambda_state"].items():
        parts = k.rsplit("_", 1)
        if len(parts) == 2:
            ds_name, nid_str = parts
            try:
                nid = int(nid_str)
            except ValueError:
                continue
            trainer.lambda_state[(ds_name, nid)] = v
    print(f"Loaded {len(trainer.lambda_state)} frozen lambda vectors")

    all_lam = torch.stack(list(trainer.lambda_state.values()))
    print(f"Frozen λ: mean={all_lam.mean():.2f}, max={all_lam.max():.1f}, "
          f"zeros={int((all_lam == 0).sum())}/{all_lam.numel()} ({100*(all_lam == 0).float().mean():.1f}%)")

    # Monkey-patch _update_lambda to be a no-op
    def _frozen_update_lambda(self, net_key, ergodic_rates, r_min):
        num_nodes = ergodic_rates.shape[0]
        return self._get_lambda(net_key, num_nodes).to(ergodic_rates.device)
    trainer._update_lambda = _types.MethodType(_frozen_update_lambda, trainer)

    # Training loop
    print(f"\nStarting FROZEN-lambda training:")
    print(f"  {args.num_outer} outer iters, {args.inner_steps} SGD steps each")
    print(f"  Lambda is FROZEN from checkpoint outer={ckpt['outer']}")
    print(f"  Eval every {args.eval_every} outer iters")
    print("=" * 70)

    eval_history = []
    history = []
    sgd_count = int(ckpt["sgd_count"])
    train_it = iter(train_loader)
    K = args.n_samples_per_network

    total_sgd = args.num_outer * args.inner_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=total_sgd, eta_min=args.lr_min)

    for outer in range(args.num_outer):
        t0 = time.time()

        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch = next(train_it)

        tiled = _tile_batch(batch, K).to(device) if K > 1 else batch.to(device)
        rollout_info = trainer.rollout(tiled)

        for _inner in range(args.inner_steps):
            step_info = trainer.train_step()
            history.append(step_info)
            scheduler.step()
            sgd_count += 1

        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        log.info(f"[outer {outer:4d}] sgd={sgd_count:5d} loss={step_info['loss']:.4f} "
                 f"cos={step_info['cos_mean']:.3f} lr={lr_now:.1e} ({dt:.1f}s)")

        if (outer > 0 and outer % args.eval_every == 0) or outer == args.num_outer - 1:
            log.info(f"\n--- Eval at outer={outer} (frozen lambda) ---")
            metrics = evaluate_on_test(
                score_net, mc_sampler, test_loader, device,
                n_samples_per_input=args.eval_samples,
                max_batches=args.eval_networks,
                eval_timeslots=args.eval_timeslots,
                lambda_state=trainer.lambda_state,
            )
            if metrics:
                metrics["outer"] = outer
                metrics["sgd_count"] = sgd_count
                eval_history.append(metrics)
                log.info(f"  Mean={metrics['mean_rate']:.4f}  p5={metrics['p5_rate']:.4f}  "
                         f"p1={metrics['p1_rate']:.4f}  Vio={metrics['mean_vio']:.4f}  "
                         f"Feas={metrics['feasibility']:.1%}")

                with open(output_dir / "eval_history.jsonl", "a") as f:
                    f.write(json.dumps(metrics) + "\n")

            torch.save({
                "state_dict": score_net.state_dict(),
                "lambda_state": {
                    f"{k[0]}_{k[1]}": v for k, v in trainer.lambda_state.items()
                },
                "outer": outer,
                "sgd_count": sgd_count,
                "metrics": metrics,
                "args": vars(args),
                "orig_args": orig_args,
                "frozen_lambda": True,
            }, output_dir / "latest_model.pt")
            log.info("")

    log.info("=" * 70)
    log.info("Frozen-lambda training complete.")
    print(f"Output: {output_dir}")

    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f)
    with open(output_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f)

    if eval_history:
        print(f"\n{'Outer':>6} {'SGD':>7} {'Mean':>7} {'p5':>7} {'p1':>7} "
              f"{'Vio':>7} {'Feas':>7}")
        print("-" * 55)
        for e in eval_history:
            print(f"{e['outer']:6d} {e['sgd_count']:7d} {e['mean_rate']:7.4f} "
                  f"{e['p5_rate']:7.4f} {e['p1_rate']:7.4f} "
                  f"{e['mean_vio']:7.4f} {e['feasibility']:6.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
