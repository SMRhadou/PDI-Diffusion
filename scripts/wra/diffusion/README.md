# WRA Diffusion Scripts

Scripts for training, evaluating, and visualizing constrained diffusion models on the Wireless Resource Allocation (WRA) task.

All outputs go to `outputs/wireless_resource_allocation-wra/<category>/<timestamp> - <label>/` via the `experiment_dir()` helper in `_experiment_paths.py`.

---

## Files

### `_experiment_paths.py` — Output directory convention

Provides `experiment_dir(category, label)` which returns a timestamped output path. Categories:

| Category | Purpose |
|---|---|
| `energy_sampler` | MC inference, baseline comparisons, ablation studies |
| `score_net_train` | PDI-Net training runs |
| `score_net_eval` | Inference with a trained score net |
| `diagnostics` | One-off probes, GIF generation |

```python
from _experiment_paths import experiment_dir
out = experiment_dir("energy_sampler", "my_experiment")
```

---

### `wra_baselines.py` — Baseline comparison framework

Compares constrained power allocation methods on the WRA test set. Produces LaTeX tables, CDF plots, bar charts, and power scatter plots.

**Methods available:**
- `pdi` — PDI with MC score estimation (no neural net)
- `pdi_unc` — Unconstrained PDI (lambda=0)
- `pdl` — Primal-Dual Langevin (no diffusion)
- `pdm` — Projected Diffusion Model
- `dps` — Diffusion Posterior Sampling
- `ddim` — Vanilla trained DDIM (requires `--ddim-checkpoint-dir`)
- PDI-Net — Trained score net (auto-added via `--pdi-net-checkpoints`)
- PD Expert — Forward-pass GNN (via `--pd-expert-checkpoint`)

**Ablation studies** (flags):
- `--schedule-sweep` — Compare noise schedules (cosine, DDPM linear, etc.)
- `--tail-sweep` — Test warm-starting lambda from tail-averaged reference
- `--lambda-study` — Fixed vs adaptive lambda comparison + dual evolution plots
- `--temperature-scatter` — Scatter plots at different inverse_beta values

**Example — Run PDI-Net baselines (no MC):**
```bash
micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
    --methods none \
    --dataset wra_medium_outdoor_high_density \
    --pdi-net-checkpoints \
        "outputs/.../best_model.pt" \
        "outputs/.../best_model_obj.pt" \
    --pdi-net-hidden 256 --pdi-net-layers 8 --pdi-net-K 2 \
    --inverse-beta 1.0 --dual-step-size 0.02 --dual-lambda-init 10.0 \
    --energy-mc-samples 8 --num-channel-realizations 50 \
    --n-samples-per-input 200 --max-batches 64 \
    --sub-batch 10 --chunk-size 50 --eval-timeslots 500 \
    --label pdi_net_baselines_K200
```

**Example — MC ceiling + PDL:**
```bash
micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
    --methods pdi,pdl \
    --dataset wra_medium_outdoor_high_density \
    --inverse-beta 1.0 --dual-step-size 0.02 --dual-lambda-init 10.0 \
    --n-samples-per-input 200 --max-batches 64 \
    --sub-batch 10 --chunk-size 50 \
    --label baselines_mc_pdl
```

**Key parameters:**
- `--max-batches` — Number of test networks to evaluate over
- `--n-samples-per-input` — K policy samples per network
- `--sub-batch` — Sub-batch size for shared lambda (0 = single shared lambda across all K)
- `--chunk-size` — Process score estimation in chunks to fit GPU memory
- `--dual-lambda-decay` — Weight decay on lambda: `lambda <- (1-decay)*lambda + step*violation`

---

### `train_energy_score.py` — PDI-Net training (lambda-conditioned)

Distills MC score targets into a neural network (PortfolioGNN or UGNN backbone wrapped in ScoreNetWithLambda). The model is conditioned on the dual variable lambda as an extra input channel.

Supports warm-starting from a pretrained vanilla DDIM checkpoint.

```bash
# Train from scratch
micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score.py \
    --dataset wra_medium_outdoor_high_density \
    --inverse-beta 1.0 --dual-step-size 0.05 --dual-lambda-init 10.0 \
    --label my_training_run

# Finetune from pretrained DDIM
micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score.py \
    --dataset wra_medium_outdoor_high_density \
    --ddim-warmup-checkpoint outputs/.../checkpoint_dir \
    --ddim-warmup-iters 100 \
    --label finetuned_from_ddim
```

Outputs to `outputs/.../score_net_train/<timestamp> - <label>/`:
- `best_model.pt` — Best by distillation loss
- `best_model_obj.pt` — Best by feasibility-weighted objective
- `eval_history.jsonl` — Per-epoch evaluation metrics
- `train.log` — Full training log

---

### `train_energy_score_dual.py` — PDI-Net training (external lambda)

Alternative training approach where the model has **no lambda input**. Instead, a per-network lambda is maintained externally and updated via dual ascent between training phases.

```bash
micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score_dual.py \
    --dataset wra_medium_outdoor_high_density \
    --dual-lr 0.1 --lambda-init 10.0 \
    --label dual_update_v1
```

---

### `calibrate_lambda_prior.py` — Lambda prior calibration

Runs a short MC inference rollout and emits per-timestep quantiles of the dual variable lambda. Used to set `--prior-mu-min` / `--prior-mu-max` for `train_energy_score.py`.

```bash
micromamba run -n gdiff python scripts/wra/diffusion/calibrate_lambda_prior.py \
    --dataset wra_medium_outdoor_high_density \
    --inverse-beta 1.0 --dual-step-size 0.05 \
    --label calibrate_v1
```

---

### `generate_scatter_gifs.py` — Animated policy evolution

Generates animated GIFs showing how power allocations evolve during the reverse diffusion process (or PDL optimization). Each frame shows scatter plots of `(p_i/P_max, p_j/P_max)` with optional score-field streamlines and per-user rate/lambda annotations.

Produces per-method GIFs + a combined comparison GIF + histogram GIFs (power, rate, lambda distributions).

```bash
micromamba run -n gdiff python scripts/wra/diffusion/generate_scatter_gifs.py \
    --ced-net-checkpoint "outputs/.../best_model_obj.pt" \
    --dataset wra_medium_outdoor_high_density \
    --K 200 --dual-step-size 0.02 --dual-lambda-init 10.0 \
    --sub-batch 10 --chunk-size 50 \
    --label scatter_evolution
```

---

### `train.sh` — Hydra-based vanilla DDIM training

Launches vanilla (unconstrained) DDIM training via the Hydra CLI. Controlled by environment variables:

```bash
WRA_PRESET=medium WRA_CUDA_DEVICE=0 bash scripts/wra/diffusion/train.sh
```

---

### `legacy/` — Archived scripts

Contains ~45 old probe, sweep, plot, and shell scripts that were used during development. Kept for reference but not actively maintained.
