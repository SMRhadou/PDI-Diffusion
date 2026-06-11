# Portfolio Experiment Workflow

## Files

| File | Purpose |
|------|---------|
| `_experiment_paths.py` | Helper for output directory naming |
| `baselines.py` | PD-Langevin, constraint helpers, blend_to_feasible |
| `calibrate_lambda.py` | Calibrate lambda prior by running MC sampler and recording lambda trajectory |
| `mc_probe.py` | Sweep MC ceiling over (ib, dual_step) grids. Reports feasibility, return, Neff |
| `pdl_probe.py` | Sweep PD-Langevin over (primal_lr, dual_lr) grids |
| `pdl_dual_trace.py` | Visualize dual variable trajectories for PDL |
| `probe_problem.py` | Inspect problem instance (budgets, returns, covariance structure) |
| `train_portfolio.py` | Train score net with CED (main training script) |
| `test_all_methods.py` | Evaluate all methods on held-out test set (CED-trained, MC ceiling, PDL, unconstrained, equal weight) |
| `figures.py` | Generate paper figures and LaTeX table from one or more instances |
| `rerun_eval.py` | Re-evaluate a saved checkpoint |
| `run_portfolio_experiment.py` | End-to-end experiment runner (older, mostly superseded) |

## Typical Workflow

### 1. Calibrate lambda prior

```bash
micromamba run -n gdiff python scripts/portfolio/calibrate_lambda.py \
  --size crypto --ib 2000 --dual-step 500 --T 500 --K 512 --B 128 \
  --no-normalize --shared-lambda --dual-lambda-decay 0.001 --seed 42
```

Outputs proposed `mu_min` and `mu_max` for the exponential lambda prior.

### 2. Sweep MC ceiling

Find best (ib, dual_step) combination:

```bash
micromamba run -n gdiff python scripts/portfolio/mc_probe.py \
  --size large --structure sectors --constraint-type variance --budget-type uniform \
  --gamma 3.0 --num-sectors 10 \
  --ib-grid "2000" --dual-step-grid "100,300,400,600" \
  --K 512 --B 256 --T 500 --no-normalize --dual-lambda-decay 0.001 \
  --shared-lambda --seed 42
```

Note: mc_probe doesn't have `--size crypto`. Use `--size large` with `--structure sectors --constraint-type variance --budget-type uniform --gamma 3.0 --num-sectors 10`.

### 3. Generate figures for a config

```bash
micromamba run -n gdiff python scripts/portfolio/figures.py \
  --size crypto --ib 2000 --dual-step 300 --no-normalize \
  --dual-lambda-decay 0.001 --K 512 --B 256 --T 500 \
  --num-instances 1 --seed 42 --label crypto_ds300
```

Add `--ced-ckpt <path>` to include the trained CED net in the comparison.

### 4. Train score net

```bash
micromamba run -n gdiff python scripts/portfolio/train_portfolio.py \
  --size crypto --ib 2000 --dual-step 500 --no-normalize \
  --dual-lambda-decay 0.001 --dual-lambda-max 10000 \
  --batch-size 128 --mc-samples-train 512 --hidden 128 --num-layers 6 \
  --lr 3e-4 --mu-min 150 --mu-max 1500 --rho-max 0.7 \
  --target-clip-norm 20.0 --perturb-fraction 0.5 --perturb-lambda-std 0.3 \
  --minibatch-size 128 --tagconv-K 2 \
  --backbone gnn --num-outer 400 --eval-every 50 --eval-B 64 \
  --lam0 0.1 --num-rollouts-per-outer 4 --num-instances 200 \
  --num-val-instances 10 \
  --label v12d_dense
```

Checkpoints saved incrementally: `best_sharpe.pt`, `best_feas.pt`, `best_pareto.pt`, `best_vio.pt`, `last.pt`.

### 5. Test on held-out instances

```bash
micromamba run -n gdiff python scripts/portfolio/test_all_methods.py \
  --ckpt <path/to/score_net_best_feas.pt> \
  --size crypto --num-test 20 --test-seed-start 3000 \
  --B 256 --T 500 --K 512
```

Runs CED-trained, MC ceiling, PDL, unconstrained MC, equal weight on 20 instances and prints mean +/- std table.

## Key Parameters

| Parameter | What it controls | Current best |
|-----------|-----------------|-------------|
| `ib` (inverse_beta) | Temperature of IS weights | 2000 |
| `dual-step` | Dual ascent step size | 300-500 |
| `dual-lambda-decay` | Weight decay on lambda | 0.001 |
| `no-normalize` | Use raw (c-b) not (c/b - 1) | True |
| `mu-min / mu-max` | Lambda prior range for training | 150 / 1500 |
| `K` | MC candidates for IS score estimate | 512 |
| `tagconv-K` | Polynomial filter hops in GNN | 2 |
| `minibatch-size` | Individual (x_t, t, lambda) points per SGD step | 128 |

## Seed Conventions

| Range | Usage |
|-------|-------|
| 0-199 | Training instances |
| 1000-1009 | Validation instances |
| 3000+ | Test instances |
