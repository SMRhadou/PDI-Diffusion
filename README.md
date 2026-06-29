# Constrained Diffusion Model with Primal-Dual Inference

Code for constrained diffusion models using primal-dual inference (PDI). The method integrates Lagrangian dual variables into diffusion-based generative models to enforce hard constraints during sampling.

## Applications

- **Wireless Resource Allocation (WRA):** Power allocation in wireless networks subject to minimum-rate constraints
- **Portfolio Optimization:** Constrained portfolio management
- **Synthetic Experiments:** Constrained sampling from mixture of Gaussians

## Repository Structure

```
src/pdi/
    diffusion/          # DDPM, DDIM, energy-guided diffusion
    models/
        ugnn/           # U-Net GNN (denoising backbone)
        components/     # Graph convolutions, pooling, attention
    trainers/
        energy_score/   # Energy-guided score training with dual variables
    datasets/wra/       # Wireless channel data loading
    conf/               # Hydra configs (dataset, model, diffusion)

scripts/
    wra/diffusion/      # WRA training & evaluation scripts
    portfolio/          # Portfolio experiment scripts
    synthetic/          # Synthetic experiment scripts
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.9+, PyTorch 2.5+, and PyTorch Geometric 2.5+. 

## Usage

### WRA: Train dual-variable-conditioned score network

```bash
micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score.py \
  --dataset wra_medium_outdoor_high_density \
  --backbone portfolio_gnn \
  --hidden 256 --num-layers 8 --gnn-K 2 \
  --inverse-beta 1.0  --energy-mc-samples 8 --num-channel-realizations 50 \
  --num-outer 400  --inner-steps 10 --inner-steps-max 30 \
  --lr 0.001 --lr-min 1e-05 --batch-size 4 --minibatch-size 128  \
  --dual-step-size 0.05 --dual-lambda-init 10.0 --dual-lambda-max 50.0 \
  --dual-num-channel-realizations 500 \
  --n-samples-per-network 50 \
  --train-mc-samples 8 --rho-max 0.7 --warmup-iters 500 \
  --target-clip-norm 20.0 --target-normalize-eps 0.1 \
  --buffer-capacity 4096 --grad-clip-norm 1.0 \
  --perturb-fraction 0.5  --perturb-x-std 0.1 --perturb-lambda-std 0.5 \
  --num-rollouts-per-outer 1 \
  --prior-mu-min 2.0 --prior-mu-max 10.0 \
  --eval-every 50 --eval-samples 200 --eval-networks 32 --eval-timeslots 500 \
  --seed 42 \
  --label pdi-net
```

### WRA: Train with outer-loop dual updates (DT)

```bash
micromamba run -n gdiff python scripts/wra/diffusion/train_energy_score_dual.py \
    --dataset wra_medium_outdoor_high_density \
    --hidden 256 --num-layers 8 --gnn-K 2 \
    --inverse-beta 1.0 --energy-mc-samples 8 --num-channel-realizations 50 \
    --num-outer 4000 --inner-steps 30 --num-rollouts-per-outer 1 \
    --lr 0.001 --lr-min 1e-05 \
    --batch-size 32 --n-samples-per-network 20 \
    --train-mc-samples 8 --target-clip-norm 20.0 \
    --buffer-capacity 4096 --grad-clip-norm 1.0 \
    --perturb-fraction 0.5 --perturb-x-std 0.1 \
    --dual-lr 0.05 --lambda-init 10.0 --lambda-max 50.0 \
    --lambda-update-chunk 32 \
    --eval-every 20 --eval-samples 20 --eval-networks 32 --eval-timeslots 50 \
    --seed 42 \
    --label dual_training
```

### WRA: Evaluate baselines and methods

```bash
micromamba run -n gdiff python scripts/wra/diffusion/wra_baselines.py \
    --methods pdi,pdl,pdm,dps \
    --dataset wra_medium_outdoor_high_density \
    --pdi-net-checkpoints <your_path> \
    --dt-checkpoints <your_path> \
    --pdi-net-hidden 256 \
    --pdi-net-layers 8 \
    --pdi-net-K 2 \
    --inverse-beta 1.0 \
    --dual-step-size 0.05 \
    --dual-lambda-init 10.0 \
    --energy-mc-samples 8 \
    --num-channel-realizations 50 \
    --n-samples-per-input 200 \
    --max-batches 64 \
    --sub-batch 20 \
    --chunk-size 50 \
    --eval-timeslots 500 \
	--pdl-num-iters 500 \
	--pdl-primal-lr 1e-4 \
	--pdl-dual-lr 1.0 \
	--pdl-lambda-init 0.0 \
    --num-evolution-trials 50 \
    --dual-lambda-decay 0.0 \
    --pdm-iters 100 \
    --pdm-rho 10.0 \
    --label pdi_net_baselines
```

## Citation

```bibtex
@software{pdi_diffusion_2026,
  title={Constrained Diffusion Models with Primal-Dual Inference},
  author={Samar Hadou, Yigit Berkay Uslu, and Alejandro Ribeiro},
  year={2026},
  url={https://arxiv.org/pdf/2606.17192}
}
```
