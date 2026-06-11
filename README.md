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

Requires Python 3.9+, PyTorch 2.5+, and PyTorch Geometric 2.5+. See [INSTALL.md](INSTALL.md) for GPU/CUDA setup.

## Usage

### WRA: Train energy-guided score network

```bash
python scripts/wra/diffusion/train_energy_score.py \
    --dataset wra_medium_outdoor_high_density \
    --inverse-beta 1.0 --dual-step-size 0.05 \
    --dual-lambda-init 10.0 --label my_run
```

### WRA: Train with outer-loop dual updates

```bash
python scripts/wra/diffusion/train_energy_score_dual.py \
    --dataset wra_medium_outdoor_high_density \
    --label dual_run
```

### WRA: Evaluate baselines and methods

```bash

```

## Citation

```bibtex
@software{pdi_diffusion_2026,
  title={Constrained Diffusion Models with Primal-Dual Inference},
  author={Samar Hadou, Yigit Berkay Uslu, Alejandro Ribeiro},
  year={2026},
  url={}
}
```
