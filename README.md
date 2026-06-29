# Constrained Diffusion Model with Primal-Dual Inference

Code for [constrained diffusion models using primal-dual inference](https://arxiv.org/pdf/2606.17192) (PDI). The method integrates Lagrangian dual variables into diffusion-based generative models to enforce average constraints during sampling. 

## Applications

- **Wireless Resource Allocation (WRA):** Power allocation in wireless networks subject to minimum-rate constraints (based on our other [repo](https://github.com/yigit-uslu/Graph-Signal-Generative-Diffusion-Modeling))
- **Portfolio Optimization:** Constrained portfolio management
- **Synthetic Experiments:** Constrained sampling from mixture of Gaussians

## Repository Structure

```
src/pdi/
    diffusion/          # DDPM, DDIM, energy-guided diffusion
    models/             # Architectures
    trainers/
        energy_score/   # Energy-guided score training with dual variables
    datasets/wra/       # Wireless channel data loading
    conf/               # Hydra configs (dataset, model, diffusion)

scripts/
    wra/                # WRA training & evaluation scripts
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

```
bash scripts/wra/training.sh
```

### WRA: Train with outer-loop dual updates (DT)

```
bash scripts/wra/training_dual.sh
```

### WRA: Evaluate baselines and methods

```
bash scripts/wra/baselines.sh
```

## Citation

```bibtex
@article{pdi_diffusion_2026,
  title={Constrained Diffusion Models with Primal-Dual Inference},
  author={Samar Hadou, Yigit Berkay Uslu, Alejandro Ribeiro},
  year={2026},
  url={https://arxiv.org/pdf/2606.17192}
}
```
