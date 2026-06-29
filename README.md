# Constrained Diffusion Model with Primal-Dual Inference

Code for [constrained diffusion models using primal-dual inference](https://arxiv.org/pdf/2606.17192) (PDI). The method integrates Lagrangian dual variables into diffusion-based generative models to enforce average constraints during sampling. 

## Applications

- **Wireless Resource Allocation (WRA):** Power allocation in wireless networks subject to minimum-rate constraints (based on our other [repo](https://github.com/yigit-uslu/Graph-Signal-Generative-Diffusion-Modeling))
- **Portfolio Optimization:** Constrained portfolio management
- **Synthetic Experiments:** Constrained sampling from mixture of Gaussians

## Repository Structure

```
src/pdi/
    cli/                # Entry points (train, WRA channel analysis, PD expert, dataset building)
    diffusion/          # DDPM, DDIM, energy-guided diffusion
    models/             # Architectures (GNN, MLP, UGNN)
    trainers/
        energy_score/   # Energy-guided score training with dual variables
    tasks/              # Task definitions and evaluators (WRA, portfolio)
    datasets/wra/       # Wireless channel data loading and PD sample management
    conf/               # Hydra configs (dataset, model, diffusion, trainer, wra_generation)

scripts/
    wra/                # WRA training & evaluation scripts
    wra/medium-large/   # PD expert training and dataset generation pipeline
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

### WRA: Data generation with ST baseline (PD expert → dataset → ST)

```bash
# Step 1: Analyze wireless channels
bash scripts/wra/medium-large/sophisticated-oarfish-9/1_analyze_channels.sh

# Step 2: Train primal-dual expert
bash scripts/wra/medium-large/sophisticated-oarfish-9/2_train_pd.sh

# Step 3: Build diffusion dataset from PD expert samples
bash scripts/wra/medium-large/sophisticated-oarfish-9/3_build_dataset.sh

# Step 4: Train a baseline diffusion model using supervised training (ST)
bash scripts/wra/medium-large/sophisticated-oarfish-9/4_train_diffusion.sh
```

### WRA: Train dual-variable-conditioned score network for PDI implementation (Our proposal)

```bash
bash scripts/wra/training.sh
```

### WRA: Dual training (DT) baseline

```bash
bash scripts/wra/training_dual.sh
```

### WRA: Evaluate baselines and methods

```bash
bash scripts/wra/baselines.sh
```

## Citation

```bibtex
@article{pdi_diffusion_2026,
  title={Constrained Diffusion Models with Primal-Dual Inference},
  author={Hadou, Samar and Uslu, Yigit Berkay and Ribeiro, Alejandro},
  year={2026},
  url={https://arxiv.org/pdf/2606.17192}
}
```
