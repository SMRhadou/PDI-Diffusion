# Constrained Diffusion Model with Primal-Dual Inference

Code for [constrained diffusion models using primal-dual inference](https://arxiv.org/pdf/2606.17192) (PDI). The method integrates Lagrangian dual variables into diffusion-based generative models to enforce average constraints during sampling. 

## Applications

- **Wireless Resource Allocation (WRA):** Power allocation in wireless networks subject to minimum-rate constraints (based on our other [repo](https://github.com/yigit-uslu/Graph-Signal-Generative-Diffusion-Modeling))
- **Portfolio Optimization:** Constrained portfolio management
- **Synthetic Experiments:** Constrained sampling from mixture of Gaussians

## Repository Structure

```
src/pdi/
    cli/                # Entry points for wra (train, PD expert, dataset building)
    diffusion/          # DDPM, DDIM, dual-variable-conditioned diffusion
    models/             # Architectures
    trainers/
        energy_score/   # Score training for dual-variable-conditioned models
    tasks/              # Task definitions and evaluators 
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
# 1. Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# 2. Create env
micromamba create -n gdiff python=3.11 -c conda-forge -y

# 3. Install PyTorch (adjust cu128 to match your CUDA: cu124, cu121, etc.)
micromamba run -n gdiff pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 4. Install PyG + extensions
micromamba run -n gdiff pip install torch-geometric
micromamba run -n gdiff pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 5. Install project dependencies
micromamba run -n gdiff pip install -r requirements.txt

# 6. Verify
micromamba run -n gdiff python -c "import torch; import torch_geometric; \
  print(f'torch={torch.__version__}, cuda={torch.version.cuda}'); print('OK')"
```

## WRA Usage

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

## Portfolio Usage
### Portfolio: Train dual-variable-conditioned score network for PDI implementation
```bash
bash scripts/portfolio/train.sh
```

### Portfolio: Evaluate PDI along with baselines
```bash
bash scripts/portfolio/eval.sh "outputs/portfolio/score_net_train/2026-06-30_15-31-38 - crypto_ib2000_ds300_mumax1500_rho0.7_v13_K1_cosine/score_net_best_pareto.pt" old_config
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
