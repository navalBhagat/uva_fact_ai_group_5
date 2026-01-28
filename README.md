# DP-LORA: Differentially Private Fine-Tuning of Diffusion Models

Reproducibility implementation of
[Differentially Private Fine-Tuning of Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf).

This repository implements differentially private fine-tuning of latent diffusion models using LoRA (Low-Rank Adaptation) to reduce privacy-performance tradeoffs. The code supports CelebA-HQ and MNIST/EMNIST datasets with configurable privacy budgets (epsilon values).

## Setup

### Prerequisites
- **OS**: Linux with NVIDIA GPU (A100, V100, or equivalent recommended)
- **Python**: 3.10+
- **Disk Space**: 
  - CelebA-HQ dataset: ~120 GB
  - Model checkpoints: ~5 GB
  - Output/logs: ~50 GB (varies with experiments)
  - Total recommended: **~200 GB**

### Environment
Install the Python environment using Anaconda:
```bash
bash dp_lora/setup_env.sh
```

This installs PyTorch, Lightning, LoRA dependencies, and other requirements. Verify the installation with:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Datasets
**CelebA-HQ**

Download and extract the dataset:
```bash
python scripts/setup_celeba_hq.py
```
This downloads from Hugging Face and saves to `~/.cache/CelebAHQ/images/`. 

_Note_: In case the dataset should be downloaded elsewhere, use the `--root` argument and
specify the desired root directory. By default, this is set to `~/.cache/CelebAHQ`. If you do change this, ensure that the corresponding change is also applied in `dp_lora/ldm/data/celebahq.py`.

**(E)MNIST**

The EMNIST and MNIST datasets are supported via `torchvision` and
will download automatically. The code directs the download to `$TMPDIR` if
available; otherwise, it defaults to the `~/.cache` directory.

_Note:_ for EMNIST, `torchvision` uses a broken download link, so the dataloader
patches this.

**Models**

Download the cin256 checkpoint:
```bash
cd dp_lora/models/ldm/cin256
wget https://ommer-lab.com/files/latent-diffusion/cin.zip
unzip cin.zip
```

## Running Experiments

### Quick Start: CelebA-HQ with SLURM
There are three stages to run for each ablation: 

1. **Fine-tune**: `jobs/celebahq/finetuning/`
2. **Sample**: `jobs/celebahq/sampling/` (matching ablation from step 1)
3. **Evaluate**: `jobs/celebahq/fid/` (matching ablation from step 1)

### SLURM Job Submission
The `jobs` directory contains SLURM scripts. Before running:
- Ensure the `logs/` and `output/celebahq` directories exist in `dp_lora`: `mkdir -p dp_lora/{logs,output/celebahq}`
- Update checkpoint paths in the sampling scripts after running finetuning
- **Important**: Update hardcoded paths in SLURM scripts. Replace `~/fact/dp_lora` with your actual project path (e.g., `$HOME/projects/uva_fact_ai_group_5/dp_lora`)

For example, to run the training with epsilon=1

**Finetuning**:
```bash
sbatch jobs/celebahq/finetuning/celebahq_eps1.job
```

Then update the checkpoint for the sampling script based on the logs in
`logs/celebahq/`. 

**Sampling**: 
```bash
sbatch jobs/celebahq/sampling/celebahq_eps1.job
```

**FID Evaluation**:

_Note_: The following only needs to be run once for all experiments with this dataset.

```bash
sbatch jobs/celebahq/fid/compute_celebahq_stats.job
```

Then run the FID calculation for your experiment: 

```bash
sbatch jobs/celebahq/fid/celebahq_eps1.job
```

### MNIST Experiments
Pre-training + fine-tuning (not used for sampling):

1. Pre-train autoencoder: `sbatch jobs/mnist/pt_autoencoder_emnist.job`
2. Pre-train LDM: `sbatch jobs/mnist/pt_ldm_emnist.job` (update autoencoder
   path)
3. Fine-tune: `sbatch jobs/mnist/ft_mnist_eps*.job` (choose epsilon)
4. Count parameters: `sbatch jobs/mnist/count_params_mnist.job` (check SLURM
   output)

### Adding a New Experiment

Start by creating a new config file in `dp_lora/reproducibility_experiments`.
See the existing configs for reference, and edit it as required. 

Create new SLURM scripts (assuming the corresponding pre-trained checkpoints
are available): 

1) Finetuning: `jobs/<dataset>/finetuning/<experiment>.job`:
```bash
#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=<exp_name>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm/<exp_name>/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate ldm
cd $HOME/projects/uva_fact_ai_group_5/dp_lora

srun python main.py \
    --base ./reproducibility_experiments/<dataset>/<experiment>.yaml \
    -t \
    --gpus "0," \
    --accelerator gpu \
    -l logs/<dataset>
```

2) Sampling: `jobs/<dataset>/sampling/<experiment>.job`
```bash
#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=<exp_name>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=slurm/<exp_name>/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate ldm
cd $HOME/<>/uva_fact_ai_group_5/dp_lora

export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Update CKPT path based on finetuning logs
srun python ./sampling/conditional_sampling.py \
    --yaml ./reproducibility_experiments/<dataset>/<experiment>.yaml \
    --ckpt logs/<dataset>/<run_name>/checkpoints/last.ckpt \
    --output output/<dataset>/<experiment>.pt \
    --num_samples 10000 \
    --batch_size 200 \
    --decoder_batch_size 25 \
    --classes 0 1
```

3) FID: `jobs/<dataset>/fid/compute_<dataset>_stats.job` and `jobs/<dataset>/fid/<experiment>.job`

Compute dataset statistics (run once per dataset):
```bash
#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=<dataset>_stats
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm/<dataset>_stats/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate ldm
cd $HOME/<>/uva_fact_ai_group_5/dp_lora

export PYTHONPATH="${PYTHONPATH}:${PWD}"

# CelebA-HQ stats (256x256)
srun python fid/compute_dataset_stats.py \
    --dataset ldm.data.<dataset>.<Class> \
    --args size:256 \
    --output output/<dataset>/<dataset>_train_stats_256.npz
```

Compute FID for experiment:
```bash
#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=<exp_name>_fid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=slurm/<exp_name>_fid/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate ldm
cd $HOME/<>/uva_fact_ai_group_5/dp_lora

export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Compute sample stats
srun python ./fid/compute_samples_stats.py \
    --samples output/<dataset>/<exp_name>.pt \
    --output output/<dataset>/<exp_name>_stats.npz

# Calculate FID
srun python ./fid/compute_fid.py \
    --path1 output/<dataset>/<dataset>_train_stats_256.npz \
    --path2 output/<dataset>/<exp_name>_stats.npz
```

**Note**: Adapt the script based on the architecture you are using. These
experiments can also be run locally using just the `python` command. If you need
to pretrain your own autoencoder or LDM, please refer to the README inside the
`dp_lora` directory. 

## Directory Structure

```
dp_lora/                          # Main codebase (models, training, utilities)
├── reproducibility_experiments/  # Training configurations
│   ├── mnist/
│   ├── celebahq/
├── models/                       # Model checkpoints
├── peft/                         # LoRA implementation
├── ldm/                          # Diffusion model code
├── sampling/                     # Sampling scripts
└── fid/                          # FID evaluation

jobs/                             # SLURM job scripts
├── celebahq/
│   ├── finetuning/
│   ├── sampling/
│   └── fid/
└── mnist/

scripts/                           # Utility scripts
```

## Workflow

The typical experiment pipeline follows this sequence:

```
Config File
    ↓
Fine-tuning (main.py) → Checkpoint saved to logs/<dataset>/
    ↓
Sampling (conditional_sampling.py) → Generated samples output/<dataset>/<exp_name>.pt
    ↓
FID Evaluation
    ├─ Dataset stats (compute_dataset_stats.py) → output/<dataset>/train_stats_256.npz
    ├─ Sample stats (compute_samples_stats.py) → output/<dataset>/<exp_name>_stats.npz
    └─ Compute FID (compute_fid.py) → FID score
```

## Troubleshooting

### Runtime Issues

**Out of Memory (OOM) errors**
- Reduce `--batch_size`
- Reduce `--decoder_batch_size` in sampling scripts
- Use bigger GPU partition (gpu_a100 instead of gpu_mig)