# DP-LORA: Differentially Private Fine-Tuning of Diffusion Models

Reproducibility implementation of [Differentially Private Fine-Tuning of Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf).

## Setup

### Environment
Install the Python environment using Anaconda:
```bash
bash dp_lora/setup_env.sh
```
**Requirements**: Linux with NVIDIA GPU.

### Datasets
**CelebA-HQ**: Download and extract the dataset:
```bash
python scripts/extract_celeba_hq.py
```
This downloads from Hugging Face and saves to `~/.cache/CelebAHQ/images/`.

**Models**: Download the cin256 checkpoint:
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
- Ensure the `logs/` and `output/celebahq` directory exists in `dp_lora`
- Update the checkpoint in the sampling script after running finetuning.

For example, to run the training with epsilon=1

**Finetuning**:
```bash
sbatch jobs/celebahq/finetuning/celebahq_eps1.job
```

Then update the checkpoint for the sampling script based on the logs in `logs/celebahq/`. 

**Sampling**: 
```bash
sbatch jobs/celebahq/sampling/celebahq_eps1.job
```

**FID Evaluation**:
```bash
sbatch jobs/celebahq/fid/compute_celebahq_stats.job
```

```bash
sbatch jobs/celebahq/fid/celebahq_eps1.job
```

### MNIST Experiments
Pre-training + fine-tuning (not used for sampling):

1. Pre-train autoencoder: `sbatch jobs/mnist/pt_autoencoder_emnist.job`
2. Pre-train LDM: `sbatch jobs/mnist/pt_ldm_emnist.job` (update autoencoder path)
3. Fine-tune: `sbatch jobs/mnist/ft_mnist_eps*.job` (choose epsilon)
4. Count parameters: `sbatch jobs/mnist/count_params_mnist.job` (check SLURM output)

### Adding a New Experiment

Start by creating a new config file in `dp_lora/reproducibility_experiments`. See the existing configs for reference, and edit it as required. 

Create new SLURM scripts (assuming the corresponding pre-trained checkpoints are available): 

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
cd ~/fact/dp_lora
srun python main.py --base ./reproducibility_experiments/<dataset>/<experiment>.yaml -t --gpus "0," --accelerator gpu -l logs/<dataset>
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
cd ~/fact/dp_lora

export PYTHONPATH="${PYTHONPATH}:${PWD}"

srun python ./sampling/conditional_sampling.py \
    --yaml ./reproducibility_experiments/<dataset>/<experiment>.yaml \
    --ckpt <home>/uva_fact_ai_group_5/dp_lora/logs/<dataset>/<>/checkpoints/last.ckpt \
    --output output/<dataset>/<experiment>.pt \
    --num_samples 10000 \
    --batch_size 200 \
    --decoder_batch_size 25 \
    --classes 0 1
```

3) FID: `jobs/<dataset>/fid/compute_<dataset>_stats.job` and `jobs/<dataset>/fid/<experiment>.job`
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
cd ~/fact/dp_lora

export PYTHONPATH="${PYTHONPATH}:${PWD}"

# CelebA-HQ stats (256x256)
srun python fid/compute_dataset_stats.py \
    --dataset ldm.data.<dataset>.<Class> \
    --args size:256 \
    --output output/<dataset>/<dataset>_train_stats_256.npz

```

```bash
#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=<exp_name>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=slurm/<exp_name>/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate ldm
cd ~/fact/dp_lora

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

**Note**: Adapt the script based on the architecture you are using. These experiments can also be run locally using just the `python` command. If you need to pretrain your own autoencoder or LDM, please refer to the README inside the `dp_lora` directory. 

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

scripts/                           d# Utility scripts
```


