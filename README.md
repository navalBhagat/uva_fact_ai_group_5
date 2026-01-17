# Reproducibility Study: DP-LORA

This repository contains the code and documentation for the reproducibility study of [Differentially Private Fine-Tuning of
Diffusion
Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf).

## Installation

The directory `dp_lora` contains the code required to pretrain and finetune the models. To set up the Python environment using Anaconda, run the script: 
```
dp_lora/setup_env.sh
```
_Note_: The environment only installs on Linux with an Nvidia GPU available.

### Dataset Setup

To run experiments with CelebA-HQ, you need to extract and prepare the dataset. Run the extraction script:
```
python scripts/extract_celeba_hq.py
```

This script will:
- Download the CelebA-HQ dataset from Hugging Face (`korexyz/celeba-hq-256x256`)
- Extract images from parquet files
- Save them as PNG files to `~/.cache/CelebAHQ/images/`
- Create the necessary metadata files for training

### Model Setup

To download and set up the CIN256 pretrained LDM checkpoint, run:
```bash
cd dp_lora/models/ldm/cin256
wget https://ommer-lab.com/files/latent-diffusion/cin.zip
unzip cin.zip
```

This will download the `cin256/model.ckpt` file which is required for our experiments.

## Usage

The `jobs` directory contains SLURM job scripts for the different experiments.

*TO DO:* add a list of job scripts and explain which results they produce.