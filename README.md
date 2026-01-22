# Reproducibility Study: DP-LORA

This repository contains the code and documentation for the reproducibility
study of [Differentially Private Fine-Tuning of Diffusion
Models](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf).

## Installation

The directory `dp_lora` contains the code required to pretrain and finetune the
models. To set up the Python environment using Anaconda, run the script:
```
dp_lora/setup_env.sh
```
_Note_: The environment only installs on Linux with an NVIDIA GPU available.

### Dataset Setup

To run experiments with CelebA-HQ, you need to extract and prepare the dataset.
Run the extraction script:

```
python scripts/extract_celeba_hq.py
```

This script will:
- Download the CelebA-HQ dataset from Hugging Face (`korexyz/celeba-hq-256x256`)
- Extract images from parquet files
- Save them as PNG files to `~/.cache/CelebAHQ/images/`
- Create the necessary metadata files for training

### Model Setup

To download and set up the `cin256` pretrained LDM checkpoint, run:

```bash
cd dp_lora/models/ldm/cin256
wget https://ommer-lab.com/files/latent-diffusion/cin.zip
unzip cin.zip
```

This will download the `cin256/model.ckpt` file which is required for our
experiments. This model will serve as the pre-trained part, which will be
fine-tuned using CelebA-HQ.

## Usage

After installing the environment and downloading the datasets, the experiments
can be conducted. Note that for most of the job scripts include paths to
checkpoints or other files. For most steps, it is required to change the paths
in the job scripts.

The `jobs` directory contains SLURM job scripts for the different experiments.
Inside, you will find the `backup`, which contains the scripts experiments that
were conducted in the paper.

There are three job scripts to run for obtaining the results:
1. First, use the `jobs/backup/finetuning` scripts to fine-tune the `cin256`
   checkpoint using CelebA-HQ. The name of each job script explains which
   configuration is used. For example, ` ft_celebahq_eps1.job` fine-tunes with
   $\epsilon = 1$ and the standard configuration, while
   `ft_celebahq_eps10_r16.job` fine-tunes with $\epsilon = 10$ and $r = 16$ ($r$
   being the LoRA rank). **Important**: make sure to edit the script to change
   the path where you saved the `cin256` checkpoint.
2. The checkpoint created by the fine-tuning script can then be used for
   sampling, for which the scripts can be found in the `jobs/backup/sampling` directory.
   Make sure to run the job script with the same name as the one used for
   fine-tuning. Once again, it is important to set the path to the checkpoint
   you obtained after fine-tuning in the script.
3. Lastly, calculate the FID score of the samples by running the corresponding
   script in the `jobs/backup/fid` directory. Here too, change the path inside
   the script to where the `.pt` file was saved after sampling.

### Using EMNIST/MNIST

The original paper created a model that was pre-trained on EMNIST and fine-tuned
using MNIST. However, this model was not used for any sampling afterwards. To
train and fine-tune this model, there are three job scripts to run:

1. In `jobs/autoencoder`, use the `autoencoder_kl_emnist32_4x4x3.job` script to
   start pre-training the autoencoder. This autoencoder will be used for the
   latent diffusion model. The Python script should automatically download the
   required dataset.
2. In `jobs/latent-diffusion` use the `emnist32-conditional.job` script to train
   the latent diffusion model. Make sure to update the path at the end of this
   script to the location of the checkpoint obtained after training the
   autoencoder.
3. Lastly, fine-tune the latent diffusion model on MNIST using the scripts in
   `jobs/finetuning`. There are three scripts for MNIST, one for each $\epsilon
   \in \{1,5,10\}$. For example, `mnist32-class-dp-eps10.job` fine-tunes the
   latent diffusion model with $\epsilon=10$.

After this model has been fine-tuned, the parameters can be counted. To do this,
use the `jobs/count_params_mnist.job` script. Make sure to update the paths in
this script to the places where the checkpoints were saved. Also make sure to
remove any commands for checkpoints you do not have. The SLURM output file will
contain the parameter counts.