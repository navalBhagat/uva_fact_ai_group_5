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

*Note*: The environment only installs on Linux with an NVIDIA GPU available.

### Dataset Setup

To run experiments with CelebA-HQ, you need to extract and prepare the dataset.
Run the extraction script:

```
python scripts/extract_celeba_hq.py
```

This script will:

* Download the CelebA-HQ dataset from Hugging Face (`korexyz/celeba-hq-256x256`)
* Extract images from parquet files
* Save them as PNG files to `~/.cache/CelebAHQ/images/`
* Create the necessary metadata files for training

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
Inside, you will find the `celebahq`, which contains the scripts experiments that
were conducted in the paper related to the CelebA-HQ dataset.

There are three job scripts to run for obtaining the results. Note that
a `logs` directory will be created to store the checkpoints.

1. First, use the `jobs/celebahq/finetuning` scripts to fine-tune the `cin256`
   checkpoint using CelebA-HQ. The name of each job script explains which
   configuration is used. For example, ` ft_celebahq_eps1.job` fine-tunes with
   $\epsilon = 1$ and the standard configuration, while
   `ft_celebahq_eps10_r16.job` fine-tunes with $\epsilon = 10$ and $r = 16$ ($r$
   being the LoRA rank). **Important**: make sure to edit the script to change
   the path where you saved the `cin256` checkpoint.

2. The checkpoint created by the fine-tuning script can then be used for
   sampling, for which the scripts can be found in the `jobs/celebahq/sampling`
   directory. Make sure to run the job script with the same name as the one used
   for fine-tuning. Once again, it is important to set the path to the
   checkpoint you obtained after fine-tuning in the script.

3. Lastly, calculate the FID score of the samples by running the corresponding
   script in the `jobs/celebahq/fid` directory. Here too, change the path inside
   the script to where the `.pt` file was saved after sampling.

### Using EMNIST/MNIST

The original paper created a model that was pre-trained on EMNIST and fine-tuned
using MNIST. However, this model was not used for any sampling afterwards. The
scripts for this model can be found in `jobs/mnist`. To train and fine-tune this
model, there are three job scripts to run:

1. Use the `jobs/mnist/pt_autoencoder_emnist.job` script to start pre-training the
   autoencoder. This autoencoder will be used for the latent diffusion model. The
   Python script should automatically download the required dataset.

2. Use the `jobs/mnist/pt_ldm_emnist.job` script to train
   the latent diffusion model. Make sure to update the path at the end of this
   script to the location of the checkpoint obtained after training the
   autoencoder.

3. Lastly, fine-tune the latent diffusion model on MNIST using the scripts in
   `jobs/mnist`. There are three scripts for MNIST, one for each $\epsilon
   \in {1,5,10}$. For example, `mnist32-class-dp-eps10.job` fine-tunes the
   latent diffusion model with $\epsilon=10$. The names of these scripts start
   with `ft_`.

After this model has been fine-tuned, the parameters can be counted. To do this,
use the `jobs/mnist/count_params_mnist.job` script. Make sure to update the paths in
this script to the places where the checkpoints were saved. Also make sure to
remove any commands for checkpoints you do not have. The SLURM output file will
contain the parameter counts.

## Directory structure

The directory `dp_lora` contains the adapted original code, `jobs` contains the
job scripts and `scripts` contains small scripts.

```
.
├── dp_lora
│   ├── callbacks
│   ├── configs
│   │   ├── autoencoder
│   │   ├── finetuning
│   │   └── latent-diffusion
│   ├── data
│   ├── fid
│   │   └── utils
│   ├── ldm
│   │   ├── data
│   │   ├── models
│   │   │   └── diffusion
│   │   ├── modules
│   │   │   ├── diffusionmodules
│   │   │   ├── distributions
│   │   │   ├── encoders
│   │   │   ├── image_degradation
│   │   │   │   └── utils
│   │   │   └── losses
│   │   └── privacy
│   ├── models
│   │   ├── first_stage_models
│   │   │   ├── kl-f16
│   │   │   ├── kl-f32
│   │   │   ├── kl-f4
│   │   │   ├── kl-f8
│   │   │   ├── vq-f16
│   │   │   ├── vq-f4
│   │   │   ├── vq-f4-noattn
│   │   │   ├── vq-f8
│   │   │   └── vq-f8-n256
│   │   └── ldm
│   │       ├── bsr_sr
│   │       ├── celeba256
│   │       ├── cin256
│   │       ├── ffhq256
│   │       ├── inpainting_big
│   │       ├── layout2img-openimages256
│   │       ├── lsun_beds256
│   │       ├── lsun_churches256
│   │       ├── semantic_synthesis256
│   │       ├── semantic_synthesis512
│   │       └── text2img256
│   ├── peft
│   │   ├── tuners
│   │   │   ├── adalora
│   │   │   ├── adaption_prompt
│   │   │   ├── ia3
│   │   │   ├── loha
│   │   │   ├── lokr
│   │   │   ├── lora
│   │   │   ├── mixed
│   │   │   ├── multitask_prompt_tuning
│   │   │   ├── oft
│   │   │   ├── p_tuning
│   │   │   ├── poly
│   │   │   ├── prefix_tuning
│   │   │   └── prompt_tuning
│   │   └── utils
│   ├── sampling
│   └── scripts
├── jobs
│   ├── celebahq
│   │   ├── fid
│   │   ├── finetuning
│   │   └── sampling
│   └── mnist
└── scripts
```
