#!/bin/sh

#SBATCH --job-name=train_gan_gpu
#SBATCH --chdir=//work/projects/acnets/EEGModalNet/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/acnets/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --error=/work/projects/acnets/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

alias micromamba=~/.local/bin/micromamba

# SECTION Run pipeline
micromamba run -n EEGModalNet python -m src.EEGModalNet.pipeline.train_gan_gpu
