#!/bin/sh

#SBATCH --job-name=train_gan
#SBATCH --chdir=//work/projects/acnets/EEGModalNet/
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/work/projects/acnets/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --error=/work/projects/acnets/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

alias micromamba=~/.local/bin/micromamba

# SECTION Run pipeline
micromamba run -n EEGModalNet python -m src.EEGModalNet.pipeline.train_gan
