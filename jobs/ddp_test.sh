#!/bin/sh

#SBATCH --job-name=ddp_test
#SBATCH --chdir=//work/projects/acnets/EEGModalNet/
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=/work/projects/acnets/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --error=/work/projects/acnets/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

alias micromamba=~/.local/bin/micromamba

# SECTION Run pipeline
micromamba run -n EEGModalNet python -m src.EEGModalNet.pipeline.ddp_test
