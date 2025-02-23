#!/bin/sh

#SBATCH --job-name=train_classifier
#SBATCH --chdir=//work/projects/acnets/EEGModalNet/
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=32GB
#SBATCH --output=/work/projects/acnets/EEGModalNet/logs/wd_%j.log
#SBATCH --error=/work/projects/acnets/EEGModalNet/logs/wd_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

alias micromamba=~/.local/bin/micromamba

# SECTION Run pipeline
micromamba run -n EEGModalNet python -m src.EEGModalNet.pipeline.preprocess
