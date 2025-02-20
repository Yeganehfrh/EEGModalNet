#!/bin/sh

#SBATCH --job-name=train_classifier
#SBATCH --chdir=//work/projects/acnets/EEGModalNet/
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --output=/work/projects/acnets/EEGModalNet/logs/classifier_%j.log
#SBATCH --error=/work/projects/acnets/EEGModalNet/logs/classifier_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

alias micromamba=~/.local/bin/micromamba

# SECTION Run pipeline
micromamba run -n EEGModalNet python -m src.EEGModalNet.pipeline.gender_classifier
