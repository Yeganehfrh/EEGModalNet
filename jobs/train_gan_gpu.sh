#!/bin/sh

#SBATCH --job-name=train_gan_gpu
#SBATCH --chdir=EEGModalNet/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --error=EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

git pull

# SECTION Run pipeline
pixi run python -m src.EEGModalNet.pipeline.train_gan_con
