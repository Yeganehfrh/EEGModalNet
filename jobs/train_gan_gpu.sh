#!/bin/sh

#SBATCH --job-name=train_gan_gpu
#SBATCH --chdir=/home/users/mansarinia/EEGModalNet/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/users/mansarinia/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --error=/home/users/mansarinia/EEGModalNet/logs/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

git pull

# SECTION Run pipeline
pixi run python -m src.EEGModalNet.pipeline.train_gan_con
