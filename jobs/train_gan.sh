#!/bin/sh

#SBATCH --job-name=train_gan_pipeline
#SBATCH --chdir=/home/users/mansarinia/Yeganeh/EEGNet/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/mansarinia/Yeganeh/EEGNet/tmp/train_gan_pipeline_%j.log
#SBATCH --error=/home/users/mansarinia/Yeganeh/EEGNet/tmp/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

# Note just pull the latest codes
git pull

# NOTE you need micromamba installed, use the following commands to install it:
#      "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

alias micromamba=~/.local/bin/micromamba
micromamba self-update -c conda-forge -y
micromamba create -f environment.yml -y

# SECTION Run pipeline
micromamba run -n eegnet-keras3 python -m src.EEGNet.pipeline.train_gan
