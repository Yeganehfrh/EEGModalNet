#!/bin/sh

#SBATCH --job-name=train_gan
#SBATCH --chdir=/home/users/mansarinia/Yeganeh/EEGModalNet/
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/mansarinia/Yeganeh/EEGModalNet/tmp/train_gan_pipeline_%j.log
#SBATCH --error=/home/users/mansarinia/Yeganeh/EEGModalNet/tmp/train_gan_pipeline_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

# Note just pull the latest codes
git pull

# NOTE you need micromamba installed, use the following commands to install it:
#      "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

alias micromamba=~/.local/bin/micromamba
# micromamba self-update -c conda-forge -y
# micromamba create -f environment.yml -y

export KERAS_BACKEND=torch
# SECTION Run pipeline
micromamba run -n EEGModalNet python -m src.EEGModalNet.pipeline.train_gan
