#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --chdir=/home/users/mansarinia/EEGModalNet/
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --output=/home/users/mansarinia/EEGModalNet/logs/classifier_%j.log
#SBATCH --error=/home/users/mansarinia/EEGModalNet/logs/classifier_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

set -euo pipefail

# SECTION Run pipeline
pixi run python -m src.EEGModalNet.pipeline.classifier --features yaregan --task gender --n-epochs 500 --batch-size 128 --epochs 80 90 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 --no-save
# pixi run python -m src.EEGModalNet.pipeline.classifier --features cbra --task gender --n-epochs 1000 --batch-size 128
# pixi run python -m src.EEGModalNet.pipeline.classifier --features raw --task gender --n-epochs 1000 --batch-size 128
# pixi run python -m src.EEGModalNet.pipeline.classifier --features yaregan --task age --n-epochs 1000 --batch-size 128
# pixi run python -m src.EEGModalNet.pipeline.classifier --features cbra --task age --n-epochs 1000 --batch-size 128
# pixi run python -m src.EEGModalNet.pipeline.classifier --features raw --task age --n-epochs 1000 --batch-size 128
