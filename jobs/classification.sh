#!/bin/bash

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

set -euo pipefail

# SECTION Run pipeline
pixi run python -m src.EEGModalNet.pipeline.classifier --features yaregan --task gender --n-epochs 1000 --batch-size 128
pixi run python -m src.EEGModalNet.pipeline.classifier --features cbra --task gender --n-epochs 1000 --batch-size 128
pixi run python -m src.EEGModalNet.pipeline.classifier --features raw --task gender --n-epochs 1000 --batch-size 128
pixi run python -m src.EEGModalNet.pipeline.classifier --features yaregan --task age --n-epochs 1000 --batch-size 128
pixi run python -m src.EEGModalNet.pipeline.classifier --features cbra --task age --n-epochs 1000 --batch-size 128
pixi run python -m src.EEGModalNet.pipeline.classifier --features raw --task age --n-epochs 1000 --batch-size 128
