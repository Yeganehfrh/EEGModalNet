# Yare-GAN: Yet Another Resting-State EEG GAN
This repository contains the official implementation of Yare-GAN — Yet Another Resting-state EEG GAN — a generative model for synthesizing multi-channel resting-state EEG data using Wasserstein GAN with Gradient Penalty (WGAN-GP).
The method and findings are described in our preprint on [arXiv](https://arxiv.org/abs/2503.02636v3).


## 📁 Repository Structure
├── .vscode/                      # Editor settings
├── jobs/                         # Job/benchmarking scripts
├── notebooks/                    # Experiment notebooks
├── run/                          # Training and evaluation runners
├── src/EEGModalNet/              # Main source code
│   ├── data/                     # Data loading & handling
│   ├── models/                   # WGAN, critic, classifier definitions
│   ├── pipeline/                 # Training pipeline components
│   ├── preprocessing/            # EEG preprocessing tools
│   ├── __init__.py
│   └── utils.py
├── .gitattributes
├── .gitignore
├── README.md
├── environment.yml              # Dependencies
└── pyproject.toml               # Project metadata & build system


## Setup

Install the required packages using the following command:

```bash
mamba env create -f environment.yml
mamba activate EEGModalNet
```

## Citation
If you find this work useful, please cite our preprint:

```bash
@article{farahzadi2025yare,
  title={YARE-GAN: Yet Another Resting State EEG-GAN},
  author={Farahzadi, Yeganeh and Ansarinia, Morteza and Kekecs, Zoltan},
  journal={arXiv preprint arXiv:2503.02636},
  year={2025}
}
```
