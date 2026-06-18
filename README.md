# REST-GAN: Resting-State EEG Synthesis and Transfer-Learning Generative Adversarial Network
This repository contains the implementation of REST-GAN, a WGAN-GP-based framework for resting-state EEG with two linked purposes: generating physiologically plausible multi-channel EEG signals and learning compact representations that transfer to downstream classification tasks without fine-tuning.

The method and findings are described in our preprint, *A Deep Generative Model for Resting-State EEG Synthesis and Transferable Representation Learning*, available on [arXiv](https://arxiv.org/abs/2503.02636).

> [!NOTE]
> This repository was previously named `YARE-GAN` (*Yet Another Resting-State EEG GAN*). The old URL redirects here.

## 📁 Repository Structure

```bash
├── .vscode/                      # Editor settings
├── jobs/                         # Slurm job scripts
├── notebooks/                    # Experiment notebooks
├── src/EEGModalNet/              # Main source code
│   ├── data/                     # Data loading & handling
│   ├── models/                   # GAN models definitions
│   ├── pipeline/                 # End-to-end execution logic
│   ├── preprocessing/            # EEG preprocessing tools
│   ├── __init__.py
│   └── utils.py                  # Utility functions
├── .gitattributes
├── .gitignore
├── README.md
├── environment.yml              # Dependencies
└── pyproject.toml               # Project metadata & build system
```


## Setup

Install the required packages using the following command:

```bash
pixi shell
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
