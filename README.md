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
