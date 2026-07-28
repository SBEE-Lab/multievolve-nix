# Overview

<p align="center">
  <img src="multievolve/multievolve_workflow.png" alt="Workflow image" width="600">
</p>

Official repository for MULTI-evolve (model-guided, universal, targeted installation of multi-mutants), an end-to-end framework for efficiently engineering hyperactive multi-mutants. 

**The MULTI-evolve Python package has the following uses:**

1. Implement the workflow for the MULTI-evolve framework including: training neural networks, proposing multi-mutants, generating MULTI-assembly mutagenic oligos for gene synthesis of proposed multi-mutants, implementing the language model zero-shot ensemble approach to nominate single mutants to experimentally test.

3. Streamlined comparison of various data splitting methods, sequence featurizations, and machine learning models.

## Installation

### Linux

We used PyTorch 2.6.0 with CUDA 12.4 for our experiments. To run the scripts in this repository, we recommend using a conda environment.  Clone the repository, navigate to the root directory, and run the following commands to install the environment and package:
```bash
cd MULTI-evolve
conda env create -f env.yml
conda activate multievolve
pip install -e .
```
Check what torch+cuda version was installed by running:
```bash
python -c "import torch; print(torch.__version__)"
```

Then, run the following command, replacing `<VERSION>` with your torch version (e.g., `2.6.0+cu124`):
```bash
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv torch-geometric \
    --find-links https://data.pyg.org/whl/torch-<VERSION>.html \
    --no-build-isolation
```

For example, if your torch version is 2.6.0+cu124, you would run:
```bash
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv torch-geometric \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html \
    --no-build-isolation
```

### Mac ARM-based

We used PyTorch 2.2.2 for our experiments. To run the scripts in this repository, we recommend using a conda environment.  Clone the repository, navigate to the root directory, and run the following commands to install the environment and package:
```bash
cd MULTI-evolve
conda env create -f env_mac.yml
conda activate multievolve
pip install -e .
```
Then, run:
```bash
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv torch-geometric \
    --find-links https://data.pyg.org/whl/torch-2.2.2+cpu.html \
    --no-build-isolation
```

## Usage

The workflow for the MULTI-evolve framework is as follows:
1. Train fully connected neural networks to predict the fitness of a given sequence.
2. Choose the best performing neural network and use it to predict combinatorial variants.
3. For the chosen multi-mutants, generate the MULTI-assembly mutagenic oligos for gene synthesis.

In certain iterations, the MULTI-evolve framework involves using a protein language model zero-shot ensemble approach to nominate single mutants to evaluate.

### Interactive Web App

MULTI-evolve can be run as a interactive web app using Streamlit.

In the root directory of the repository run:
```bash
conda activate multievolve
streamlit run app.py
```

With the Nix flake, the packaged app can also be run directly:
```bash
nix run .#multievolve-streamlit
```

For NixOS deployments, import the flake module and keep the service on its
loopback default:
```nix
{
  imports = [ inputs.multievolve.nixosModules.multievolve-streamlit ];

  services.multievolve-streamlit = {
    enable = true;
    port = 8501;
    workingDirectory = "/var/lib/multievolve-streamlit";
  };
}
```

`workingDirectory` is the service's dedicated writable state root and is also
used for `HOME` and `MULTIEVOLVE_ROOT`. Set this option, rather than overriding
those environment variables, when state should live elsewhere. Because the unit
uses `ProtectHome` and `PrivateTmp`, choose a persistent path outside `/home`,
`/root`, `/tmp`, and `/var/tmp`.

The Streamlit application accepts project, experiment, and export names that
start with a letter or number and contain only letters, numbers, `.`, `_`, or
`-`. Uploaded files must use the corresponding FASTA, CSV, PDB, or CIF suffix
and a unique basename using the same character set within each submission.
These boundaries keep browser uploads inside the selected project directory.

The Streamlit application does not provide authentication or per-user workspace
isolation. All sessions share the configured state root. Do not expose it by
setting `host = "0.0.0.0"` and `openFirewall = true` on an untrusted network.
Put non-loopback deployments behind an authenticated TLS reverse proxy and
restrict network access.

CUDA execution additionally requires an NVIDIA driver compatible with the
packaged CUDA runtime and a working NixOS NVIDIA/OpenGL configuration. Systems
with restricted device permissions may need:

```nix
services.multievolve-streamlit.extraGroups = [ "video" "render" ];
```

Model checkpoints are downloaded below the service's writable home by default.
For pre-populated checkpoints, use a service-readable path outside `/home` and
point the relevant cache environment variable, such as `TORCH_HOME`, at it. The
service's strict filesystem protection keeps such paths read-only unless they
are explicitly added to `ReadWritePaths`.
<p align="center">
  <img src="multievolve/streamlit_1.png" alt="GUI interface image 1" width="600">
</p>

### Command-line

See the [Scripts README](scripts/README.md) to learn how to use MULTI-evolve via the Command-line.

## Repository Structure

```
multievolve/                    # Main package
├── featurizers/                # Sequence featurization modules
├── predictors/                 # ML model training and prediction
│   └── sweep_configs/          # Hyperparameter sweep configurations
├── proposers/                  # Variant proposal modules
├── splitters/                  # Data splitting strategies
└── utils/                      # Utility functions

data/                           # Example datasets
notebooks/                      # Tutorial and benchmarking notebooks
scripts/                        # Command-line workflow scripts

sweep_results/                  # Content-addressed training runs
└── <experiment>/
    ├── manifest.json           # Input, seed, software, and fold identity
    ├── jobs/                   # Atomic fold/config checkpoints
    └── results.csv             # Deterministically reconstructed sweep table

proteins/                       # Cache directory (auto-generated)
└── <protein_name>/
    ├── inputs/                 # Immutable content-addressed input evidence
    │   └── <input-identity>/
    ├── feature_cache/          # Cached featurized sequences by featurizer type
    ├── model_cache/            # Cached predictor objects by dataset
    │   └── <dataset>/
    │       ├── objects/        # Saved models
    │       └── results/        # Model comparison results
    ├── proposers/              # Evaluated proposed sequences
    │   ├── checkpoints/        # Hash-validated final ensemble folds
    │   └── results/
    └── split_cache/            # Cached splitter objects by dataset
        └── <dataset>/
```

## Training and comparing various machine learning models

The MULTI-evolve package can be used to compare different data splitting methods, sequence featurizations, and machine learning models. In addition, the package can be used to perform zero-shot predictions with protein language models (ESM, ESM-IF). Examples are provided in the ```notebooks/examples``` folder. 

## Contributors

Vincent Q. Tran ([VincentQTran](https://github.com/VincentQTran/)), Matthew Nemeth ([mnemeth66](https://github.com/mnemeth66)), and Brian Hie ([brianhie](https://github.com/brianhie)).

## Citation

MULTI-evolve was developed by the Patrick Hsu Lab. If you use this code for your research, please cite our paper:

```
@ARTICLE
author={Tran, Vincent Q. and Nemeth, Matthew and Bartie, Liam J. and Chandrasekaran, Sita S. and Fanton, Alison and Moon, Hyungseok C. and Hie, Brian L. and Konermann, Silvana and Hsu, Patrick D.},
title={Rapid directed evolution guided by protein language models and epistatic interactions},
year={2026},
journal={Science},
DOI={https://doi.org/10.1126/science.aea1820}
```
