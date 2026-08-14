# verl Phase-Aware Offloading Experiments

This repository contains the experiment drivers, configurations, reports, and
summarized results for single-GPU phase-aware offloading experiments.

## Modified verl

The experiments use a modified version of verl based on upstream v0.4.1.

- Upstream commit: `8d9e350ea58c7ad4b50dd14d9dcb50577242c55f`
- Planned base tag: `experiment-base-v0.4.1`
- Planned modified branch: `single-gpu-phase-offloading`

The modified verl repository will be linked at `src/verl` as a Git submodule.

## Environment

- Ubuntu 22.04 family
- Python 3.10.12
- CUDA runtime 11.8
- PyTorch 2.4.1+cu118
- verl 0.4.1 plus local phase-aware offloading changes

## Clone

```bash
git clone --recurse-submodules git@github.com:2Ju1/verl-research.git
cd verl-research
```

For an existing clone:

```bash
git submodule update --init --recursive
```

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -r constraints-cu118.txt \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-lock.txt
pip install -e src/verl
```

## Project layout

```text
benchmarks/offload/  Experiment runners and configurations
data/                Small input datasets
reports/             Experiment reports
results/             Curated final results (to be added)
src/verl/            Modified verl Git submodule
```

Raw outputs, caches, models, temporary files, and virtual environments are not
tracked by Git.
