# DDPG Experiment using Stable Baselines 3

This experiment attempts to learn a motion planning task for a 2 joint robotic arm using the DDPG algorithm

## Setup

`pipenv` should be installed to run the following instructions. Pip can be used to install these libraries to a base python environment. Make sure python3.8 is installed on your system before using pipenv.

```bash
cd ddpg-test
git submodule update --init --recursive
pipenv shell
pipenv update
```

## Usage

```bash
./ddpg-test.py
```
