# DDPG Experiment using Stable Baselines 3

This experiment attempts to learn a motion planning task for a 2 joint robotic arm using the DDPG algorithm

## Setup

`pipenv` should be installed to run the following instructions. Pip can be used to install these libraries to a base python environment. Make sure python3.8 is installed on your system before using pipenv.

```bash
cd ddpg-test
git submodule update --init --recursive
pipenv shell
pipenv update --pre --clear
cd pybullet-gym
pip install -e .
cd ..
```

## Usage

```bash
python ddpg-test.py
```

Optional arguments:

```text
  -h, --help            show this help message and exit
  -f FILENAME, --file FILENAME
                        load model data from specified zip file
  -o FILENAME, --output FILENAME
                        save model data to specified zip file
  -t N, --timesteps N   train for N timesteps
```

## Common Issues

If `pipenv` refuses to install dependencies, they can be installed using

```bash
pipenv update --pre --skip-lock
```
