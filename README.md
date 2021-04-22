# DDPG Experiment using Stable Baselines 3

This experiment attempts to learn a motion planning task for a 2 joint robotic arm using the DDPG algorithm

## Setup

`pipenv` should be installed to run the following instructions. Pip can be used to install these libraries to a base python environment. Make sure python 3.8 is installed on your system before using pipenv.

```bash
cd ddpg-test
git submodule update --init --recursive
pipenv shell
pipenv update --pre --clear
```

## Enjoy pretrained model (sfujimoto TD3)

```bash
python enjoy.py
```

## Usage sfujimoto_test

```bash
cd sfujimoto_test

# train a model
python main.py --env ReacherBulletEnv-v0 --save_model
```

## Usage stable_baselines3_test

```bash
# train a model and view it
python ddpg-test.py

# load an already trained model
python ddpg-test.py -f my_trained_model.zip

# train a model using TD3 for 20000 timesteps and save it
python ddpg-test.py -o my_trained_model.zip -t 20000 --td3
```

Optional arguments:

```text
  -h, --help            show this help message and exit
  -f FILENAME, --file FILENAME
                        load model data from specified zip file
  -o FILENAME, --output FILENAME
                        save model data to specified zip file
  -t N, --timesteps N   train for N timesteps
  --td3                 use TD3 instead of DDPG
```

## Common Issues

If `pipenv` refuses to install dependencies, they can be installed using

```bash
pipenv update --pre --skip-lock
```
