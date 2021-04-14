#!/usr/bin/env python3

import gym
import time
import argparse
import pybulletgym  # provides ReacherPyBulletEnv-v0
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def main(load_model='', save_model='', timesteps=10000):
    env = gym.make('ReacherPyBulletEnv-v0')

    if not load_model or not isinstance(load_model, str):
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

        print(f'Training for %{timesteps} timesteps')
        model.learn(total_timesteps=timesteps, log_interval=10)
        if save_model and isinstance(save_model, str):
            print(f'Saving trained model to %{save_model}')
            model.save(save_model)
        env = model.get_env()

    else:
        print(f'Loading model from %{load_model}')
        model = DDPG.load(load_model)

    # Needed to display the environment
    print("Env render")
    env.render("human")
    time.sleep(0.1) # Sometimes the env doesn't render after training. This might fix it?

    # Show env in action
    obs = env.reset()
    print("Env reset")
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        time.sleep(0.1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', action='store', default='', metavar='FILENAME', help='load model data from specified zip file')
    parser.add_argument('-o', '--output', action='store', default='', metavar='FILENAME', help='save model data to specified zip file')
    parser.add_argument('-t', '--timesteps', action='store', default='10000', metavar='N', help='train for N timesteps')
    args = parser.parse_args()

    # Run
    main(load_model=args.file,
         save_model=args.output,
         timesteps=int(args.timesteps))
