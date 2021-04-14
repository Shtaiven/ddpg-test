#!/usr/bin/env python3

import gym
import time
import argparse
import pybulletgym  # provides ReacherPyBulletEnv-v0
import numpy as np

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def main(load_model='', save_model='', timesteps=10000, use_td3=False):
    env = gym.make('ReacherPyBulletEnv-v0')
    model_type = DDPG
    if use_td3:
        model_type = TD3

    # Needed to display the environment
    print("Env render")
    env.render("human")

    if not load_model or not isinstance(load_model, str):
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = model_type("MlpPolicy", env, action_noise=action_noise, verbose=1)

        print(f'Training for {timesteps} timesteps')
        model.learn(total_timesteps=timesteps, log_interval=10)
        if save_model and isinstance(save_model, str):
            print(f'Saving trained model to {save_model}')
            model.save(save_model)
        env = model.get_env()

    else:
        print(f'Loading model from {load_model}')
        model = model_type.load(load_model)

    # Show env in action
    obs = env.reset()
    dones = False
    dones_prev = dones
    while True:
        if not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render("human")
            if dones != dones_prev:
                print('Done')
            dones_prev = dones
        time.sleep(0.1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', action='store', default='', metavar='FILENAME', help='load model data from specified zip file')
    parser.add_argument('-o', '--output', action='store', default='', metavar='FILENAME', help='save model data to specified zip file')
    parser.add_argument('-t', '--timesteps', action='store', default='10000', metavar='N', help='train for N timesteps')
    parser.add_argument('--td3', action='store_true', help='use TD3 instead of DDPG')
    args = parser.parse_args()

    # Run
    main(load_model=args.file,
         save_model=args.output,
         timesteps=int(args.timesteps),
         use_td3=args.td3)
