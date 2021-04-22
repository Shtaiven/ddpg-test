#!/usr/bin/env python3
import gym
import time
import argparse
import stable_baselines3_test.custom_envs  # provides CustomReacherBulletEnv-v0
import numpy as np

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn

from stable_baselines3_test.policies import CustomTD3Policy


def main(load_model='', save_model='', timesteps=100000, use_td3=False):
    env = gym.make('CustomReacherBulletEnv-v0')
    model_type = DDPG
    if use_td3:
        model_type = TD3
    print(f'Using model type {model_type.__name__}')

    # Needed to display the environment
    print("Env render")
    env.render("human", close=True)

    if not load_model or not isinstance(load_model, str):
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Set hyperparameters based on rl-baselines-zoo
        # ReacherBulletEnv-v0:
        #     env_wrapper: utils.wrappers.TimeFeatureWrapper
        #     n_timesteps: !!float 1e6
        #     policy: 'MlpPolicy'
        #     gamma: 0.99
        #     buffer_size: 1000000
        #     noise_type: 'normal'
        #     noise_std: 0.1
        #     learning_starts: 10000
        #     batch_size: 100
        #     learning_rate: !!float 1e-3
        #     train_freq: 1000
        #     gradient_steps: 1000
        #     policy_kwargs: "dict(layers=[400, 300])"
        model = model_type(CustomTD3Policy,
                           env,
                           learning_rate=float(1e-3),
                           learning_starts=1000000,
                           gamma=0.99,
                           buffer_size=1000000,
                           batch_size=100,
                           train_freq=1000,
                           gradient_steps=1000,
                           action_noise=action_noise,
                           verbose=1)

        print(f'Training for {timesteps} timesteps')
        model.learn(total_timesteps=timesteps,
                    eval_env=env,
                    log_interval=10)
        if save_model and isinstance(save_model, str):
            print(f'Saving trained model to {save_model}')
            model.save(save_model)

    else:
        print(f'Loading model from {load_model}')
        model = model_type.load(load_model)

    # Show env in action
    obs = env.reset()
    dones = False
    dones_prev = dones
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        if dones != dones_prev:
            # reset the env
            print('Done')
            dones = False
            time.sleep(2.99)
            obs = env.reset()
        dones_prev = dones
        time.sleep(0.01)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', action='store', default='', metavar='FILENAME', help='load model data from specified zip file')
    parser.add_argument('-o', '--output', action='store', default='', metavar='FILENAME', help='save model data to specified zip file')
    parser.add_argument('-t', '--timesteps', action='store', default='100000', metavar='N', help='train for N timesteps')
    parser.add_argument('--td3', action='store_true', help='use TD3 instead of DDPG')
    args = parser.parse_args()

    # Run
    main(load_model=args.file,
         save_model=args.output,
         timesteps=int(args.timesteps),
         use_td3=args.td3)
