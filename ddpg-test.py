#!/usr/bin/env python3

import gym
import time
# import robo_gym
import pybulletgym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def main(load_model=False):
    # env = gym.make("EndEffectorPositioningUR10Sim-v0", ip='localhost', gui=True)
    env = gym.make('ReacherPyBulletEnv-v0')

    if not load_model:
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
        model.learn(total_timesteps=10000, log_interval=10)
        model.save("ddpg_pendulum")
        env = model.get_env()

        # del model # remove to demonstrate saving and loading
    elif load_model:
        model = DDPG.load("ddpg_pendulum")

    env.render("human")
    obs = env.reset()
    print("Env reset")
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        time.sleep(0.1)

if __name__ == "__main__":
    load_model = True
    main(load_model=load_model)
