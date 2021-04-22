#!/usr/bin/env python3

import gym
import time

import sfujimoto_test.TD3 as TD3
import stable_baselines3_test.custom_envs


def main(load_model=''):
    env = gym.make('CustomReacherBulletEnv-v0')

    # Needed to display the environment
    print("Env render")
    env.render("human", close=True)

    print(f'Loading model from {load_model}')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = 0.2 * max_action
    kwargs["noise_clip"] = 0.5 * max_action
    kwargs["policy_freq"] = 2
    policy = TD3.TD3(**kwargs)
    policy.load(filename=load_model)

    # Show env in action
    obs = env.reset()
    dones = False
    dones_prev = dones
    print("Ctrl-C to quit")
    while True:
        action = policy.select_action(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        if dones != dones_prev:
            # reset the env
            dones = False
            time.sleep(0.99)
            obs = env.reset()
        dones_prev = dones
        time.sleep(0.01)


if __name__ == "__main__":
    model_name = "sfujimoto_test/models/TD3_ReacherBulletEnv-v0_0"
    main(model_name)
