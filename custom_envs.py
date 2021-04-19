import gym
from gym.envs.registration import registry


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(
    id='CustomReacherBulletEnv-v0',
    entry_point='gym_manipulator_envs:ReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)
