from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
from pybullet_envs.robot_manipulators import Reacher
from pybullet_envs.env_bases import Camera


class CustomCamera(Camera):
    def __init__(self, env):
        super(CustomCamera, self).__init__(env)

    def move_and_look_at(self, x, y, z, distance=None, pitch=None, yaw=None):
        lookat = [x, y, z]
        camInfo = self.env._p.getDebugVisualizerCamera()

        if distance is None:
            distance = camInfo[10]

        if pitch is None:
            pitch = camInfo[9]

        if yaw is None:
            yaw = camInfo[8]

        self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

class ReacherBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = Reacher()
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
        self.camera = CustomCamera(self)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        electricity_cost = (
            -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot)
                    )  # work torque*angular_velocity
            - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
        self.rewards = [
            float(self.potential - potential_old),
            float(electricity_cost),
            float(stuck_joint_cost)
        ]
        target_distance = np.linalg.norm(self.robot.to_target_vec)

        # taken from OpenAI Reacher-v2 env created using MuJoCo
        # https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py
        # distance_cost = -target_distance
        # action_cost = -np.square(a).sum()
        # self.rewards = [
        #     float(distance_cost),
        #     float(action_cost)
        # ]

        self.HUD(state, a, False)

        done = target_distance < 0.01  # in cm?

        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        x, y, z = [0., 0., 0.]
        self.camera.move_and_look_at(x, y, z, 0.4, -89.99, 0.)
