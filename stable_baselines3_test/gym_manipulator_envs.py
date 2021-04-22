from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
from robot_manipulators import Reacher
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

        # ---- Begin Reacher-v2 ----
        # vec = self.robot.to_target_vec
        # reward_dist = - np.linalg.norm(vec)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl

        # self.robot.apply_action(a)
        # self.scene.global_step()

        # ob = self.robot.calc_state()
        # self.HUD(ob, a, False)
        # done = False
        # return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        # ---- End Reacher-v2 ----

        # ---- Begin original ReacherBulletEnv ----
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
        self.HUD(state, a, False)
        return state, sum(self.rewards), False, {}
        # ---- End original ReacherBulletEnv ----

    def camera_adjust(self):
        x, y, z = [0., 0., 0.]
        self.camera.move_and_look_at(x, y, z, 0.4, -89.99, 0.)
