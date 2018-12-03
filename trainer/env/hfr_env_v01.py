"""This file implements a gym environment.

In each episode, there will be 2 robots, one chaser and one runner.
In this environment the runner will not move.

"""
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
from robot.robot_manager import RobotType, RobotManager

IMAGE_W = 320
IMAGE_H = 240


def convert_to_list(obj):
    try:
        iter(obj)
        return obj
    except TypeError:
        return [obj]


class HumanFollowingGymEnv(gym.Env):
    """The gym environment for human following robots.

    It simulates the human following task.

    The action space includes:
    1. velocity for left driving wheel(s)
    2. velocity for right driving wheel(s)

    The state space includes:
    1. the depth image from the head camera (size = IMAGE_H * IMAGE_W)
    2. position (size=3) and orientation (size=3) change between time t and t-1

    The reward function is based on the distance between the robot and the target.

    Each episode ends when:
    1. max_steps has passed
    2. either robot has fallen
    3. distance between robots is less than 1 meter
    whichever occurs first.

    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 25
    }

    def __init__(self,
                 robot_type=RobotType.HSR,
                 target_distance_min=2.0,
                 target_distance_max=5.0,
                 target_angle_max=np.pi / 6,
                 min_distance=1.0,
                 max_steps=500,
                 time_step=0.01,
                 action_repeat=5,
                 reward_scale=1.0,
                 reward_action_coef=0.0,
                 reward_orient_coef=2.0,
                 render=False):
        """Initialize the human following environment."""

        if render:
            self._pybullet_client = p.connect(p.GUI)
        else:
            self._pybullet_client = p.connect(p.DIRECT)
        self._robot_type = robot_type
        self._max_steps = max_steps
        self._time_step = time_step
        self._action_repeat = action_repeat
        self._reward_action_coef = reward_action_coef
        self._reward_orient_coef = reward_orient_coef
        self._reward_scale = reward_scale
        self._env_step_counter = 0
        self._render = render
        self._last_xy_yaw = None
        self._chaser = None
        self._target = None
        self._target_distance_range = [target_distance_min,
                                       target_distance_max]
        self._target_angle_max = target_angle_max
        self._min_distance = min_distance
        self.state_dim = 3
        self.action_dim = 2
        action_bound = np.ones(self.action_dim)
        self.action_space = spaces.Box(low=-action_bound,
                                       high=action_bound,
                                       dtype=np.float64)

    def _reset_env(self):
        p.resetSimulation(self._pybullet_client)
        p.setTimeStep(self._time_step)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf')

    def _get_global_view(self):
        base_pos, _ = self._chaser.get_pose()
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=10,
            yaw=0,
            pitch=-50,
            roll=0,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(IMAGE_W) / IMAGE_H,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
            width=IMAGE_W,
            height=IMAGE_H,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        return np.array(px)[:, :, :3]

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array" or self._chaser is None:
            return np.array([])
        if self._chaser is None:
            return np.array([])
        rgb_array, _ = self._chaser.get_camera_image(IMAGE_W, IMAGE_H)
        rgb_array = rgb_array[:, :, :3]
        global_view = self._get_global_view()
        return np.concatenate([rgb_array, global_view], axis=1)

    def close(self):
        p.disconnect(self._pybullet_client)

    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        r_min, r_max = self._target_distance_range
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(-self._target_angle_max,
                                  self._target_angle_max)
        # x-axis going out of the front face of the robot.
        chaser_x = 0
        chaser_y = 0
        target_x = r * np.sin(theta)
        target_y = r * np.cos(theta)
#        print('target(x,y) = ({}, {})'.format(target_x, target_y))
        chaser_yaw = 0
        target_yaw = np.random.uniform(-np.pi, np.pi)
        if (
                (self._chaser is None) or
                (self._target is None) or
                (self._chaser.is_fallen()) or
                (self._target.is_fallen())
        ):
            self._reset_env()
            self._chaser = RobotManager.create_robot(
                self._robot_type, chaser_x, chaser_y, chaser_yaw)
            self._target = RobotManager.create_robot(
                self._robot_type, target_x, target_y, target_yaw)
        else:
            self.set_chaser_xy_yaw(chaser_x, chaser_y, chaser_yaw)
            self.set_target_xy_yaw(target_x, target_y, target_yaw)
        self._env_step_counter = 0
        self._last_xy_yaw = None
        if self._render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        return self._get_observation()

    def set_target_xy_yaw(self, x, y, yaw):
        self._target.set_xy_yaw(x, y, yaw)

    def set_chaser_xy_yaw(self, x, y, yaw):
        self._chaser.set_xy_yaw(x, y, yaw)

    def _get_observation(self):
        rgb_array, depth_array = self._chaser.get_camera_image(IMAGE_W, IMAGE_H)
        (x, y), yaw = self._chaser.get_xy_yaw()
        if self._last_xy_yaw:
            xy_yaw_diff = [x - self._last_xy_yaw[0],
                           y - self._last_xy_yaw[1],
                           yaw - self._last_xy_yaw[2]]
        else:
            xy_yaw_diff = [.0, .0, .0]
        self._last_xy_yaw = [x, y, yaw]
        return depth_array, xy_yaw_diff

    def _get_distance(self):
        (chaser_x, chaser_y), _ = self._chaser.get_xy_yaw()
        (target_x, target_y), _ = self._target.get_xy_yaw()
        distance_sqr = ((target_x - chaser_x) ** 2 + (target_y - chaser_y) ** 2)
        distance = np.sqrt(distance_sqr)
        return distance

    def _get_orientation_diff(self):
        (target_x, target_y), _ = self._target.get_xy_yaw()
        (chaser_x, chaser_y), chaser_yaw = self._chaser.get_xy_yaw()
        theta = np.arctan2(target_x - chaser_x, target_y - chaser_y)
#        self._chaser.show_text_over_head('yaw={0:.4f}, theta={1:.4f}'.format(
#            chaser_yaw * 180 / np.pi, theta * 180 / np.pi))
        orientation_diff = min(abs(theta + chaser_yaw), abs(theta - chaser_yaw))
#        self._chaser.show_text_over_head('diff={0:.4f}'.format(
#            orientation_diff * 180.0 / np.pi))
        return orientation_diff

    def _get_reward(self, action):
        distance_r = -self._get_distance()
        orientation_r = -self._get_orientation_diff()
        action_r = -np.linalg.norm(action)
        fall_r = -10000 if self._chaser.is_fallen() else 0
        return ((fall_r + distance_r +
                 self._reward_orient_coef * orientation_r +
                 self._reward_action_coef * action_r) * self._reward_scale)

    def _is_episode_over(self):
        return ((self._env_step_counter >= self._max_steps) or
                (self._chaser.is_fallen()) or
                (self._target.is_fallen()) or
                (self._get_distance() < self._min_distance))

    def step(self, action):
        if len(action) > self.action_dim:
            chaser_ctrl = convert_to_list(np.clip(action[:self.action_dim],
                                                  self.action_space.low,
                                                  self.action_space.high))
            target_ctrl = convert_to_list(np.clip(action[self.action_dim:],
                                                  self.action_space.low,
                                                  self.action_space.high))
        else:
            chaser_ctrl = convert_to_list(np.clip(action,
                                                  self.action_space.low,
                                                  self.action_space.high))
            target_ctrl = [0.] * self.action_dim
        for i in range(self._action_repeat):
            self._chaser.apply_control(chaser_ctrl)
            self._target.apply_control(target_ctrl)
            p.stepSimulation()
        self._env_step_counter += 1
        observation = self._get_observation()
        reward = self._get_reward(chaser_ctrl)
        done = self._is_episode_over()
#        self._chaser.show_text_over_head('reward={0:.4f}'.format(reward))
        info = {'step': self._env_step_counter,
                'chaser': self._chaser.get_pose(),
                'target': self._target.get_pose()}
        return observation, reward, done, info
