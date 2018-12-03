from trainer.env.hfr_env_v01 import HumanFollowingGymEnv
import numpy as np
import pybullet as p
from robot.robot_manager import RobotType

env = HumanFollowingGymEnv(robot_type=RobotType.R2D2,
                           target_angle_max=np.pi / 6,
                           render=True)
print(env.action_space.high)
print(env.action_space.low)
env.reset()

chaser_l_wheel_vel_slider = p.addUserDebugParameter(
    'chaserLeftWheelVelocity', -1.0, 1.0, 0.3)
chaser_r_wheel_vel_slider = p.addUserDebugParameter(
    'chaserRightWheelVelocity', -1.0, 1.0, -0.3)
target_l_wheel_vel_slider = p.addUserDebugParameter(
    'targetLeftWheelVelocity', -1.0, 1.0, 0)
target_r_wheel_vel_slider = p.addUserDebugParameter(
    'targetRightWheelVelocity', -1.0, 1.0, 0)
target_position_ctrl = p.addUserDebugParameter('targetPositionCtrl', 0, 1, 0)
target_x_slider = p.addUserDebugParameter('targetX', -5, 5, 0)
target_y_slider = p.addUserDebugParameter('targetY', -5, 5, 2)

step = 0
while True:

    chaser_l_wheel_vel = p.readUserDebugParameter(chaser_l_wheel_vel_slider)
    chaser_r_wheel_vel = p.readUserDebugParameter(chaser_r_wheel_vel_slider)
    target_l_wheel_vel = p.readUserDebugParameter(target_l_wheel_vel_slider)
    target_r_wheel_vel = p.readUserDebugParameter(target_r_wheel_vel_slider)
    target_position_flag = p.readUserDebugParameter(target_position_ctrl)
    target_x = p.readUserDebugParameter(target_x_slider)
    target_y = p.readUserDebugParameter(target_y_slider)
    if target_position_flag > 0:
        env.set_target_xy_yaw(target_x, target_y, 0)
    control_msg = [chaser_l_wheel_vel, chaser_r_wheel_vel,
                   target_l_wheel_vel, target_r_wheel_vel]
#    control_msg = [chaser_l_wheel_vel, chaser_r_wheel_vel]

    observation, reward, done, info = env.step(np.array(control_msg))
#    print('depth.min() = {}, depth.max() = {}'.format(
#        observation[0].min(), observation[0].max()
#    ))

    step = info['step']
    print('Step={}, reward={}, observation={}, done={}'.format(
        step, reward, observation[1], done))
    chaser_xyz, chaser_rpy = info['chaser']
    target_xyz, target_rpy = info['target']
    print('chaser_xyz={}, chaser_rpy={}'.format(chaser_xyz, chaser_rpy))
    print('target_xyz={}, target_rpy={}'.format(target_xyz, target_rpy))

    if done:
        env.reset()
