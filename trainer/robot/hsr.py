"""This file implements Toyota's HSR robot model.

This implementation is for human following tasks, the controls dimensions are:
1. left driving wheel angular velocity (velocity control)
2. right driving wheel angular velocity (velocity control)

"""
import pybullet as p
from robot_interface import RobotInterface

BASE_ROLL_JOINT = 1
R_DRIVE_WHEEL_JNT = 2
L_DRIVE_WHEEL_JNT = 3
TORSO_LIFT_JNT = 12
HEAD_PAN_JNT = 13
HEAD_TILT_JNT = 14
HEAD_CAMERA_JNT = 22
ARM_LIFT_JNT = 23
ARM_FLEX_JNT = 24
ARM_ROLL_JNT = 25
WRIST_FLEX_JNT = 26
WRIST_ROLL_JNT = 27
HAND_MOTOR_JNT = 30
HAND_L_PROXIMAL_JNT = 31
HAND_L_SPRING_PROXIMAL_JNT = 32
HAND_L_MIMIC_DISTAL_JNT = 33
HAND_L_DISTAL_JNT = 34
HAND_R_PROXIMAL_JNT = 37
HAND_R_SPRING_PROXIMAL_JNT = 38
HAND_R_MIMIC_DISTAL_JNT = 39
HAND_R_DISTAL_JNT = 40

MAX_WHEEL_VELOCITY = 20.8
MAX_WHEEL_FORCE = 11.067

URDF_ROOT = './urdf/hsr'


class HSR(RobotInterface):
    """Toyota's HSR robot."""

    def __init__(self, start_position, start_orientation):
        """Load URDF and initialize the robot. """
        p.setAdditionalSearchPath(URDF_ROOT)
        self._z = 0.0184
        self.id = p.loadURDF('hsr_description/robots/hsrb4s.urdf',
                             start_position + [self._z],
                             start_orientation,
                             flags=p.URDF_USE_INERTIA_FROM_FILE)

    def keep_other_joints_fixed(self):
        """Fix other joints pose."""
        p.setJointMotorControl2(self.id, TORSO_LIFT_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, ARM_FLEX_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, ARM_LIFT_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, ARM_ROLL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, WRIST_FLEX_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, WRIST_ROLL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_MOTOR_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_L_PROXIMAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_L_SPRING_PROXIMAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_L_MIMIC_DISTAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_L_DISTAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_R_PROXIMAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_R_SPRING_PROXIMAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_R_MIMIC_DISTAL_JNT, p.POSITION_CONTROL, 0)
        p.setJointMotorControl2(self.id, HAND_R_DISTAL_JNT, p.POSITION_CONTROL, 0)

    def get_camera_target_positions(self):
        """Get head camera and target object positions. """
        robot = self.id

        torso_lift_jnt_info = p.getJointInfo(robot, TORSO_LIFT_JNT)
        base_translation_torso_lift = torso_lift_jnt_info[14]
        base_rotation_torso_lift = torso_lift_jnt_info[15]

        head_pan_jnt_info = p.getJointInfo(robot, HEAD_PAN_JNT)
        torso_lift_translation_head_pan = head_pan_jnt_info[14]
        head_pan_angle, _, _, _ = p.getJointState(robot, HEAD_PAN_JNT)
        torso_lift_rotation_head_pan = p.getQuaternionFromEuler([0, 0, head_pan_angle])

        head_tilt_jnt_info = p.getJointInfo(robot, HEAD_TILT_JNT)
        head_pan_translation_head_tilt = head_tilt_jnt_info[14]
        head_tilt_angle, _, _, _ = p.getJointState(robot, HEAD_TILT_JNT)
        head_pan_rotation_head_tilt = p.getQuaternionFromEuler([0, -head_tilt_angle, 0])

        head_camera_jnt_info = p.getJointInfo(robot, HEAD_CAMERA_JNT)
        head_tilt_translation_camera = head_camera_jnt_info[14]
        head_tilt_rotation_camera = head_camera_jnt_info[15]

        camera_translation_obj = [0, 0, 1]
        camera_rotation_obj = [0, 0, 0, 1]

        world_translation_base, world_rotation_base = p.getBasePositionAndOrientation(robot)
        world_translation_torso, world_rotation_torso = (
            p.multiplyTransforms(world_translation_base, world_rotation_base,
                                 base_translation_torso_lift, base_rotation_torso_lift))
        world_translation_head_pan, world_rotation_head_pan = (
            p.multiplyTransforms(world_translation_torso, world_rotation_torso,
                                 torso_lift_translation_head_pan, torso_lift_rotation_head_pan))
        world_translation_head_tilt, world_rotation_head_tilt = (
            p.multiplyTransforms(world_translation_head_pan, world_rotation_head_pan,
                                 head_pan_translation_head_tilt, head_pan_rotation_head_tilt))
        world_translation_camera, world_rotation_camera = (
        p.multiplyTransforms(world_translation_head_tilt, world_rotation_head_tilt,
                             head_tilt_translation_camera, head_tilt_rotation_camera))
        world_translation_obj, world_rotation_obj = (
        p.multiplyTransforms(world_translation_camera, world_rotation_camera,
                             camera_translation_obj, camera_rotation_obj))

        return (world_translation_camera, world_rotation_camera,
                world_translation_obj, world_rotation_obj)

    def apply_wheel_velocity(self, velocities):
        """Apply velocity control to both driving wheels."""
        p.setJointMotorControlArray(self.id, [L_DRIVE_WHEEL_JNT, R_DRIVE_WHEEL_JNT],
                                    p.VELOCITY_CONTROL, targetVelocities=velocities,
                                    forces=[MAX_WHEEL_VELOCITY] * 2)

    def apply_control(self, control):
        """Apply controls to HSR."""
        self.keep_other_joints_fixed()
        wheel_ctrl = [x * MAX_WHEEL_VELOCITY for x in control]
        self.apply_wheel_velocity(wheel_ctrl)
