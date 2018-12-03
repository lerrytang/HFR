"""R2D2.

This implementation is for human following tasks, the controls dimensions are:
1. left driving wheel angular velocity (velocity control)
2. right driving wheel angular velocity (velocity control)

"""
import os
import pybullet as p
import robot_interface

R_FRONT_WHEEL_JNT = 2
R_BACK_WHEEL_JNT = 3
L_FRONT_WHEEL_JNT = 6
L_BACK_WHEEL_JNT = 7
HEAD_SWIVEL_JNT = 8
CAMERA_JNT = 9

MAX_WHEEL_VELOCITY = 10.
MAX_WHEEL_FORCE = 30

URDF_ROOT = os.path.join(os.path.dirname(robot_interface.__file__), 'urdf/r2d2')


class R2D2(robot_interface.RobotInterface):
    """R2D2 robot from pybullet_data."""

    def __init__(self, x, y, yaw):
        """Load URDF and initialize the robot. """
        p.setAdditionalSearchPath(URDF_ROOT)
        self.z = 0.4705
        self.id = p.loadURDF('r2d2.urdf',
                             [x, y, self.z],
                             p.getQuaternionFromEuler([0, 0, yaw]),
                             flags=p.URDF_USE_INERTIA_FROM_FILE)

    def get_camera_target_positions(self):
        """Get head camera and target object positions. """

        robot = self.id

        (world_translation_base,
         world_rotation_base) = p.getBasePositionAndOrientation(robot)

        head_swirl_jnt_info = p.getJointInfo(robot, HEAD_SWIVEL_JNT)
        base_translation_head = head_swirl_jnt_info[14]
        rad, _, _, _ = p.getJointState(robot, HEAD_SWIVEL_JNT)
        base_rotation_head = p.getQuaternionFromEuler([0, 0, rad])
        world_translation_head, world_rotation_head = p.multiplyTransforms(
            world_translation_base, world_rotation_base,
            base_translation_head, base_rotation_head)

        camera_jnt_info = p.getJointInfo(robot, CAMERA_JNT)
        head_translation_camera = camera_jnt_info[14]
        head_rotation_camera = camera_jnt_info[15]
        world_translation_camera, world_rotation_camera = p.multiplyTransforms(
            world_translation_head, world_rotation_head,
            head_translation_camera, head_rotation_camera)

        camera_translation_obj = [0, 1, 0]
        camera_rotation_obj = p.getQuaternionFromEuler([0, -1.75, 0])
        world_translation_obj, world_rotation_obj = p.multiplyTransforms(
            world_translation_camera, world_rotation_camera,
            camera_translation_obj, camera_rotation_obj)

        return (world_translation_camera, world_rotation_camera,
                world_translation_obj, world_rotation_obj)

    def apply_wheel_velocity(self, velocities):
        """Apply velocity control to both driving wheels."""
        p.setJointMotorControlArray(self.id,
                                    [L_FRONT_WHEEL_JNT, R_FRONT_WHEEL_JNT,
                                     L_BACK_WHEEL_JNT, R_BACK_WHEEL_JNT],
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=velocities,
                                    forces=[MAX_WHEEL_FORCE] * 4)

    def apply_control(self, control):
        """Apply controls to HSR."""
        wheel_ctrl = [x * MAX_WHEEL_VELOCITY for x in control]
        wheel_ctrl += wheel_ctrl
        self.apply_wheel_velocity(wheel_ctrl)
