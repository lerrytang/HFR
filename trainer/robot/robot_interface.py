"""Interface for robot.

This file defines an interface for human following robots.

"""
import numpy as np
import pybullet as p


class RobotInterface(object):
    """Human following robot."""

    id = 0
    z = 0
    text_id = -1

    def print_joints_info(self):
        """Print all joints info."""
        for jnt_id in range(p.getNumJoints(self.id)):
            print(p.getJointInfo(self.id, jnt_id))

    def get_camera_target_positions(self):
        """Get head camera and target object positions.

        Return poses for camera and object.

        Returns:
           (camera_translation, camera_rotation,
            object_translation, object_rotation),
            all coordinates are in the global frame.
        """
        raise NotImplementedError('Not implemented')

    def get_camera_image(self, img_w, img_h):
        """Get image from camera.

        Return RGB and depth images.

        Args:
            img_w: int, image width
            img_h: int, image height
        Returns:
            (rgb_array, depth_array), both are numpy arrays
        """
        far = 1000
        near = 0.01
        camera_pos, _, target_pos, _ = self.get_camera_target_positions()
        view_matrix = p.computeViewMatrix( camera_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            60, 1.0 * img_w / img_h, near, far)
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            img_w, img_h, view_matrix, proj_matrix, shadow=1,
            lightDirection=[1, 1, 1], renderer=p.ER_TINY_RENDERER)
        """
        suggested by Erwin
        
        Bullet uses OpenGL to render, and the convention is non-linear z-buffer.

        See https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer

        far=1000.//depends on projection matrix, this is default
        near=0.01//depends on projection matrix
        depth = far * near / (far - (far - near) * depthImg)//depthImg is the depth from Bullet 'getCameraImage'

        Note that this 'true' z value is still not really the distance from 3d point to the camera center,
        but the projected distance onto the near plane.
        If you want the actual distance from camera to 3d point, you need a bit more work.
        depth/=math.cos(alpha)

        See PyBullet https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py
        """
        depth_img = far * near / (far - (far - near) * depth_img)
        depth_img = np.clip(depth_img, 0, 20.0)
        return rgb_img, depth_img

    def set_xy_yaw(self, x, y, yaw):
        """Set robot's (x, y) and yaw."""
        self.set_pose([x, y, self.z], [0, 0, yaw])

    def get_xy_yaw(self):
        """Get robot's current position and orientation."""
        xyz, rpy = self.get_pose()
        return xyz[:2], rpy[-1]

    def get_pose(self):
        """Get the robot's current pose."""
        xyz, rpy = p.getBasePositionAndOrientation(self.id)
        rpy = p.getEulerFromQuaternion(rpy)
        return xyz, rpy

    def set_pose(self, position, orientation):
        """Set the robot to the specified pose.

        Set the robot to the specified pose.

        Args:
            position: vec3, [x, y, z].
            orientation: vec3, [roll, pitch, yaw]
        """
        p.resetBasePositionAndOrientation(
            self.id, position, p.getQuaternionFromEuler(orientation))

    def is_fallen(self):
        """Test whether robot has fallen."""
        camera_pos, _, _, _ = self.get_camera_target_positions()
        if camera_pos[2] < 0.4:
            return True
        else:
            return False

    def show_text_over_head(self, text):
        """Show some text over robot's head"""
        camera_pos, _, _, _ = self.get_camera_target_positions()
        camera_pos = list(camera_pos)
        camera_pos[1] -= 0.1
        camera_pos[2] += 0.2
        if self.text_id >= 0:
            p.removeUserDebugItem(self.text_id)
            self.text_id = -1
        self.text_id = p.addUserDebugText(
            text, camera_pos, textColorRGB=[0, 0, 0.8], textSize=1, lifeTime=0)

    def apply_control(self, control):
        """Apply control to the robot.

        Velocity control, the dimension of controls depends on the robot.

        Args:
            control: numpy.array

        """
        raise NotImplementedError('Not implemented')
