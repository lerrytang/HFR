import unittest
import numpy as np
import pybullet as p
import pybullet_data
from trainer.robot.robot_manager import RobotManager
from trainer.robot.robot_manager import RobotType


class R2D2_test(unittest.TestCase):

    def setUp(self):
        self.client = p.connect(p.DIRECT)
        p.resetSimulation(self.client)
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf')
        self.r2d2 = RobotManager.create_robot(
            RobotType.R2D2, x=0, y=0, yaw=0)

    def tearDown(self):
        p.disconnect(self.client)

    def test_pose(self):
        for _ in range(10):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            yaw = np.random.uniform(-np.pi, np.pi)
            self.r2d2.set_xy_yaw(x, y, yaw)
            (robot_x, robot_y), orientation = self.r2d2.get_xy_yaw()
            self.assertAlmostEqual(x, robot_x)
            self.assertAlmostEqual(y, robot_y)
            self.assertAlmostEqual(yaw, orientation)

    def test_fallen(self):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        yaw = np.random.uniform(-np.pi, np.pi)
        # standing pose
        self.r2d2.set_pose([x, y, self.r2d2.z], [0, 0, yaw])
        self.assertFalse(self.r2d2.is_fallen())
        rad = np.pi / 2.5
        z = self.r2d2.z * np.cos(rad)
        # falling forward by 72 degree
        self.r2d2.set_pose([x, y, z], [0, rad, yaw])
        self.assertTrue(self.r2d2.is_fallen(), 'z={}'.format(z))
        # falling side way by 72 degree
        self.r2d2.set_pose([x, y, z], [rad, 0, yaw])
        self.assertTrue(self.r2d2.is_fallen(), 'z={}'.format(z))


if __name__ == '__main__':
    unittest.main()
