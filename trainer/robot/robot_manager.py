"""Robot manager.

This file implements a robot manager who is responsible for
creating robot and place robot according to a specified pose.

"""
import hsr
import r2d2


class RobotManager(object):
    """Robot manager."""

    @classmethod
    def create_robot(cls, robot_type, x, y, yaw):
        """Create a robot according to the type.

        Load URDF file and return the id of a new robot.

        Args:
            robot_type: RobotType
            x: float, x position
            y: float, y position
            yaw: float, yaw
        Returns:
            HumanFollowingRobot object.
        Raises:
            ValueError, raised when robot_type is unknown.

        """
        if robot_type == 'HSR':
            robot = hsr.HSR(x, y, yaw)
        elif robot_type == 'R2D2':
            robot = r2d2.R2D2(x, y, yaw)
        else:
            raise ValueError('Unknown robot type: {}'.format(robot_type))
        return robot
