"""sensor_msgs/LaserScan -> the arrays build_observation expects."""
import numpy as np


def scan_to_arrays(msg, lidar_yaw=0.0):
    """Ranges and beam angles relative to the robot's forward axis.

    lidar_yaw is the LiDAR frame's yaw within base_link (0.0 when it points
    forward, pi when the sensor is mounted backwards). Invalid readings are left
    untouched: build_observation already treats inf/NaN/<=0 as max range.
    """
    ranges = np.asarray(msg.ranges, dtype=float)
    angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment + lidar_yaw
    return ranges, angles
