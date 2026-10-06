import numpy as np
from sensor_msgs.msg import LaserScan

from martha_nav.robots import ROBOTS
from martha_nav.ros.scan_adapter import scan_to_arrays
from martha_nav.sim2d.geometry import draw_box, empty_grid, raycast
from martha_nav.sim2d.observation import build_observation

LIDAR_MAX = ROBOTS['martha'].lidar_range


def make_scan(ranges, angle_min=-np.pi, angle_max=np.pi):
    msg = LaserScan()
    msg.angle_min = float(angle_min)
    msg.angle_max = float(angle_max)
    msg.angle_increment = float((angle_max - angle_min) / len(ranges))
    msg.range_min, msg.range_max = 0.15, LIDAR_MAX
    msg.ranges = [float(r) for r in ranges]
    return msg


def test_angles_follow_the_message_fields():
    msg = make_scan([1.0, 2.0, 3.0, 4.0])
    ranges, angles = scan_to_arrays(msg)
    assert np.allclose(ranges, [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(angles, msg.angle_min + np.arange(4) * msg.angle_increment)


def test_lidar_yaw_rotates_the_angles():
    msg = make_scan([1.0, 2.0])
    _, angles = scan_to_arrays(msg, lidar_yaw=np.pi / 2)
    assert np.allclose(angles, [-np.pi / 2, np.pi / 2])


def test_golden_same_observation_from_a_laserscan_and_from_sim_arrays():
    """The ROS path and the 2D simulator must produce the identical vector."""
    grid = empty_grid(10.0, 10.0)
    draw_box(grid, 7.0, 5.0, 0.4, 0.4)
    draw_box(grid, 5.0, 8.0, 3.0, 0.2)
    sim_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    sim_ranges = raycast(grid, 5.0, 5.0, sim_angles, LIDAR_MAX)
    sim_obs = build_observation(sim_ranges, sim_angles, (0.2, -0.1), (1.4, 0.3), (0.5, 0.0))

    msg = make_scan(np.where(sim_ranges >= LIDAR_MAX, np.inf, sim_ranges))
    ros_ranges, ros_angles = scan_to_arrays(msg)
    ros_obs = build_observation(ros_ranges, ros_angles, (0.2, -0.1), (1.4, 0.3), (0.5, 0.0))

    # LaserScan stores ranges as float32, so the two paths differ only by that
    # rounding (measured: 6e-8 on 10 of the 96 values).
    assert np.allclose(sim_obs, ros_obs, atol=1e-6)
    assert np.array_equal(sim_obs[90:], ros_obs[90:])     # everything but the LiDAR is exact
