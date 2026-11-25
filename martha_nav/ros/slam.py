"""slam_toolbox as the launch files run it: its packaged parameters plus Martha's.

Its map goes to /slam_map, so it never replaces the /map the global planner uses.
"""
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from martha_nav.robots import ROBOTS


def own_reach(robot, margin=0.03):
    """m from the LiDAR within which it sees the robot's own posts (0 without posts).

    slam_toolbox maps every beam it is given: the tower's posts would be stamped into the map
    around every pose it keeps, as specks where there is nothing."""
    return max((np.hypot(abs(x - robot.lidar_offset_x) + sx / 2, abs(y) + sy / 2) + margin
                for x, y, sx, sy in robot.posts), default=0.0)


def slam_toolbox(mode, scan_topic, use_sim_time, map_file='', max_range=8.0,
                 robot=ROBOTS['martha_tower']):
    """mode 'mapping' or 'localization'; map_file is the saved posegraph without extension,
    whose origin is where the mapping began; max_range is the LiDAR's
    (ROBOTS[...].lidar_range); nothing nearer than the robot's posts reach is mapped."""
    config = Path(get_package_share_directory('slam_toolbox')) / 'config'
    params = {'use_sim_time': use_sim_time, 'mode': mode, 'base_frame': 'base_link',
              'scan_topic': scan_topic, 'max_laser_range': max_range,
              'min_laser_range': float(own_reach(robot))}
    if mode == 'mapping':
        executable, defaults = 'sync_slam_toolbox_node', 'mapper_params_online_sync.yaml'
    else:
        executable, defaults = 'localization_slam_toolbox_node', 'mapper_params_localization.yaml'
        params['map_file_name'] = map_file
        params['map_start_pose'] = [0.0, 0.0, 0.0]
    return Node(package='slam_toolbox', executable=executable, name='slam_toolbox',
                output='screen', parameters=[str(config / defaults), params],
                remappings=[('/map', '/slam_map'), ('/map_metadata', '/slam_map_metadata')])
