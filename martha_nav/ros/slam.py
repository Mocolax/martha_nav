"""slam_toolbox as the launch files run it: its packaged parameters plus Martha's.

Its map goes to /slam_map, so it never replaces the /map the global planner uses.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def slam_toolbox(mode, scan_topic, use_sim_time, publish_tf=True, map_file='',
                 start_pose=(0.0, 0.0, 0.0), max_range=8.0):
    """mode 'mapping' or 'localization'; map_file is the saved posegraph without extension;
    max_range is the LiDAR's (ROBOTS[...].lidar_range)."""
    config = Path(get_package_share_directory('slam_toolbox')) / 'config'
    params = {'use_sim_time': use_sim_time, 'mode': mode, 'base_frame': 'base_link',
              'scan_topic': scan_topic, 'max_laser_range': max_range}
    if mode == 'mapping':
        executable, defaults = 'sync_slam_toolbox_node', 'mapper_params_online_sync.yaml'
        if not publish_tf:
            params['transform_publish_period'] = 0.0
    else:
        executable, defaults = 'localization_slam_toolbox_node', 'mapper_params_localization.yaml'
        params['map_file_name'] = map_file
        params['map_start_pose'] = [float(v) for v in start_pose]
    return Node(package='slam_toolbox', executable=executable, name='slam_toolbox',
                output='screen', parameters=[str(config / defaults), params],
                remappings=[('/map', '/slam_map'), ('/map_metadata', '/slam_map_metadata')])
