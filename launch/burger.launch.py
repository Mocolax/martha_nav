"""The TurtleBot3 Burger's navigation: on its Raspberry Pi for the demo, or on the PC to debug.

The robot's own bringup runs apart, on the Pi: ros2 launch turtlebot3_bringup robot.launch.py
Map a place (drive it with teleop_twist_keyboard), then save it with tools/save_map.sh:
    ros2 launch martha_nav burger.launch.py
Navigate in it, starting where the mapping began (or use 2D Pose Estimate in RViz):
    ros2 launch martha_nav burger.launch.py map:=/abs/maps/sala checkpoint:=/abs/policy.npz [speed_scale:=0.5]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from martha_nav.robots import ROBOTS
from martha_nav.ros.slam import slam_toolbox

ARGUMENTS = [
    DeclareLaunchArgument('map', default_value='',
                          description='saved map without extension; empty maps the place'),
    DeclareLaunchArgument('checkpoint', default_value='',
                          description='policy.npz from export_policy (or a .zip, with PyTorch)'),
    DeclareLaunchArgument('speed_scale', default_value='1.0'),
]


def launch_setup(context, *args, **kwargs):
    reach = ROBOTS['burger'].lidar_range
    saved_map = LaunchConfiguration('map').perform(context)
    if not saved_map:
        return [slam_toolbox('mapping', '/scan', use_sim_time=False, max_range=reach)]
    return [
        slam_toolbox('localization', '/scan', use_sim_time=False, map_file=saved_map,
                     max_range=reach),
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'map_yaml': saved_map + '.yaml'}]),
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[{'robot': 'burger'}]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'robot': 'burger',
                          'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'),
                                                        value_type=float)}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
