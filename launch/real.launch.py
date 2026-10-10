"""Martha on the real robot: ESP32, RPLIDAR, EKF and slam_toolbox.

Map a place, driving with teleop_twist_keyboard in another terminal, then save it:
    ros2 launch martha_nav real.launch.py esp32_port:=/dev/serial/by-id/... lidar_port:=/dev/serial/by-id/...
    tools/save_map.sh /abs/path/martha_nav/maps/lab_real
Navigate in it, starting on the spot where the mapping began (or use 2D Pose Estimate):
    ros2 launch martha_nav real.launch.py map:=/abs/path/martha_nav/maps/lab_real checkpoint:=/abs/best_model.zip
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from martha_nav.robots import checkpoint_robot
from martha_nav.ros.slam import slam_toolbox

ARGUMENTS = [
    DeclareLaunchArgument('map', default_value='',
                          description='saved map without extension; empty maps the place'),
    DeclareLaunchArgument('checkpoint', default_value=''),
    DeclareLaunchArgument('esp32_port', default_value='/dev/ttyUSB0'),
    DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB1'),
    DeclareLaunchArgument('rviz', default_value='false'),
]


def launch_setup(context, *args, **kwargs):
    share = Path(FindPackageShare('martha_nav').perform(context))
    saved_map = LaunchConfiguration('map').perform(context)
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    # The planners take martha_tower when the checkpoint was trained for it and martha
    # otherwise, so a policy for another robot (the Burger) is still refused. The model
    # always carries the tower, which the real robot has.
    trained = checkpoint_robot(checkpoint) if checkpoint else 'martha'
    robot = trained if trained in ('martha', 'martha_tower') else 'martha'
    urdf = ParameterValue(Command(['xacro ', str(share / 'urdf' / 'martha.urdf.xacro'),
                                   ' drive:=planar tower:=true']), value_type=str)
    actions = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': urdf}]),
        Node(package='martha_nav', executable='esp32_bridge', output='screen',
             parameters=[{'port': LaunchConfiguration('esp32_port')}]),
        Node(package='rplidar_ros', executable='rplidar_node', output='screen',
             parameters=[str(share / 'config' / 'rplidar.yaml'),
                         {'serial_port': LaunchConfiguration('lidar_port')}]),
        Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
             output='screen', parameters=[str(share / 'config' / 'ekf.yaml')],
             remappings=[('odometry/filtered', '/odom')]),
        Node(package='rviz2', executable='rviz2', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', str(share / 'rviz' / 'nav.rviz')],
             remappings=[('/gazebo_ros_lidar/out', '/scan')]),
    ]
    if not saved_map:
        return actions + [slam_toolbox('mapping', '/scan', use_sim_time=False)]
    return actions + [
        slam_toolbox('localization', '/scan', use_sim_time=False, map_file=saved_map),
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'map_yaml': saved_map + '.yaml'}]),
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[{'robot': robot}]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'robot': robot}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
