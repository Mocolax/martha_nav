"""Martha on the real robot: ESP32, RPLIDAR, EKF and slam_toolbox.

Map a place, driving with an Xbox controller (joy:=true, config/xbox_teleop.yaml) or with
teleop_twist_keyboard in another terminal, then save it:
    ros2 launch martha_nav real.launch.py joy:=true
        esp32_port:=/dev/serial/by-id/... lidar_port:=/dev/serial/by-id/...
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
    DeclareLaunchArgument('speed_scale', default_value='1.0',
                          description='fraction of the trained speeds to command, in (0, 1]'),
    DeclareLaunchArgument('inflation', default_value='',
                          description="global planner's obstacle inflation in m (empty: the "
                                      "robot's); lower it when a narrow passage closes"),
    DeclareLaunchArgument('esp32_port', default_value='/dev/ttyUSB0'),
    DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB1'),
    DeclareLaunchArgument('rviz', default_value='false'),
    DeclareLaunchArgument('joy', default_value='false',
                          description='drive with an Xbox controller: sticks, RB/LB '
                                      'faster/slower, Start stops'),
]


def launch_setup(context, *args, **kwargs):
    share = Path(FindPackageShare('martha_nav').perform(context))
    saved_map = LaunchConfiguration('map').perform(context)
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    # The planners take martha_tower when the checkpoint was trained for it and martha
    # otherwise, so a policy for another robot (the Burger) is still refused. The model
    # always carries the tower, which the real robot has.
    inflation = LaunchConfiguration('inflation').perform(context)
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
             # While mapping, the map growing is slam_toolbox's /slam_map; /map is the saved one.
             remappings=[('/gazebo_ros_lidar/out', '/scan')]
             + ([] if saved_map else [('/map', '/slam_map')])),
        Node(package='joy', executable='joy_node', output='screen',
             condition=IfCondition(LaunchConfiguration('joy')),
             parameters=[str(share / 'config' / 'xbox_teleop.yaml')]),
        Node(package='martha_nav', executable='xbox_teleop', output='screen',
             condition=IfCondition(LaunchConfiguration('joy')),
             parameters=[str(share / 'config' / 'xbox_teleop.yaml')]),
    ]
    if not saved_map:
        return actions + [slam_toolbox('mapping', '/scan', use_sim_time=False)]
    return actions + [
        slam_toolbox('localization', '/scan', use_sim_time=False, map_file=saved_map),
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'map_yaml': saved_map + '.yaml'}]),
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[{'robot': robot,
                          **({'inflation': float(inflation)} if inflation else {})}]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'robot': robot,
                          'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'),
                                                        value_type=float)}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
