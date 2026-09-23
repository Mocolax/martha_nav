"""Gazebo + Martha + map + ground-truth TF + the two navigation nodes.

ros2 launch martha_nav sim.launch.py checkpoint:=/abs/path/best_model.zip

drive:=mecanum (default) simulates the wheels and rollers through ros2_control;
drive:=planar uses gazebo_ros_planar_move, which is much faster but has no wheel
dynamics. sim_speed_factor and physics_step_size rewrite the world's physics.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from martha_nav.ros.world_speed import create_scaled_world

# The sensor plugins keep their default topics so the URDF has no ":=" in it
# (gazebo_ros2_control cannot parse a robot_description containing one).
SCAN_TOPIC = '/gazebo_ros_lidar/out'
BUMPER_TOPIC = '/gazebo_ros_bumper/bumper_states'
# With the mecanum drive the odometry comes from the controller, not from planar_move.
MECANUM_ODOM = '/mecanum_drive_controller/odometry'
MECANUM_ODOM_TF = '/mecanum_drive_controller/tf_odometry'

ARGUMENTS = [
    DeclareLaunchArgument('world', default_value='lab'),
    DeclareLaunchArgument('drive', default_value='mecanum', choices=['mecanum', 'planar']),
    DeclareLaunchArgument('gui', default_value='true'),
    DeclareLaunchArgument('rviz', default_value='false'),
    DeclareLaunchArgument('checkpoint', default_value=''),
    DeclareLaunchArgument('x', default_value='0.0'),
    DeclareLaunchArgument('y', default_value='0.0'),
    DeclareLaunchArgument('lidar_samples', default_value='360'),
    DeclareLaunchArgument('sim_speed_factor', default_value='1.0'),
    DeclareLaunchArgument('physics_step_size', default_value='0.001'),
]


def launch_setup(context, *args, **kwargs):
    share = Path(FindPackageShare('martha_nav').perform(context))
    drive = LaunchConfiguration('drive').perform(context)
    odom_remap = [('/odom', MECANUM_ODOM)] if drive == 'mecanum' else []
    world = create_scaled_world(
        share / 'worlds' / f"{LaunchConfiguration('world').perform(context)}.world",
        speed_factor=LaunchConfiguration('sim_speed_factor').perform(context),
        physics_step_size=LaunchConfiguration('physics_step_size').perform(context))

    urdf = ParameterValue(Command([
        'xacro ', str(share / 'urdf' / 'martha.urdf.xacro'),
        ' drive:=', drive,
        ' controllers_file:=', str(share / 'config' / 'controllers.yaml'),
        ' lidar_samples:=', LaunchConfiguration('lidar_samples'),
    ]), value_type=str)

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-topic', 'robot_description', '-entity', 'martha',
                            '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                            '-z', '0.05'])

    actions = [
        ExecuteProcess(
            cmd=['gzserver', str(world), '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_state.so'],
            output='screen'),
        ExecuteProcess(cmd=['gzclient'], condition=IfCondition(LaunchConfiguration('gui')),
                       output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': urdf, 'use_sim_time': True}]),
        spawn,
        Node(package='martha_nav', executable='map_publisher', output='screen',
             parameters=[{'world': LaunchConfiguration('world'), 'use_sim_time': True}]),
        Node(package='martha_nav', executable='ground_truth_tf', output='screen',
             remappings=odom_remap,
             parameters=[{'use_sim_time': True,
                          'odom_tf_topic': MECANUM_ODOM_TF if drive == 'mecanum' else ''}]),
        Node(package='martha_nav', executable='planner_node', output='screen',
             parameters=[{'use_sim_time': True}]),
        Node(package='martha_nav', executable='policy_node', output='screen',
             remappings=[('/scan', SCAN_TOPIC)] + odom_remap,
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'use_sim_time': True}]),
        Node(package='rviz2', executable='rviz2', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             parameters=[{'use_sim_time': True}]),
        RegisterEventHandler(OnShutdown(on_shutdown=lambda *_, **__: Path(world).unlink(
            missing_ok=True))),
    ]

    if drive == 'mecanum':
        # The controller manager only exists once the robot is in Gazebo, and the
        # controllers must be spawned one after the other.
        broadcaster = Node(package='controller_manager', executable='spawner', output='screen',
                           arguments=['joint_state_broadcaster'])
        controller = Node(package='controller_manager', executable='spawner', output='screen',
                          arguments=['mecanum_drive_controller'])
        actions += [
            RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[broadcaster])),
            RegisterEventHandler(OnProcessExit(target_action=broadcaster, on_exit=[controller])),
            Node(package='martha_nav', executable='cmd_vel_bridge', output='screen',
                 parameters=[{'use_sim_time': True}]),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
