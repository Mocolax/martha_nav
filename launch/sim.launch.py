"""Gazebo + Martha + map + ground-truth TF; with a checkpoint, policy.launch.py on top.

ros2 launch martha_nav sim.launch.py [world:=lab]               then policy.launch.py sim:=true
ros2 launch martha_nav sim.launch.py checkpoint:=/abs/path/best_model.zip     both at once

robot:=martha simulates Martha without the aluminium tower; by default the robot is the one the
checkpoint was trained for, or martha_tower (Martha as built) without one.

For Martha, drive:=mecanum (default) simulates the wheels and rollers through ros2_control;
drive:=planar uses gazebo_ros_planar_move, which is much faster but has no wheel
dynamics. sim_speed_factor and physics_step_size rewrite the
world's physics. Localization is Gazebo's true pose.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from martha_nav.robots import ROBOTS, checkpoint_robot
from martha_nav.ros.world_speed import create_scaled_world

# Topic names as in policy.launch.py.
MECANUM_ODOM = '/mecanum_drive_controller/odometry'
MECANUM_ODOM_TF = '/mecanum_drive_controller/tf_odometry'

ARGUMENTS = [
    DeclareLaunchArgument('world', default_value='lab'),
    DeclareLaunchArgument('drive', default_value='mecanum', choices=['mecanum', 'planar']),
    DeclareLaunchArgument('robot', default_value='',
                          description="martha or martha_tower; empty: the checkpoint's"
                                      ' or martha_tower'),
    DeclareLaunchArgument('gui', default_value='true'),
    DeclareLaunchArgument('rviz', default_value='false'),
    DeclareLaunchArgument('checkpoint', default_value=''),
    DeclareLaunchArgument('x', default_value='0.0'),
    DeclareLaunchArgument('y', default_value='0.0'),
    DeclareLaunchArgument('lidar_samples', default_value='360'),
    DeclareLaunchArgument('sim_speed_factor', default_value='1.0'),
    DeclareLaunchArgument('physics_step_size', default_value='0.001'),
    DeclareLaunchArgument('action_delay', default_value='0',
                          description='control periods ppo_local_planner holds each command back'),
    DeclareLaunchArgument('guard', default_value='true', choices=['true', 'false'],
                          description="ppo_local_planner's directional safety guard"),
]


def launch_setup(context, *args, **kwargs):
    share = Path(FindPackageShare('martha_nav').perform(context))
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    trained = checkpoint_robot(checkpoint) if checkpoint else 'martha_tower'
    name = LaunchConfiguration('robot').perform(context) or trained
    if checkpoint and name != trained:
        raise RuntimeError(f'{checkpoint} drives {trained}, not {name}')
    robot = ROBOTS[name]
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
        f" tower:={'true' if robot.posts else 'false'}",
    ]), value_type=str)

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-topic', 'robot_description', '-entity', 'robot',
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
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'world': LaunchConfiguration('world'), 'use_sim_time': True}]),
        Node(package='martha_nav', executable='gazebo_ground_truth_tf', output='screen',
             remappings=odom_remap,
             parameters=[{'use_sim_time': True,
                          'odom_tf_topic': MECANUM_ODOM_TF if drive == 'mecanum' else ''}]),
        Node(package='rviz2', executable='rviz2', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', str(share / 'rviz' / 'nav.rviz')],
             parameters=[{'use_sim_time': True}]),
        RegisterEventHandler(OnShutdown(on_shutdown=lambda *_, **__: Path(world).unlink(
            missing_ok=True))),
    ]

    if checkpoint:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch' / 'policy.launch.py')),
            launch_arguments={'checkpoint': checkpoint, 'sim': 'true', 'drive': drive,
                              'action_delay': LaunchConfiguration('action_delay'),
                              'guard': LaunchConfiguration('guard')}.items()))
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
            Node(package='martha_nav', executable='mecanum_cmd_vel_bridge', output='screen',
                 parameters=[{'use_sim_time': True}]),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
