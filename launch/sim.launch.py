"""Gazebo + Martha + map + ground-truth TF + the two navigation nodes.

ros2 launch martha_nav sim.launch.py checkpoint:=/abs/path/best_model.zip

drive:=mecanum (default) simulates the wheels and rollers through ros2_control;
drive:=planar uses gazebo_ros_planar_move, which is much faster but has no wheel
dynamics. sim_speed_factor and physics_step_size rewrite the world's physics.

Localization is Gazebo's true pose unless slam:=localization slam_map:=/abs/maps/lab,
where slam_toolbox localizes in a map made with slam:=mapping (tools/map_world.py).
Map it with the robot at the origin (x:=0 y:=0) so its map frame is Gazebo's.
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
    DeclareLaunchArgument('slam', default_value='off', choices=['off', 'mapping', 'localization']),
    DeclareLaunchArgument('action_delay', default_value='0',
                          description='control periods ppo_local_planner holds each command back'),
    DeclareLaunchArgument('slam_map', default_value='',
                          description='slam_toolbox map without extension, for slam:=localization'),
]


def launch_setup(context, *args, **kwargs):
    share = Path(FindPackageShare('martha_nav').perform(context))
    drive = LaunchConfiguration('drive').perform(context)
    slam = LaunchConfiguration('slam').perform(context)
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
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'world': LaunchConfiguration('world'), 'use_sim_time': True}]),
        Node(package='martha_nav', executable='gazebo_ground_truth_tf', output='screen',
             remappings=odom_remap,
             parameters=[{'use_sim_time': True, 'publish_map_odom': slam != 'localization',
                          'odom_tf_topic': MECANUM_ODOM_TF if drive == 'mecanum' else ''}]),
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[{'use_sim_time': True}]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             remappings=[('/scan', SCAN_TOPIC)] + odom_remap,
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'use_sim_time': True,
                          'action_delay': ParameterValue(LaunchConfiguration('action_delay'),
                                                         value_type=int)}]),
        Node(package='rviz2', executable='rviz2', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', str(share / 'rviz' / 'nav.rviz')],
             parameters=[{'use_sim_time': True}]),
        RegisterEventHandler(OnShutdown(on_shutdown=lambda *_, **__: Path(world).unlink(
            missing_ok=True))),
    ]

    if slam != 'off':
        actions.append(slam_toolbox(context, slam))
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


def slam_toolbox(context, mode):
    """slam_toolbox with its own tuned parameters, adapted to Martha. When mapping it
    publishes no TF (the robot drives on the true pose); its map goes to /slam_map so it
    does not replace the world's /map, which the planner uses."""
    config = Path(FindPackageShare('slam_toolbox').perform(context)) / 'config'
    params = {'use_sim_time': True, 'mode': mode, 'base_frame': 'base_link',
              'scan_topic': SCAN_TOPIC, 'max_laser_range': 8.0}
    if mode == 'mapping':
        executable, defaults = 'sync_slam_toolbox_node', 'mapper_params_online_sync.yaml'
        params['transform_publish_period'] = 0.0
    else:
        executable, defaults = 'localization_slam_toolbox_node', 'mapper_params_localization.yaml'
        params['map_file_name'] = LaunchConfiguration('slam_map').perform(context)
        params['map_start_pose'] = [float(LaunchConfiguration('x').perform(context)),
                                    float(LaunchConfiguration('y').perform(context)), 0.0]
    return Node(package='slam_toolbox', executable=executable, name='slam_toolbox',
                output='screen', parameters=[str(config / defaults), params],
                remappings=[('/map', '/slam_map'), ('/map_metadata', '/slam_map_metadata')])


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
