"""The navigation: the global planner (A* on /map) and the PPO local planner.

On top of a running sim.launch.py (sim:=true) or real.launch.py with a map:
    ros2 launch martha_nav policy.launch.py checkpoint:=/abs/policies/martha.npz [sim:=true]

The planners take the robot the checkpoint was trained for, martha or martha_tower; the robot
in Gazebo or on the floor has to be that one (sim.sh policy checks it). In Gazebo they run on
sim time and read the LiDAR plugin's topic and, with drive:=mecanum, the controller's odometry.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from martha_nav.robots import checkpoint_robot

# The sensor plugins keep their default topics so the URDF has no ":=" in it
# (gazebo_ros2_control cannot parse a robot_description containing one).
SCAN_TOPIC = '/gazebo_ros_lidar/out'
# With the mecanum drive the odometry comes from the controller, not from planar_move.
MECANUM_ODOM = '/mecanum_drive_controller/odometry'

ARGUMENTS = [
    DeclareLaunchArgument('checkpoint', default_value='',
                          description='policy: an exported .npz or a best_model.zip'),
    DeclareLaunchArgument('sim', default_value='false', choices=['true', 'false']),
    DeclareLaunchArgument('drive', default_value='mecanum', choices=['mecanum', 'planar'],
                          description="Gazebo's drive, with sim:=true"),
    DeclareLaunchArgument('speed_scale', default_value='1.0',
                          description='fraction of the trained speeds to command, in (0, 1]'),
    DeclareLaunchArgument('inflation', default_value='',
                          description="global planner inflation in m; empty: the robot's"),
    DeclareLaunchArgument('action_delay', default_value='0',
                          description='control periods between an action and its command'),
    DeclareLaunchArgument('guard', default_value='true', choices=['true', 'false'],
                          description='directional safety guard on the policy commands'),
]


def launch_setup(context, *args, **kwargs):
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    if not checkpoint:
        raise RuntimeError('checkpoint:= is required')
    robot = checkpoint_robot(checkpoint)
    sim = LaunchConfiguration('sim').perform(context) == 'true'
    inflation = LaunchConfiguration('inflation').perform(context)
    remappings = []
    if sim:
        remappings.append(('/scan', SCAN_TOPIC))
        if LaunchConfiguration('drive').perform(context) == 'mecanum':
            remappings.append(('/odom', MECANUM_ODOM))
    return [
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[{'use_sim_time': sim, 'robot': robot,
                          **({'inflation': float(inflation)} if inflation else {})}]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             remappings=remappings,
             parameters=[{'checkpoint': checkpoint, 'use_sim_time': sim, 'robot': robot,
                          'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'),
                                                        value_type=float),
                          'action_delay': ParameterValue(LaunchConfiguration('action_delay'),
                                                         value_type=int),
                          'guard': ParameterValue(LaunchConfiguration('guard'),
                                                  value_type=bool)}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
