"""The TurtleBot3 Burger's navigation: on its Raspberry Pi for the demo, or on the PC to debug.

The robot's own bringup runs apart, on the Pi: ros2 launch turtlebot3_bringup robot.launch.py
Map a place (drive it with teleop_twist_keyboard), then save it with tools/save_map.sh:
    ros2 launch martha_nav burger.launch.py
Navigate in it, starting where the mapping began, at x:=/y:=/yaw:= in the map (or use 2D Pose
Estimate in RViz):
    ros2 launch martha_nav burger.launch.py map:=/abs/maps/sala checkpoint:=/abs/policy.npz \
        [speed_scale:=0.5]
The real Burger plans with 0.15 m of inflation and reaches a goal at 0.20 m, as in its first runs
in a 3.7 x 3.4 m room, where the profile's 0.40 m left almost no free cells; the 2D and Gazebo
evaluations keep the profile's 0.40 m. In the 2D simulator 0.20 m of goal tolerance scored as
the trained 0.30 m, and 0.15 m within a point.
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
    DeclareLaunchArgument('speed_scale', default_value='1.0',
                          description='fraction of the trained speeds to command, in (0, 1]'),
    DeclareLaunchArgument('goal_tolerance', default_value='0.20',
                          description='m from the goal that counts as reached (trained with 0.3)'),
    DeclareLaunchArgument('guard', default_value='true',
                          description='false lets the policy drive into what enters the footprint'),
    DeclareLaunchArgument('x', default_value='0.0', description='start pose in the map, m'),
    DeclareLaunchArgument('y', default_value='0.0', description='start pose in the map, m'),
    DeclareLaunchArgument('yaw', default_value='0.0', description='start heading in the map, rad'),
    DeclareLaunchArgument('inflation', default_value='0.15',
                          description="global planner's obstacle inflation in m (not part of the "
                                      'policy: no retraining)'),
]


def launch_setup(context, *args, **kwargs):
    reach = ROBOTS['burger'].lidar_range
    saved_map = LaunchConfiguration('map').perform(context)
    start = [float(LaunchConfiguration(k).perform(context)) for k in ('x', 'y', 'yaw')]
    planner = {'robot': 'burger', **{k: float(LaunchConfiguration(k).perform(context))
                                     for k in ('inflation', 'goal_tolerance')}}
    if not saved_map:
        return [slam_toolbox('mapping', '/scan', use_sim_time=False, max_range=reach)]
    return [
        slam_toolbox('localization', '/scan', use_sim_time=False, map_file=saved_map,
                     start_pose=start, max_range=reach),
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'map_yaml': saved_map + '.yaml'}]),
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[planner]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'robot': 'burger',
                          'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'),
                                                        value_type=float),
                          'guard': ParameterValue(LaunchConfiguration('guard'), value_type=bool)}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
