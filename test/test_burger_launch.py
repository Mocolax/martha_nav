"""burger.launch.py builds its nodes: mapping without a map, navigation with one."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

from martha_nav.robots import ROBOTS

LAUNCH = Path(__file__).resolve().parents[1] / 'launch' / 'burger.launch.py'


def built(**args):
    """{executable: its parameters} of the nodes burger.launch.py builds with these arguments."""
    spec = importlib.util.spec_from_file_location('burger_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update({'map': '', 'checkpoint': '', 'speed_scale': '1.0', **args})
    nodes = [n for n in module.launch_setup(context) if isinstance(n, Node)]
    return {n.node_executable: {k: v for part in evaluate_parameters(context, n._Node__parameters)
                                if isinstance(part, dict) for k, v in part.items()}
            for n in nodes}


def test_without_a_map_it_only_maps():
    assert sorted(built()) == ['sync_slam_toolbox_node']


def test_with_a_map_it_navigates_as_the_burger():
    nodes = built(map='/maps/sala', checkpoint='/policies/burger.npz')
    assert sorted(nodes) == ['global_planner', 'localization_slam_toolbox_node',
                             'ppo_local_planner', 'world_map_publisher']


def test_slam_toolbox_reaches_as_far_as_the_burgers_lidar():
    reach = ROBOTS['burger'].lidar_range
    mapping = built()['sync_slam_toolbox_node']
    localizing = built(map='/maps/sala')['localization_slam_toolbox_node']
    assert mapping['max_laser_range'] == localizing['max_laser_range'] == reach
    assert mapping['mode'] == 'mapping' and localizing['mode'] == 'localization'
    assert localizing['map_file_name'] == '/maps/sala'


def test_both_planners_run_as_the_burger_with_the_given_policy_and_speed():
    nodes = built(map='/maps/sala', checkpoint='/policies/burger.npz', speed_scale='0.5')
    assert nodes['global_planner']['robot'] == 'burger'
    assert nodes['ppo_local_planner'] == {'checkpoint': '/policies/burger.npz', 'robot': 'burger',
                                          'speed_scale': 0.5}
    assert nodes['world_map_publisher'] == {'map_yaml': '/maps/sala.yaml'}
