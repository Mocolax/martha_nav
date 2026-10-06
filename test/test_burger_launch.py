"""burger.launch.py builds its nodes: mapping without a map, navigation with one."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch_ros.actions import Node

LAUNCH = Path(__file__).resolve().parents[1] / 'launch' / 'burger.launch.py'


def actions(**args):
    spec = importlib.util.spec_from_file_location('burger_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update({'map': '', 'checkpoint': '', 'speed_scale': '1.0', **args})
    return module.launch_setup(context)


def executables(nodes):
    return sorted(n.node_executable for n in nodes if isinstance(n, Node))


def test_without_a_map_it_only_maps():
    assert executables(actions()) == ['sync_slam_toolbox_node']


def test_with_a_map_it_navigates_as_the_burger():
    nodes = actions(map='/maps/sala', checkpoint='/policies/burger.npz')
    assert executables(nodes) == ['global_planner', 'localization_slam_toolbox_node',
                                  'ppo_local_planner', 'world_map_publisher']
