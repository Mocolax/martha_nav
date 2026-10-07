"""burger.launch.py builds its nodes: mapping without a map, navigation with one."""
from launch_nodes import built as built_nodes

from martha_nav.robots import ROBOTS


def built(**args):
    return built_nodes('burger', {'map': '', 'checkpoint': '', 'speed_scale': '1.0', **args})


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
