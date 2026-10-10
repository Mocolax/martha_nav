"""burger.launch.py builds its nodes: mapping without a map, navigation with one."""
from launch_nodes import built as built_nodes

from martha_nav.robots import ROBOTS


def built(**args):
    return built_nodes('burger', args)


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
                                          'speed_scale': 0.5, 'guard': True}
    assert nodes['world_map_publisher'] == {'map_yaml': '/maps/sala.yaml'}


def test_the_real_burger_plans_with_its_own_inflation_and_goal_tolerance():
    """Its room left no free cells at the profile's 0.40 m; Martha's launches keep the profile."""
    planner = built(map='/maps/sala')['global_planner']
    assert planner == {'robot': 'burger', 'inflation': 0.15, 'goal_tolerance': 0.20}
    planner = built(map='/maps/sala', inflation='0.40', goal_tolerance='0.3')['global_planner']
    assert planner == {'robot': 'burger', 'inflation': 0.40, 'goal_tolerance': 0.3}


def test_localization_starts_at_the_given_pose():
    slam = built(map='/maps/sala')['localization_slam_toolbox_node']
    assert slam['map_start_pose'] == (0.0, 0.0, 0.0)
    slam = built(map='/maps/sala', x='0.5', y='-0.3', yaw='0.93')['localization_slam_toolbox_node']
    assert slam['map_start_pose'] == (0.5, -0.3, 0.93)


def test_the_footprint_guard_can_be_turned_off():
    assert built(map='/maps/sala', guard='false')['ppo_local_planner']['guard'] is False
