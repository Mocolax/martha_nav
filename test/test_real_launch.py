"""real.launch.py runs Martha: a policy trained for another robot must be refused."""
from launch_nodes import built

PLANNERS = ('global_planner', 'ppo_local_planner')


def test_both_planners_are_pinned_to_martha():
    nodes = built('real', {'map': '/maps/lab', 'checkpoint': '/policies/best_model.zip',
                           'esp32_port': '/dev/ttyUSB0', 'lidar_port': '/dev/ttyUSB1',
                           'rviz': 'false'}, only=PLANNERS)
    assert sorted(nodes) == sorted(PLANNERS)
    assert all(nodes[name]['robot'] == 'martha' for name in PLANNERS)
    assert nodes['ppo_local_planner']['checkpoint'] == '/policies/best_model.zip'
