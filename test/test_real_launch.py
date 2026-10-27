"""real.launch.py runs Martha: the planners take the robot the checkpoint was trained for."""
from launch_nodes import built

PLANNERS = ('global_planner', 'ppo_local_planner')


def test_both_planners_are_pinned_to_martha():
    nodes = built('real', {'map': '/maps/lab', 'checkpoint': '/policies/best_model.zip',
                           'esp32_port': '/dev/ttyUSB0', 'lidar_port': '/dev/ttyUSB1',
                           'rviz': 'false'}, only=PLANNERS)
    assert sorted(nodes) == sorted(PLANNERS)
    assert all(nodes[name]['robot'] == 'martha' for name in PLANNERS)
    assert nodes['ppo_local_planner']['checkpoint'] == '/policies/best_model.zip'


def test_the_planners_follow_a_martha_tower_checkpoint(tmp_path):
    import json
    import numpy as np
    args = {'map': '/maps/lab', 'esp32_port': '/dev/ttyUSB0', 'lidar_port': '/dev/ttyUSB1',
            'rviz': 'false'}
    policy = tmp_path / 'martha_tower.npz'
    np.savez(policy, settings=json.dumps({'robot': 'martha_tower'}))
    nodes = built('real', {**args, 'checkpoint': str(policy)}, only=PLANNERS)
    assert all(nodes[name]['robot'] == 'martha_tower' for name in PLANNERS)
