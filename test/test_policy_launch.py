"""policy.launch.py: the planners take the robot the checkpoint was trained for."""
from launch_nodes import built

PLANNERS = ('global_planner', 'ppo_local_planner')


def test_both_planners_are_pinned_to_martha():
    nodes = built('policy', {'checkpoint': '/policies/best_model.zip'})
    assert sorted(nodes) == sorted(PLANNERS)
    assert all(nodes[name]['robot'] == 'martha' for name in PLANNERS)
    assert nodes['ppo_local_planner']['checkpoint'] == '/policies/best_model.zip'


def test_the_planners_follow_a_martha_tower_checkpoint(tmp_path):
    import json
    import numpy as np
    policy = tmp_path / 'martha_tower.npz'
    np.savez(policy, settings=json.dumps({'robot': 'martha_tower'}))
    nodes = built('policy', {'checkpoint': str(policy)})
    assert all(nodes[name]['robot'] == 'martha_tower' for name in PLANNERS)
