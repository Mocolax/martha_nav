from dataclasses import replace

from martha_nav.sim2d.reward import RewardConfig, compute_reward


def test_default_terms():
    total, t = compute_reward(0.1, False, False)
    assert t == {'progress': 0.1, 'goal': 0.0, 'collision': 0.0, 'step': -0.005}
    assert abs(total - 0.095) < 1e-12


def test_terminal_values():
    assert compute_reward(0.0, True, False)[1]['goal'] == 20.0
    assert compute_reward(0.0, False, True)[1]['collision'] == -20.0


def test_negative_progress_is_not_paid():
    assert compute_reward(-0.5, False, False)[1]['progress'] == 0.0


def test_geodesic_progress_can_be_negative_and_route_progress_cannot():
    geo = replace(RewardConfig(), progress_mode='geodesic')
    assert compute_reward(-0.3, False, False, geo)[1]['progress'] == -0.3
    assert compute_reward(-0.3, False, False, RewardConfig())[1]['progress'] == 0.0
