import math
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


def test_clearance_disabled_by_default():
    total, terms = compute_reward(0.1, False, False, clearance=0.0)
    assert 'clearance' not in terms
    assert abs(total - 0.095) < 1e-12


def test_clearance_penalty():
    cfg = RewardConfig(clearance=0.05)
    mid = compute_reward(0.0, False, False, cfg, clearance=cfg.clearance_mid)[1]
    near = compute_reward(0.0, False, False, cfg, clearance=0.0)[1]
    far = compute_reward(0.0, False, False, cfg, clearance=1.0)[1]
    assert abs(mid['clearance'] + 0.025) < 1e-9
    assert near['clearance'] < -0.04
    assert far['clearance'] > -0.001


def test_infinite_clearance_has_no_penalty():
    cfg = RewardConfig(clearance=0.05)
    assert compute_reward(0.0, False, False, cfg, clearance=math.inf)[1]['clearance'] == 0.0


def test_reverse_penalty_per_metre():
    assert 'reverse' not in compute_reward(0.0, False, False, backward=0.2)[1]
    cfg = RewardConfig(reverse=1.0)
    assert abs(compute_reward(0.0, False, False, cfg, backward=0.2)[1]['reverse'] + 0.2) < 1e-12
