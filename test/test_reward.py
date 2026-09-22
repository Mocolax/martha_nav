from martha_nav.sim2d.reward import RewardConfig, compute_reward


def test_default_terms():
    total, t = compute_reward(0.1, False, False, 2.0, 0.0)
    assert t['progress'] == 0.1 and t['step'] == -0.005
    assert t['proximity'] == 0.0 and t['turn'] == 0.0
    assert abs(total - 0.095) < 1e-12


def test_terminal_values():
    _, t = compute_reward(0.0, True, False, 2.0, 0.0)
    assert t['goal'] == 20.0
    _, t = compute_reward(0.0, False, True, 0.1, 0.0)
    assert t['collision'] == -10.0


def test_negative_progress_is_not_paid():
    _, t = compute_reward(-0.5, False, False, 2.0, 0.0)
    assert t['progress'] == 0.0


def test_optional_terms_when_enabled():
    cfg = RewardConfig(proximity=0.1, turn=0.02)
    _, t = compute_reward(0.0, False, False, 0.25, 1.0, cfg)
    assert abs(t['proximity'] + 0.05) < 1e-12      # half of proximity_dist
    assert abs(t['turn'] + 0.02) < 1e-12
