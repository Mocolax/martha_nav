import numpy as np

from martha_nav.sim2d.dynamics import Dynamics, DynamicsParams, DynamicsRanges, sample_params

never = lambda x, y, t: False  # noqa: E731


def test_command_acts_one_step_later():
    d = Dynamics(DynamicsParams())
    d.step(0.3, 0.0, never)
    assert d.v == 0.0
    d.step(0.3, 0.0, never)
    assert d.v > 0.0


def test_converges_to_gain_times_command():
    d = Dynamics(DynamicsParams(tau_v=0.3, gain_v=1.1))
    for _ in range(50):
        d.step(0.3, 0.0, never)
    assert abs(d.v - 0.33) < 1e-3


def test_acceleration_is_limited():
    d = Dynamics(DynamicsParams(tau_v=0.01, acc_v=0.5))
    d.step(0.35, 0.0, never)
    d.step(0.35, 0.0, never)
    assert d.v <= 0.5 * 0.1 + 1e-9


def test_straight_line_and_turning():
    d = Dynamics(DynamicsParams(tau_v=0.01, tau_w=0.01, acc_v=100, acc_w=100))
    for _ in range(11):
        d.step(0.2, 0.0, never)
    assert abs(d.pose[0] - 0.2) < 0.01 and abs(d.pose[1]) < 1e-9
    d.reset(np.zeros(3))
    for _ in range(11):
        d.step(0.0, 0.5, never)
    assert abs(d.pose[2] - 0.5) < 0.01


def test_collision_keeps_last_free_pose_and_stops():
    d = Dynamics(DynamicsParams(tau_v=0.01, acc_v=100))
    wall = lambda x, y, t: x > 0.05  # noqa: E731
    hit = False
    for _ in range(5):
        hit = d.step(0.35, 0.0, wall) or hit
    assert hit and d.pose[0] <= 0.05 and d.v == 0.0


def test_sampled_params_stay_in_ranges():
    r = DynamicsRanges()
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = sample_params(rng, r)
        assert r.tau[0] <= p.tau_v <= r.tau[1] and r.acc_w[0] <= p.acc_w <= r.acc_w[1]
        assert r.gain[0] <= p.gain_v <= r.gain[1]
