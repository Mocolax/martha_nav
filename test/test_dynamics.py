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


def test_lateral_command_moves_the_robot_sideways():
    d = Dynamics(DynamicsParams(tau_v=0.01, tau_w=0.01, acc_v=100, acc_w=100))
    for _ in range(11):
        d.step_holonomic(0.0, 0.2, 0.0, never)
    assert abs(d.pose[1] - 0.2) < 0.01 and abs(d.pose[0]) < 1e-9


def test_lateral_motion_follows_the_heading():
    d = Dynamics(DynamicsParams(tau_v=0.01, tau_w=0.01, acc_v=100, acc_w=100))
    d.reset([0.0, 0.0, np.pi / 2])
    for _ in range(11):
        d.step_holonomic(0.0, 0.2, 0.0, never)
    assert abs(d.pose[0] + 0.2) < 0.01          # +y of the robot is -x of the world


def test_delay_is_the_number_of_control_periods_before_a_command_acts():
    """Gazebo's mecanum obeys in the same period; a real robot may lag more than one."""
    for delay in (0, 1, 2):
        d = Dynamics(DynamicsParams(delay=delay))
        moved = []
        for _ in range(3):
            d.step(0.3, 0.0, never)
            moved.append(d.v > 0.0)
        assert moved.index(True) == delay


def test_default_ranges_draw_nothing_extra():
    """The delay is only drawn when it varies, so existing episodes stay identical."""
    rng, reference = np.random.default_rng(0), np.random.default_rng(0)
    p = sample_params(rng)
    reference.random(6)                                    # tau x2, acc x2, gain x2
    assert p.delay == 1 and rng.random() == reference.random()


def test_wide_ranges_cover_the_gazebo_mecanum():
    from martha_nav.sim2d.dynamics import WIDE_DYNAMICS
    rng = np.random.default_rng(0)
    samples = [sample_params(rng, WIDE_DYNAMICS) for _ in range(300)]
    assert {p.delay for p in samples} == {0, 1, 2}
    assert max(p.acc_v for p in samples) > 2.5 and max(p.acc_w for p in samples) > 6.5
    assert min(p.tau_v for p in samples) < 0.05
