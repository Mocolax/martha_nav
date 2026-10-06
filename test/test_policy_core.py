import numpy as np

from martha_nav.robots import ROBOTS
from martha_nav.ros.policy_core import PolicyCore, footprint_blocked
from martha_nav.sim2d.planner import Path

LIDAR_MAX, V_MAX, W_MAX = (ROBOTS['martha'].lidar_range, ROBOTS['martha'].v_max,
                           ROBOTS['martha'].w_max)


class FakeModel:
    """Stand-in for the trained model: records the observation, returns a fixed action."""

    def __init__(self, action=(1.0, 0.0)):
        self.action = np.array(action)
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = np.asarray(obs)
        return self.action, None


def straight_path():
    return Path([[0.0, 0.0], [10.0, 0.0]])


def clear_scan(n=360):
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return np.full(n, LIDAR_MAX), angles


def test_action_is_mapped_to_velocities():
    core = PolicyCore(FakeModel((1.0, -1.0)))
    ranges, angles = clear_scan()
    v, w, _ = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert (v, w) == (V_MAX, -W_MAX)


def test_observation_matches_the_contract():
    model = FakeModel()
    core = PolicyCore(model, lookahead=1.5)
    ranges, angles = clear_scan()
    core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    obs = model.last_obs
    assert obs.shape == (96,)
    assert np.isclose(obs[90], 1.5 / 3.0)        # carrot 1.5 m ahead
    assert np.isclose(obs[91], 0.0)              # straight ahead


def test_previous_action_is_fed_back_and_cleared_on_reset():
    model = FakeModel((0.5, 0.25))
    core = PolicyCore(model)
    ranges, angles = clear_scan()
    core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    core.compute(straight_path(), (0.1, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert np.allclose(model.last_obs[94:], [0.5, 0.25])
    core.reset()
    core.compute(straight_path(), (0.2, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert np.allclose(model.last_obs[94:], [0.0, 0.0])


def test_an_obstacle_ahead_blocks_forward_motion_but_allows_turning():
    """The guard must leave a way out: freezing the robot turns a near-collision
    into an episode that never ends."""
    core = PolicyCore(FakeModel((1.0, 0.6)))      # wants to drive forward and turn
    ranges, angles = clear_scan()
    ranges[180] = 0.04                            # right in front of the LiDAR
    v, w, info = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert v == 0.0 and w != 0.0
    assert info['blocked'] == ['front']


def test_an_obstacle_ahead_still_allows_reversing():
    core = PolicyCore(FakeModel((-1.0, 0.0)))
    ranges, angles = clear_scan()
    ranges[180] = 0.04
    v, _, _ = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert v < 0.0


def test_an_obstacle_behind_blocks_only_reverse():
    core = PolicyCore(FakeModel((-1.0, 0.0)))
    ranges, angles = clear_scan()
    ranges[0] = 0.30                              # beam 0 points backwards
    v, _, info = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert v == 0.0 and info['blocked'] == ['rear']
    forward = PolicyCore(FakeModel((1.0, 0.0)))
    v, _, _ = forward.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert v > 0.0


def test_footprint_check_uses_the_lidar_offset_and_tells_the_side():
    angles = np.array([0.0, np.pi])
    # Forward the footprint ends 0.28 - 0.2325 = 0.0475 m ahead of the LiDAR.
    assert 'front' in footprint_blocked(np.array([0.04, 8.0]), angles)
    assert footprint_blocked(np.array([0.5, 8.0]), angles) == set()
    # Backwards it reaches 0.28 + 0.2325 = 0.5125 m behind it.
    assert 'rear' in footprint_blocked(np.array([8.0, 0.40]), angles)
    assert footprint_blocked(np.array([0.04, 0.40]), angles) == {'front', 'rear'}
    # A point beside the robot blocks sliding that way, not driving forward.
    beside = footprint_blocked(np.array([0.30]), np.array([3 * np.pi / 4]))
    assert beside == {'left'}


def test_the_burger_guard_reacts_to_readings_its_lidar_can_produce():
    burger = ROBOTS['burger']
    angles = np.array([0.0, np.pi])
    # LiDAR and footprint centre coincide and it reads nothing under 0.12 m: the guard reaches
    # 0.07 + 0.10 = 0.17 m to the front and to the back.
    assert footprint_blocked(np.array([0.15, 3.5]), angles, burger) == {'front'}
    assert footprint_blocked(np.array([0.20, 3.5]), angles, burger) == set()
    assert 'rear' in footprint_blocked(np.array([3.5, 0.15]), angles, burger)


def test_the_burger_scan_points_start_at_its_lidar():
    core = PolicyCore(FakeModel(), robot=ROBOTS['burger'])
    points = core._scan_points(np.array([1.0]), np.zeros(1), (0.0, 0.0, 0.0))
    assert np.allclose(points, [[1.0 - 0.032, 0.0]])
    assert core._scan_points(np.array([3.5]), np.zeros(1), (0.0, 0.0, 0.0)).shape == (0, 2)


def test_the_burger_core_commands_its_own_speeds():
    burger = ROBOTS['burger']
    core = PolicyCore(FakeModel((1.0, -1.0)), robot=burger)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    v, w, _ = core.compute(straight_path(), (0.0, 0.0, 0.0), np.full(360, 3.5), angles, (0.0, 0.0))
    assert (v, w) == (burger.v_max, -burger.w_max)


def test_the_carrot_skips_a_scanned_obstacle():
    """A box on the route pushes the carrot past it instead of into it."""
    model = FakeModel()
    core = PolicyCore(model, lookahead=1.5)
    ranges, angles = clear_scan()
    ranges[180] = 1.5                             # beam 180 points forward (angle 0)
    core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert model.last_obs[90] > 1.5 / 3.0         # carrot pushed further along


def test_holonomic_core_returns_three_velocities_and_guards_the_sides():
    V_LATERAL = ROBOTS['martha'].v_lateral
    core = PolicyCore(FakeModel((0.0, 1.0, 0.0)), action_dim=3)   # wants to slide left
    ranges, angles = clear_scan()
    vx, vy, w, info = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0, 0.0))
    assert (vx, vy, w) == (0.0, V_LATERAL, 0.0) and info['blocked'] == []
    ranges[315] = 0.30                                 # beam 315 points back-left, beside the robot
    vx, vy, w, info = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0, 0.0))
    assert vy == 0.0 and 'left' in info['blocked']


def test_goal_target_steers_to_the_end_of_the_route():
    seen = {}

    class Spy:
        def predict(self, obs, deterministic=True):
            seen['obs'] = obs
            return np.zeros(2), None

    core = PolicyCore(Spy(), target='goal')
    path = Path(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 6.0]]))
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    *_, info = core.compute(path, (0.0, 0.0, 0.0), np.full(360, 5.0), angles, (0.0, 0.0))
    assert np.allclose(info['carrot'], [1.0, 6.0])


def test_recurrent_policy_keeps_its_state_between_ticks_and_resets_on_a_new_route():
    calls = []

    class Lstm:
        policy = type('P', (), {'lstm_actor': object()})()

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            calls.append((state, bool(episode_start[0])))
            return np.zeros(2), len(calls)          # the "state" is just a counter

    core = PolicyCore(Lstm())
    path = Path(np.array([[0.0, 0.0], [5.0, 0.0]]))
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    for _ in range(3):
        core.compute(path, (0.0, 0.0, 0.0), np.full(360, 5.0), angles, (0.0, 0.0))
    core.reset()                                    # ppo_local_planner calls this when the episode ends
    core.compute(path, (0.0, 0.0, 0.0), np.full(360, 5.0), angles, (0.0, 0.0))
    assert calls == [(None, True), (1, False), (2, False), (None, True)]


class Lstm:
    """Recurrent stand-in: the returned "state" counts the calls since the last reset."""

    policy = type('P', (), {'lstm_actor': object()})()

    def __init__(self):
        self.calls = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.calls.append((state, bool(episode_start[0])))
        return np.zeros(2), len(self.calls)


def test_a_replan_to_the_same_goal_keeps_the_lstm_state():
    """The planner replans when the robot strays 1 m, typically while avoiding an
    obstacle: that is when the policy needs its memory most."""
    model = Lstm()
    core = PolicyCore(model)
    ranges, angles = clear_scan()
    first = Path([[0.0, 0.0], [5.0, 0.0]])
    replanned = Path([[0.0, 1.0], [2.0, 1.0], [5.0, 0.0]])
    core.compute(first, (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    core.compute(first, (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    core.compute(replanned, (0.0, 1.0, 0.0), ranges, angles, (0.0, 0.0))
    assert model.calls == [(None, True), (1, False), (2, False)]


def test_a_new_goal_starts_a_new_episode():
    model = Lstm()
    core = PolicyCore(model)
    ranges, angles = clear_scan()
    core.compute(Path([[0.0, 0.0], [5.0, 0.0]]), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    core.compute(Path([[0.0, 0.0], [0.0, 5.0]]), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert model.calls == [(None, True), (None, True)]


def test_progress_along_a_replanned_route_counts():
    """After a replan the arc length restarts on the new route; progress on it is progress."""
    core = PolicyCore(FakeModel())
    ranges, angles = clear_scan()
    first = Path([[0.0, 0.0], [10.0, 0.0]])
    for x in (0.0, 1.0, 2.0):
        core.compute(first, (x, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    detour = Path([[2.0, 0.0], [2.0, 2.0], [10.0, 2.0], [10.0, 0.0]])
    core.compute(detour, (2.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    *_, info = core.compute(detour, (2.0, 0.5, 0.0), ranges, angles, (0.0, 0.0))
    assert np.isclose(info['s'], 0.5)


def test_action_delay_holds_each_command_that_many_ticks():
    """Gazebo's mecanum obeys at once; one tick of delay gives back the 2D robot's lag."""
    core = PolicyCore(FakeModel((1.0, 0.0)), action_delay=1)
    ranges, angles = clear_scan()
    first, *_ = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    second, *_ = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert (first, second) == (0.0, V_MAX)
    core.reset()
    again, *_ = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert again == 0.0
