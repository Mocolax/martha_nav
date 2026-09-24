import numpy as np

from martha_nav.ros.policy_core import PolicyCore, footprint_blocked
from martha_nav.sim2d.observation import LIDAR_MAX, V_MAX, W_MAX
from martha_nav.sim2d.planner import Path


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


def test_the_carrot_skips_a_scanned_obstacle():
    """A box on the route pushes the carrot past it instead of into it."""
    model = FakeModel()
    core = PolicyCore(model, lookahead=1.5)
    ranges, angles = clear_scan()
    ranges[180] = 1.5                             # beam 180 points forward (angle 0)
    core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert model.last_obs[90] > 1.5 / 3.0         # carrot pushed further along


def test_holonomic_core_returns_three_velocities_and_guards_the_sides():
    from martha_nav.sim2d.observation import V_LATERAL
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
