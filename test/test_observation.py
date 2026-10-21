import numpy as np

from martha_nav.robots import ROBOTS
from martha_nav.sim2d.geometry import draw_box, empty_grid, post_ranges, raycast
from martha_nav.sim2d.observation import (OBS_DIM, action_to_cmd, build_observation, post_rays,
                                          post_sectors, reduce_scan)

MARTHA, BURGER = ROBOTS['martha'], ROBOTS['burger']
LIDAR_MAX, V_MAX, W_MAX, V_LATERAL = MARTHA.lidar_range, MARTHA.v_max, MARTHA.w_max, MARTHA.v_lateral


def test_sector_zero_is_front_and_sectors_grow_counter_clockwise():
    ranges = np.array([1.0, 2.0, 3.0])
    angles = np.array([0.0, np.deg2rad(4.0), np.deg2rad(-4.0)])
    s = reduce_scan(ranges, angles)
    assert s[0] == 1.0 and s[1] == 2.0 and s[89] == 3.0
    assert s[45] == LIDAR_MAX


def test_invalid_readings_count_as_max_range():
    s = reduce_scan(np.array([np.inf, np.nan, 0.0, 20.0]), np.zeros(4))
    assert s[0] == LIDAR_MAX


def test_minimum_per_sector():
    s = reduce_scan(np.array([2.0, 1.5]), np.deg2rad([0.5, -1.0]))
    assert s[0] == 1.5


def test_reduction_is_independent_of_beam_density():
    g = empty_grid(10.0, 10.0)
    draw_box(g, 6.5, 5.0, 0.4, 0.4)
    a360 = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    a720 = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    s360 = reduce_scan(raycast(g, 5.0, 5.0, a360, LIDAR_MAX), a360)
    s720 = reduce_scan(raycast(g, 5.0, 5.0, a720, LIDAR_MAX), a720)
    assert np.max(np.abs(s360 - s720)) < 0.1


def test_observation_layout_and_bounds():
    obs = build_observation(np.full(180, 4.0), np.linspace(-np.pi, np.pi, 180, endpoint=False),
                            velocity=(0.35, -0.8), waypoint_rel=(0.0, 1.5), prev_action=(0.5, -1.0))
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.allclose(obs[:90], 0.8)                               # default encoding: 4 / (4 + 1)
    assert np.isclose(obs[90], 0.5) and np.isclose(obs[91], 0.5)   # 1.5/3 m, +90 deg
    assert np.allclose(obs[92:94], [1.0, -1.0])
    assert np.allclose(obs[94:], [0.5, -1.0])
    far = build_observation(np.full(4, 1.0), np.zeros(4), (0.5, 0), (-10.0, 0.0), (0, 0))
    assert far.min() >= -1.0 and far.max() <= 1.0 and np.isclose(far[90], 1.0)


def test_action_mapping_is_asymmetric():
    assert action_to_cmd([1.0, 1.0]) == (0.35, 0.8)
    assert action_to_cmd([-1.0, -1.0]) == (-0.15, -0.8)
    assert action_to_cmd([5.0, 0.0]) == (0.35, 0.0)


def test_inverse_lidar_encoding_gives_more_resolution_up_close():
    angles = np.zeros(2)
    near = build_observation(np.array([0.3, 0.3]), angles, (0, 0), (1, 0), (0, 0))
    far = build_observation(np.array([0.5, 0.5]), angles, (0, 0), (1, 0), (0, 0))
    assert np.isclose(near[0], 0.3 / 1.3) and np.isclose(far[0], 0.5 / 1.5)
    assert far[0] - near[0] > 0.1
    empty = build_observation(np.array([np.inf]), np.zeros(1), (0, 0), (1, 0), (0, 0))
    assert np.isclose(empty[45], 8.0 / 9.0)


def test_holonomic_action_adds_a_lateral_command():
    from martha_nav.sim2d.observation import obs_dim
    assert action_to_cmd([1.0, 0.0]) == (V_MAX, 0.0)                 # (v, w)
    vx, vy, w = action_to_cmd([1.0, -1.0, 0.5])                      # (vx, vy, w)
    assert (vx, vy) == (V_MAX, -V_LATERAL) and np.isclose(w, 0.5 * W_MAX)
    assert obs_dim(2) == 96 and obs_dim(3) == 98


def test_holonomic_observation_carries_three_velocities_and_actions():
    from martha_nav.sim2d.observation import obs_dim
    obs = build_observation(np.full(4, 8.0), np.zeros(4), velocity=(0.35, 0.25, -0.8),
                            waypoint_rel=(1.5, 0.0), prev_action=(0.1, 0.2, 0.3))
    assert obs.shape == (obs_dim(3),)
    assert np.allclose(obs[92:95], [1.0, 1.0, -1.0])                 # vx, vy, w normalised
    assert np.allclose(obs[95:], [0.1, 0.2, 0.3])


def test_the_burger_scales_actions_and_velocities_to_its_own_limits():
    assert action_to_cmd([1.0, -1.0], BURGER) == (0.22, -1.5)
    assert action_to_cmd([-1.0, 0.0], BURGER) == (-0.22, 0.0)
    obs = build_observation(np.full(4, 2.0), np.zeros(4), (0.11, 0.75), (1.0, 0.0), (0, 0),
                            robot=BURGER)
    assert np.allclose(obs[92:94], [0.5, 0.5])


def test_an_empty_sector_reads_the_robots_range():
    assert reduce_scan(np.array([np.inf]), np.zeros(1), max_range=3.5)[0] == 3.5
    empty = build_observation(np.array([np.inf]), np.zeros(1), (0, 0), (1, 0), (0, 0), robot=BURGER)
    assert np.isclose(empty[45], 3.5 / 4.5)


TOWER = ROBOTS['martha_tower']


def test_the_tower_posts_shadow_six_sectors_a_side_and_a_robot_without_posts_none():
    idx, values = post_sectors(TOWER)
    degrees = np.minimum(idx * 4, 360 - idx * 4)           # both sides folded onto 0-180
    assert sorted(degrees) == sorted([148, 152, 156, 160, 164, 168] * 2)
    # Each sector reads the face of its post nearest the LiDAR.
    # 160 deg is in both shadows and reads the nearer, front post.
    assert np.allclose(values[degrees <= 160], 0.226, atol=0.001)
    assert np.allclose(values[degrees > 160], 0.391, atol=0.001)
    assert len(post_sectors(MARTHA)[0]) == 0 and not post_rays(MARTHA, np.zeros(3)).any()


def test_the_post_shadow_covers_every_beam_that_hits_a_post_with_room_to_spare():
    angles = np.deg2rad(np.arange(-180, 180, 0.1))
    hits = np.isfinite(post_ranges(TOWER, angles))
    shadow = post_rays(TOWER, angles)
    assert shadow[hits].all()
    # Turning the posts by up to 2.5 deg (the LiDAR's mounting yaw) keeps them in the shadow.
    for turn in (-2.5, 2.5):
        assert post_rays(TOWER, angles + np.deg2rad(turn))[hits].all()


def test_the_tower_observation_ignores_what_the_lidar_reads_on_its_posts():
    """On the robot, the A2M8 reads each post in only part of its scans, flickering."""
    rng = np.random.default_rng(0)
    angles = np.deg2rad(np.arange(-180, 180, 0.5))
    room = np.full(angles.shape, 3.0)
    posts = post_ranges(TOWER, angles)
    seen = np.where(np.isfinite(posts), posts, room)
    shadow = post_rays(TOWER, angles)
    variants = [seen, room,                                    # always seen, never seen
                np.where(shadow, np.inf, room), np.where(shadow, 0.0, room),
                np.where(shadow & (rng.random(angles.shape) < 0.5), np.inf, seen),
                np.where(shadow, rng.uniform(0.05, 8.0, angles.shape), room)]
    obs = [build_observation(r, angles, (0.0, 0.0), (1.0, 0.0), (0.0, 0.0), robot=TOWER)
           for r in variants]
    for o in obs[1:]:
        assert np.array_equal(o, obs[0])
    # Outside the shadow the LiDAR still counts: an obstacle at 1 m dead behind.
    behind = np.where(np.abs(angles) > np.deg2rad(178), 1.0, seen)
    assert not np.array_equal(build_observation(behind, angles, (0.0, 0.0), (1.0, 0.0), (0.0, 0.0),
                                                robot=TOWER), obs[0])

