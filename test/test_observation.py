import numpy as np

from martha_nav.sim2d.geometry import draw_box, empty_grid, raycast
from martha_nav.sim2d.observation import (LIDAR_MAX, OBS_DIM, action_to_cmd, build_observation,
                                          reduce_scan)


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
    assert np.allclose(obs[:90], 0.5)
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
    near = build_observation(np.array([0.3, 0.3]), angles, (0, 0), (1, 0), (0, 0), lidar_encoding='inverse')
    far = build_observation(np.array([0.5, 0.5]), angles, (0, 0), (1, 0), (0, 0), lidar_encoding='inverse')
    assert np.isclose(near[0], 0.3 / 1.3) and np.isclose(far[0], 0.5 / 1.5)
    assert far[0] - near[0] > 0.1                        # linear /8 m gives only 0.025
    empty = build_observation(np.array([np.inf]), np.zeros(1), (0, 0), (1, 0), (0, 0),
                              lidar_encoding='inverse')
    assert np.isclose(empty[45], 8.0 / 9.0)


def test_unknown_lidar_encoding_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_observation(np.ones(1), np.zeros(1), (0, 0), (1, 0), (0, 0), lidar_encoding='log')
