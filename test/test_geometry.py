import numpy as np
import pytest

from martha_nav.robots import ROBOTS
from martha_nav.sim2d.geometry import (draw_box, draw_circle, empty_grid, footprint_collides,
                                       footprint_points, post_ranges, raycast)


def test_grid_cells_and_outside_is_occupied():
    g = empty_grid(2.0, 1.0)
    assert g.shape == (20, 40)
    assert not g.occupied(np.array([1.0]), np.array([0.5]))[0]
    assert g.occupied(np.array([-0.1]), np.array([0.5]))[0]
    assert g.occupied(np.array([1.0]), np.array([5.0]))[0]


def test_draw_box_and_circle():
    g = empty_grid(4.0, 4.0)
    draw_box(g, 1.0, 1.0, 0.5, 0.5)
    draw_circle(g, 3.0, 3.0, 0.3)
    assert g.occupied(np.array([1.0, 3.0, 2.0]), np.array([1.0, 3.0, 2.0])).tolist() == [True, True, False]


def test_rotated_box():
    g = empty_grid(4.0, 4.0)
    draw_box(g, 2.0, 2.0, 2.0, 0.2, yaw=np.pi / 2)   # long along y after rotation
    assert g.occupied(np.array([2.0]), np.array([2.9]))[0]
    assert not g.occupied(np.array([2.9]), np.array([2.0]))[0]


def test_raycast_hits_wall_at_expected_distance():
    g = empty_grid(10.0, 4.0)
    draw_box(g, 5.05, 2.0, 0.1, 4.0)                 # wall face at x = 5.0
    d = raycast(g, 1.0, 2.0, np.array([0.0, np.pi]), max_range=8.0)
    assert abs(d[0] - 4.0) <= 0.05
    assert abs(d[1] - 1.0) <= 0.05                   # grid border counts as occupied


def test_raycast_returns_max_range_when_nothing_is_hit():
    g = empty_grid(20.0, 20.0)
    d = raycast(g, 10.0, 10.0, np.array([0.0]), max_range=8.0)
    assert d[0] == 8.0


def test_footprint_collision_respects_orientation():
    g = empty_grid(4.0, 4.0)
    draw_box(g, 2.0, 2.0 + 0.30, 2.0, 0.05)          # thin wall 0.30 m above the centre
    martha = footprint_points(ROBOTS['martha'])
    assert not footprint_collides(g, 2.0, 2.0, 0.0, martha)          # half-width 0.205 < 0.30
    assert footprint_collides(g, 2.0, 2.0, np.pi / 2, martha)        # half-length 0.28 reaches the wall cells


def test_crop_keeps_world_coordinates():
    g = empty_grid(10.0, 10.0)
    draw_box(g, 7.0, 7.0, 0.5, 0.5)
    c = g.crop(5.0, 5.0, 9.0, 9.0)
    assert c.shape[0] < g.shape[0]
    assert c.occupied(np.array([7.0, 6.0]), np.array([7.0, 6.0])).tolist() == [True, False]
    assert c.occupied(np.array([2.0]), np.array([2.0]))[0]     # outside the crop is occupied


def test_drawing_only_touches_cells_near_the_shape():
    g = empty_grid(10.0, 10.0)
    draw_box(g, 5.0, 5.0, 1.0, 0.5, yaw=0.3)
    draw_circle(g, 1.0, 1.0, 0.3)
    rows, cols = np.nonzero(g.occ)
    xs, ys = g.cell_center(rows, cols)
    assert np.all((np.hypot(xs - 5.0, ys - 5.0) < 0.6) | (np.hypot(xs - 1.0, ys - 1.0) < 0.31))
    assert 0.5 * 0.9 < g.occ.sum() * g.resolution ** 2 < (0.5 + np.pi * 0.09) * 1.2


def test_the_lidar_sees_the_tower_posts_behind_it():
    angles = np.deg2rad(np.arange(-180, 180))
    assert np.isinf(post_ranges(ROBOTS['martha'], angles)).all()
    r = post_ranges(ROBOTS['martha_tower'], angles)
    seen = np.abs(np.rad2deg(angles[np.isfinite(r)]))
    # The posts and the lift's motors, from 132 to 166 deg a side. The right motor's box is the
    # nearest, 0.19 m away (its corner; the A2M8 read 0.21 m there), and the right rear post
    # 0.40 m, as the A2M8 read it.
    assert seen.min() >= 132 and seen.max() <= 166
    assert 0.18 < r[np.isfinite(r)].min() < 0.19
    assert r[np.argmin(np.abs(angles + np.deg2rad(164)))] == pytest.approx(0.40, abs=0.01)


def test_slam_toolbox_maps_nothing_of_the_tower():
    from martha_nav.ros.slam import own_reach
    tower = ROBOTS['martha_tower']
    r = post_ranges(tower, np.deg2rad(np.arange(-180, 180, 0.1)))
    assert r[np.isfinite(r)].max() < own_reach(tower) < 0.5
    assert own_reach(ROBOTS['martha']) == 0.0


def test_a_circle_thinner_than_a_cell_still_fills_its_cell():
    grid = empty_grid(1.0, 1.0)
    draw_circle(grid, 0.512, 0.537, 0.02)       # off every cell centre
    assert grid.occ.sum() == 1 and grid.occupied(0.512, 0.537)
