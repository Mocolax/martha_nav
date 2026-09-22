import numpy as np

from martha_nav.sim2d.geometry import draw_box, draw_circle, empty_grid, footprint_collides, raycast


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
    assert not footprint_collides(g, 2.0, 2.0, 0.0)          # half-width 0.205 < 0.30
    assert footprint_collides(g, 2.0, 2.0, np.pi / 2)        # half-length 0.28 reaches the wall cells


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
