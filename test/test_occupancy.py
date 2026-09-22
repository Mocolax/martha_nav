import numpy as np

from martha_nav.ros.occupancy import grid_to_msg, msg_to_grid
from martha_nav.sim2d.geometry import draw_box, empty_grid


def sample_grid():
    g = empty_grid(2.0, 1.0, origin=(-1.0, -0.5))
    draw_box(g, 0.5, 0.0, 0.2, 0.2)
    return g


def test_grid_to_msg_layout():
    g = sample_grid()
    msg = grid_to_msg(g)
    assert msg.header.frame_id == 'map'
    assert msg.info.resolution == 0.05
    assert (msg.info.width, msg.info.height) == (g.shape[1], g.shape[0])
    assert (msg.info.origin.position.x, msg.info.origin.position.y) == (-1.0, -0.5)
    assert len(msg.data) == g.occ.size
    assert set(msg.data) == {0, 100}


def test_round_trip_preserves_cells_and_origin():
    g = sample_grid()
    back = msg_to_grid(grid_to_msg(g))
    assert np.array_equal(back.occ, g.occ)
    assert back.origin == g.origin and back.resolution == g.resolution


def test_unknown_cells_count_as_occupied():
    msg = grid_to_msg(sample_grid())
    msg.data = [-1] * len(msg.data)
    assert msg_to_grid(msg).occ.all()
