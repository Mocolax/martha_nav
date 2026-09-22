import numpy as np
import pytest

from martha_nav.sim2d.planner import PlanningGrid
from martha_nav.sim2d.worlds import rasterize_world, world_shapes, WORLDS_DIR

ALL_WORLDS = ['four_rooms', 'hall', 'multi', 'roblab', 'room', 'tube', 'lab']


def test_lab_shapes_parsed():
    shapes = world_shapes(WORLDS_DIR / 'lab.world')
    assert len(shapes) == 13
    assert all(s[0] == 'box' for s in shapes)


def test_lab_raster_matches_its_7_5_by_9_5_m_footprint():
    g = rasterize_world('lab')
    h, w = g.shape
    assert abs(w * g.resolution - 7.65) < 0.1      # 7.5 m + wall thickness
    assert abs(h * g.resolution - 9.65) < 0.1


def test_nested_model_poses_are_composed():
    # tube.world places walls through <model><pose>, not through collision poses.
    g = rasterize_world('tube')
    assert g.occ.mean() > 0.02


@pytest.mark.parametrize('name', ALL_WORLDS)
def test_every_world_has_a_large_free_region(name):
    pg = PlanningGrid(rasterize_world(name))
    area = len(pg.largest_component()) * pg.grid.resolution ** 2
    assert area > 20.0


def test_raster_is_cached_and_read_only():
    a = rasterize_world('room')
    assert a is rasterize_world('room')
    with pytest.raises(ValueError):
        a.occ[0, 0] = True
    b = a.copy()
    b.occ[0, 0] = not b.occ[0, 0]
    assert np.any(a.occ != b.occ)
