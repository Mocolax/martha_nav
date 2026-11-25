import numpy as np
import pytest

from martha_nav.robots import ROBOTS
from martha_nav.ros.policy_core import footprint_blocked
from martha_nav.ros.virtual_obstacles import merge, painted

RES, ORIGIN = 0.05, (-1.0, -1.0)            # a 2 x 2 m map, x and y from -1 to 1 m


def write_map(tmp_path, paint=True, backup=True):
    """A wall at x = 0.9 m since the mapping; a table at x 0.5-0.7, y -0.1-0.1 m painted after."""
    def pgm(path, image):
        path.write_bytes(b'P5\n%d %d\n255\n' % image.shape[::-1] + image.tobytes())

    def cells(x0, x1, y0, y1):
        c0, c1 = round((x0 - ORIGIN[0]) / RES), round((x1 - ORIGIN[0]) / RES)
        r0, r1 = round((y0 - ORIGIN[1]) / RES), round((y1 - ORIGIN[1]) / RES)
        return slice(40 - r1, 40 - r0), slice(c0, c1)     # the image's first row is the top

    mapped = np.full((40, 40), 254, np.uint8)
    mapped[cells(0.9, 0.95, -1.0, 1.0)] = 0
    now = mapped.copy()
    if paint:
        now[cells(0.5, 0.7, -0.1, 0.1)] = 0
    pgm(tmp_path / 'm.pgm', now)
    if backup:
        pgm(tmp_path / 'm.pgm.bak', mapped)
    (tmp_path / 'm.yaml').write_text(f'image: m.pgm\nresolution: {RES}\norigin: [{ORIGIN[0]}, '
                                     f'{ORIGIN[1]}, 0]\n')
    return tmp_path / 'm.yaml'


def test_only_what_was_painted_after_the_mapping_counts(tmp_path):
    grid = painted(write_map(tmp_path))
    assert grid.occ.sum() == 4 * 4                         # the 0.2 x 0.2 m table, not the wall
    assert grid.occupied(0.6, 0.0) and not grid.occupied(0.92, 0.0)
    for name, kwargs in (('no_backup', {'backup': False}), ('clean', {'paint': False})):
        (tmp_path / name).mkdir()
        assert painted(write_map(tmp_path / name, **kwargs)) is None


@pytest.fixture
def table(tmp_path):
    return painted(write_map(tmp_path))


def test_a_beam_reads_the_nearer_of_the_lidar_and_the_painted_table(table):
    angles = np.array([0.0, 0.0, 0.0, np.pi / 2])
    scan = np.array([3.0, 0.2, np.inf, 3.0])                # room, a real leg, nothing, room
    out = merge(scan, angles, table, (0.0, 0.0, 0.0), 0.1, reach=2.5)
    # The LiDAR is 0.1 m ahead of the robot, the table 0.5 m: 0.4 m, to half a cell.
    assert out[0] == pytest.approx(0.4, abs=RES / 2) and out[2] == pytest.approx(0.4, abs=RES / 2)
    assert out[1] == 0.2 and out[3] == 3.0


def test_the_robots_heading_turns_the_beams(table):
    # Below the table, facing +y: the beam ahead meets its lower edge at y = -0.1 m.
    out = merge(np.array([3.0, 3.0]), np.array([0.0, np.pi]), table, (0.6, -0.6, np.pi / 2),
                0.1, reach=2.5)
    assert out[0] == pytest.approx(0.4, abs=RES / 2) and out[1] == 3.0


def test_nothing_is_painted_beyond_reach_or_outside_the_map(table):
    assert merge(np.array([3.0]), np.array([0.0]), table, (0.0, 0.0, 0.0), 0.1, reach=0.3)[0] == 3.0
    # Leaving the map is not an obstacle, unlike Grid.occupied's border.
    assert merge(np.array([3.0]), np.array([np.pi]), table, (-0.8, 0.0, 0.0), 0.1, reach=2.5)[0] == 3.0


def test_the_guard_stops_short_of_a_painted_table(table):
    tower = ROBOTS['martha_tower']
    angles = np.deg2rad(np.arange(-180, 180, 0.5))
    room = np.full(angles.shape, 3.0)
    pose = (0.5 - tower.length / 2 - 0.03, 0.0, 0.0)       # the table 3 cm ahead of the front
    assert footprint_blocked(room, angles, tower) == set()
    assert 'front' in footprint_blocked(merge(room, angles, table, pose, tower.lidar_offset_x, 2.5),
                                        angles, tower)
