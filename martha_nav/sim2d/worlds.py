"""Rasterise the static boxes and cylinders of a Gazebo .world into a Grid."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from martha_nav.sim2d.geometry import RESOLUTION, draw_box, draw_circle, empty_grid

WORLDS_DIR = Path(__file__).resolve().parents[2] / 'worlds'
SKIP_MODELS = {'ground_plane', 'sun'}


def _pose(elem):
    """(x, y, yaw) of an element's <pose>, identity if absent."""
    node = elem.find('pose')
    if node is None or not node.text:
        return np.zeros(3)
    v = [float(t) for t in node.text.split()]
    return np.array([v[0], v[1], v[5]])


def _compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def world_shapes(path):
    """List of ('box', x, y, sx, sy, yaw) and ('cylinder', x, y, r) in world coordinates."""
    root = ET.parse(path).getroot()
    shapes = []
    for model in root.iter('model'):
        if model.get('name') in SKIP_MODELS:
            continue
        m_pose = _pose(model)
        for link in model.findall('link'):
            l_pose = _compose(m_pose, _pose(link))
            for col in link.findall('collision'):
                pose = _compose(l_pose, _pose(col))
                box = col.find('geometry/box/size')
                cyl = col.find('geometry/cylinder/radius')
                if box is not None:
                    sx, sy, _ = (float(t) for t in box.text.split())
                    shapes.append(('box', pose[0], pose[1], sx, sy, pose[2]))
                elif cyl is not None:
                    shapes.append(('cylinder', pose[0], pose[1], float(cyl.text)))
    return shapes


def _extent(shape):
    if shape[0] == 'box':
        _, x, y, sx, sy, yaw = shape
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        hx, hy = (c * sx + s * sy) / 2, (s * sx + c * sy) / 2
        return x - hx, y - hy, x + hx, y + hy
    _, x, y, r = shape
    return x - r, y - r, x + r, y + r


@lru_cache(maxsize=None)
def rasterize_world(name, resolution=RESOLUTION):
    """Grid of worlds/<name>.world clipped to the bounding box of its shapes."""
    shapes = world_shapes(WORLDS_DIR / f'{name}.world')
    ext = np.array([_extent(s) for s in shapes])
    x0, y0 = ext[:, 0].min(), ext[:, 1].min()
    x1, y1 = ext[:, 2].max(), ext[:, 3].max()
    grid = empty_grid(x1 - x0, y1 - y0, origin=(x0, y0), resolution=resolution)
    for shape in shapes:
        if shape[0] == 'box':
            draw_box(grid, *shape[1:])
        else:
            draw_circle(grid, *shape[1:])
    grid.occ.setflags(write=False)
    return grid
