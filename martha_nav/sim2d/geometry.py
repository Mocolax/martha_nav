"""Occupancy grid, LiDAR raycasting and footprint collision. Pure numpy."""
from dataclasses import dataclass, field

import numpy as np

RESOLUTION = 0.05          # m per cell


@dataclass
class Grid:
    """Boolean occupancy grid. occ[row, col]; row grows with y, col with x.

    origin is the world (x, y) of the lower-left corner of cell (0, 0).
    Everything outside the grid counts as occupied.
    """

    occ: np.ndarray
    origin: tuple
    resolution: float = RESOLUTION
    _padded: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def shape(self):
        return self.occ.shape

    def copy(self):
        return Grid(self.occ.copy(), self.origin, self.resolution)

    def crop(self, x0, y0, x1, y1):
        """Copy of the cells covering [x0, x1] x [y0, y1], clipped to the grid."""
        (r0, r1), (c0, c1) = self._window(x0, y0, x1, y1)
        origin = (self.origin[0] + c0 * self.resolution, self.origin[1] + r0 * self.resolution)
        return Grid(self.occ[r0:r1, c0:c1].copy(), origin, self.resolution)

    def _window(self, x0, y0, x1, y1):
        """Row and column slices (start, stop) covering a world rectangle."""
        (ra, ca), (rb, cb) = self.to_cell(x0, y0), self.to_cell(x1, y1)
        rows, cols = self.occ.shape
        return ((int(np.clip(ra, 0, rows)), int(np.clip(rb + 1, 0, rows))),
                (int(np.clip(ca, 0, cols)), int(np.clip(cb + 1, 0, cols))))

    def to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return row, col

    def cell_center(self, row, col):
        x = self.origin[0] + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(row) + 0.5) * self.resolution
        return x, y

    def occupied(self, x, y):
        # A one-cell occupied border lets out-of-grid points clip onto it.
        if self._padded is None:
            self._padded = np.pad(self.occ, 1, constant_values=True)
        row, col = self.to_cell(x, y)
        rows, cols = self.occ.shape
        return self._padded[np.clip(row + 1, 0, rows + 1), np.clip(col + 1, 0, cols + 1)]


def empty_grid(width, height, origin=(0.0, 0.0), resolution=RESOLUTION):
    rows = int(np.ceil(height / resolution))
    cols = int(np.ceil(width / resolution))
    return Grid(np.zeros((rows, cols), dtype=bool), tuple(origin), resolution)


def _local_centers(grid, cx, cy, reach):
    """Window slices around (cx, cy) and the cell-centre coordinates inside it."""
    (r0, r1), (c0, c1) = grid._window(cx - reach, cy - reach, cx + reach, cy + reach)
    xs, ys = grid.cell_center(np.arange(r0, r1)[:, None], np.arange(c0, c1)[None, :])
    return (slice(r0, r1), slice(c0, c1)), xs, ys


def draw_box(grid, cx, cy, sx, sy, yaw=0.0):
    """Mark cells whose centers fall inside a rotated rectangle."""
    win, xs, ys = _local_centers(grid, cx, cy, np.hypot(sx, sy) / 2)
    dx, dy = xs - cx, ys - cy
    c, s = np.cos(yaw), np.sin(yaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    grid.occ[win] |= (np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)
    grid._padded = None


def draw_circle(grid, cx, cy, radius):
    win, xs, ys = _local_centers(grid, cx, cy, radius)
    grid.occ[win] |= (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    grid._padded = None


def raycast(grid, ox, oy, angles, max_range):
    """Distance from (ox, oy) along each world angle to the first occupied cell."""
    step = grid.resolution / 2
    ts = np.arange(step, max_range + step, step)
    angles = np.asarray(angles, dtype=float)
    xs = ox + np.cos(angles)[:, None] * ts[None, :]
    ys = oy + np.sin(angles)[:, None] * ts[None, :]
    hits = grid.occupied(xs, ys)
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), np.minimum(ts[first], max_range), max_range)


def footprint_points(robot, spacing=RESOLUTION / 2):
    """Points on the perimeter of the robot's contact rectangle, in base_link."""
    hx, hy = robot.length / 2, robot.width / 2
    xs = np.linspace(-hx, hx, int(np.ceil(robot.length / spacing)) + 1)
    ys = np.linspace(-hy, hy, int(np.ceil(robot.width / spacing)) + 1)
    top = np.stack([xs, np.full_like(xs, hy)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, -hy)], axis=1)
    left = np.stack([np.full_like(ys, -hx), ys], axis=1)
    right = np.stack([np.full_like(ys, hx), ys], axis=1)
    points = np.concatenate([top, bottom, left, right])
    points[:, 0] += robot.footprint_offset_x
    return points


def footprint_collides(grid, x, y, theta, footprint):
    """True when the robot rectangle's perimeter touches an occupied cell."""
    c, s = np.cos(theta), np.sin(theta)
    px = x + c * footprint[:, 0] - s * footprint[:, 1]
    py = y + s * footprint[:, 0] + c * footprint[:, 1]
    return bool(grid.occupied(px, py).any())
