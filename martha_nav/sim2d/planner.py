"""Grid shortest paths (Dijkstra on an inflated grid), route geometry and carrot."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

INFLATION = 0.40   # m; a path in the inflated grid implies a free gap >= 0.8 m


class Path:
    """Polyline route with arc-length parametrisation."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])

    def point_at(self, s):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, len(self.seg_len) - 1))
        if self.seg_len[i] == 0.0:
            return self.points[i].copy()
        t = (s - self.s[i]) / self.seg_len[i]
        return self.points[i] + t * (self.points[i + 1] - self.points[i])

    def project(self, x, y, s_hint=None, back=1.0, fwd=2.0):
        """Arc length of the closest route point, searched near s_hint if given."""
        a, b = self.points[:-1], self.points[1:]
        ab = b - a
        denom = np.maximum((ab ** 2).sum(axis=1), 1e-12)
        t = np.clip(((np.array([x, y]) - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = ((proj - np.array([x, y])) ** 2).sum(axis=1)
        s_proj = self.s[:-1] + t * self.seg_len
        if s_hint is not None:
            window = (s_proj >= s_hint - back) & (s_proj <= s_hint + fwd)
            if window.any():
                d2 = np.where(window, d2, np.inf)
        return float(s_proj[int(np.argmin(d2))])


def carrot(path, s, lookahead, scan_points=None, clearance=0.4, step=0.1):
    """Route point `lookahead` ahead of s, pushed forward past scanned obstacles.

    Returns (point (2,), s_carrot).
    """
    s_c = min(s + lookahead, path.length)
    if scan_points is None or len(scan_points) == 0:
        return path.point_at(s_c), s_c
    while True:
        p = path.point_at(s_c)
        if np.min(np.hypot(*(scan_points - p).T)) >= clearance or s_c >= path.length:
            return p, s_c
        s_c = min(s_c + step, path.length)


class PlanningGrid:
    """8-connected graph over the free cells of an inflated occupancy grid."""

    def __init__(self, grid, inflation=INFLATION):
        self.grid = grid
        clearance = ndimage.distance_transform_edt(~grid.occ) * grid.resolution
        self.free = clearance > inflation
        rows, cols = self.free.shape
        self.node_of = -np.ones(self.free.shape, dtype=np.int64)
        free_rc = np.argwhere(self.free)
        self.node_of[free_rc[:, 0], free_rc[:, 1]] = np.arange(len(free_rc))
        self.rc = free_rc
        src, dst, w = [], [], []
        for dr, dc, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            a = self.node_of[r0:r1, c0:c1]
            b = self.node_of[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ok = (a >= 0) & (b >= 0)
            src.append(a[ok])
            dst.append(b[ok])
            w.append(np.full(int(ok.sum()), cost * grid.resolution))
        n = len(free_rc)
        self.graph = coo_matrix(
            (np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(n, n)
        ).tocsr()
        _, labels = connected_components(self.graph, directed=False)
        self.labels = labels

    def node_at(self, x, y):
        row, col = self.grid.to_cell(x, y)
        if 0 <= row < self.free.shape[0] and 0 <= col < self.free.shape[1]:
            return int(self.node_of[row, col])
        return -1

    def node_xy(self, node):
        x, y = self.grid.cell_center(self.rc[node, 0], self.rc[node, 1])
        return np.array([float(x), float(y)])

    def largest_component(self):
        """Node ids of the largest connected free region."""
        if len(self.labels) == 0:
            return np.array([], dtype=np.int64)
        biggest = np.bincount(self.labels).argmax()
        return np.flatnonzero(self.labels == biggest)

    def distances_from(self, node, limit=np.inf):
        return dijkstra(self.graph, directed=False, indices=node,
                        return_predecessors=True, limit=limit)

    def route(self, start_xy, goal_xy, pred=None, start_node=None):
        """Path from start to goal through free cells, or None."""
        s = self.node_at(*start_xy) if start_node is None else start_node
        g = self.node_at(*goal_xy)
        if s < 0 or g < 0:
            return None
        if pred is None:
            _, pred = self.distances_from(s)
        if g != s and pred[g] < 0:
            return None
        nodes = [g]
        while nodes[-1] != s:
            nodes.append(pred[nodes[-1]])
        pts = [self.node_xy(n) for n in reversed(nodes)]
        pts[0] = np.asarray(start_xy, dtype=float)
        pts[-1] = np.asarray(goal_xy, dtype=float)
        if len(pts) == 1:
            pts.append(pts[0].copy())
        return Path(np.array(pts))
