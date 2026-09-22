"""Episode generation: static map, start/goal, static route, surprise obstacles."""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from martha_nav.sim2d.geometry import Grid, draw_box, draw_circle, empty_grid
from martha_nav.sim2d.planner import INFLATION, PlanningGrid
from martha_nav.sim2d.worlds import rasterize_world

WALL = 0.15


def _room(w, h):
    """Empty room with interior w x h; returns (grid, outer_w, outer_h)."""
    W, H = w + 2 * WALL, h + 2 * WALL
    g = empty_grid(W, H)
    draw_box(g, W / 2, WALL / 2, W, WALL)
    draw_box(g, W / 2, H - WALL / 2, W, WALL)
    draw_box(g, WALL / 2, H / 2, WALL, H)
    draw_box(g, W - WALL / 2, H / 2, WALL, H)
    return g, W, H


def _random_flip(g, rng):
    occ = g.occ
    if rng.random() < 0.5:
        occ = occ[:, ::-1]
    if rng.random() < 0.5:
        occ = occ[::-1, :]
    return Grid(np.ascontiguousarray(occ), g.origin, g.resolution)


def open_room(rng):
    g, _, _ = _room(rng.uniform(4, 8), rng.uniform(4, 8))
    return g


def corridor(rng):
    length, width = rng.uniform(6, 12), rng.uniform(1.2, 2.5)
    g, _, _ = _room(length, width) if rng.random() < 0.5 else _room(width, length)
    return g


def doorway(rng):
    w1, w2, h = rng.uniform(3, 5), rng.uniform(3, 5), rng.uniform(3, 5)
    g, W, H = _room(w1 + WALL + w2, h)
    gap = rng.uniform(0.85, 1.2)
    y_gap = rng.uniform(WALL + 0.3, H - WALL - 0.3 - gap)
    x = WALL + w1 + WALL / 2
    draw_box(g, x, y_gap / 2, WALL, y_gap)
    top = y_gap + gap
    draw_box(g, x, (top + H) / 2, WALL, H - top)
    return _random_flip(g, rng)


def l_turn(rng):
    g, W, H = _room(rng.uniform(5, 8), rng.uniform(5, 8))
    c1, c2 = rng.uniform(1.2, 2.0), rng.uniform(1.2, 2.0)
    x0, y0 = WALL + c1, WALL + c2
    draw_box(g, (x0 + W) / 2, (y0 + H) / 2, W - x0, H - y0)
    return _random_flip(g, rng)


def furniture_walls(rng):
    g, W, H = _room(rng.uniform(4, 8), rng.uniform(4, 8))
    for _ in range(rng.integers(2, 7)):
        sx, sy = rng.uniform(0.4, 1.5), rng.uniform(0.4, 0.8)
        side = rng.integers(4)
        if side == 0:
            draw_box(g, rng.uniform(WALL, W - WALL), WALL + sy / 2, sx, sy)
        elif side == 1:
            draw_box(g, rng.uniform(WALL, W - WALL), H - WALL - sy / 2, sx, sy)
        elif side == 2:
            draw_box(g, WALL + sy / 2, rng.uniform(WALL, H - WALL), sy, sx)
        else:
            draw_box(g, W - WALL - sy / 2, rng.uniform(WALL, H - WALL), sy, sx)
    return g


def furniture_center(rng):
    g, W, H = _room(rng.uniform(5, 8), rng.uniform(5, 8))
    for _ in range(rng.integers(1, 5)):
        draw_box(g, rng.uniform(WALL + 1.0, W - WALL - 1.0), rng.uniform(WALL + 1.0, H - WALL - 1.0),
                 rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5), rng.uniform(0, np.pi))
    return g


def narrow_passage(rng):
    g, W, H = _room(rng.uniform(5, 8), rng.uniform(4, 7))
    gap, thick = rng.uniform(0.85, 1.0), rng.uniform(0.5, 1.5)
    y_gap = rng.uniform(WALL + 0.5, H - WALL - 0.5 - gap)
    x = W / 2
    draw_box(g, x, y_gap / 2, thick, y_gap)
    top = y_gap + gap
    draw_box(g, x, (top + H) / 2, thick, H - top)
    return _random_flip(g, rng)


TEMPLATES = {
    'open_room': open_room,
    'corridor': corridor,
    'doorway': doorway,
    'l_turn': l_turn,
    'furniture_walls': furniture_walls,
    'furniture_center': furniture_center,
    'narrow_passage': narrow_passage,
}
WORLD_SOURCES = ('four_rooms', 'hall', 'multi', 'roblab', 'room', 'tube')
TRAIN_SOURCES = ('corridor', 'doorway', 'l_turn', 'furniture_walls', 'furniture_center',
                 'narrow_passage') + WORLD_SOURCES
EVAL_ONLY_SOURCES = ('lab',)


@dataclass
class Obstacle:
    kind: str            # 'box' or 'cylinder'
    x: float
    y: float
    sx: float = 0.0      # box size x
    sy: float = 0.0      # box size y
    yaw: float = 0.0
    radius: float = 0.0  # cylinder

    @property
    def extent(self):
        return self.radius if self.kind == 'cylinder' else float(np.hypot(self.sx, self.sy) / 2)

    def draw(self, grid):
        if self.kind == 'box':
            draw_box(grid, self.x, self.y, self.sx, self.sy, self.yaw)
        else:
            draw_circle(grid, self.x, self.y, self.radius)


@dataclass
class ScenarioConfig:
    sources: tuple = TRAIN_SOURCES
    obstacle_mode: str = 'mixed'     # 'mixed' | 'none' | 'always'
    p_no_obstacles: float = 0.2
    max_obstacles: int = 4
    route_min: float = 3.0
    route_max: float = 12.0
    inflation: float = INFLATION
    obstacle_attempts: int = 20
    start_clearance: float = 1.0
    goal_clearance: float = 0.6
    lateral_offset: float = 0.5


@dataclass
class Scenario:
    source: str
    static: Grid
    full: Grid
    start: np.ndarray          # (x, y, yaw)
    goal: np.ndarray           # (x, y)
    path: object               # planner.Path over the static map
    shortest: float            # route length with obstacles, for SPL
    obstacles: list = field(default_factory=list)
    obstacles_dropped: bool = False


@lru_cache(maxsize=None)
def _world_planning(name, inflation):
    grid = rasterize_world(name)
    return grid, PlanningGrid(grid, inflation)


def static_map(source, rng, inflation=INFLATION):
    if source in TEMPLATES:
        grid = TEMPLATES[source](rng)
        return grid, PlanningGrid(grid, inflation)
    return _world_planning(source, inflation)


def _sample_obstacle(rng, path, start, goal, cfg):
    for _ in range(50):
        s = rng.uniform(0.0, path.length)
        p = path.point_at(s)
        t = path.point_at(min(s + 0.1, path.length)) - path.point_at(max(s - 0.1, 0.0))
        n = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-9)
        c = p + n * rng.uniform(-cfg.lateral_offset, cfg.lateral_offset)
        if rng.random() < 0.5:
            ob = Obstacle('box', c[0], c[1], sx=rng.uniform(0.2, 0.6), sy=rng.uniform(0.2, 0.6),
                          yaw=rng.uniform(0, np.pi))
        else:
            ob = Obstacle('cylinder', c[0], c[1], radius=rng.uniform(0.1, 0.3))
        if (np.hypot(*(c - start[:2])) - ob.extent >= cfg.start_clearance
                and np.hypot(*(c - goal)) - ob.extent >= cfg.goal_clearance):
            return ob
    return None


def _detour(full, path, start, goal, cfg, margin=2.0):
    """Route around the obstacles, planned on a crop around the static route (faster)."""
    lo = path.points.min(axis=0) - margin
    hi = path.points.max(axis=0) + margin
    return PlanningGrid(full.crop(lo[0], lo[1], hi[0], hi[1]), cfg.inflation).route(start[:2], goal)


def _n_obstacles(rng, cfg):
    if cfg.obstacle_mode == 'none':
        return 0
    if cfg.obstacle_mode == 'mixed' and rng.random() < cfg.p_no_obstacles:
        return 0
    return int(rng.integers(1, cfg.max_obstacles + 1))


def generate(seed, cfg=ScenarioConfig()):
    """Deterministic episode for an integer seed."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        source = str(cfg.sources[rng.integers(len(cfg.sources))])
        static, pg = static_map(source, rng, cfg.inflation)
        nodes = pg.largest_component()
        if len(nodes) == 0:
            continue
        start_node = int(nodes[rng.integers(len(nodes))])
        dist, pred = pg.distances_from(start_node, limit=cfg.route_max)
        cand = np.flatnonzero((dist >= cfg.route_min) & (dist <= cfg.route_max))
        if len(cand) == 0:
            continue
        goal = pg.node_xy(int(cand[rng.integers(len(cand))]))
        start_xy = pg.node_xy(start_node)
        path = pg.route(start_xy, goal, pred=pred, start_node=start_node)
        start = np.array([start_xy[0], start_xy[1], rng.uniform(-np.pi, np.pi)])
        break
    else:
        raise RuntimeError(f'no valid start/goal for seed {seed}')

    n = _n_obstacles(rng, cfg)
    if n == 0:
        return Scenario(source, static, static, start, goal, path, path.length)
    for _ in range(cfg.obstacle_attempts):
        obstacles = [o for o in (_sample_obstacle(rng, path, start, goal, cfg) for _ in range(n)) if o]
        full = static.copy()
        for ob in obstacles:
            ob.draw(full)
        detour = _detour(full, path, start, goal, cfg)
        if detour is not None and detour.length <= 1.5 * path.length + 2.0:
            return Scenario(source, static, full, start, goal, path, detour.length, obstacles)
    return Scenario(source, static, static, start, goal, path, path.length, obstacles_dropped=True)
