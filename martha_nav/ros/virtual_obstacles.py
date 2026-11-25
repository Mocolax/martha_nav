"""Obstacles painted into a saved map (tools/paint_map.py), shown to the policy as LiDAR hits.

A table above the scan plane shows to the LiDAR as its legs only. Painted into the map, it
keeps the global planner's routes away; merged into the scan here, the policy and the safety
guard see it too: every beam reads the nearer of what the LiDAR measured and the painted cells
along it. Only what was painted counts (the .pgm against the .pgm.bak paint_map keeps), not the
whole map: furniture moved since the mapping, or a localization a few cm off, would otherwise
put walls and chairs where there are none.
"""
from pathlib import Path

import numpy as np
import yaml

from martha_nav.ros.world_map_publisher import read_pgm
from martha_nav.sim2d.geometry import Grid

OCCUPIED = 100    # pgm values below this are occupied (map_saver writes 0, paint_map too)


def painted(map_yaml):
    """Grid of the cells painted since the mapping (occupied now, not in the .bak), or None."""
    meta = yaml.safe_load(Path(map_yaml).read_text())
    pgm = Path(map_yaml).parent / meta['image']
    backup = pgm.with_name(pgm.name + '.bak')
    if not backup.exists():
        return None
    occ = (read_pgm(pgm) < OCCUPIED) & (read_pgm(backup) >= OCCUPIED)
    if not occ.any():
        return None
    return Grid(np.ascontiguousarray(np.flipud(occ)), tuple(float(v) for v in meta['origin'][:2]),
                float(meta['resolution']))


def merge(ranges, angles, grid, pose, lidar_offset_x, reach):
    """ranges with every beam cut at the first painted cell within reach m of the LiDAR.

    angles are from the robot's forward axis, as scan_to_arrays gives them; pose is the robot's
    (x, y, yaw) in the map. Beyond the grid nothing is painted (unlike Grid.occupied).
    """
    ranges = np.asarray(ranges, dtype=float)
    x, y, yaw = pose
    ox, oy = x + lidar_offset_x * np.cos(yaw), y + lidar_offset_x * np.sin(yaw)
    step = grid.resolution / 2
    ts = np.arange(step, reach + step, step)
    world = yaw + np.asarray(angles, dtype=float)
    row, col = grid.to_cell(ox + np.cos(world)[:, None] * ts, oy + np.sin(world)[:, None] * ts)
    rows, cols = grid.occ.shape
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    hits = inside & grid.occ[np.clip(row, 0, rows - 1), np.clip(col, 0, cols - 1)]
    seen = hits.any(axis=1)
    virtual = np.where(seen, ts[hits.argmax(axis=1)], np.inf)
    measured = np.where(np.isfinite(ranges) & (ranges > 0), ranges, np.inf)
    return np.where(seen, np.minimum(measured, virtual), ranges)
