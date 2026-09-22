"""nav_msgs/OccupancyGrid <-> sim2d.geometry.Grid."""
import numpy as np
from nav_msgs.msg import OccupancyGrid

from martha_nav.sim2d.geometry import Grid


def grid_to_msg(grid, frame_id='map', stamp=None):
    """Occupied cells become 100, free cells 0, row-major from the origin."""
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution)
    msg.info.height, msg.info.width = (int(n) for n in grid.occ.shape)
    msg.info.origin.position.x = float(grid.origin[0])
    msg.info.origin.position.y = float(grid.origin[1])
    msg.info.origin.orientation.w = 1.0
    msg.data = np.where(grid.occ, 100, 0).astype(np.int8).ravel().tolist()
    return msg


def msg_to_grid(msg, occupied_threshold=50):
    """Unknown (-1) counts as occupied: the planner must not route through it."""
    data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    occ = (data >= occupied_threshold) | (data < 0)
    origin = (float(msg.info.origin.position.x), float(msg.info.origin.position.y))
    return Grid(np.ascontiguousarray(occ), origin, float(msg.info.resolution))
