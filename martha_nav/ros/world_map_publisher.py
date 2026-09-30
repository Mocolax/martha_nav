"""Publish a latched /map: a rasterised .world, or a map saved on the real robot."""
import re
from pathlib import Path

import numpy as np
import yaml
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from martha_nav.ros.common import LATCHED, run_node
from martha_nav.ros.occupancy import grid_to_msg
from martha_nav.sim2d.geometry import Grid
from martha_nav.sim2d.worlds import rasterize_world


def read_pgm(path):
    """Binary 8-bit PGM (P5), as map_saver and GIMP write it."""
    data = Path(path).read_bytes()
    tokens = []
    for match in re.finditer(rb'#[^\n]*|(\S+)', data):
        if match.group(1):
            tokens.append(match.group(1))
        if len(tokens) == 4:
            break
    if tokens[0] != b'P5':
        raise ValueError(f'{path}: only binary PGM (P5) is supported')
    width, height = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data, np.uint8, width * height, match.end() + 1).reshape(height, width)


def load_map_yaml(path):
    """map_saver's .yaml + .pgm -> Grid. Only white pixels are free: unknown (205) and
    occupied block the planner, as msg_to_grid treats unknown cells."""
    meta = yaml.safe_load(Path(path).read_text())
    image = read_pgm(Path(path).parent / meta['image'])
    occ = np.flipud(image < 250)           # the image starts at the top, the grid at the origin
    origin = (float(meta['origin'][0]), float(meta['origin'][1]))
    return Grid(np.ascontiguousarray(occ), origin, float(meta['resolution']))


class WorldMapPublisher(Node):
    def __init__(self):
        super().__init__('world_map_publisher')
        world = self.declare_parameter('world', 'lab').value
        map_yaml = self.declare_parameter('map_yaml', '').value
        frame = self.declare_parameter('frame_id', 'map').value
        grid = load_map_yaml(map_yaml) if map_yaml else rasterize_world(world)
        msg = grid_to_msg(grid, frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.pub.publish(msg)
        self.get_logger().info(f'published {map_yaml or world}: '
                               f'{msg.info.width}x{msg.info.height} cells @ {msg.info.resolution} m')


def main():
    run_node(WorldMapPublisher)


if __name__ == '__main__':
    main()
