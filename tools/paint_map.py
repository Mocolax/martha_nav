#!/usr/bin/env python3
"""Paint what the LiDAR cannot see (a table above its scan plane) into a saved map, from RViz.

The global planner routes over the map's .pgm; localization uses the .posegraph, which this
leaves alone. With the map running (./robot.sh <map>), click two opposite corners of a table
with RViz's Publish Point tool: the rectangle between them, widened by --margin, turns black
in maps/<map>.pgm and the planner gets the new map at once. Click as many tables as needed;
the first change keeps the map as it was in maps/<map>.pgm.bak.

In the container (./robot.sh shell):
    python3 tools/paint_map.py <map> [--margin 0.05]
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout, not the install

import rclpy  # noqa: E402
from geometry_msgs.msg import PointStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402

from martha_nav.ros.common import LATCHED  # noqa: E402
from martha_nav.ros.occupancy import grid_to_msg  # noqa: E402
from martha_nav.ros.world_map_publisher import load_map_yaml, read_pgm  # noqa: E402

MAPS = Path(__file__).resolve().parents[1] / 'maps'


def paint(pgm, meta, a, b, margin):
    """The image with the rectangle between map points a and b (m), widened by margin, black."""
    res, (ox, oy) = float(meta['resolution']), meta['origin'][:2]
    (x0, x1), (y0, y1) = sorted((a[0], b[0])), sorted((a[1], b[1]))
    cols = np.clip([int((x0 - margin - ox) // res), int((x1 + margin - ox) // res) + 1],
                   0, pgm.shape[1])
    rows_up = np.clip([int((y0 - margin - oy) // res), int((y1 + margin - oy) // res) + 1],
                      0, pgm.shape[0])
    out = pgm.copy()
    # The image's first row is the map's top: row r from the bottom is image row h - 1 - r.
    out[pgm.shape[0] - rows_up[1]:pgm.shape[0] - rows_up[0], cols[0]:cols[1]] = 0
    return out


def write_pgm(path, image):
    path.write_bytes(b'P5\n%d %d\n255\n' % (image.shape[1], image.shape[0]) + image.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('map', help='name in maps/, without extension')
    parser.add_argument('--margin', type=float, default=0.05, help='m added on every side')
    args = parser.parse_args()
    yaml_path = MAPS / f'{args.map}.yaml'
    meta = yaml.safe_load(yaml_path.read_text())
    pgm_path = yaml_path.parent / meta['image']

    rclpy.init()
    node = rclpy.create_node('paint_map')
    pub = node.create_publisher(OccupancyGrid, '/map', LATCHED)
    corners = []

    def on_click(msg):
        corners.append((msg.point.x, msg.point.y))
        if len(corners) == 1:
            print(f'corner ({msg.point.x:.2f}, {msg.point.y:.2f}): now the opposite one')
            return
        a, b = corners
        corners.clear()
        backup = pgm_path.with_name(pgm_path.name + '.bak')
        if not backup.exists():
            shutil.copy(pgm_path, backup)
        write_pgm(pgm_path, paint(read_pgm(pgm_path), meta, a, b, args.margin))
        out = grid_to_msg(load_map_yaml(yaml_path))
        out.header.stamp = node.get_clock().now().to_msg()
        pub.publish(out)
        print(f'painted ({a[0]:.2f}, {a[1]:.2f}) - ({b[0]:.2f}, {b[1]:.2f}) into {pgm_path.name}')

    node.create_subscription(PointStamped, '/clicked_point', on_click, 10)
    print(f'Click two opposite corners of each table with Publish Point ({pgm_path.name}). '
          'Ctrl-C to stop.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
