#!/usr/bin/env python3
"""Turn the LiDAR straight by hand, watching how far its scan is turned, live.

The tower's rear posts mirror each other about base_link's x axis, so with the LiDAR straight
(turned exactly 180 degrees, as the URDF says) they show at the same angle on either side of
it. Their mean angle is what the LiDAR is turned by beyond that; this prints it a few times a
second, with which way to turn the LiDAR, while you turn it. It assumes the LiDAR sits on the
centreline: 1 cm to the side looks like 1.5 degrees of turn.

Robot in open space: nothing but the robot within --radius of the LiDAR. With the RPLIDAR
publishing /scan (./robot.sh mapping), in the container (./robot.sh shell):
    python3 tools/align_lidar.py [--window 30]
"""
import argparse
import collections
import math
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from check_tower_posts import blobs  # also puts this checkout on sys.path
from martha_nav.robots import ROBOTS  # noqa: E402

GOOD = 1.0   # deg: straight enough, well inside post_pad (the estimate wanders by about this)
SAME = 0.015  # m: the two rear posts' ranges differ by less than this
NEAREST = 0.25  # m: the rear posts are farther, the front posts and the lift's motors nearer


def merge(blobs_):
    """One blob from several pieces of the same post, weighted by how often each shows."""
    w = [b['rate'] for b in blobs_]
    return {'bearing': math.atan2(sum(wi * math.sin(b['bearing']) for wi, b in zip(w, blobs_)),
                                  sum(wi * math.cos(b['bearing']) for wi, b in zip(w, blobs_))),
            'range': sum(wi * b['range'] for wi, b in zip(w, blobs_)) / sum(w)}


def rear_posts(found):
    """The left and right rear posts' blobs (LiDAR frame), or None if not both are seen.

    In the LiDAR's own frame the rear posts lie ahead of it, near its 0 degrees: the left and
    right blobs beyond the front posts (NEAREST m) at the same range (within SAME m), unlike
    the part between the left posts. A post that flickers can break into several blobs.
    """
    near = [b for b in found if abs(b['bearing']) < math.radians(35) and b['range'] > NEAREST]
    pairs = sorted(((abs(lb['range'] - rb['range']), lb, rb) for lb in near for rb in near
                    if lb['bearing'] < 0 < rb['bearing']), key=lambda p: p[0])
    if not pairs or pairs[0][0] > SAME:
        return None
    return [merge([b for b in near if abs(b['range'] - p['range']) <= SAME
                   and b['bearing'] * p['bearing'] > 0]) for p in pairs[0][1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--window', type=int, default=30, help='scans averaged (10 a second)')
    parser.add_argument('--radius', type=float, default=0.6, help='m from the LiDAR')
    parser.add_argument('--min-rate', type=float, default=0.1,
                        help='share of scans a beam must see a post in')
    args = parser.parse_args()
    x, _, sx, _ = min(ROBOTS['martha_tower'].posts)       # a rear post
    face = x + sx / 2                                       # its face towards the LiDAR

    rclpy.init()
    node = rclpy.create_node('align_lidar')
    window = collections.deque(maxlen=args.window)
    last = [time.monotonic()]                     # when the latest scan came

    def receive(scan):
        window.append(scan)
        last[0] = time.monotonic()

    node.create_subscription(LaserScan, '/scan', receive, qos_profile_sensor_data)
    print('Turn the LiDAR (seen from above) until it says STRAIGHT. Ctrl-C to stop.\n')
    shown, line = 0.0, 'waiting for /scan'
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - shown < 0.3 or (len(window) < args.window // 2
                                                  and time.monotonic() - last[0] <= 1.0):
                continue
            shown = time.monotonic()
            if shown - last[0] > 1.0:     # a USB plug that slips: the driver stops publishing
                print(f'\r\033[Kno /scan for {shown - last[0]:.0f} s: is the LiDAR still plugged '
                      'in? (restart its node)', end='', flush=True)
                window.clear()
                continue
            posts = rear_posts(blobs(list(window), args.radius, args.min_rate, 0.0)[1])
            if posts is None:            # a post that flickers drops out for a moment
                print(f'\r\033[K{line}  (rear posts lost)', end='', flush=True)
                continue
            bl, br = posts
            turn = -math.degrees(bl['bearing'] + br['bearing']) / 2      # + = anticlockwise
            half = (br['bearing'] - bl['bearing']) / 2
            lidar_x = face + (bl['range'] + br['range']) / 2 * math.cos(half)
            col = max(-10, min(10, round(turn)))
            bar = ''.join('|' if i == 0 else '#' if i == col else '.' for i in range(-10, 11))
            todo = ('STRAIGHT' if abs(turn) <= GOOD else
                    f'-> {abs(turn):.1f} deg {"clockwise" if turn > 0 else "anticlockwise"}')
            line = f'turned {turn:+5.1f} deg [{bar}] {todo:<26} x {lidar_x:.3f} m'
            print(f'\r\033[K{line}', end='', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
