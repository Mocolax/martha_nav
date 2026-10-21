#!/usr/bin/env python3
"""Where the tower's posts really are, measured with the LiDAR, against ROBOTS['martha_tower'].

Everything the LiDAR sees within --radius of itself, steadily over a few seconds, is taken
as the robot's own structure. Each such blob is reported with its bearing, nearest range,
width and how often it shows, matched to the profile's nearest post, and turned into a post
centre in base_link; at the end it prints the posts= line for robots.py. Left/right pairs
also give the LiDAR's mounting yaw, which post_pad has to cover.

Robot in open space: nothing but the robot within --radius of the LiDAR. With the RPLIDAR
publishing /scan (tools/teleop_test.sh, or real.launch.py), in the container:
    ./tools/ct_ros python3 tools/check_tower_posts.py [--scans 60] [--radius 0.6]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout, not the install

import rclpy  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402

from martha_nav.robots import ROBOTS  # noqa: E402
from martha_nav.ros.scan_adapter import scan_to_arrays  # noqa: E402
from martha_nav.sim2d.observation import post_shadows  # noqa: E402


def collect(count, timeout):
    rclpy.init()
    node = rclpy.create_node('check_tower_posts')
    scans = []
    node.create_subscription(LaserScan, '/scan', scans.append, qos_profile_sensor_data)
    end = node.get_clock().now().nanoseconds + timeout * 1e9
    while len(scans) < count and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return scans


def blobs(scans, radius, min_rate):
    """Runs of beams that see something within radius in at least min_rate of the scans."""
    angles = scan_to_arrays(scans[0])[1]
    ranges = np.array([scan_to_arrays(s)[0] for s in scans if len(s.ranges) == len(angles)])
    near = np.isfinite(ranges) & (ranges > 0) & (ranges <= radius)
    rate = near.mean(axis=0)
    steady = rate >= min_rate
    if steady.all() or not steady.any():
        return angles, []
    start = int(np.argmin(steady))  # begin on a gap, so a run across +-pi stays whole
    order = np.roll(np.arange(len(angles)), -start)
    runs, current = [], []
    for i in order:
        if steady[i]:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    found = []
    for run in runs:
        hits = ranges[:, run][near[:, run]]
        bearing = math.atan2(np.sin(angles[run]).mean(), np.cos(angles[run]).mean())
        width = len(run) * abs(angles[1] - angles[0])
        found.append({'bearing': bearing, 'range': float(np.median(hits)),
                      'nearest': float(np.percentile(hits, 5)), 'width': width,
                      'rate': float(rate[run].max())})
    return angles, found


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scans', type=int, default=60)
    parser.add_argument('--radius', type=float, default=0.6, help='m from the LiDAR')
    parser.add_argument('--min-rate', type=float, default=0.2,
                        help='share of scans a beam must see something in')
    args = parser.parse_args()

    robot = ROBOTS['martha_tower']
    scans = collect(args.scans, timeout=args.scans / 5 + 5)
    if not scans:
        print('No /scan in time: is the RPLIDAR node running?')
        return 1
    print(f'{len(scans)} scans, {len(scans[0].ranges)} beams, range_min '
          f'{scans[0].range_min:.2f} m')
    _, found = blobs(scans, args.radius, args.min_rate)

    sx, sy = robot.posts[0][2:] if robot.posts else (0.04, 0.02)
    expected = [(x, y, mid) for (x, y, *_), (mid, _, _) in zip(robot.posts, post_shadows(robot))]
    print(f'\nExpected (robots.py, LiDAR at x = {robot.lidar_offset_x} m):')
    for x, y, mid in expected:
        print(f'  post ({x:+.3f}, {y:+.3f})  bearing {math.degrees(mid):+7.1f} deg')

    # Match each blob to the profile's nearest post, keeping the steadiest blob per post.
    matched = {}
    for b in found:
        if expected:
            k = min(range(len(expected)), key=lambda i: abs(wrap(b['bearing'] - expected[i][2])))
            b['post'] = k
            if k not in matched or b['rate'] > matched[k]['rate']:
                matched[k] = b

    # A left/right pair mirrors about x in base_link; the LiDAR's yaw turns both scan bearings
    # the same way (scan bearing = base_link bearing - yaw), so the pair's mean is -yaw.
    yaws = [-wrap(matched[k]['bearing'] + matched[j]['bearing']) / 2
            for k in matched for j in matched
            if j > k and expected[k][0] == expected[j][0] and expected[k][1] == -expected[j][1]]
    yaw = sum(yaws) / len(yaws) if yaws else 0.0

    print(f'\nSeen within {args.radius} m (centres in base_link, LiDAR yaw '
          f'{math.degrees(yaw):+.1f} deg taken out):')
    measured = {}
    for b in sorted(found, key=lambda b: b['bearing']):
        # The posts stand behind the LiDAR, which sees their front face: the centre lies half
        # the post's length further back.
        bearing = b['bearing'] + yaw
        x = robot.lidar_offset_x + b['range'] * math.cos(bearing) - sx / 2
        y = b['range'] * math.sin(bearing)
        line = (f'  bearing {math.degrees(b["bearing"]):+7.1f} deg, range {b["range"]:.3f} m '
                f'(nearest {b["nearest"]:.3f}), width {math.degrees(b["width"]):4.1f} deg, '
                f'seen in {b["rate"]:4.0%} -> centre ({x:+.3f}, {y:+.3f})')
        if 'post' in b:
            k = b['post']
            line += f', post {k} {math.degrees(wrap(b["bearing"] - expected[k][2])):+.1f} deg off'
            if matched.get(k) is b:
                measured[k] = {'x': x, 'y': y}
        print(line)
    if not found:
        print('  nothing: the posts are inside range_min, or --radius is too small')

    missing = [k for k in range(len(expected)) if k not in measured]
    for k in missing:
        x, y, _ = expected[k]
        print(f'\n  post ({x:+.3f}, {y:+.3f}) not seen: closer than the A2M8 measures '
              '(0.15 m)? Measure it with a tape from the LiDAR\'s centre.')

    if yaws:
        print(f'\nLiDAR mounting yaw from the left/right pairs: {math.degrees(yaw):+.1f} deg '
              f'(post_pad covers {math.degrees(robot.post_pad):.1f} deg)')
        if abs(yaw) > robot.post_pad:
            print('  -> more than post_pad: turn the LiDAR straight, or raise post_pad')

    if measured and not missing:
        posts = ', '.join(f'({m["x"]:.3f}, {m["y"]:.3f}, {sx}, {sy})'
                          for _, m in sorted(measured.items()))
        print(f'\nFor robots.py (rounded to the tape, and keep them symmetric if they are):'
              f'\n    posts=({posts})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
