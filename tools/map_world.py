"""Map a Gazebo world with slam_toolbox, for the localization test.

With `ros2 launch martha_nav sim.launch.py x:=0 y:=0 slam:=mapping` running (the robot
drives on its true pose meanwhile), visit the world's fixed points in a nearest-neighbour
tour, then save slam_toolbox's map for `slam:=localization slam_map:=<out>`:

    ./tools/ct_ros python3 tools/map_world.py --out /home/ros/ros2_ws/src/martha_nav/maps/lab
"""
import argparse
import math
from pathlib import Path

import rclpy
from slam_toolbox.srv import SerializePoseGraph

from martha_nav.ros.evaluate_gazebo import EvaluateGazebo
from martha_nav.sim2d.scenarios import load_points


def tour(points, here=(0.0, 0.0)):
    """Nearest-neighbour order, starting from the robot."""
    left, order = list(points), []
    while left:
        here = min(left, key=lambda p: math.dist(p, here))
        left.remove(here)
        order.append(here)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='lab')
    ap.add_argument('--out', required=True, help='absolute path, without extension')
    ap.add_argument('--goal-timeout', type=float, default=60.0, help='simulated seconds per point')
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = EvaluateGazebo()
    try:
        node.cancel_goal()
        points = tour(load_points(args.world))
        for i, point in enumerate(points, 1):
            node.send_goal(point)
            end = node.now() + args.goal_timeout
            while node.status not in ('succeeded', 'failed') and node.now() < end:
                node.spin(0.2)
            node.get_logger().info(f'{i}/{len(points)} {point}: {node.status}')
            node.cancel_goal()
        client = node.create_client(SerializePoseGraph, '/slam_toolbox/serialize_map')
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('slam_toolbox is not running: launch with slam:=mapping')
        future = client.call_async(SerializePoseGraph.Request(filename=args.out))
        rclpy.spin_until_future_complete(node, future, timeout_sec=60.0)
        if future.result() is None or future.result().result != SerializePoseGraph.Response.RESULT_SUCCESS:
            raise RuntimeError(f'slam_toolbox could not save {args.out}')
        node.get_logger().info(f'map saved: {args.out}.posegraph / .data')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
