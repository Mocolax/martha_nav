"""Episodic evaluation in Gazebo, against a running sim.launch.py.

ros2 run martha_nav gazebo_eval --ros-args -p episodes:=100 -p out:=/tmp/eval.csv
ros2 run martha_nav gazebo_eval --ros-args -p mode:=points -p out:=/tmp/eval_points.csv

mode "seeds" plays the generated episodes of the reserved evaluation seeds; mode
"points" plays the hand-placed start/goal pairs of the previous package. Both build
the episode with scenarios.generate, so the same seed is the same episode as in the
2D simulator and the comparison is paired.
"""
import csv
import math
import time
from dataclasses import replace
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String

from martha_nav.learning.evaluate import eval_seeds
from martha_nav.sim2d.scenarios import ScenarioConfig, generate, point_pairs

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)

BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static><link name="link">
<collision name="c"><geometry><box><size>{sx} {sy} 0.6</size></box></geometry></collision>
<visual name="v"><geometry><box><size>{sx} {sy} 0.6</size></box></geometry></visual>
</link></model></sdf>"""

# Visual only, no collision: the LiDAR must not see the marker.
GOAL_SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="goal_marker"><static>true</static><link name="link">
<visual name="v"><geometry><cylinder><radius>0.15</radius><length>0.02</length></cylinder></geometry>
<material><ambient>0.1 0.8 0.2 1</ambient><diffuse>0.1 0.8 0.2 1</diffuse></material></visual>
<visual name="pole"><pose>0 0 0.5 0 0 0</pose>
<geometry><cylinder><radius>0.02</radius><length>1.0</length></cylinder></geometry>
<material><ambient>0.1 0.8 0.2 1</ambient><diffuse>0.1 0.8 0.2 1</diffuse></material></visual>
</link></model></sdf>"""

CYLINDER_SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static><link name="link">
<collision name="c"><geometry><cylinder><radius>{r}</radius><length>0.6</length></cylinder>
</geometry></collision>
<visual name="v"><geometry><cylinder><radius>{r}</radius><length>0.6</length></cylinder>
</geometry></visual></link></model></sdf>"""


class GazeboEval(Node):
    def __init__(self):
        super().__init__('gazebo_eval')
        self.world = self.declare_parameter('world', 'lab').value
        self.mode = self.declare_parameter('mode', 'seeds').value
        self.condition = self.declare_parameter('condition', 'obstacles').value
        episodes = self.declare_parameter('episodes', 100).value
        self.out = self.declare_parameter('out', '/tmp/eval_gazebo.csv').value
        self.timeout = self.declare_parameter('episode_timeout', 120.0).value
        self.settle = self.declare_parameter('settle_seconds', 1.5).value
        # Same rule as the 2D environment, so the two columns measure the same failure.
        self.no_progress = self.declare_parameter('no_progress_seconds', 15.0).value
        self.progress_epsilon = self.declare_parameter('progress_epsilon', 0.10).value

        obstacles = {'obstacles': 'always', 'clean': 'none', 'mixed': 'mixed'}[self.condition]
        self.cfg = ScenarioConfig(sources=(self.world,), obstacle_mode=obstacles)
        if self.mode == 'points':
            pairs = point_pairs(self.world)
            self.cfg = replace(self.cfg, point_pairs=pairs)
            episodes = len(pairs)
        self.seeds = eval_seeds(episodes)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', LATCHED)
        self.create_subscription(String, '/nav_status', self.on_status, LATCHED)
        self.create_subscription(ContactsState, '/bumper_states', self.on_contact, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.position = None
        self.spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete = self.create_client(DeleteEntity, '/delete_entity')
        self.set_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.status = 'idle'
        self.contact = False

    # ---- callbacks ----
    def on_status(self, msg):
        self.status = msg.data

    def on_contact(self, msg):
        if msg.states:
            self.contact = True

    def on_odom(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    # ---- gazebo helpers ----
    def call(self, client, request):
        client.wait_for_service()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        return future.result()

    def teleport(self, x, y, yaw):
        request = SetEntityState.Request()
        request.state.name = 'martha'
        request.state.pose.position.x, request.state.pose.position.y = float(x), float(y)
        request.state.pose.position.z = 0.05
        request.state.pose.orientation.z = math.sin(yaw / 2)
        request.state.pose.orientation.w = math.cos(yaw / 2)
        request.state.reference_frame = 'world'
        self.call(self.set_state, request)

    def spawn_goal_marker(self, goal):
        """A green post at the goal, so the demo is readable in gzclient."""
        request = SpawnEntity.Request()
        request.name, request.xml = 'goal_marker', GOAL_SDF
        request.initial_pose.position.x = float(goal[0])
        request.initial_pose.position.y = float(goal[1])
        request.initial_pose.position.z = 0.01
        self.call(self.spawn, request)
        return ['goal_marker']

    def spawn_obstacles(self, obstacles):
        names = []
        for i, ob in enumerate(obstacles):
            name = f'obstacle_{i}'
            sdf = (BOX_SDF.format(name=name, sx=ob.sx, sy=ob.sy) if ob.kind == 'box'
                   else CYLINDER_SDF.format(name=name, r=ob.radius))
            request = SpawnEntity.Request()
            request.name, request.xml = name, sdf
            request.initial_pose.position.x = float(ob.x)
            request.initial_pose.position.y = float(ob.y)
            request.initial_pose.position.z = 0.3
            request.initial_pose.orientation.z = math.sin(ob.yaw / 2)
            request.initial_pose.orientation.w = math.cos(ob.yaw / 2)
            self.call(self.spawn, request)
            names.append(name)
        return names

    def clear_obstacles(self, names):
        for name in names:
            request = DeleteEntity.Request()
            request.name = name
            self.call(self.delete, request)

    def send_goal(self, goal):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(goal[0]), float(goal[1])
        msg.pose.orientation.w = 1.0
        for _ in range(3):                      # the planner may still be discovering us
            self.goal_pub.publish(msg)
            self.spin(0.2)

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ---- episodes ----
    def run_episode(self, seed):
        scenario = generate(seed, self.cfg)
        self.teleport(*scenario.start)
        names = self.spawn_obstacles(scenario.obstacles) + self.spawn_goal_marker(scenario.goal)
        self.status, self.contact = 'idle', False
        self.spin(self.settle)
        self.contact = False                    # ignore contacts caused by the teleport
        self.send_goal(scenario.goal)
        start = time.time()
        outcome = 'timeout'
        anchor, anchor_time = self.position, time.time()
        while time.time() - start < self.timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.contact:
                outcome = 'collision'
                break
            if self.status == 'succeeded':
                outcome = 'success'
                break
            if self.status == 'failed':
                outcome = 'failed'
                break
            if self.position is not None:
                moved = anchor is None or math.dist(self.position, anchor) > self.progress_epsilon
                if moved:
                    anchor, anchor_time = self.position, time.time()
                elif time.time() - anchor_time > self.no_progress:
                    outcome = 'stalled'
                    break
        self.clear_obstacles(names)
        return {'episode_seed': seed, 'outcome': outcome, 'source': scenario.source,
                'mode': self.mode, 'n_obstacles': len(scenario.obstacles),
                'route_length': round(scenario.path.length, 3),
                'shortest': round(scenario.shortest, 3),
                'seconds': round(time.time() - start, 2)}

    def run(self):
        rows = []
        for seed in self.seeds:
            row = self.run_episode(seed)
            rows.append(row)
            done = len(rows)
            share = sum(r['outcome'] == 'success' for r in rows) / done
            self.get_logger().info(
                f'{done}/{len(self.seeds)} seed {seed}: {row["outcome"]} (éxito {share:.2%})')
            self.write(rows)
        counts = {o: sum(r['outcome'] == o for r in rows) / len(rows)
                  for o in sorted({r['outcome'] for r in rows})}
        self.get_logger().info(f'done: {counts} -> {self.out}')

    def write(self, rows):
        Path(self.out).parent.mkdir(parents=True, exist_ok=True)
        with open(self.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    rclpy.init()
    node = GazeboEval()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
