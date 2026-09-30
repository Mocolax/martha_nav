"""Episodic evaluation in Gazebo, against a running sim.launch.py.

ros2 run martha_nav evaluate_gazebo --ros-args -p episodes:=100 -p out:=/tmp/eval.csv
ros2 run martha_nav evaluate_gazebo --ros-args -p mode:=points -p out:=/tmp/eval_points.csv

mode "seeds" plays the generated episodes of the reserved evaluation seeds; mode
"points" plays the hand-placed start/goal pairs of the previous package. Both build
the episode with scenarios.generate, so the same seed is the same episode as in the
2D simulator and the comparison is paired. Episodes end by the 2D environment's rules,
in simulated time (the "seconds" column too).

The loc_err_* columns compare the localization (TF map -> base_link) with Gazebo's
true pose: zero with sim.launch.py's default, the test with slam:=localization.
"""
import csv
import math
from dataclasses import replace
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ContactsState, ModelStates
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener

from martha_nav.ros.common import LATCHED, run_node, yaw_of
from martha_nav.sim2d.dynamics import DT
from martha_nav.sim2d.env import EnvConfig, episode_steps, eval_seeds
from martha_nav.sim2d.planner import RouteProgress
from martha_nav.sim2d.scenarios import CONDITIONS, ScenarioConfig, generate, point_pairs

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


REACHED = 0.5          # m: the planner's 0.3 m goal tolerance plus the localization error


def episode_outcome(contact, status, to_goal, progress, elapsed, time_limit, no_progress_time):
    """NavEnv's end rules in its order, or None while the episode goes on. to_goal is the true
    distance: under SLAM the planner succeeds on the estimated pose, which may be wrong."""
    if contact:
        return 'collision'
    if status == 'succeeded':
        return 'success' if to_goal <= REACHED else 'lost'
    if status == 'failed':
        return 'failed'
    if progress.seconds_without_progress >= no_progress_time:
        return 'stalled'
    if elapsed >= time_limit:
        return 'timeout'
    return None


def localization_error(estimate, truth):
    """(metres, radians) between two (x, y, yaw) poses."""
    yaw = (estimate[2] - truth[2] + math.pi) % (2 * math.pi) - math.pi
    return math.dist(estimate[:2], truth[:2]), abs(yaw)


class EvaluateGazebo(Node):
    def __init__(self):
        super().__init__('evaluate_gazebo',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.world = self.declare_parameter('world', 'lab').value
        self.mode = self.declare_parameter('mode', 'seeds').value
        self.condition = self.declare_parameter('condition', 'obstacles').value
        episodes = self.declare_parameter('episodes', 100).value
        self.out = self.declare_parameter('out', '/tmp/eval_gazebo.csv').value
        self.settle = self.declare_parameter('settle_seconds', 1.5).value
        self.no_progress = self.declare_parameter('no_progress_seconds',
                                                  EnvConfig().no_progress_time).value

        self.cfg = ScenarioConfig(sources=(self.world,), obstacle_mode=CONDITIONS[self.condition])
        if self.mode == 'points':
            pairs = point_pairs(self.world)
            self.cfg = replace(self.cfg, point_pairs=pairs)
            episodes = len(pairs)
        self.seeds = eval_seeds(episodes)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', LATCHED)
        self.cancel_pub = self.create_publisher(Empty, '/cancel_goal', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.create_subscription(String, '/nav_status', self.on_status, LATCHED)
        self.create_subscription(ContactsState, '/bumper_states', self.on_contact, 10)
        # Position from Gazebo itself: the odometry topic depends on the drive, and a
        # missing subscription would silently disable the stall rule.
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_states, 10)
        self.position = self.yaw = None
        self.spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete = self.create_client(DeleteEntity, '/delete_entity')
        self.set_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.status = None
        self.contact = False

    # ---- callbacks ----
    def on_status(self, msg):
        self.status = msg.data

    def on_contact(self, msg):
        if msg.states:
            self.contact = True

    def on_states(self, msg):
        if 'martha' in msg.name:
            pose = msg.pose[msg.name.index('martha')]
            self.position = (pose.position.x, pose.position.y)
            self.yaw = yaw_of(pose.orientation)

    def estimated_pose(self):
        """Where the localization thinks the robot is, or None before it knows."""
        try:
            tf = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:                      # noqa: BLE001 - no TF yet
            return None
        t = tf.transform
        return t.translation.x, t.translation.y, yaw_of(t.rotation)

    # ---- gazebo helpers ----
    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'{client.srv_name} is not available: is Gazebo running?')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.success:
            reason = getattr(result, 'status_message', 'no answer')
            raise RuntimeError(f'{client.srv_name} failed ({reason}); restart the simulation')
        return result

    def teleport(self, x, y, yaw):
        request = SetEntityState.Request()
        request.state.name = 'martha'
        request.state.pose.position.x, request.state.pose.position.y = float(x), float(y)
        request.state.pose.position.z = 0.05
        request.state.pose.orientation.z = math.sin(yaw / 2)
        request.state.pose.orientation.w = math.cos(yaw / 2)
        request.state.reference_frame = 'world'
        self.call(self.set_state, request)

    def set_initial_pose(self, x, y, yaw):
        """As the operator does in RViz: a teleport is a kidnapping for a localizer."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        self.initial_pose_pub.publish(msg)

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

    def cancel_goal(self):
        """Make the planner drop its goal and wait until it says so: the policy then stops."""
        self.status = None
        while self.status != 'idle':
            self.get_logger().info('waiting for the planner to cancel the goal', once=True)
            self.cancel_pub.publish(Empty())
            self.spin(0.2)
        self.spin(0.3)                          # the policy's zero command reaches the wheels

    def now(self):
        """Simulated seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def spin(self, seconds):
        end = self.now() + seconds
        while self.now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ---- episodes ----
    def run_episode(self, seed):
        scenario = generate(seed, self.cfg)
        self.cancel_goal()                      # or the robot drives the last goal from here
        self.teleport(*scenario.start)
        while self.position is None or math.dist(self.position, scenario.start[:2]) > 0.05:
            self.spin(0.05)                     # or the localizer reads the jump as odometry
        self.spin(0.2)
        self.set_initial_pose(*scenario.start)
        names = self.spawn_obstacles(scenario.obstacles) + self.spawn_goal_marker(scenario.goal)
        self.spin(self.settle)
        self.contact = False                    # ignore contacts caused by the teleport
        self.send_goal(scenario.goal)
        # As NavEnv: progress on the static route sampled every DT, timeout by its length.
        progress = RouteProgress(scenario.path)
        time_limit = episode_steps(scenario.path.length) * DT
        start = sample = now = self.now()
        outcome, errors = None, []
        while outcome is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            now = self.now()
            while self.position is not None and sample + DT <= now:
                sample += DT
                progress.update(*self.position)
                estimate = self.estimated_pose()
                if estimate is not None:
                    errors.append(localization_error(estimate, (*self.position, self.yaw)))
            outcome = episode_outcome(self.contact, self.status,
                                      math.dist(self.position, scenario.goal), progress, now - start,
                                      time_limit, self.no_progress)
        self.clear_obstacles(names)
        return {'episode_seed': seed, 'outcome': outcome, 'source': scenario.source,
                'mode': self.mode, 'n_obstacles': len(scenario.obstacles),
                'route_length': round(scenario.path.length, 3),
                'shortest': round(scenario.shortest, 3),
                'seconds': round(now - start, 2),
                'loc_err_mean': round(sum(e for e, _ in errors) / len(errors), 3) if errors else None,
                'loc_err_max': round(max(e for e, _ in errors), 3) if errors else None,
                'yaw_err_max_deg': round(math.degrees(max(y for _, y in errors)), 1) if errors else None}

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
        self.cancel_goal()
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
    run_node(EvaluateGazebo, EvaluateGazebo.run)


if __name__ == '__main__':
    main()
