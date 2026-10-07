"""/plan + /scan + /odom -> /cmd_vel at 10 Hz, running the trained policy."""
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from martha_nav.robots import ROBOTS
from martha_nav.ros.common import LATCHED, run_node, yaw_of
from martha_nav.ros.numpy_policy import NumpyPolicy, settings_of
from martha_nav.ros.policy_core import PolicyCore
from martha_nav.ros.scan_adapter import scan_to_arrays
from martha_nav.sim2d.planner import Path

STALE = 0.3      # s; older sensor data stops the robot
LATENCY_REPORT = 10.0   # s between latency log lines


def load_policy(checkpoint):
    """(model, settings): an exported .npz runs on numpy alone; a .zip needs PyTorch."""
    if checkpoint.endswith('.npz'):
        policy = NumpyPolicy(checkpoint)
        return policy, policy.settings
    import torch

    from martha_nav.learning.evaluate import load_model, trained_env_config
    torch.set_num_threads(1)
    settings = settings_of(trained_env_config(checkpoint))
    return load_model(checkpoint), settings


def to_twist(velocities, scale=1.0):
    """(v, w) or (vx, vy, w) -> Twist, every component times scale."""
    cmd = Twist()
    cmd.linear.x = scale * float(velocities[0])
    if len(velocities) == 3:
        cmd.linear.y = scale * float(velocities[1])
    cmd.angular.z = scale * float(velocities[-1])
    return cmd


class PpoLocalPlanner(Node):
    def __init__(self):
        super().__init__('ppo_local_planner')
        checkpoint = self.declare_parameter('checkpoint', '').value
        if not checkpoint:
            raise RuntimeError('parameter "checkpoint" is required')
        self.speed_scale = self.declare_parameter('speed_scale', 1.0).value
        if not 0 < self.speed_scale <= 1:
            raise RuntimeError(f'speed_scale must be in (0, 1], not {self.speed_scale}')
        model, settings = load_policy(checkpoint)
        wanted = self.declare_parameter('robot', '').value
        if wanted and wanted != settings['robot']:
            raise RuntimeError(f"{checkpoint} drives {settings['robot']}, not {wanted}")
        self.core = PolicyCore(
            model,
            lookahead=self.declare_parameter('lookahead', 1.5).value,
            action_dim=settings['action_dim'],
            target=settings['target'],
            action_delay=self.declare_parameter('action_delay', 0).value,
            robot=ROBOTS[settings['robot']])
        self.action_dim = self.core.action_dim
        self.get_logger().info(f"robot {settings['robot']}, action space {self.action_dim}D, "
                               f'action delay {self.core.action_delay}, speed x{self.speed_scale}')
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.scan = self.scan_time = self.odom_time = self.scan_stamp = None
        self.latencies = []
        self.velocity = (0.0, 0.0)
        self.path = None
        self.status = 'idle'
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PathMsg, '/plan', self.on_plan, LATCHED)
        self.create_subscription(String, '/nav_status', self.on_status, LATCHED)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.carrot_pub = self.create_publisher(PointStamped, '/carrot', 10)
        self.create_timer(0.1, self.tick)
        self.create_timer(LATENCY_REPORT, self.report_latency)
        self.get_logger().info(f'policy loaded: {checkpoint}')

    # ---- inputs ----
    def on_scan(self, msg):
        self.scan = scan_to_arrays(msg)
        self.scan_time = self.get_clock().now()
        self.scan_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

    def on_odom(self, msg):
        twist = msg.twist.twist
        self.velocity = ((twist.linear.x, twist.linear.y, twist.angular.z) if self.action_dim == 3
                         else (twist.linear.x, twist.angular.z))
        self.odom_time = self.get_clock().now()

    def on_plan(self, msg):
        # No reset here: the core tells a replan (same goal, memory kept) from a new goal.
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path = Path(np.array(points)) if len(points) >= 2 else None

    def on_status(self, msg):
        self.status = msg.data
        if self.status != 'active':
            # The episode is over: the next goal must not start on this route or memory.
            self.path = None
            self.core.reset()

    def pose(self):
        try:
            tf = self.buffer.lookup_transform(self.map_frame, self.base_frame,
                                              rclpy.time.Time(), Duration(seconds=0.2))
        except Exception:                      # noqa: BLE001 - TF errors are expected at startup
            return None
        q = tf.transform.rotation
        return tf.transform.translation.x, tf.transform.translation.y, yaw_of(q)

    # ---- control ----
    def stop(self, reason):
        self.cmd_pub.publish(Twist())
        self.get_logger().warning(reason, throttle_duration_sec=2.0)

    def fresh(self, stamp, limit=STALE):
        return stamp is not None and (self.get_clock().now() - stamp) < Duration(seconds=limit)

    def tick(self):
        if self.status != 'active' or self.path is None:
            self.cmd_pub.publish(Twist())
            return
        if not self.fresh(self.scan_time) or not self.fresh(self.odom_time):
            return self.stop('stale sensor data')
        pose = self.pose()
        if pose is None:
            return self.stop('no map -> base transform')
        ranges, angles = self.scan
        *velocities, info = self.core.compute(self.path, pose, ranges, angles, self.velocity)
        if info['blocked']:
            self.get_logger().warning(f"obstacle inside the footprint ({info['blocked']}), "
                                      'blocking those directions', throttle_duration_sec=2.0)
        self.cmd_pub.publish(to_twist(velocities, self.speed_scale))
        self.latencies.append((self.get_clock().now() - self.scan_stamp).nanoseconds * 1e-9)
        if info['carrot'] is not None:
            point = PointStamped()
            point.header.frame_id = self.map_frame
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x, point.point.y = float(info['carrot'][0]), float(info['carrot'][1])
            self.carrot_pub.publish(point)

    def report_latency(self):
        """Scan stamp -> /cmd_vel: the delay the policy acts with on this machine and network."""
        if self.latencies:
            self.get_logger().info(f'latency scan -> cmd_vel: mean {np.mean(self.latencies):.3f} s,'
                                   f' max {np.max(self.latencies):.3f} s')
            self.latencies = []


def main():
    run_node(PpoLocalPlanner)


if __name__ == '__main__':
    main()
