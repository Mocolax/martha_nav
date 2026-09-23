"""A* over /map -> /plan, driven by /goal_pose and the map -> base_link transform."""
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from martha_nav.ros.occupancy import msg_to_grid
from martha_nav.ros.planner_core import PlannerCore

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.core = PlannerCore(
            inflation=self.declare_parameter('inflation', 0.40).value,
            replan_distance=self.declare_parameter('replan_distance', 1.0).value,
            goal_tolerance=self.declare_parameter('goal_tolerance', 0.3).value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, LATCHED)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.plan_pub = self.create_publisher(PathMsg, '/plan', LATCHED)
        self.status_pub = self.create_publisher(String, '/nav_status', LATCHED)
        self.status = None
        self.create_timer(0.2, self.tick)

    def on_map(self, msg):
        self.core.set_map(msg_to_grid(msg))
        self.get_logger().info(f'map received: {msg.info.width}x{msg.info.height}')

    def on_goal(self, msg):
        self.core.set_goal(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def pose(self):
        try:
            tf = self.buffer.lookup_transform(self.map_frame, self.base_frame,
                                              rclpy.time.Time(), Duration(seconds=0.2))
        except Exception:                      # noqa: BLE001 - TF errors are expected at startup
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def tick(self):
        pose = self.pose()
        if pose is None:
            return
        previous, path = self.status, self.core.path
        self.status = self.core.update(*pose)
        if self.status != previous:
            self.status_pub.publish(String(data=self.status))
            self.get_logger().info(f'status: {self.status}')
        if self.core.path is not None and self.core.path is not path:
            self.plan_pub.publish(self.to_msg(self.core.path))

    def to_msg(self, path):
        msg = PathMsg()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path.points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg


def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
