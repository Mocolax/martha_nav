"""Grid planner over /map -> /plan, driven by /goal_pose and the map -> base_link transform.

std_msgs/Empty on /cancel_goal drops the goal: the status goes idle and the policy stops.
"""
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener

from martha_nav.ros.common import LATCHED, run_node
from martha_nav.ros.occupancy import msg_to_grid
from martha_nav.ros.planner_core import PlannerCore


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        self.core = PlannerCore(
            inflation=self.declare_parameter('inflation', 0.40).value,
            replan_distance=self.declare_parameter('replan_distance', 1.0).value,
            goal_tolerance=self.declare_parameter('goal_tolerance', 0.3).value,
            snap_distance=self.declare_parameter('snap_distance', 0.5).value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, LATCHED)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.create_subscription(Empty, '/cancel_goal', self.on_cancel, 10)
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

    def on_cancel(self, _msg):
        self.core.cancel()
        self.plan_pub.publish(self.to_msg(None))
        self.status = None                     # always answer, so a cancel can be awaited
        self.publish_status('idle')
        self.get_logger().info('goal cancelled')

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
        path = self.core.path
        status = self.core.update(*pose)
        # The route first: the policy must never see 'active' while it holds an old route.
        if self.core.path is not None and self.core.path is not path:
            self.plan_pub.publish(self.to_msg(self.core.path))
        self.publish_status(status)

    def publish_status(self, status):
        if status != self.status:
            self.status = status
            self.status_pub.publish(String(data=status))
            self.get_logger().info(f'status: {status}')

    def to_msg(self, path):
        """nav_msgs/Path of a route; None gives an empty one."""
        msg = PathMsg()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in ([] if path is None else path.points):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg


def main():
    run_node(GlobalPlanner)


if __name__ == '__main__':
    main()
