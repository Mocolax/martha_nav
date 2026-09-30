"""What every node of the package shares."""
import math

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

# Late subscribers still get the last message: /map, /plan, /nav_status, /goal_pose.
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def run_node(node_class, body=rclpy.spin):
    """main() of a node: body(node) until Ctrl-C, then destroy it while ROS is still up.

    rclpy's own SIGINT handler shuts ROS down first, so destroy_node could not publish
    (the policy's final stop) and every node printed an ExternalShutdownException.
    """
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = node_class()
    try:
        body(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
