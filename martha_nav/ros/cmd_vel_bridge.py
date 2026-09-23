"""/cmd_vel (Twist) -> the TwistStamped reference the mecanum controller expects.

The Humble mecanum_drive_controller does not subscribe to /cmd_vel: it takes a
TwistStamped on <controller>/reference. Reused from the previous package.
"""
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node

REFERENCE_TOPIC = '/mecanum_drive_controller/reference'


def to_twist_stamped(twist, frame_id, stamp):
    msg = TwistStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.twist = twist
    return msg


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.frame_id = self.declare_parameter('frame_id', 'base_link').value
        output = self.declare_parameter('output_topic', REFERENCE_TOPIC).value
        self.pub = self.create_publisher(TwistStamped, output, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)

    def on_cmd_vel(self, msg):
        self.pub.publish(to_twist_stamped(msg, self.frame_id, self.get_clock().now().to_msg()))


def main():
    rclpy.init()
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
