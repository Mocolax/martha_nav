"""/cmd_vel (Twist) -> the TwistStamped reference the mecanum controller expects.

The Humble mecanum_drive_controller does not subscribe to /cmd_vel: it takes a
TwistStamped on <controller>/reference.
"""
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node

from martha_nav.ros.common import run_node

REFERENCE_TOPIC = '/mecanum_drive_controller/reference'


def to_twist_stamped(twist, frame_id, stamp):
    msg = TwistStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.twist = twist
    return msg


class MecanumCmdVelBridge(Node):
    def __init__(self):
        super().__init__('mecanum_cmd_vel_bridge')
        self.frame_id = self.declare_parameter('frame_id', 'base_link').value
        output = self.declare_parameter('output_topic', REFERENCE_TOPIC).value
        self.pub = self.create_publisher(TwistStamped, output, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)

    def on_cmd_vel(self, msg):
        self.pub.publish(to_twist_stamped(msg, self.frame_id, self.get_clock().now().to_msg()))


def main():
    run_node(MecanumCmdVelBridge)


if __name__ == '__main__':
    main()
