from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist

from martha_nav.ros.mecanum_cmd_vel_bridge import REFERENCE_TOPIC, to_twist_stamped


def test_conversion_keeps_the_twist_and_stamps_it():
    twist = Twist()
    twist.linear.x, twist.angular.z = 0.25, -0.4
    stamp = Time(sec=12, nanosec=5)
    msg = to_twist_stamped(twist, 'base_link', stamp)
    assert msg.header.frame_id == 'base_link' and msg.header.stamp == stamp
    assert (msg.twist.linear.x, msg.twist.angular.z) == (0.25, -0.4)
    assert msg.twist.linear.y == 0.0          # the policy never commands vy


def test_reference_topic_is_the_controller_one():
    assert REFERENCE_TOPIC == '/mecanum_drive_controller/reference'
