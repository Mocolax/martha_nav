import rclpy
from geometry_msgs.msg import Twist  # noqa: F401 - ROS must be importable

from martha_nav.ros.common import run_node, yaw_of


def test_ctrl_c_shuts_the_node_down_quietly():
    """No ExternalShutdownException traceback, and the node still gets to clean up."""
    calls = []

    class Node:
        def destroy_node(self):
            calls.append('destroyed while ROS is up' if rclpy.ok() else 'too late')

    def body(node):
        raise KeyboardInterrupt

    run_node(Node, body)
    assert calls == ['destroyed while ROS is up'] and not rclpy.ok()


def test_yaw_of_a_quaternion():
    import math
    q = type('Q', (), {'x': 0.0, 'y': 0.0, 'z': math.sin(0.4), 'w': math.cos(0.4)})()
    assert math.isclose(yaw_of(q), 0.8)
