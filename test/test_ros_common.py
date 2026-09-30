import pytest
import rclpy

from martha_nav.ros.common import run_node, yaw_of

QUIET_NODE = """
import os, signal, threading
from rclpy.node import Node
from martha_nav.ros.common import run_node

class Quiet(Node):                    # no timer and nothing to receive, like world_map_publisher
    def __init__(self):
        super().__init__('quiet')
        threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT)).start()

run_node(Quiet)
print('stopped')
"""


def test_ctrl_c_stops_an_idle_node_quietly():
    """ros2 launch ... & in a script starts the nodes with SIGINT ignored, and a node with
    nothing to do never wakes up by itself: Ctrl-C must still end it, without a traceback."""
    import signal
    import subprocess
    import sys
    done = subprocess.run([sys.executable, '-c', QUIET_NODE], capture_output=True, text=True,
                          timeout=20, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    assert done.returncode == 0 and 'stopped' in done.stdout
    assert 'Traceback' not in done.stderr


def test_yaw_of_a_quaternion():
    import math
    q = type('Q', (), {'x': 0.0, 'y': 0.0, 'z': math.sin(0.4), 'w': math.cos(0.4)})()
    assert math.isclose(yaw_of(q), 0.8)


class Stub:
    def destroy_node(self):
        pass


def test_an_error_from_ros_shutting_down_under_the_executor_is_not_a_crash():
    """Ctrl-C can shut ROS down between two executor iterations; the next one then raises
    RCLError ('the given context is not valid') instead of ExternalShutdownException."""
    def body(node):
        rclpy.shutdown()
        raise RuntimeError('failed to initialize wait set: the given context is not valid')

    run_node(Stub, body)


def test_a_real_error_still_crashes_the_node():
    def body(node):
        raise ValueError('a bug')

    with pytest.raises(ValueError):
        run_node(Stub, body)
