import math

import pytest
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist

from martha_nav.ros import esp32_bridge
from martha_nav.ros.esp32_bridge import format_cmd, imu_msg, odometry_msg, parse_line


def test_command_line_the_firmware_parses():
    assert format_cmd(0.25, -0.1, 0.8) == b'cmd_vel,0.250,-0.100,0.800\n'


def test_parse_odom_keeps_a_missing_gyro_as_nan():
    kind, values = parse_line('odom,0.1,-0.2,0.3,nan')
    assert kind == 'odom' and values[:3] == [0.1, -0.2, 0.3] and math.isnan(values[3])


@pytest.mark.parametrize('line, parsed', [
    ('battery,12.34', ('battery', [12.34])),
    ('motor_overcurrent', ('event', 'motor_overcurrent')),
    ('dom,0.1,-0.2,0.3,0.4', None),     # the partial first line after opening the port
    ('odom,0.1,-0.2', None),            # missing fields
    ('odom,0.1,x,0.3,0.4', None),
])
def test_parse_line(line, parsed):
    assert parse_line(line) == parsed


def test_odometry_carries_velocities_for_the_ekf():
    msg = odometry_msg(0.2, -0.1, 0.5, Time(sec=3))
    assert (msg.header.frame_id, msg.child_frame_id) == ('odom', 'base_link')
    twist = msg.twist.twist
    assert (twist.linear.x, twist.linear.y, twist.angular.z) == (0.2, -0.1, 0.5)
    assert msg.twist.covariance[0] == 0.02 and msg.twist.covariance[35] == 0.05


def test_imu_is_yaw_rate_only():
    msg = imu_msg(0.3, Time(sec=3))
    assert msg.header.frame_id == 'base_link' and msg.angular_velocity.z == 0.3
    assert msg.angular_velocity_covariance[8] == 0.02
    assert msg.orientation_covariance[0] == -1.0 and msg.linear_acceleration_covariance[0] == -1.0


class FakeSerial:
    def __init__(self, *_args, **_kwargs):
        self.rx, self.tx = b'', []

    def read(self, size):
        data, self.rx = self.rx[:size], self.rx[size:]
        return data

    def write(self, data):
        self.tx.append(data)

    def close(self):
        pass


class Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(esp32_bridge.serial, 'Serial', FakeSerial)
    rclpy.init()
    node = esp32_bridge.Esp32Bridge()
    node.odom_pub, node.imu_pub = Recorder(), Recorder()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def test_lines_split_across_reads_are_joined(node):
    node.serial.rx = b'ery,12.1\nodom,0.1,0.0,0.2,0.05\nodom,0.2,0.0,0.'
    node.read()
    assert len(node.odom_pub.msgs) == 1 and len(node.imu_pub.msgs) == 1
    node.serial.rx = b'3,nan\n'
    node.read()
    assert node.odom_pub.msgs[-1].twist.twist.angular.z == 0.3
    assert len(node.imu_pub.msgs) == 1        # no gyro in that line: no /imu


def test_cmd_vel_and_reset_reach_the_port(node):
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.angular.z = 0.3, 0.1, -0.5
    node.on_cmd_vel(twist)
    node.on_reset(None, esp32_bridge.Trigger.Response())
    assert node.serial.tx == [b'cmd_vel,0.300,0.100,-0.500\n', b'reset\n']
