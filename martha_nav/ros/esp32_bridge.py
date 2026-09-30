"""/cmd_vel -> the ESP32 over serial; wheel odometry and gyro back to ROS.

The firmware owns the protections and the speed ceiling: this node only translates.
Protocol: docs/robot-real-design.md, section 4.7.
"""
import math
from contextlib import suppress

import serial
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger

from martha_nav.ros.common import run_node

UNKNOWN = 1e6  # variance of what is not measured
# Same variances as martha/cmd_vel_serial_bridge.py: the EKF was tuned with them.
TWIST_VARIANCE = [0.02, 0.05, UNKNOWN, UNKNOWN, UNKNOWN, 0.05]
GYRO_Z_VARIANCE = 0.02
DATA_FIELDS = {'odom': 4, 'battery': 1}


def format_cmd(vx, vy, wz):
    return f'cmd_vel,{vx:.3f},{vy:.3f},{wz:.3f}\n'.encode()


def parse_line(line):
    """('odom', [vx, vy, wz, gz]), ('battery', [volts]), ('event', line), or None.

    None is a garbled line, typically the partial first line after opening the port.
    """
    kind, _, rest = line.partition(',')
    if kind not in DATA_FIELDS:
        return ('event', line) if line.isidentifier() else None
    try:
        values = [float(v) for v in rest.split(',')]
    except ValueError:
        return None
    return (kind, values) if len(values) == DATA_FIELDS[kind] else None


def odometry_msg(vx, vy, wz, stamp):
    """Velocities only: the EKF integrates the pose."""
    msg = Odometry()
    msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, 'odom', 'base_link'
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z = vx, vy, wz
    msg.twist.covariance = [0.0] * 36
    for i, variance in enumerate(TWIST_VARIANCE):
        msg.twist.covariance[i * 7] = variance
    return msg


def imu_msg(gz, stamp):
    """Yaw rate only; covariance[0] = -1 marks orientation and acceleration as absent."""
    msg = Imu()
    msg.header.stamp, msg.header.frame_id = stamp, 'base_link'
    msg.angular_velocity.z = gz
    msg.angular_velocity_covariance = [UNKNOWN, 0.0, 0.0, 0.0, UNKNOWN, 0.0, 0.0, 0.0, GYRO_Z_VARIANCE]
    msg.orientation_covariance[0] = -1.0
    msg.linear_acceleration_covariance[0] = -1.0
    return msg


class Esp32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')
        port = self.declare_parameter('port', '/dev/ttyUSB0').value
        self.serial = serial.Serial(port, 115200, timeout=0)
        self.buffer = b''
        self.last_odom = self.get_clock().now()
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odometry', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_service(Trigger, '~/reset', self.on_reset)
        self.create_timer(0.01, self.read)

    def on_cmd_vel(self, msg):
        self.serial.write(format_cmd(msg.linear.x, msg.linear.y, msg.angular.z))

    def on_reset(self, _request, response):
        self.serial.write(b'reset\n')
        response.success = True
        response.message = 'reset sent; the answer (ready / reset_blocked) is in this log'
        return response

    def read(self):
        *lines, self.buffer = (self.buffer + self.serial.read(4096)).split(b'\n')
        now = self.get_clock().now()
        for raw in lines:
            parsed = parse_line(raw.decode(errors='replace').strip())
            if parsed is None:
                continue
            kind, values = parsed
            if kind == 'odom':
                self.last_odom = now
                stamp = now.to_msg()
                self.odom_pub.publish(odometry_msg(*values[:3], stamp))
                if math.isfinite(values[3]):
                    self.imu_pub.publish(imu_msg(values[3], stamp))
            elif kind == 'battery':
                self.get_logger().info(f'battery {values[0]:.2f} V', throttle_duration_sec=30.0)
            else:
                self.get_logger().warning(f'ESP32: {values}')
        if now - self.last_odom > Duration(seconds=1.0):
            self.get_logger().warning('no odometry from the ESP32 for 1 s',
                                      throttle_duration_sec=5.0)

    def destroy_node(self):
        with suppress(serial.SerialException):   # the port may be what failed (unplugged)
            self.serial.write(format_cmd(0.0, 0.0, 0.0))
        self.serial.close()
        super().destroy_node()


def main():
    run_node(Esp32Bridge)


if __name__ == '__main__':
    main()
