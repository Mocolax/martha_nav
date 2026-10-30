"""Step response of Martha's mecanum drive in Gazebo: vx, vy and w, one at a time.

Sends /cmd_vel at 10 Hz (sim time) and records the true body velocity from /gazebo/model_states.
python3 tools/step_response.py out.csv   (inside the container, with sim.launch.py running)
"""
import csv
import math
import sys

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter

STEPS = [('vx', 0.35), ('vx', -0.15), ('vy', 0.25), ('vy', -0.25), ('w', 0.8), ('w', -0.8),
         ('vy', 0.125), ('vy', -0.125)]
PRE, ON, POST = 1.0, 3.0, 2.0


class StepTest(Node):
    def __init__(self, out):
        super().__init__('step_test', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_states, 10)
        self.create_timer(0.1, self.tick)
        self.file = open(out, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['t', 'step', 'axis', 'target', 'cmd', 'vx', 'vy', 'w', 'x', 'y', 'yaw'])
        self.t0 = None
        self.cmd = (None, 0, 0.0, 0.0)  # step index, axis, target, commanded now
        self.last_log = -1.0
        self.done = False

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        t = self.now()
        if t <= 0:
            return
        if self.t0 is None:
            self.t0 = t
        k, rest = divmod(t - self.t0, PRE + ON + POST)
        k = int(k)
        if k >= len(STEPS):
            self.pub.publish(Twist())
            self.done = True
            return
        axis, target = STEPS[k]
        value = target if PRE <= rest < PRE + ON else 0.0
        msg = Twist()
        if axis == 'vx':
            msg.linear.x = value
        elif axis == 'vy':
            msg.linear.y = value
        else:
            msg.angular.z = value
        self.pub.publish(msg)
        self.cmd = (k, axis, target, value)

    def on_states(self, msg):
        if 'robot' not in msg.name or self.t0 is None:
            return
        t = self.now()
        if t - self.last_log < 0.01:    # 100 Hz is plenty
            return
        self.last_log = t
        i = msg.name.index('robot')
        p, tw = msg.pose[i], msg.twist[i]
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        vx = c * tw.linear.x + s * tw.linear.y
        vy = -s * tw.linear.x + c * tw.linear.y
        k, axis, target, value = self.cmd
        self.writer.writerow([f'{t - self.t0:.3f}', k, axis, target, value, f'{vx:.4f}',
                              f'{vy:.4f}', f'{tw.angular.z:.4f}', f'{p.position.x:.3f}',
                              f'{p.position.y:.3f}', f'{yaw:.4f}'])


def main():
    rclpy.init()
    node = StepTest(sys.argv[1])
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.file.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
