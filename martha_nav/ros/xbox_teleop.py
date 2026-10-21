"""/joy (Xbox controller) -> /cmd_vel for driving Martha by hand, with speed levels and a stop.

No button to hold: the sticks drive whenever the robot is not stopped. RB raises the speed
and LB lowers it, a step per press, as q/z in teleop_twist_keyboard, between a floor and
the firmware's ceiling. Start stops the robot, and it stays stopped (zero commands, sticks
ignored) until Start is pressed again. Nothing is sent without /joy, so a lost controller
ends in the firmware's 500 ms timeout. Axes, buttons and speeds: config/xbox_teleop.yaml.
"""
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy

from martha_nav.ros.common import run_node


class Presses:
    """Which buttons went down in this message (a press, not a hold)."""

    def __init__(self):
        self.held = set()

    def update(self, buttons):
        down = {i for i, b in enumerate(buttons) if b}
        new = down - self.held
        self.held = down
        return new


class SpeedLevel:
    """Stick scales (vx, vy, wz at full stick) multiplied by step per RB, divided per LB; each
    axis stops at its own maximum while the others keep rising."""

    def __init__(self, scale, scale_max, step, lowest=0.25):
        self.scale, self.scale_max, self.step = list(scale), list(scale_max), step
        self.top = max(m / s for m, s in zip(scale_max, scale))  # the last axis at its max
        self.lowest = lowest
        self.factor = 1.0

    def change(self, steps):
        self.factor = min(max(self.factor * self.step ** steps, self.lowest), self.top)

    @property
    def scales(self):
        return [min(s * self.factor, m) for s, m in zip(self.scale, self.scale_max)]


def stick_command(axes, cfg, scales):
    """(vx, vy, wz) from the sticks at the given full-stick scales."""
    return tuple(axes[cfg[axis]] * s if cfg[axis] < len(axes) else 0.0
                 for axis, s in zip(('axis_x', 'axis_y', 'axis_yaw'), scales))


class XboxTeleop(Node):
    def __init__(self):
        super().__init__('xbox_teleop')
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in [
            ('axis_x', 1), ('axis_y', 0), ('axis_yaw', 2), ('button_faster', 10),
            ('button_slower', 9), ('button_stop', 6), ('scale', [0.20, 0.15, 0.5]),
            ('scale_max', [1.0, 1.0, 2.0]), ('speed_step', 1.25)]}
        self.speed = SpeedLevel(self.cfg['scale'], self.cfg['scale_max'], self.cfg['speed_step'])
        self.presses = Presses()
        self.stopped = False
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Joy, '/joy', self.on_joy, 10)
        self.get_logger().info('sticks drive, RB faster, LB slower, Start stops / resumes')
        self.log_speed()

    def log_speed(self):
        vx, vy, wz = self.speed.scales
        self.get_logger().info(f'speed: vx {vx:.2f} m/s, vy {vy:.2f} m/s, wz {wz:.2f} rad/s')

    def on_joy(self, msg):
        new = self.presses.update(msg.buttons)
        if self.cfg['button_stop'] in new:
            self.stopped = not self.stopped
            self.get_logger().warning('STOPPED: Start to drive again' if self.stopped
                                      else 'driving')
        steps = (self.cfg['button_faster'] in new) - (self.cfg['button_slower'] in new)
        if steps:
            self.speed.change(steps)
            self.log_speed()
        cmd = Twist()
        if not self.stopped:
            cmd.linear.x, cmd.linear.y, cmd.angular.z = stick_command(msg.axes, self.cfg,
                                                                       self.speed.scales)
        self.pub.publish(cmd)


def main():
    run_node(XboxTeleop)


if __name__ == '__main__':
    main()
