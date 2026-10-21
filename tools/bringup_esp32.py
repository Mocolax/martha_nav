#!/usr/bin/env python3
"""Bring-up checks of firmware/firmware.ino on the real robot, straight over serial (no ROS).

The checks of docs/robot-real-design.md section 8, in order, asking before anything moves:
link and arming, battery against the multimeter, gyro at rest, wheel polarity per axis
and PI tracking (wheels in the air), cmd_vel timeout, and a turn on the floor that checks
the gyro's sign against the wheels. Ctrl-C stops the wheels and exits.

Stop esp32_bridge first (it holds the port). Only the standard library, runs on the host:
    python3 tools/bringup_esp32.py [--port /dev/serial/by-id/...] [--csv runs/bringup.csv]
"""
import argparse
import csv
import fcntl
import math
import os
import select
import struct
import sys
import termios
import time

DEFAULT_PORT = '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'
CMD_PERIOD = 0.05          # 20 Hz, well inside the firmware's 500 ms timeout
FIRMWARE_GAIN = 4.9        # BATTERY_GAIN in firmware.ino


class Esp32:
    def __init__(self, port):
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = attrs[5] = termios.B115200
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        for line in (termios.TIOCM_DTR, termios.TIOCM_RTS):  # idle: no reset, no boot mode
            try:
                fcntl.ioctl(self.fd, termios.TIOCMBIC, struct.pack('I', line))
            except OSError:
                pass  # not a real UART (tests)
        self.buffer = b''
        self.odom = []     # (time, vx, vy, wz, gz)
        self.battery = None
        self.events = []

    def send(self, line):
        os.write(self.fd, (line + '\n').encode())

    def poll(self, seconds):
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            if select.select([self.fd], [], [], min(left, 0.02))[0]:
                self.buffer += os.read(self.fd, 4096)
                *lines, self.buffer = self.buffer.split(b'\n')
                for raw in lines:
                    self.parse(raw.decode(errors='replace').strip())

    def parse(self, line):
        kind, _, rest = line.partition(',')
        try:
            values = [float(v) for v in rest.split(',')] if rest else []
        except ValueError:
            return
        if kind == 'odom' and len(values) == 4:
            self.odom.append((time.monotonic(), *values))
        elif kind == 'battery' and len(values) == 1:
            self.battery = values[0]
        elif line.isidentifier():
            self.events.append(line)
            if line != 'cmd_vel_timeout':  # also follows every stop: check_timeout reports it
                print(f'    ESP32: {line}')

    def drive(self, vx, vy, wz, seconds):
        """Commands at 20 Hz for `seconds`; returns the odometry received meanwhile."""
        start, first = time.monotonic(), len(self.odom)
        while time.monotonic() - start < seconds:
            self.send(f'cmd_vel,{vx:.3f},{vy:.3f},{wz:.3f}')
            self.poll(CMD_PERIOD)
        return [(t - start, *rest) for t, *rest in self.odom[first:]]

    def stop(self):
        for _ in range(3):
            self.send('cmd_vel,0,0,0')
            self.poll(CMD_PERIOD)


def mean(values):
    return sum(values) / len(values) if values else float('nan')


def ask(prompt):
    return input(f'\n>>> {prompt} ')


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        verdict = 'SKIP' if ok is None else ('OK' if ok else 'FAIL')
        self.rows.append((name, verdict, detail))
        print(f'  [{verdict}] {name}: {detail}')

    def print(self):
        print('\n=== Summary ===')
        width = max(len(name) for name, _, _ in self.rows)
        for name, verdict, detail in self.rows:
            print(f'  {verdict:4}  {name:{width}}  {detail}')


def check_link(esp, report):
    print('\n--- 1. Link and arming')
    esp.poll(8.0)  # opening the port can reboot it: ~5 s of gyro calibration first
    if not esp.odom:
        report.add('link', False, 'no odom lines: wrong port, firmware.ino not flashed, '
                                  'or esp32_bridge still running')
        return False
    report.add('link', True, f'{len(esp.odom)} odom lines in 8 s (expected ~400)')
    if 'imu_missing' in esp.events or all(math.isnan(o[4]) for o in esp.odom[-50:]):
        report.add('imu', False, 'imu_missing / gz is NaN: check SDA 25, SCL 26 and the IMU power')
    esp.events.clear()
    esp.send('reset')
    esp.poll(1.0)
    if 'ready' in esp.events:
        report.add('arming', True, 'reset -> ready')
        return True
    report.add('arming', False, f'reset -> {esp.events or "no answer"}: reset_blocked means '
                                'battery < 11.5 V or GPIO23 low (R4 / RV1, design section 7)')
    return False


def check_battery(esp, report):
    print('\n--- 2. Battery')
    esp.poll(1.5)
    if esp.battery is None:
        report.add('battery', False, 'no battery line in 1.5 s')
        return
    print(f'    ESP32 reads {esp.battery:.2f} V')
    answer = ask('Multimeter voltage at the battery (Enter to skip):').strip().replace(',', '.')
    if not answer:
        report.add('battery', None, f'{esp.battery:.2f} V, not compared')
        return
    measured = float(answer)
    error = (esp.battery - measured) / measured * 100
    detail = f'ESP32 {esp.battery:.2f} V, multimeter {measured:.2f} V ({error:+.1f} %)'
    if abs(error) > 3:
        detail += f'; set BATTERY_GAIN = {FIRMWARE_GAIN * measured / esp.battery:.3f}'
    report.add('battery', abs(error) <= 3, detail)


def check_gyro_still(esp, report):
    print('\n--- 3. Gyro at rest (do not touch the robot)')
    first = len(esp.odom)
    esp.poll(3.0)
    gz = [o[4] for o in esp.odom[first:] if not math.isnan(o[4])]
    if not gz:
        report.add('gyro at rest', False, 'no gyro readings')
        return
    drift = math.degrees(mean(gz))
    report.add('gyro at rest', abs(drift) < 0.5,
               f'mean {drift:+.2f} deg/s ({drift * 60:+.0f} deg/min if it were not relearned)')


AXES = [  # name, command, index of the odom field it should move
    ('forward  vx=+0.10', (0.10, 0.0, 0.0), 1),
    ('left     vy=+0.10', (0.0, 0.10, 0.0), 2),
    ('turn ccw wz=+0.50', (0.0, 0.0, 0.50), 3),
]


def check_polarity(esp, report, log):
    print('\n--- 4. Polarity per axis (wheels in the air)')
    print('    Watch the wheels on each axis: forward = all four forward; left = mecanum')
    print('    strafe left; ccw = left wheels back, right wheels forward.')
    for name, command, field in AXES:
        samples = esp.drive(*command, seconds=2.0)
        esp.stop()
        esp.poll(0.8)
        log(name, samples)
        steady = [s[field] for s in samples if s[0] > 1.0]
        target = command[field - 1]
        ratio = mean(steady) / target
        others = [abs(mean([s[i] for s in samples if s[0] > 1.0])) for i in (1, 2, 3) if i != field]
        ok = 0.8 <= ratio <= 1.2 and max(others) < 0.3 * abs(target)
        detail = f'measured {mean(steady):+.3f} for {target:+.3f} ({ratio:.2f}x), cross-talk ' \
                 f'{max(others):.3f}'
        if ratio < 0:
            detail += ' -> wheels turned the wrong way: MOTOR_SIGN; right way: ENCODER_SIGN'
        report.add(f'polarity {name}', ok, detail)


def check_tracking(esp, report, log):
    print('\n--- 5. PI tracking, vx = 0.20 step (wheels in the air)')
    target = 0.20
    samples = esp.drive(target, 0.0, 0.0, seconds=3.0)
    esp.stop()
    esp.poll(0.8)
    log('step vx=0.20', samples)
    vx = [(s[0], s[1]) for s in samples]
    steady = mean([v for t, v in vx if t > 2.0])
    rise = next((t for t, v in vx if v >= 0.9 * target), float('nan'))
    overshoot = max(0.0, (max(v for _, v in vx) - target) / target * 100)
    error = (steady - target) / target * 100
    ok = abs(error) < 10 and overshoot < 25 and rise < 1.0
    detail = f'rise to 90 % {rise:.2f} s, overshoot {overshoot:.0f} %, steady error {error:+.1f} %'
    if overshoot >= 25:
        detail += ' -> lower KP'
    elif abs(error) >= 10 or not rise < 1.0:
        detail += ' -> raise KI (or KP if it is slow)'
    report.add('PI step', ok, detail)


def check_timeout(esp, report):
    print('\n--- 6. cmd_vel timeout (wheels in the air)')
    esp.drive(0.20, 0.0, 0.0, seconds=1.0)
    esp.events.clear()
    stopped_at, first = time.monotonic(), len(esp.odom)
    esp.poll(1.5)  # no commands: the firmware must stop on its own after 500 ms
    late = [o[1] for o in esp.odom[first:] if o[0] - stopped_at > 1.0]
    ok = 'cmd_vel_timeout' in esp.events and late and abs(mean(late)) < 0.02
    report.add('timeout', ok, f'event {"seen" if "cmd_vel_timeout" in esp.events else "missing"}, '
                              f'vx after 1 s {mean(late):+.3f}')


def check_turn_on_floor(esp, report, log):
    print('\n--- 7. Turn on the floor: gyro sign against the wheels')
    answer = ask('Robot on the floor with ~1 m clear around it. Enter to turn ~60 deg ccw, '
                 '"s" to skip:')
    if answer.strip().lower() == 's':
        report.add('gyro turning', None, 'skipped')
        return
    samples = esp.drive(0.0, 0.0, 0.5, seconds=2.0)
    esp.stop()
    esp.poll(1.0)
    log('floor turn wz=+0.50', samples)
    steady = [s for s in samples if s[0] > 0.8]
    gz = mean([s[4] for s in steady])
    wz = mean([s[3] for s in steady])
    ok = gz > 0.25 and 0.7 <= gz / wz <= 1.3
    detail = f'gyro {gz:+.3f} rad/s, wheels {wz:+.3f} rad/s, gyro/wheels {gz / wz:.2f}'
    if gz < 0:
        detail += ' -> gyro sign: IMU_UP_SIGN = -1 in firmware.ino'
    elif gz / wz < 0.7:
        detail += ' -> wheels slip or LX_PLUS_LY is too small: the EKF trusts the gyro here'
    report.add('gyro turning', ok, detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', default=DEFAULT_PORT)
    parser.add_argument('--csv', help='write every odom sample of the motion checks here')
    args = parser.parse_args()

    rows = []

    def log(phase, samples):
        rows.extend((phase, *s) for s in samples)

    esp = Esp32(args.port)
    report = Report()
    try:
        armed = check_link(esp, report)
        check_battery(esp, report)
        check_gyro_still(esp, report)
        if not armed:
            report.add('motion checks', None, 'not armed, skipped')
        elif ask('Wheels in the air? Enter to start spinning them, "s" to skip:').lower() != 's':
            check_polarity(esp, report, log)
            check_tracking(esp, report, log)
            check_timeout(esp, report)
            check_turn_on_floor(esp, report, log)
    except KeyboardInterrupt:
        print('\n    interrupted')
    finally:
        esp.stop()
    report.print()
    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['phase', 't', 'vx', 'vy', 'wz', 'gz'])
            writer.writerows(rows)
        print(f'\n  odometry samples in {args.csv}')
    return 0 if all(v != 'FAIL' for _, v, _ in report.rows) else 1


if __name__ == '__main__':
    sys.exit(main())
