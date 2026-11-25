"""Planar motion, (v, w) or holonomic (vx, vy, w), with first-order lag, acceleration limits
and a 1-step delay."""
from dataclasses import dataclass

import numpy as np

DT = 0.1        # s, control period (10 Hz)
SUBSTEPS = 5


@dataclass
class DynamicsRanges:
    tau: tuple = (0.05, 0.4)      # s
    acc_v: tuple = (0.3, 1.0)     # m/s^2
    acc_w: tuple = (1.0, 3.0)     # rad/s^2
    gain: tuple = (0.9, 1.1)
    delay: tuple = (1, 1)         # control periods before a command acts, inclusive
    # Empty: the same draw as gain_v (vy) or from gain (w), as before these existed.
    gain_vy: tuple = ()
    gain_w: tuple = ()
    # rad/s of turn per m/s of commanded vy: on Martha a lateral command also turns her.
    yaw_leak: tuple = (0.0, 0.0)


# Also covers Gazebo's mecanum, which obeys in the same period with 2.5 m/s^2 and
# 6.5 rad/s^2 (docs/resultados.md, "Causa de la brecha").
WIDE_DYNAMICS = DynamicsRanges(tau=(0.02, 0.4), acc_v=(0.3, 3.0), acc_w=(1.0, 7.0), delay=(0, 2))

# Martha on the lab floor with the claw tower (docs/dinamica-real.md, 2026-11-25): steps of
# 1.2 s from standstill, wheel encoders and gyro against the LiDAR's own motion (ICP). Forward
# gives 0.6-1.3 of the command depending on the direction, turning 0.45-0.85, sideways only
# 0.1-0.35 and it turns her by 0.3-0.9 rad/s per m/s, either way; tau 0.15-0.6 s.
REAL_DYNAMICS = DynamicsRanges(tau=(0.15, 0.6), acc_v=(0.5, 2.5), acc_w=(1.0, 5.0),
                               gain=(0.6, 1.3), gain_w=(0.4, 0.9), gain_vy=(0.1, 0.35),
                               yaw_leak=(-0.9, 0.9), delay=(0, 2))


@dataclass
class DynamicsParams:
    tau_v: float = 0.1
    tau_w: float = 0.1
    acc_v: float = 1.0
    acc_w: float = 3.0
    gain_v: float = 1.0
    gain_w: float = 1.0
    delay: int = 1
    gain_vy: float = None         # None: gain_v
    yaw_leak: float = 0.0         # rad/s per m/s of commanded vy


def sample_params(rng, r=DynamicsRanges()):
    p = DynamicsParams(
        tau_v=rng.uniform(*r.tau), tau_w=rng.uniform(*r.tau),
        acc_v=rng.uniform(*r.acc_v), acc_w=rng.uniform(*r.acc_w),
        gain_v=rng.uniform(*r.gain), gain_w=rng.uniform(*(r.gain_w or r.gain)),
    )
    # Drawn only when they vary or are set, so the default ranges leave the episodes as they were.
    p.delay = int(r.delay[0] if r.delay[0] == r.delay[1] else rng.integers(r.delay[0], r.delay[1] + 1))
    if r.gain_vy:
        p.gain_vy = rng.uniform(*r.gain_vy)
    if r.yaw_leak[0] != r.yaw_leak[1]:
        p.yaw_leak = rng.uniform(*r.yaw_leak)
    return p


class Dynamics:
    """Integrates the robot pose; the command sent at step k acts at step k + delay."""

    def __init__(self, params, dt=DT, substeps=SUBSTEPS):
        self.p = params
        self.h = dt / substeps
        self.substeps = substeps
        self.reset(np.zeros(3))

    def reset(self, pose):
        self.pose = np.array(pose, dtype=float)
        self.v = 0.0
        self.vy = 0.0
        self.w = 0.0
        self.pending = [(0.0, 0.0, 0.0)] * self.p.delay    # commands still on their way

    def _approach(self, current, target, tau, acc):
        alpha = 1.0 - np.exp(-self.h / tau)
        limit = acc * self.h
        dv = min(max((target - current) * alpha, -limit), limit)
        return current + dv

    def step(self, cmd_v, cmd_w, collides):
        """Advance one control period with the (v, w) command space."""
        return self.step_holonomic(cmd_v, 0.0, cmd_w, collides)

    def step_holonomic(self, cmd_v, cmd_vy, cmd_w, collides):
        """Advance one control period. collides(x, y, theta) -> bool.

        Returns True on collision; the pose then stays at the last free pose
        and the velocities are zeroed.
        """
        self.pending.append((cmd_v, cmd_vy, cmd_w))
        acting_v, acting_vy, acting_w = self.pending.pop(0)
        target_v = self.p.gain_v * acting_v
        target_vy = (self.p.gain_v if self.p.gain_vy is None else self.p.gain_vy) * acting_vy
        target_w = self.p.gain_w * acting_w + self.p.yaw_leak * acting_vy
        for _ in range(self.substeps):
            self.v = self._approach(self.v, target_v, self.p.tau_v, self.p.acc_v)
            self.vy = self._approach(self.vy, target_vy, self.p.tau_v, self.p.acc_v)
            self.w = self._approach(self.w, target_w, self.p.tau_w, self.p.acc_w)
            x, y, th = self.pose
            nxt = np.array([x + (self.v * np.cos(th) - self.vy * np.sin(th)) * self.h,
                            y + (self.v * np.sin(th) + self.vy * np.cos(th)) * self.h,
                            th + self.w * self.h])
            nxt[2] = (nxt[2] + np.pi) % (2 * np.pi) - np.pi
            if collides(*nxt):
                self.v = self.vy = self.w = 0.0
                return True
            self.pose = nxt
        return False
