"""Unicycle (v, w) model with first-order lag, acceleration limits and 1-step delay."""
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


@dataclass
class DynamicsParams:
    tau_v: float = 0.1
    tau_w: float = 0.1
    acc_v: float = 1.0
    acc_w: float = 3.0
    gain_v: float = 1.0
    gain_w: float = 1.0


def sample_params(rng, r=DynamicsRanges()):
    return DynamicsParams(
        tau_v=rng.uniform(*r.tau), tau_w=rng.uniform(*r.tau),
        acc_v=rng.uniform(*r.acc_v), acc_w=rng.uniform(*r.acc_w),
        gain_v=rng.uniform(*r.gain), gain_w=rng.uniform(*r.gain),
    )


class Dynamics:
    """Integrates the robot pose; the command sent at step k acts at step k+1."""

    def __init__(self, params, dt=DT, substeps=SUBSTEPS, holonomic=False):
        self.p = params
        self.h = dt / substeps
        self.substeps = substeps
        self.holonomic = holonomic
        self.reset(np.zeros(3))

    def reset(self, pose):
        self.pose = np.array(pose, dtype=float)
        self.v = 0.0
        self.vy = 0.0
        self.w = 0.0
        self.pending = (0.0, 0.0, 0.0)

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
        target_v = self.p.gain_v * self.pending[0]
        target_vy = self.p.gain_v * self.pending[1]
        target_w = self.p.gain_w * self.pending[2]
        self.pending = (cmd_v, cmd_vy, cmd_w)
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
