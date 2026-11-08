"""The simple baseline PPO is compared to: a heading PID towards the carrot plus if/else rules.

It reads the same observation as the policy and answers the same predict(), so the 2D
evaluator and the ROS local planner run it unchanged, on the same Dijkstra route and carrot.
A run whose config.yaml says `controller: pid` loads it instead of a trained model.
The integral term is zero: a heading error has no steady offset for it to remove, so in
practice it is a PD.
"""
import numpy as np

from martha_nav.sim2d.observation import N_SECTORS, post_sectors

KP, KD = 1.5, 0.3      # w = KP * heading error - KD * measured w (rad/s)
STOP = 0.25            # m from the front bumper: stop and turn towards the freer side
SLOW = 0.8             # m: below this, the speed falls linearly to zero at STOP
TURN = 0.6             # fraction of w_max for the turn in place
SIDE = 0.25            # m from a side of the body: closer, steer away from that side
K_SIDE = 12.0          # rad/s per m inside SIDE (tuned on training seeds: runs/_pid_tune.py)
SWEEP = 0.03           # m beyond the body's turning radius that must be clear to turn in place
SIDE_CAP = 2.0         # m: farther readings count as this when comparing the sides
MARGIN = 0.05          # m beyond each side of the body that counts as "ahead"

ANGLES = np.arange(N_SECTORS) * 2 * np.pi / N_SECTORS    # sector centres, 0 = front, CCW
LEFT = (ANGLES > np.radians(20)) & (ANGLES < np.radians(160))
RIGHT = (ANGLES > np.radians(200)) & (ANGLES < np.radians(340))


class PidController:
    def __init__(self, robot):
        self.robot = robot
        self.own = np.zeros(N_SECTORS, dtype=bool)
        self.own[post_sectors(robot)[0]] = True       # the tower's posts are not obstacles
        self.half_length, self.half_width = robot.length / 2, robot.width / 2
        self.radius = np.hypot(self.half_length, self.half_width)   # swept when turning in place

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.atleast_2d(np.asarray(obs, dtype=float))
        if obs.shape[1] != N_SECTORS + 6:
            raise ValueError('the PID baseline drives (v, w) only')
        r = self.robot
        enc = np.minimum(obs[:, :N_SECTORS], 0.999)
        d = enc / (1 - enc)                            # undo encode_lidar: metres from the LiDAR
        heading = obs[:, N_SECTORS + 1] * np.pi        # angle to the carrot
        w_meas = obs[:, N_SECTORS + 3] * r.w_max

        # Scan points in the footprint's frame; the posts are left out.
        x = d * np.cos(ANGLES) + r.lidar_offset_x - r.footprint_offset_x
        y = d * np.sin(ANGLES)
        seen = ~self.own
        ahead = seen & (x > 0) & (np.abs(y) <= self.half_width + MARGIN)
        beside = seen & (np.abs(x) <= self.half_length)
        gap = np.where(ahead, x, np.inf).min(axis=1) - self.half_length
        left = np.where(beside & (y > 0), y, np.inf).min(axis=1) - self.half_width
        right = np.where(beside & (y < 0), -y, np.inf).min(axis=1) - self.half_width
        room = np.where(seen, np.hypot(x, y), np.inf).min(axis=1) - self.radius
        side = np.where(self.own, np.nan, np.minimum(d, SIDE_CAP))
        left_freer = np.nanmean(side[:, LEFT], axis=1) >= np.nanmean(side[:, RIGHT], axis=1)

        # The PID towards the carrot, pushed away from a wall that is too close.
        w = (KP * heading - KD * w_meas
             - K_SIDE * np.clip(SIDE - left, 0, None) + K_SIDE * np.clip(SIDE - right, 0, None))
        v = r.v_max * np.maximum(np.cos(heading), 0.0)
        v = np.where(gap < SLOW, v * np.clip((gap - STOP) / (SLOW - STOP), 0, 1), v)
        # Blocked ahead: turn in place when the body can, otherwise back out straight.
        blocked = gap < STOP
        can_turn = room > SWEEP
        w = np.where(blocked & can_turn, np.where(left_freer, 1, -1) * TURN * r.w_max, w)
        w = np.where(blocked & ~can_turn, 0.0, w)
        a_v = np.where(blocked & ~can_turn, -1.0, v / r.v_max)

        action = np.stack([a_v, np.clip(w / r.w_max, -1, 1)], axis=1)
        return action.astype(np.float32), state
