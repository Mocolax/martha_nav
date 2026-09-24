"""Local-planner logic, without ROS: carrot, observation, action and safety stop."""
import numpy as np

from martha_nav.sim2d.dynamics import DT
from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH
from martha_nav.sim2d.observation import (GOAL_MAX, LIDAR_MAX, WAYPOINT_MAX, action_to_cmd,
                                          build_observation)
from martha_nav.sim2d.planner import carrot


def footprint_blocked(ranges, angles, margin=0.05):
    """Which sides of the footprint a scan point has entered.

    Returns a set from {'front', 'rear', 'left', 'right'}, empty when clear. Ranges
    are measured from the LiDAR, which sits LIDAR_OFFSET_X ahead of the footprint
    centre, so the rectangle is shifted by that amount. The side matters because a
    guard that blocks every motion leaves the robot frozen against the obstacle.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    valid = np.isfinite(ranges) & (ranges > 0)
    x = ranges[valid] * np.cos(angles[valid]) + LIDAR_OFFSET_X
    y = ranges[valid] * np.sin(angles[valid])
    half_x, half_y = ROBOT_LENGTH / 2 + margin, ROBOT_WIDTH / 2 + margin
    inside = (np.abs(x) <= half_x) & (np.abs(y) <= half_y)
    # Classify by the dominant axis of the intrusion, in units of the half extents:
    # a point dead ahead blocks driving forward, not sliding sideways.
    nx, ny = x[inside] / half_x, y[inside] / half_y
    along_x = np.abs(nx) >= np.abs(ny)
    sides = set()
    for name, mask in (('front', along_x & (nx >= 0)), ('rear', along_x & (nx < 0)),
                       ('left', ~along_x & (ny >= 0)), ('right', ~along_x & (ny < 0))):
        if bool(mask.any()):
            sides.add(name)
    return sides


class PolicyCore:
    """One control step: from a route and a scan to (v, w)."""

    def __init__(self, model, lookahead=1.5, carrot_clearance=0.4, lidar_encoding='inverse',
                 action_dim=2, stuck_signal=False, no_progress_time=15.0, target='carrot'):
        self.model = model
        self.lookahead = lookahead
        self.carrot_clearance = carrot_clearance
        self.lidar_encoding = lidar_encoding
        self.action_dim = action_dim
        self.stuck_signal = stuck_signal
        self.no_progress_time = no_progress_time
        self.target = target
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(self.action_dim)
        self.s = self.s_best = 0.0
        self.since_progress = 0

    def compute(self, path, pose, ranges, angles, velocity):
        """pose is (x, y, yaw) in the map frame.

        Returns (v, w, info), or (vx, vy, w, info) with the holonomic action space.
        """
        x, y, yaw = pose
        blocked = footprint_blocked(ranges, angles)
        self.s = path.project(x, y, s_hint=self.s)
        # Same counter as the 2D environment, so the stuck signal means the same thing.
        gain = max(0.0, self.s - self.s_best)
        self.s_best = max(self.s_best, self.s)
        self.since_progress = 0 if gain > 1e-3 else self.since_progress + 1
        if self.target == 'goal':
            point, scale = path.points[-1], GOAL_MAX
        else:
            points = self._scan_points(ranges, angles, pose)
            point, _ = carrot(path, self.s, self.lookahead, points, self.carrot_clearance)
            scale = WAYPOINT_MAX
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy)
        stuck = (min(self.since_progress * DT / self.no_progress_time, 1.0)
                 if self.stuck_signal else None)
        obs = build_observation(ranges, angles, velocity, rel, self.prev_action,
                                self.lidar_encoding, stuck, scale)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        self.prev_action = action
        commands = list(action_to_cmd(action))
        # Directional guard: stop the motion that would hit, keep the one that escapes.
        if 'front' in blocked:
            commands[0] = min(commands[0], 0.0)
        if 'rear' in blocked:
            commands[0] = max(commands[0], 0.0)
        if len(commands) == 3:
            if 'left' in blocked:
                commands[1] = min(commands[1], 0.0)
            if 'right' in blocked:
                commands[1] = max(commands[1], 0.0)
        return (*commands, {'blocked': sorted(blocked), 's': self.s, 'carrot': point,
                            'stuck': stuck})

    def _scan_points(self, ranges, angles, pose):
        """Scan hits in map coordinates, for the carrot's obstacle skipping."""
        x, y, yaw = pose
        ranges = np.asarray(ranges, dtype=float)
        angles = np.asarray(angles, dtype=float)
        hit = np.isfinite(ranges) & (ranges > 0) & (ranges < LIDAR_MAX)
        if not hit.any():
            return np.empty((0, 2))
        ox = x + LIDAR_OFFSET_X * np.cos(yaw)
        oy = y + LIDAR_OFFSET_X * np.sin(yaw)
        world = yaw + angles[hit]
        return np.stack([ox + ranges[hit] * np.cos(world), oy + ranges[hit] * np.sin(world)], axis=1)
