"""Local-planner logic, without ROS: carrot, observation, action and safety stop."""
import numpy as np

from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH
from martha_nav.sim2d.observation import LIDAR_MAX, action_to_cmd, build_observation
from martha_nav.sim2d.planner import carrot


def footprint_blocked(ranges, angles, margin=0.05):
    """True when a scan point falls inside the robot rectangle plus a margin.

    Ranges are measured from the LiDAR, which sits LIDAR_OFFSET_X ahead of the
    footprint centre, so the rectangle is shifted by that amount.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    valid = np.isfinite(ranges) & (ranges > 0)
    x = ranges[valid] * np.cos(angles[valid]) + LIDAR_OFFSET_X
    y = ranges[valid] * np.sin(angles[valid])
    inside = (np.abs(x) <= ROBOT_LENGTH / 2 + margin) & (np.abs(y) <= ROBOT_WIDTH / 2 + margin)
    return bool(inside.any())


class PolicyCore:
    """One control step: from a route and a scan to (v, w)."""

    def __init__(self, model, lookahead=1.5, carrot_clearance=0.4, lidar_encoding='inverse'):
        self.model = model
        self.lookahead = lookahead
        self.carrot_clearance = carrot_clearance
        self.lidar_encoding = lidar_encoding
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(2)
        self.s = 0.0

    def compute(self, path, pose, ranges, angles, velocity):
        """pose is (x, y, yaw) in the map frame. Returns (v, w, info)."""
        x, y, yaw = pose
        if footprint_blocked(ranges, angles):
            self.prev_action = np.zeros(2)
            return 0.0, 0.0, {'blocked': True, 's': self.s, 'carrot': None}

        self.s = path.project(x, y, s_hint=self.s)
        points = self._scan_points(ranges, angles, pose)
        point, _ = carrot(path, self.s, self.lookahead, points, self.carrot_clearance)
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy)
        obs = build_observation(ranges, angles, velocity, rel, self.prev_action,
                                self.lidar_encoding)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        self.prev_action = action
        v, w = action_to_cmd(action)
        return v, w, {'blocked': False, 's': self.s, 'carrot': point}

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
