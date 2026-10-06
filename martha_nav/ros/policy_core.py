"""Local-planner logic, without ROS: carrot, observation, action and safety stop."""
import numpy as np

from martha_nav.learning import is_recurrent
from martha_nav.robots import ROBOTS
from martha_nav.sim2d.observation import GOAL_MAX, WAYPOINT_MAX, action_to_cmd, build_observation
from martha_nav.sim2d.planner import RouteProgress, carrot


def footprint_blocked(ranges, angles, robot=ROBOTS['martha']):
    """Which sides of the footprint a scan point has entered.

    Returns a set from {'front', 'rear', 'left', 'right'}, empty when clear. Ranges are
    measured from the LiDAR, so the points are moved into the footprint's frame first,
    and the guard reacts robot.guard_margin beyond the rectangle. The side matters
    because a guard that blocks every motion leaves the robot frozen against the obstacle.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    valid = np.isfinite(ranges) & (ranges > 0)
    x = (ranges[valid] * np.cos(angles[valid]) + robot.lidar_offset_x
         - robot.footprint_offset_x)
    y = ranges[valid] * np.sin(angles[valid])
    half_x, half_y = robot.length / 2 + robot.guard_margin, robot.width / 2 + robot.guard_margin
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

    def __init__(self, model, lookahead=1.5, carrot_clearance=0.4, action_dim=2, target='carrot',
                 action_delay=0, robot=ROBOTS['martha']):
        self.model = model
        self.robot = robot
        self.lookahead = lookahead
        self.carrot_clearance = carrot_clearance
        self.action_dim = action_dim
        self.target = target
        # Ticks each command is held back: a robot that obeys at once (Gazebo's mecanum)
        # gets the lag of the one the policy was trained on.
        self.action_delay = action_delay
        # An LSTM policy needs its hidden state carried from one tick to the next.
        self.recurrent = is_recurrent(model)
        self.reset()

    def reset(self):
        """Start a new episode: forget the route, the previous action and the LSTM state."""
        self.prev_action = np.zeros(self.action_dim)
        self.progress = None
        self.lstm_state, self.episode_start = None, True
        self.pending = [(0.0,) * self.action_dim] * self.action_delay

    def _follow(self, path):
        """A new goal starts a new episode; a replan to the same goal only swaps the route."""
        if self.progress is not None and not np.allclose(path.points[-1],
                                                         self.progress.path.points[-1]):
            self.reset()
        if self.progress is None:
            self.progress = RouteProgress(path)
        elif path is not self.progress.path:
            self.progress.reroute(path)

    def compute(self, path, pose, ranges, angles, velocity):
        """pose is (x, y, yaw) in the map frame.

        Returns (v, w, info), or (vx, vy, w, info) with the holonomic action space.
        """
        x, y, yaw = pose
        blocked = footprint_blocked(ranges, angles, self.robot)
        self._follow(path)
        # The progress along the route places the carrot.
        self.progress.update(x, y)
        if self.target == 'goal':
            point, scale = path.points[-1], GOAL_MAX
        else:
            points = self._scan_points(ranges, angles, pose)
            point, _ = carrot(path, self.progress.s, self.lookahead, points, self.carrot_clearance)
            scale = WAYPOINT_MAX
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy)
        obs = build_observation(ranges, angles, velocity, rel, self.prev_action, scale,
                                robot=self.robot)
        if self.recurrent:
            action, self.lstm_state = self.model.predict(
                obs, state=self.lstm_state, episode_start=np.array([self.episode_start]),
                deterministic=True)
            self.episode_start = False
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        self.prev_action = action
        self.pending.append(action_to_cmd(action, self.robot))
        commands = list(self.pending.pop(0))
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
        return (*commands, {'blocked': sorted(blocked), 's': self.progress.s, 'carrot': point})

    def _scan_points(self, ranges, angles, pose):
        """Scan hits in map coordinates, for the carrot's obstacle skipping."""
        x, y, yaw = pose
        ranges = np.asarray(ranges, dtype=float)
        angles = np.asarray(angles, dtype=float)
        hit = np.isfinite(ranges) & (ranges > 0) & (ranges < self.robot.lidar_range)
        if not hit.any():
            return np.empty((0, 2))
        ox = x + self.robot.lidar_offset_x * np.cos(yaw)
        oy = y + self.robot.lidar_offset_x * np.sin(yaw)
        world = yaw + angles[hit]
        return np.stack([ox + ranges[hit] * np.cos(world), oy + ranges[hit] * np.sin(world)], axis=1)
