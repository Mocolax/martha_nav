"""The single observation contract shared by the 2D env and the ROS nodes."""
import numpy as np

from martha_nav.robots import ROBOTS

N_SECTORS = 90
WAYPOINT_MAX = 3.0  # m, carrot distance scale
GOAL_MAX = 12.0     # m, goal distance scale when the policy sees the goal instead (longest route)
OBS_DIM = N_SECTORS + 6          # the default (vx, w) action space


def obs_dim(action_dim=2):
    """90 LiDAR sectors, the waypoint, the measured velocity and the previous action."""
    return N_SECTORS + 2 + 2 * action_dim


def reduce_scan(ranges, angles, max_range=8.0):
    """Minimum range per 4-degree sector.

    angles are relative to the robot's forward axis. Sector 0 is centred on
    the front and sectors grow counter-clockwise. Invalid readings (inf, NaN,
    <= 0) and empty sectors count as max_range.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    width = 2 * np.pi / N_SECTORS
    valid = np.isfinite(ranges) & (ranges > 0)
    r = np.where(valid, np.minimum(ranges, max_range), max_range)
    idx = (np.floor(np.mod(angles + width / 2, 2 * np.pi) / width).astype(int)) % N_SECTORS
    out = np.full(N_SECTORS, max_range)
    np.minimum.at(out, idx, r)
    return out


def encode_lidar(sectors):
    """Metres -> [0, 1) as d / (d + 1 m): finer up close, where it matters."""
    return sectors / (sectors + 1.0)


def build_observation(ranges, angles, velocity, waypoint_rel, prev_action,
                      waypoint_max=WAYPOINT_MAX, robot=ROBOTS['martha']):
    """96-value observation in [-1, 1] (98 with the holonomic action space).

    ranges/angles: raw scan, angles relative to the robot's forward axis.
    velocity: measured (v, w). waypoint_rel: (dx, dy) in the robot frame.
    prev_action: last action sent, already in [-1, 1].
    waypoint_max: distance scale, WAYPOINT_MAX for the carrot or GOAL_MAX for the goal.
    robot: the profile whose LiDAR range and velocity limits scale the values.
    """
    lidar = encode_lidar(reduce_scan(ranges, angles, robot.lidar_range))
    dx, dy = waypoint_rel
    wp = [min(np.hypot(dx, dy), waypoint_max) / waypoint_max, np.arctan2(dy, dx) / np.pi]
    vel = ([velocity[0] / robot.v_max, velocity[1] / robot.v_lateral, velocity[2] / robot.w_max]
           if len(velocity) == 3 else [velocity[0] / robot.v_max, velocity[1] / robot.w_max])
    obs = np.concatenate([lidar, wp, vel, np.asarray(prev_action, dtype=float)])
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def action_to_cmd(action, robot=ROBOTS['martha']):
    """[-1, 1]^n -> velocities: (v, w), or (vx, vy, w) with the holonomic space."""
    values = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    a_v = values[0]
    v = float(a_v * (robot.v_max if a_v >= 0 else robot.v_reverse))
    if len(values) == 2:
        return v, float(values[1] * robot.w_max)
    return v, float(values[1] * robot.v_lateral), float(values[2] * robot.w_max)
