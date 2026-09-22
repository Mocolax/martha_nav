"""The single observation contract shared by the 2D env and the ROS nodes."""
import numpy as np

N_SECTORS = 90
LIDAR_MAX = 8.0     # m, RPLIDAR A2M8
WAYPOINT_MAX = 3.0  # m
V_MAX = 0.35        # m/s forward
V_REVERSE = 0.15    # m/s backward
W_MAX = 0.8         # rad/s
OBS_DIM = N_SECTORS + 6


def reduce_scan(ranges, angles):
    """Minimum range per 4-degree sector.

    angles are relative to the robot's forward axis. Sector 0 is centred on
    the front and sectors grow counter-clockwise. Invalid readings (inf, NaN,
    <= 0) and empty sectors count as LIDAR_MAX.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    width = 2 * np.pi / N_SECTORS
    valid = np.isfinite(ranges) & (ranges > 0)
    r = np.where(valid, np.minimum(ranges, LIDAR_MAX), LIDAR_MAX)
    idx = (np.floor(np.mod(angles + width / 2, 2 * np.pi) / width).astype(int)) % N_SECTORS
    out = np.full(N_SECTORS, LIDAR_MAX)
    np.minimum.at(out, idx, r)
    return out


def build_observation(ranges, angles, velocity, waypoint_rel, prev_action):
    """96-value observation in [-1, 1].

    ranges/angles: raw scan, angles relative to the robot's forward axis.
    velocity: measured (v, w). waypoint_rel: (dx, dy) in the robot frame.
    prev_action: last action sent, already in [-1, 1].
    """
    lidar = reduce_scan(ranges, angles) / LIDAR_MAX
    dx, dy = waypoint_rel
    wp = [min(np.hypot(dx, dy), WAYPOINT_MAX) / WAYPOINT_MAX, np.arctan2(dy, dx) / np.pi]
    vel = [velocity[0] / V_MAX, velocity[1] / W_MAX]
    obs = np.concatenate([lidar, wp, vel, np.asarray(prev_action, dtype=float)])
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def action_to_cmd(action):
    """[-1, 1]^2 -> (v, w); reverse is capped lower than forward on purpose."""
    a_v, a_w = np.clip(action, -1.0, 1.0)
    v = a_v * (V_MAX if a_v >= 0 else V_REVERSE)
    return float(v), float(a_w * W_MAX)
