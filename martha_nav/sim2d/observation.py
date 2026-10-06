"""The single observation contract shared by the 2D env and the ROS nodes."""
import numpy as np

N_SECTORS = 90
LIDAR_MAX = 8.0     # m, RPLIDAR A2M8
WAYPOINT_MAX = 3.0  # m, carrot distance scale
GOAL_MAX = 12.0     # m, goal distance scale when the policy sees the goal instead (longest route)
V_MAX = 0.35        # m/s forward
V_REVERSE = 0.15    # m/s backward
V_LATERAL = 0.25    # m/s sideways, only with the holonomic action space
W_MAX = 0.8         # rad/s
OBS_DIM = N_SECTORS + 6          # the default (vx, w) action space


def obs_dim(action_dim=2):
    """90 LiDAR sectors, the waypoint, the measured velocity and the previous action."""
    return N_SECTORS + 2 + 2 * action_dim


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


def encode_lidar(sectors):
    """Metres -> [0, 1) as d / (d + 1 m): finer up close, where it matters."""
    return sectors / (sectors + 1.0)


def build_observation(ranges, angles, velocity, waypoint_rel, prev_action,
                      waypoint_max=WAYPOINT_MAX):
    """96-value observation in [-1, 1] (98 with the holonomic action space).

    ranges/angles: raw scan, angles relative to the robot's forward axis.
    velocity: measured (v, w). waypoint_rel: (dx, dy) in the robot frame.
    prev_action: last action sent, already in [-1, 1].
    waypoint_max: distance scale, WAYPOINT_MAX for the carrot or GOAL_MAX for the goal.
    """
    lidar = encode_lidar(reduce_scan(ranges, angles))
    dx, dy = waypoint_rel
    wp = [min(np.hypot(dx, dy), waypoint_max) / waypoint_max, np.arctan2(dy, dx) / np.pi]
    vel = ([velocity[0] / V_MAX, velocity[1] / V_LATERAL, velocity[2] / W_MAX]
           if len(velocity) == 3 else [velocity[0] / V_MAX, velocity[1] / W_MAX])
    obs = np.concatenate([lidar, wp, vel, np.asarray(prev_action, dtype=float)])
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def action_to_cmd(action):
    """[-1, 1]^n -> velocities: (v, w), or (vx, vy, w) with the holonomic space.

    Reverse is capped lower than forward on purpose, and sideways lower than both:
    the mecanum wheels are least efficient moving laterally.
    """
    values = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    a_v = values[0]
    v = float(a_v * (V_MAX if a_v >= 0 else V_REVERSE))
    if len(values) == 2:
        return v, float(values[1] * W_MAX)
    return v, float(values[1] * V_LATERAL), float(values[2] * W_MAX)
