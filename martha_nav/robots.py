"""The robots the policy can drive: everything the code assumes about the body and the LiDAR."""
import json
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Robot:
    name: str
    length: float              # m, contact rectangle along x of base_link
    width: float               # m, along y
    footprint_offset_x: float  # m, centre of that rectangle in base_link
    lidar_offset_x: float      # m, LiDAR position in base_link
    lidar_range: float         # m, farthest reading; beyond it the scan reads "nothing"
    lidar_min: float           # m, nearest reading; closer also reads "nothing" (0: no limit)
    lidar_rate: float          # Hz, scans per second
    v_max: float               # m/s forward
    v_reverse: float           # m/s backward
    v_lateral: float           # m/s sideways; 0 for a robot that cannot slide
    w_max: float               # rad/s
    inflation: float           # m, obstacle inflation of the global planner; not part of a
                               # trained model: it plans the routes the model follows
    guard_margin: float        # m, how far beyond the contact rectangle the safety guard reacts
    posts: tuple = ()          # ((x, y, side), ...) m: square posts of the robot's own structure
                               # that cross the LiDAR's scan plane, in base_link. The LiDAR sees
                               # them; the safety guard ignores them

    @property
    def holonomic(self):
        return self.v_lateral > 0


ROBOTS = {
    # Mecanum wheels, RPLIDAR A2M8. Reverse and sideways are capped lower than forward.
    'martha': Robot('martha', length=0.56, width=0.41, footprint_offset_x=0.0,
                    lidar_offset_x=0.2325, lidar_range=8.0, lidar_min=0.0, lidar_rate=10.0,
                    v_max=0.35, v_reverse=0.15, v_lateral=0.25, w_max=0.8, inflation=0.40,
                    guard_margin=0.05),
    # TurtleBot3 Burger: turtlebot3_burger.urdf (body 0.140 m centred at x = -0.032, wheels
    # 0.178 m across, base_scan at x = -0.032) and its spec sheet (0.22 m/s; 2.84 rad/s, capped
    # at 1.5 so the action keeps its resolution). LDS-01: 3.5 m, 0.12 m, 5 Hz (an LDS-02 is
    # 8.0 m and 0.16 m). Its guard reaches 0.17 m ahead of the LiDAR, past the 0.12 m blind zone.
    'burger': Robot('burger', length=0.14, width=0.178, footprint_offset_x=-0.032,
                    lidar_offset_x=-0.032, lidar_range=3.5, lidar_min=0.12, lidar_rate=5.0,
                    v_max=0.22, v_reverse=0.22, v_lateral=0.0, w_max=1.5, inflation=0.40,
                    guard_margin=0.10),
}

# Martha with the aluminium tower: four 20 x 20 mm posts 0.28 m apart (outer faces) along x
# and 0.24 m along y, rising 0.565 m from the chassis through the LiDAR's scan plane. The
# body and the footprint do not change; the LiDAR now sees the posts behind it.
ROBOTS['martha_tower'] = replace(
    ROBOTS['martha'], name='martha_tower',
    posts=tuple((x, y, 0.02) for x in (0.13, -0.13) for y in (0.11, -0.11)))


def checkpoint_robot(path):
    """The robot a checkpoint was trained for: its config.yaml (.zip) or its settings (.npz)."""
    path = Path(path)
    if path.suffix == '.npz':
        return json.loads(str(np.load(path)['settings']))['robot']
    config = path.with_name('config.yaml')
    if not config.exists():
        return 'martha'
    return yaml.safe_load(config.read_text())['env'].get('robot', 'martha')


def check_profile(recorded, name, path):
    """Refuse a checkpoint trained for another version of robot `name` (None: not recorded).

    The inflation is left out: training with 0.30 or 0.40 m gave the same policy, while planning
    with 0.40 m is what helps (docs/resultados.md)."""
    if recorded is None:
        return
    # A field added later with a default (posts) counts as that default in older recordings,
    # and the JSON round trip turns the current tuples into the lists a recording holds.
    defaults = {f.name: f.default for f in fields(Robot) if f.default is not MISSING}
    recorded = json.loads(json.dumps({**defaults, **recorded}))
    current = json.loads(json.dumps(asdict(ROBOTS[name])))
    changed = [k for k in {**current, **recorded}
               if k != 'inflation' and recorded.get(k) != current.get(k)]
    if changed:
        raise ValueError(f"{path} was trained for a different {name} profile (changed: "
                         f"{', '.join(changed)}): retrain and re-export")
