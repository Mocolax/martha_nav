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
    posts: tuple = ()          # ((x, y, sx, sy), ...) m: rectangular posts of the robot's own
                               # structure that cross the LiDAR's scan plane, centre and size in
                               # base_link. The LiDAR sees them; the safety guard ignores them
    post_pad: float = 0.05     # rad (~3 deg) the posts' shadow is widened by in the observation:
                               # the mounting yaw and the posts' real placement

    @property
    def holonomic(self):
        return self.v_lateral > 0


ROBOTS = {
    # Mecanum wheels, RPLIDAR A2M8. Reverse and sideways are capped lower than forward.
    'martha': Robot('martha', length=0.56, width=0.41, footprint_offset_x=0.0,
                    lidar_offset_x=0.2325, lidar_range=8.0, lidar_min=0.0, lidar_rate=10.0,
                    v_max=0.35, v_reverse=0.15, v_lateral=0.25, w_max=0.8, inflation=0.40,
                    guard_margin=0.05),
}

# Martha as built, with the aluminium tower and the claw lift (tools/check_tower_posts.py checks
# it against the LiDAR, tools/align_lidar.py turns the LiDAR straight). Four 20 x 40 mm posts,
# the long side along x, cross the LiDAR's scan plane behind it: the front pair centred at
# x = -0.033 m, the rear pair at -0.208 m, y = +-0.11 m (measured 2026-10-21 with the LiDAR and
# a tape). Since 2026-11-24 (150 raw scans) the LiDAR sits 0.195 m ahead of base_link, turned
# 180 degrees, and the claw lift's two motors stand on the front posts, fitted to what the A2M8
# sees: the left one 0.122-0.166 m to the side with its face at x = 0.025 m, the right one
# 0.125-0.157 m to the side reaching x = 0.057 m. On the left, between the posts, a part along
# their inner face returns a reading now and then (0.34 m from the LiDAR): a 1 cm box. The body
# and the footprint are Martha's.
ROBOTS['martha_tower'] = replace(
    ROBOTS['martha'], name='martha_tower', lidar_offset_x=0.195,
    posts=tuple((x, y, 0.04, 0.02) for x in (-0.033, -0.208) for y in (0.11, -0.11))
    + ((0.006, 0.144, 0.038, 0.044), (0.022, -0.141, 0.070, 0.032),
       (-0.1205, 0.095, 0.135, 0.010)))
# The same Martha forbidden to reverse: the posts shadow most of what is behind it, and it can
# still turn in place or slide sideways to get out of a dead end.
ROBOTS['martha_tower_fwd'] = replace(ROBOTS['martha_tower'], name='martha_tower_fwd', v_reverse=0.0)


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

    The inflation is left out: it plans the routes the model follows, it is not part of the
    trained model."""
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
