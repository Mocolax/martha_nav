"""Gymnasium environment: a robot following a carrot along an A* route."""
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from martha_nav.robots import ROBOTS
from martha_nav.sim2d.dynamics import DT, Dynamics, DynamicsRanges, sample_params
from martha_nav.sim2d.geometry import footprint_collides, footprint_points, raycast
from martha_nav.sim2d.observation import (GOAL_MAX, WAYPOINT_MAX, action_to_cmd, build_observation,
                                          obs_dim)
from martha_nav.sim2d.planner import DistanceField, RouteProgress, carrot
from martha_nav.sim2d.reward import RewardConfig, compute_reward
from martha_nav.sim2d.scenarios import ScenarioConfig, generate

TRAIN_SEED_LIMIT = 1_000_000   # training episode seeds are < this; evaluation seeds are >=


def eval_seeds(n, offset=0):
    """Evaluation episode seeds; disjoint from training seeds by construction."""
    return [TRAIN_SEED_LIMIT + offset + i for i in range(n)]


def episode_steps(route_length, robot=ROBOTS['martha']):
    """Control steps before a timeout: three times the route at full speed, plus 10 s."""
    return int(np.ceil((3 * route_length / robot.v_max + 10) / DT))


@dataclass
class EnvConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    dynamics: DynamicsRanges = field(default_factory=DynamicsRanges)
    reward: RewardConfig = field(default_factory=RewardConfig)
    carrot_range: tuple = (1.0, 2.5)
    carrot_clearance: float = 0.4
    goal_tolerance: float = 0.3
    no_progress_time: float = 15.0
    n_rays: int = 180
    lidar_noise: tuple = (0.01, 0.02)
    lidar_dropout: float = 0.01
    vel_noise: float = 0.05
    # 2 -> (v, w); 3 -> (vx, vy, w), which uses Martha's mecanum wheels sideways.
    action_dim: int = 2
    # 'carrot': a point on the A* route ahead of the robot. 'goal': the goal itself, so the
    # deployed policy needs no global planner. The route still shapes the reward in training.
    target: str = 'carrot'
    robot: str = 'martha'        # a key of martha_nav.robots.ROBOTS
    episode_seeds: tuple = ()    # evaluation: play exactly these seeds, in order
    record_trajectory: bool = False  # evaluation: the pose every step, in the episode's info


class NavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=None):
        self.cfg = cfg or EnvConfig()
        self.robot = ROBOTS[self.cfg.robot]
        if self.cfg.action_dim == 3 and not self.robot.holonomic:
            raise ValueError(f'{self.robot.name} cannot slide sideways: use action_dim 2')
        self.footprint = footprint_points(self.robot)
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim(self.cfg.action_dim),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.cfg.action_dim,), np.float32)
        self.ray_angles = np.linspace(-np.pi, np.pi, self.cfg.n_rays, endpoint=False)
        self._seed_index = 0

    # ---- episode ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.cfg.episode_seeds:
            seeds = self.cfg.episode_seeds
            self.episode_seed = int(seeds[self._seed_index % len(seeds)])
            self._seed_index += 1
        else:
            self.episode_seed = int(self.np_random.integers(TRAIN_SEED_LIMIT))
        self.sc = generate(self.episode_seed, self.cfg.scenario)
        self.rng = np.random.default_rng([self.episode_seed, 1])
        self.dyn = Dynamics(sample_params(self.rng, self.cfg.dynamics))
        self.dyn.reset(self.sc.start)
        self.lookahead = self.rng.uniform(*self.cfg.carrot_range)
        self.lidar_sigma = self.rng.uniform(*self.cfg.lidar_noise)
        # A LiDAR slower than the control loop repeats its last scan in between (random phase).
        self.scan_every = max(1, round(1 / (self.robot.lidar_rate * DT)))
        self.scan_phase = int(self.rng.integers(self.scan_every)) if self.scan_every > 1 else 0
        self.progress = RouteProgress(self.sc.path)
        self.steps = 0
        self.max_steps = episode_steps(self.sc.path.length, self.robot)
        self.prev_action = np.zeros(self.cfg.action_dim)
        self.travelled = 0.0
        self.terms = defaultdict(float)
        self.trajectory = []
        self._record()
        if self.cfg.reward.progress_mode == 'geodesic':
            # Cropped around the route: the whole world costs up to 50 ms per reset.
            lo = self.sc.path.points.min(axis=0) - 2.0
            hi = self.sc.path.points.max(axis=0) + 2.0
            self.field = DistanceField(self.sc.full.crop(lo[0], lo[1], hi[0], hi[1]), self.sc.goal)
            self.geo = self.field(*self.sc.start[:2])
            if self.geo is None:
                self.geo = self.sc.shortest
        self._scan()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        before = self.dyn.pose[:2].copy()
        commands = action_to_cmd(action, self.robot)
        step = self.dyn.step_holonomic if len(commands) == 3 else self.dyn.step
        collided = step(*commands,
                        lambda x, y, t: footprint_collides(self.sc.full, x, y, t, self.footprint))
        x, y, _ = self.dyn.pose
        self.travelled += float(np.hypot(*(self.dyn.pose[:2] - before)))
        self.steps += 1
        self._record()
        gain = self.progress.update(x, y)
        to_goal = np.hypot(x - self.sc.goal[0], y - self.sc.goal[1])
        reached = not collided and to_goal < self.cfg.goal_tolerance
        stalled = (not (collided or reached)
                   and self.progress.seconds_without_progress >= self.cfg.no_progress_time)
        if (self.steps + self.scan_phase) % self.scan_every == 0:
            self._scan()
        reward, terms = compute_reward(self._reward_progress(gain, x, y), reached, collided,
                                       self.cfg.reward)
        for k, v in terms.items():
            self.terms[k] += v
        self.prev_action = action
        # A stall truncates the episode: its value is bootstrapped, not punished.
        terminated = bool(collided or reached)
        truncated = not terminated and (self.steps >= self.max_steps or stalled)
        info = {}
        if terminated or truncated:
            outcome = ('collision' if collided else 'success' if reached
                       else 'stalled' if stalled else 'timeout')
            info = self._summary(outcome)
        return self._obs(), float(reward), terminated, truncated, info

    def _record(self):
        if self.cfg.record_trajectory:
            x, y, yaw = self.dyn.pose
            self.trajectory.append({'t': self.steps * DT, 'x': x, 'y': y, 'yaw': yaw})

    def _reward_progress(self, route_gain, x, y):
        """Metres of progress for the reward. The stall rule keeps using the route."""
        if self.cfg.reward.progress_mode != 'geodesic':
            return route_gain
        d = self.field(x, y)
        if d is None:            # off the field for a step: no credit, keep the last value
            return 0.0
        gain, self.geo = self.geo - d, d
        return gain

    # ---- sensing ---------------------------------------------------------
    def _scan(self):
        x, y, th = self.dyn.pose
        reach, offset = self.robot.lidar_range, self.robot.lidar_offset_x
        ox, oy = x + offset * np.cos(th), y + offset * np.sin(th)
        r = raycast(self.sc.full, ox, oy, th + self.ray_angles, reach)
        r = r + self.rng.normal(0.0, self.lidar_sigma, r.shape)
        r[self.rng.random(r.shape) < self.cfg.lidar_dropout] = reach
        if self.robot.lidar_min > 0:
            r[r < self.robot.lidar_min] = reach        # too close to measure, as the real sensor
        self.ranges = np.clip(r, 0.0, reach)
        hit = self.ranges < reach
        a = th + self.ray_angles[hit]
        self.scan_points = np.stack([ox + self.ranges[hit] * np.cos(a),
                                     oy + self.ranges[hit] * np.sin(a)], axis=1)

    def _obs(self):
        x, y, th = self.dyn.pose
        if self.cfg.target == 'goal':
            point, scale = self.sc.goal, GOAL_MAX
        else:
            point, _ = carrot(self.sc.path, self.progress.s, self.lookahead, self.scan_points,
                              self.cfg.carrot_clearance)
            scale = WAYPOINT_MAX
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy)
        noise = 1.0 + self.rng.normal(0.0, self.cfg.vel_noise, 3)
        vel = ((self.dyn.v * noise[0], self.dyn.vy * noise[1], self.dyn.w * noise[2])
               if self.cfg.action_dim == 3 else (self.dyn.v * noise[0], self.dyn.w * noise[2]))
        return build_observation(self.ranges, self.ray_angles, vel, rel, self.prev_action,
                                 scale, robot=self.robot)

    def _summary(self, outcome):
        success = outcome == 'success'
        spl = self.sc.shortest / max(self.sc.shortest, self.travelled) if success else 0.0
        info = {
            'outcome': outcome,
            'episode_seed': self.episode_seed,
            'source': self.sc.source,
            'n_obstacles': len(self.sc.obstacles),
            'obstacles_dropped': self.sc.obstacles_dropped,
            'route_length': self.sc.path.length,
            'shortest': self.sc.shortest,
            'travelled': self.travelled,
            'spl': spl,
            'steps': self.steps,
        }
        info.update({f'r_{k}': v for k, v in self.terms.items()})
        if self.cfg.record_trajectory:
            info['trajectory'] = self.trajectory
        return info
