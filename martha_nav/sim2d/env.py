"""Gymnasium environment: Martha following a carrot along an A* route."""
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from martha_nav.sim2d.dynamics import DT, Dynamics, DynamicsRanges, sample_params
from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, footprint_collides, raycast
from martha_nav.sim2d.observation import (LIDAR_MAX, V_MAX, action_to_cmd, build_observation,
                                          obs_dim)
from martha_nav.sim2d.planner import carrot
from martha_nav.sim2d.reward import RewardConfig, compute_reward
from martha_nav.sim2d.scenarios import ScenarioConfig, generate

TRAIN_SEED_LIMIT = 1_000_000   # training episode seeds are < this; evaluation seeds are >=


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
    lidar_encoding: str = 'inverse'  # see observation.encode_lidar; 'linear' in full_cnn_s0
    # 2 -> (v, w); 3 -> (vx, vy, w), which uses Martha's mecanum wheels sideways.
    action_dim: int = 2
    # Adds the time without advancing to the observation, so a memoryless policy can
    # tell that it is blocked instead of rediscovering the same dead end every step.
    stuck_signal: bool = False
    episode_seeds: tuple = ()    # evaluation: play exactly these seeds, in order


class NavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=None):
        self.cfg = cfg or EnvConfig()
        self.observation_space = spaces.Box(
            -1.0, 1.0, (obs_dim(self.cfg.action_dim, self.cfg.stuck_signal),), np.float32)
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
        self.dyn = Dynamics(sample_params(self.rng, self.cfg.dynamics),
                            holonomic=self.cfg.action_dim == 3)
        self.dyn.reset(self.sc.start)
        self.lookahead = self.rng.uniform(*self.cfg.carrot_range)
        self.lidar_sigma = self.rng.uniform(*self.cfg.lidar_noise)
        self.s = self.s_best = 0.0
        self.steps = self.since_progress = 0
        self.max_steps = int(np.ceil((3 * self.sc.path.length / V_MAX + 10) / DT))
        self.prev_action = np.zeros(self.cfg.action_dim)
        self.travelled = 0.0
        self.terms = defaultdict(float)
        self._scan()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        before = self.dyn.pose[:2].copy()
        commands = action_to_cmd(action)
        step = self.dyn.step_holonomic if len(commands) == 3 else self.dyn.step
        collided = step(*commands, lambda x, y, t: footprint_collides(self.sc.full, x, y, t))
        x, y, _ = self.dyn.pose
        self.travelled += float(np.hypot(*(self.dyn.pose[:2] - before)))
        self.steps += 1
        self.s = self.sc.path.project(x, y, s_hint=self.s)
        gain = max(0.0, self.s - self.s_best)
        self.s_best = max(self.s_best, self.s)
        self.since_progress = 0 if gain > 1e-3 else self.since_progress + 1
        to_goal = np.hypot(x - self.sc.goal[0], y - self.sc.goal[1])
        reached = not collided and to_goal < self.cfg.goal_tolerance
        stalled = not (collided or reached) and self.since_progress * DT >= self.cfg.no_progress_time
        self._scan()
        reward, terms = compute_reward(gain, reached, collided, float(self.ranges.min()),
                                       action[-1] - self.prev_action[-1], self.cfg.reward, stalled)
        for k, v in terms.items():
            self.terms[k] += v
        self.prev_action = action
        # A stall is free (truncated, bootstrapped) unless it is penalised; then it is terminal.
        stall_ends = stalled and self.cfg.reward.stalled != 0.0
        terminated = collided or reached or stall_ends
        truncated = not terminated and (self.steps >= self.max_steps or stalled)
        info = {}
        if terminated or truncated:
            outcome = ('collision' if collided else 'success' if reached
                       else 'stalled' if stalled else 'timeout')
            info = self._summary(outcome)
        return self._obs(), float(reward), terminated, truncated, info

    # ---- sensing ---------------------------------------------------------
    def _scan(self):
        x, y, th = self.dyn.pose
        ox, oy = x + LIDAR_OFFSET_X * np.cos(th), y + LIDAR_OFFSET_X * np.sin(th)
        r = raycast(self.sc.full, ox, oy, th + self.ray_angles, LIDAR_MAX)
        r = r + self.rng.normal(0.0, self.lidar_sigma, r.shape)
        r[self.rng.random(r.shape) < self.cfg.lidar_dropout] = LIDAR_MAX
        self.ranges = np.clip(r, 0.0, LIDAR_MAX)
        hit = self.ranges < LIDAR_MAX
        a = th + self.ray_angles[hit]
        self.scan_points = np.stack([ox + self.ranges[hit] * np.cos(a),
                                     oy + self.ranges[hit] * np.sin(a)], axis=1)

    def _obs(self):
        x, y, th = self.dyn.pose
        point, _ = carrot(self.sc.path, self.s, self.lookahead, self.scan_points,
                          self.cfg.carrot_clearance)
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy)
        noise = 1.0 + self.rng.normal(0.0, self.cfg.vel_noise, 3)
        vel = ((self.dyn.v * noise[0], self.dyn.vy * noise[1], self.dyn.w * noise[2])
               if self.cfg.action_dim == 3 else (self.dyn.v * noise[0], self.dyn.w * noise[2]))
        stuck = (min(self.since_progress * DT / self.cfg.no_progress_time, 1.0)
                 if self.cfg.stuck_signal else None)
        return build_observation(self.ranges, self.ray_angles, vel, rel, self.prev_action,
                                 self.cfg.lidar_encoding, stuck)

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
        return info
