"""Reward terms. Every term is logged apart."""
import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    progress: float = 1.0        # per metre of progress, measured as progress_mode says
    # 'route': new-record arc length along the A* route (never negative).
    # 'geodesic': drop in geodesic distance to the goal on the map with the obstacles,
    # potential-based shaping (Ng et al. 1999), so moving away costs what coming back pays.
    progress_mode: str = 'route'
    goal: float = 20.0
    collision: float = -20.0
    step: float = -0.005
    clearance: float = 0.0          # per-step scale; 0 = off
    clearance_mid: float = 0.15     # m from the contact rectangle at half strength
    clearance_steepness: float = 15.0  # 1/m
    # Per metre driven backward. Route progress never goes negative, so without it reversing
    # costs only time, and the LiDAR's blind spot is behind the robot. 0 = off.
    reverse: float = 0.0


def compute_reward(progress_gain, reached, collided, cfg=RewardConfig(), clearance=math.inf,
                   backward=0.0):
    """Return (total, terms). progress_gain and backward (driven in reverse) are in metres."""
    gain = progress_gain if cfg.progress_mode == 'geodesic' else max(progress_gain, 0.0)
    terms = {
        'progress': cfg.progress * gain,
        'goal': cfg.goal if reached else 0.0,
        'collision': cfg.collision if collided else 0.0,
        'step': cfg.step,
    }
    if cfg.clearance > 0:
        # Privileged map clearance is training-only, never part of the observation.
        z = cfg.clearance_steepness * (cfg.clearance_mid - clearance)
        terms['clearance'] = -cfg.clearance * 0.5 * (1 + math.tanh(z / 2))
    if cfg.reverse > 0:
        terms['reverse'] = -cfg.reverse * backward
    return sum(terms.values()), terms
