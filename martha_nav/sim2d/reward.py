"""Reward terms. Edit RewardConfig to experiment; every term is logged apart."""
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
    proximity: float = 0.0       # optional, off by default
    proximity_dist: float = 0.5  # m
    turn: float = 0.0            # optional, off by default; per unit of |delta a_w|
    stalled: float = 0.0         # optional; when non-zero a stall ends the episode (terminal)


def compute_reward(progress_gain, reached, collided, min_range, delta_turn, cfg=RewardConfig(),
                   stalled=False):
    """Return (total, terms). progress_gain is in metres; see RewardConfig.progress_mode."""
    gain = progress_gain if cfg.progress_mode == 'geodesic' else max(progress_gain, 0.0)
    terms = {
        'progress': cfg.progress * gain,
        'goal': cfg.goal if reached else 0.0,
        'collision': cfg.collision if collided else 0.0,
        'step': cfg.step,
        'proximity': -cfg.proximity * max(0.0, 1.0 - min_range / cfg.proximity_dist),
        'turn': -cfg.turn * abs(delta_turn),
        'stalled': cfg.stalled if stalled else 0.0,
    }
    return sum(terms.values()), terms
