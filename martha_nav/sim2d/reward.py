"""Reward terms. Every term is logged apart."""
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


def compute_reward(progress_gain, reached, collided, cfg=RewardConfig()):
    """Return (total, terms). progress_gain is in metres; see RewardConfig.progress_mode."""
    gain = progress_gain if cfg.progress_mode == 'geodesic' else max(progress_gain, 0.0)
    terms = {
        'progress': cfg.progress * gain,
        'goal': cfg.goal if reached else 0.0,
        'collision': cfg.collision if collided else 0.0,
        'step': cfg.step,
    }
    return sum(terms.values()), terms
