"""Reward terms. Edit RewardConfig to experiment; every term is logged apart."""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    progress: float = 1.0        # per metre of new-record route progress
    goal: float = 20.0
    collision: float = -20.0     # -10 in full_cnn_s0; see docs/resultados.md (A/B/C)
    step: float = -0.005
    proximity: float = 0.0       # optional, off by default
    proximity_dist: float = 0.5  # m
    turn: float = 0.0            # optional, off by default; per unit of |delta a_w|
    stalled: float = 0.0         # optional; when non-zero a stall ends the episode (terminal)


def compute_reward(progress_gain, reached, collided, min_range, delta_turn, cfg=RewardConfig(),
                   stalled=False):
    """Return (total, terms). progress_gain is metres beyond the episode's best."""
    terms = {
        'progress': cfg.progress * max(progress_gain, 0.0),
        'goal': cfg.goal if reached else 0.0,
        'collision': cfg.collision if collided else 0.0,
        'step': cfg.step,
        'proximity': -cfg.proximity * max(0.0, 1.0 - min_range / cfg.proximity_dist),
        'turn': -cfg.turn * abs(delta_turn),
        'stalled': cfg.stalled if stalled else 0.0,
    }
    return sum(terms.values()), terms
