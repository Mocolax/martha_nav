import numpy as np

from martha_nav.learning.evaluate import eval_seeds, run_episodes, summarize, wilson
from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig
from martha_nav.sim2d.scenarios import ScenarioConfig

OPEN = EnvConfig(scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none'))


def test_eval_seeds_are_disjoint_from_training():
    assert min(eval_seeds(10)) >= TRAIN_SEED_LIMIT


def test_wilson_interval():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88


class Pursuit:
    """Stand-in for a trained model (same predict signature as SB3)."""

    def predict(self, obs, deterministic=True):
        ang = obs[:, 91]
        return np.stack([np.where(np.abs(ang) < 0.25, 1.0, 0.0), np.clip(4 * ang, -1, 1)], 1), None


def test_run_episodes_plays_each_seed_once():
    seeds = eval_seeds(6)
    rows = run_episodes(Pursuit(), OPEN, seeds, n_envs=2)
    assert [r['episode_seed'] for r in rows] == seeds
    s = summarize(rows)
    assert s['episodes'] == 6 and s['success'] >= 5 / 6
