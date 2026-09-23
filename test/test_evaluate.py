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


def test_evaluation_uses_the_lidar_encoding_the_model_was_trained_with(tmp_path):
    from martha_nav.learning.evaluate import _trained_lidar_encoding
    assert _trained_lidar_encoding(tmp_path / 'best_model.zip') == 'inverse'   # no config: default
    (tmp_path / 'config.yaml').write_text('env:\n  n_rays: 180\n')
    assert _trained_lidar_encoding(tmp_path / 'best_model.zip') == 'linear'    # runs before the option
    (tmp_path / 'config.yaml').write_text('env:\n  lidar_encoding: inverse\n')
    assert _trained_lidar_encoding(tmp_path / 'best_model.zip') == 'inverse'


def test_point_mode_builds_one_episode_per_pair():
    from dataclasses import replace

    from martha_nav.sim2d.scenarios import point_pairs
    pairs = point_pairs('lab')
    cfg = replace(OPEN, scenario=replace(OPEN.scenario, sources=('lab',), point_pairs=pairs,
                                         obstacle_mode='always'))
    seeds = eval_seeds(len(pairs))
    rows = run_episodes(Pursuit(), cfg, seeds, n_envs=2)
    assert len(rows) == len(pairs)
    assert {r['source'] for r in rows} == {'lab'}
