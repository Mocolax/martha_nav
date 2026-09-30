import numpy as np

from martha_nav.learning.evaluate import run_episodes, summarize, wilson
from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig, eval_seeds
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


def test_evaluation_uses_the_env_the_model_was_trained_with(tmp_path):
    from martha_nav.learning.evaluate import trained_env_config
    model = tmp_path / 'best_model.zip'
    assert trained_env_config(model) == EnvConfig()                             # no config: defaults
    (tmp_path / 'config.yaml').write_text('env:\n  n_rays: 180\n')
    old = trained_env_config(model)                                             # before the options
    assert (old.lidar_encoding, old.action_dim, old.stuck_signal, old.target) == (
        'linear', 2, False, 'carrot')
    (tmp_path / 'config.yaml').write_text(
        'env:\n  lidar_encoding: inverse\n  action_dim: 3\n  stuck_signal: true\n  target: goal\n')
    new = trained_env_config(model)
    assert (new.lidar_encoding, new.action_dim, new.stuck_signal, new.target) == (
        'inverse', 3, True, 'goal')


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


def test_recurrent_models_keep_their_hidden_state():
    """A recurrent policy is called with its state and the episode-start flags."""
    import numpy as np

    from martha_nav.learning.evaluate import run_episodes

    class Recurrent:
        policy = type('P', (), {'lstm_actor': object()})()

        def __init__(self):
            self.saw_state, self.saw_starts = False, False

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            self.saw_state = self.saw_state or state is not None
            self.saw_starts = self.saw_starts or episode_start is not None
            ang = obs[:, 91]
            action = np.stack([np.where(np.abs(ang) < 0.25, 1.0, 0.0), np.clip(4 * ang, -1, 1)], 1)
            return action, ('state',)

    model = Recurrent()
    rows = run_episodes(model, OPEN, eval_seeds(4), n_envs=2)
    assert len(rows) == 4 and model.saw_state and model.saw_starts


def test_cli_names_the_csv_after_its_sources_and_honours_episodes(tmp_path, monkeypatch):
    """--episodes is never overridden, and a CSV is only called 'train' for the training set."""
    from martha_nav.learning import evaluate
    from martha_nav.sim2d.scenarios import point_pairs
    played = []
    monkeypatch.setattr(evaluate, 'load_model', lambda path: None)
    monkeypatch.setattr(evaluate, 'run_episodes', lambda model, cfg, seeds, n_envs: played.append(
        len(seeds)) or [{'episode_seed': s, 'outcome': 'success', 'spl': 1.0} for s in seeds])
    model = str(tmp_path / 'best_model.zip')
    assert evaluate.main(['--model', model, '--episodes', '3']) is None     # evaluate_2d exits 0
    evaluate.main(['--model', model, '--episodes', '3', '--sources', 'room', 'hall', 'tube'])
    evaluate.main(['--model', model, '--points', 'lab', '--episodes', '500'])
    evaluate.main(['--model', model, '--points', 'lab'])
    assert played == [3, 3, 500, len(point_pairs('lab'))]
    assert sorted(p.name for p in tmp_path.glob('eval_*.csv')) == [
        'eval_obstacles_lab-points.csv', 'eval_obstacles_room-hall-tube.csv',
        'eval_obstacles_train.csv']
