import numpy as np
import pytest

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
    (tmp_path / 'config.yaml').write_text(
        'env:\n  lidar_encoding: inverse\n  action_dim: 3\n  target: goal\n')
    cfg = trained_env_config(model)
    assert (cfg.action_dim, cfg.target) == (3, 'goal')


def test_evaluation_drives_the_robot_the_model_was_trained_for(tmp_path):
    from martha_nav.learning.evaluate import trained_env_config
    model = tmp_path / 'best_model.zip'
    (tmp_path / 'config.yaml').write_text('env:\n  lidar_encoding: inverse\n')
    assert trained_env_config(model).robot == 'martha'
    (tmp_path / 'config.yaml').write_text('env:\n  robot: burger\n')
    cfg = trained_env_config(model)
    assert cfg.robot == 'burger' and cfg.scenario.inflation == 0.20


def test_a_model_trained_for_another_profile_of_its_robot_is_refused(tmp_path):
    from dataclasses import asdict

    import yaml

    from martha_nav.learning.evaluate import trained_env_config
    from martha_nav.robots import ROBOTS
    model = tmp_path / 'best_model.zip'
    profile = asdict(ROBOTS['burger'])

    def write(**extra):
        (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'burger'}, **extra}))

    write(robot_profile=profile)
    assert trained_env_config(model).robot == 'burger'
    write()                                                        # run from before profiles
    assert trained_env_config(model).robot == 'burger'
    write(robot_profile={**profile, 'guard_margin': 0.15})
    with pytest.raises(ValueError, match=r'different burger profile \(changed: guard_margin\)'):
        trained_env_config(model)


def test_a_model_trained_with_a_removed_option_is_refused(tmp_path):
    import pytest

    from martha_nav.learning.evaluate import trained_env_config
    model = tmp_path / 'best_model.zip'
    for env in ('lidar_encoding: linear', 'lidar_encoding: inverse\n  stuck_signal: true',
                'n_rays: 180'):                    # before config.yaml recorded the encoding
        (tmp_path / 'config.yaml').write_text(f'env:\n  {env}\n')
        with pytest.raises(ValueError, match='no longer'):
            trained_env_config(model)


def test_cli_builds_the_point_pairs_with_the_robots_inflation(tmp_path, monkeypatch):
    from martha_nav.learning import evaluate
    inflations, real = [], evaluate.point_pairs

    def spy(world, **kwargs):
        inflations.append(kwargs.get('inflation'))
        return real(world, **kwargs)

    monkeypatch.setattr(evaluate, 'point_pairs', spy)
    monkeypatch.setattr(evaluate, 'load_model', lambda path: None)
    monkeypatch.setattr(evaluate, 'run_episodes', lambda model, cfg, seeds, n_envs: [
        {'episode_seed': s, 'outcome': 'success', 'spl': 1.0, 'trajectory': []} for s in seeds])
    (tmp_path / 'config.yaml').write_text('env:\n  robot: burger\n')
    evaluate.main(['--model', str(tmp_path / 'best_model.zip'), '--points', 'lab', '--episodes', '2'])
    assert inflations == [0.20]


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
        len(seeds)) or [{'episode_seed': s, 'outcome': 'success', 'spl': 1.0, 'trajectory': []}
                         for s in seeds])
    model = str(tmp_path / 'best_model.zip')
    assert evaluate.main(['--model', model, '--episodes', '3']) is None     # evaluate_2d exits 0
    evaluate.main(['--model', model, '--episodes', '3', '--sources', 'room', 'hall', 'tube'])
    evaluate.main(['--model', model, '--points', 'lab', '--episodes', '500'])
    evaluate.main(['--model', model, '--points', 'lab'])
    assert played == [3, 3, 500, len(point_pairs('lab'))]
    side = ('_traj.csv', '_route.csv', '_obstacles.csv')
    assert sorted(p.name for p in tmp_path.glob('eval_*.csv') if not p.name.endswith(side)) == [
        'eval_obstacles_lab-points.csv', 'eval_obstacles_room-hall-tube.csv',
        'eval_obstacles_train.csv']


def test_evaluation_uses_the_dynamics_the_model_was_trained_with(tmp_path):
    import json
    from dataclasses import asdict

    import yaml

    from martha_nav.learning.evaluate import trained_env_config
    from martha_nav.sim2d.dynamics import WIDE_DYNAMICS, DynamicsRanges
    model = tmp_path / 'best_model.zip'
    env = json.loads(json.dumps(asdict(EnvConfig(dynamics=WIDE_DYNAMICS))))   # as train saves it
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': env}))
    assert trained_env_config(model).dynamics == WIDE_DYNAMICS
    del env['dynamics']['delay']                                     # runs from before the delay
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': env}))
    assert trained_env_config(model).dynamics.delay == DynamicsRanges().delay


def test_cli_writes_each_episode_trajectory_route_and_obstacles(tmp_path, monkeypatch):
    """Beside the per-episode CSV: what the robot did, the A* route it was given and the boxes."""
    import csv

    from martha_nav.learning import evaluate
    from martha_nav.sim2d.scenarios import generate
    monkeypatch.setattr(evaluate, 'load_model', lambda path: Pursuit())
    model = str(tmp_path / 'best_model.zip')
    evaluate.main(['--model', model, '--episodes', '2', '--sources', 'open_room', '--n-envs', '1'])
    read = lambda name: list(csv.DictReader(open(tmp_path / name)))
    rows = read('eval_obstacles_open_room.csv')
    assert 'trajectory' not in rows[0]
    traj = read('eval_obstacles_open_room_traj.csv')
    route = read('eval_obstacles_open_room_route.csv')
    obstacles = read('eval_obstacles_open_room_obstacles.csv')
    seed = rows[0]['episode_seed']
    steps = [p for p in traj if p['episode_seed'] == seed]
    assert len(steps) == int(rows[0]['steps']) + 1
    sc = generate(int(seed), evaluate.trained_env_config(model).scenario.__class__(
        sources=('open_room',), obstacle_mode='always'))
    first = next(p for p in route if p['episode_seed'] == seed and p['i'] == '0')
    assert (float(first['x']), float(first['y'])) == pytest.approx(tuple(sc.path.points[0]))
    assert len([o for o in obstacles if o['episode_seed'] == seed]) == len(sc.obstacles)
