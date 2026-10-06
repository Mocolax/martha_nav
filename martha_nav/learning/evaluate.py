"""Deterministic evaluation on fixed seeds; CSV rows and a summary with 95% CIs."""
import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from martha_nav.learning.policy import is_recurrent
from martha_nav.sim2d.dynamics import DynamicsRanges
from martha_nav.sim2d.env import EnvConfig, NavEnv, eval_seeds
from martha_nav.sim2d.scenarios import CONDITIONS, TRAIN_SOURCES, generate, point_pairs


def make_vec_env(cfgs):
    fns = [lambda c=c: NavEnv(c) for c in cfgs]
    return DummyVecEnv(fns) if len(fns) == 1 else SubprocVecEnv(fns)


def run_episodes(model, env_cfg, seeds, n_envs=8, deterministic=True):
    """Play every seed exactly once; returns one info dict per episode, in seed order."""
    n_envs = max(1, min(n_envs, len(seeds)))
    chunks = [tuple(c) for c in np.array_split(np.asarray(seeds), n_envs)]
    venv = make_vec_env([replace(env_cfg, episode_seeds=c) for c in chunks])
    done_count = [0] * n_envs
    rows = []
    obs = venv.reset()
    recurrent = is_recurrent(model)
    state, episode_starts = None, np.ones(n_envs, dtype=bool)
    while any(done_count[i] < len(chunks[i]) for i in range(n_envs)):
        if recurrent:
            actions, state = model.predict(obs, state=state, episode_start=episode_starts,
                                           deterministic=deterministic)
        else:
            actions, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = venv.step(actions)
        episode_starts = np.asarray(dones, dtype=bool)
        for i, done in enumerate(dones):
            if done and done_count[i] < len(chunks[i]):
                done_count[i] += 1
                rows.append({k: v for k, v in infos[i].items()
                             if k not in ('terminal_observation', 'TimeLimit.truncated')})
    venv.close()
    return sorted(rows, key=lambda r: r['episode_seed'])


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(rows):
    n = len(rows)
    out = {'episodes': n, 'spl': float(np.mean([r['spl'] for r in rows])) if n else 0.0}
    for outcome in ('success', 'collision', 'timeout', 'stalled'):
        k = sum(r['outcome'] == outcome for r in rows)
        lo, hi = wilson(k, n)
        out[outcome] = k / n if n else 0.0
        out[f'{outcome}_ci'] = (lo, hi)
    return out


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r} - {'trajectory'})   # in its own file
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_episode_files(rows, scenario_cfg, out):
    """Beside out: each episode's trajectory, A* route and obstacles."""
    stem = str(Path(out).with_suffix(''))
    traj, route, obstacles = [], [], []
    for row in rows:
        seed = row['episode_seed']
        traj += [{'episode_seed': seed, **p} for p in row['trajectory']]
        sc = generate(seed, scenario_cfg)
        route += [{'episode_seed': seed, 'i': i, 'x': x, 'y': y}
                  for i, (x, y) in enumerate(sc.path.points)]
        obstacles += [{'episode_seed': seed, **asdict(o)} for o in sc.obstacles]
    write_csv(traj, stem + '_traj.csv')
    write_csv(route, stem + '_route.csv')
    write_csv(obstacles, stem + '_obstacles.csv')


def _run_config(model_path):
    """The config.yaml train wrote next to the model, or {} when there is none."""
    path = Path(model_path).with_name('config.yaml')
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def trained_env_config(model_path):
    """The EnvConfig the model was trained with: its observation must match."""
    config = _run_config(model_path)
    if not config:
        return EnvConfig()
    saved = config['env']
    dynamics = DynamicsRanges(**{k: tuple(v) for k, v in saved.get('dynamics', {}).items()})
    # Defaults for configs that do not record an option.
    return EnvConfig(dynamics=dynamics, lidar_encoding=saved.get('lidar_encoding', 'linear'),
                     action_dim=saved.get('action_dim', 2),
                     stuck_signal=saved.get('stuck_signal', False),
                     target=saved.get('target', 'carrot'))


def load_model(path):
    """PPO, or RecurrentPPO when the run's config says it was trained with an LSTM."""
    if _run_config(path).get('recurrent', False):
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, device='cpu')
    from stable_baselines3 import PPO
    return PPO.load(path, device='cpu')


def main(argv=None):
    import torch

    torch.set_num_threads(4)   # default (all cores) starves the env workers

    ap = argparse.ArgumentParser(description='Evaluate a trained policy in the 2D simulator.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=None,
                    help='default: 500, or one per pair with --points')
    ap.add_argument('--condition', choices=list(CONDITIONS), default='obstacles')
    ap.add_argument('--sources', nargs='+', default=list(TRAIN_SOURCES))
    ap.add_argument('--n-envs', type=int, default=8)
    ap.add_argument('--points', default=None,
                    help='use the fixed start/goal pairs of this world (e.g. lab)')
    ap.add_argument('--out', default=None, help='CSV path (default: next to the model)')
    args = ap.parse_args(argv)

    cfg = trained_env_config(args.model)
    scenario = replace(cfg.scenario, sources=tuple(args.sources),
                       obstacle_mode=CONDITIONS[args.condition])
    if args.points:
        pairs = point_pairs(args.points)
        scenario = replace(scenario, sources=(args.points,), point_pairs=pairs)
    cfg = replace(cfg, scenario=scenario, record_trajectory=True)
    model = load_model(args.model)
    episodes = args.episodes or (len(scenario.point_pairs) if args.points else 500)
    rows = run_episodes(model, cfg, eval_seeds(episodes), args.n_envs)
    name = args.points + '-points' if args.points else (
        'train' if tuple(args.sources) == TRAIN_SOURCES else '-'.join(args.sources))
    out = args.out or Path(args.model).with_name(f'eval_{args.condition}_{name}.csv')
    write_episode_files(rows, scenario, out)
    write_csv(rows, out)
    s = summarize(rows)
    print(f'{args.condition}: episodes={s["episodes"]} success={s["success"]:.3f} '
          f'[{s["success_ci"][0]:.3f}, {s["success_ci"][1]:.3f}] collision={s["collision"]:.3f} '
          f'timeout={s["timeout"]:.3f} stalled={s["stalled"]:.3f} spl={s["spl"]:.3f} -> {out}')


if __name__ == '__main__':
    main()
