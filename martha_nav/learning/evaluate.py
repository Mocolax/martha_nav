"""Deterministic evaluation on fixed seeds; CSV rows and a summary with 95% CIs."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig, NavEnv
from martha_nav.sim2d.scenarios import TRAIN_SOURCES, point_pairs

CONDITIONS = {'clean': 'none', 'obstacles': 'always', 'mixed': 'mixed'}


def eval_seeds(n, offset=0):
    """Evaluation episode seeds; disjoint from training seeds by construction."""
    return [TRAIN_SEED_LIMIT + offset + i for i in range(n)]


def make_vec_env(cfgs):
    fns = [lambda c=c: NavEnv(c) for c in cfgs]
    return DummyVecEnv(fns) if len(fns) == 1 else SubprocVecEnv(fns)


def _is_recurrent(model):
    return getattr(model, 'is_recurrent', False) or 'Recurrent' in type(model).__name__


def run_episodes(model, env_cfg, seeds, n_envs=8, deterministic=True):
    """Play every seed exactly once; returns one info dict per episode, in seed order."""
    n_envs = max(1, min(n_envs, len(seeds)))
    chunks = [tuple(c) for c in np.array_split(np.asarray(seeds), n_envs)]
    venv = make_vec_env([replace(env_cfg, episode_seeds=c) for c in chunks])
    done_count = [0] * n_envs
    rows = []
    obs = venv.reset()
    recurrent = _is_recurrent(model)
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
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _trained_env(model_path):
    """The env must match training: read its settings from the run's config.yaml."""
    config = Path(model_path).with_name('config.yaml')
    if not config.exists():
        return {}
    import yaml
    saved = yaml.safe_load(config.read_text())['env']
    # Runs trained before these options existed used the linear encoding and (v, w).
    return {'lidar_encoding': saved.get('lidar_encoding', 'linear'),
            'action_dim': saved.get('action_dim', 2),
            'stuck_signal': saved.get('stuck_signal', False),
            'target': saved.get('target', 'carrot')}


def load_model(path):
    """PPO, or RecurrentPPO when the run's config says it was trained with an LSTM."""
    config = Path(path).with_name('config.yaml')
    recurrent = False
    if config.exists():
        import yaml
        recurrent = bool(yaml.safe_load(config.read_text()).get('recurrent', False))
    if recurrent:
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

    cfg = EnvConfig(**_trained_env(args.model))
    scenario = replace(cfg.scenario, sources=tuple(args.sources),
                       obstacle_mode=CONDITIONS[args.condition])
    if args.points:
        pairs = point_pairs(args.points)
        scenario = replace(scenario, sources=(args.points,), point_pairs=pairs)
    cfg = replace(cfg, scenario=scenario)
    model = load_model(args.model)
    episodes = args.episodes or (len(scenario.point_pairs) if args.points else 500)
    rows = run_episodes(model, cfg, eval_seeds(episodes), args.n_envs)
    name = args.points + '-points' if args.points else (
        'train' if tuple(args.sources) == TRAIN_SOURCES else '-'.join(args.sources))
    out = args.out or Path(args.model).with_name(f'eval_{args.condition}_{name}.csv')
    write_csv(rows, out)
    s = summarize(rows)
    print(f'{args.condition}: episodes={s["episodes"]} success={s["success"]:.3f} '
          f'[{s["success_ci"][0]:.3f}, {s["success_ci"][1]:.3f}] collision={s["collision"]:.3f} '
          f'timeout={s["timeout"]:.3f} stalled={s["stalled"]:.3f} spl={s["spl"]:.3f} -> {out}')
    return s


if __name__ == '__main__':
    main()
