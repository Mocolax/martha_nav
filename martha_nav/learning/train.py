"""Train PPO (Stable-Baselines3) on the 2D simulator.

python3 -m martha_nav.learning.train --preset gate --arch cnn --seed 0
python3 -m martha_nav.learning.train --preset full --arch cnn --seed 0
"""
import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from martha_nav.learning.evaluate import eval_seeds, run_episodes, summarize
from martha_nav.learning.policy import policy_kwargs
from martha_nav.sim2d.env import EnvConfig, NavEnv
from martha_nav.sim2d.scenarios import TRAIN_SOURCES

RUNS_DIR = Path(__file__).resolve().parents[2] / 'runs'

PRESETS = {
    # Convergence gate: open room, no obstacles, must reach >= 80% success.
    'gate': dict(sources=('open_room',), obstacle_mode='none', steps=500_000),
    # Main training: every training source, mixed obstacles.
    'full': dict(sources=TRAIN_SOURCES, obstacle_mode='mixed', steps=5_000_000),
}

PPO_PARAMS = dict(n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)
LEARNING_RATE = 3e-4


class EpisodeLogger(BaseCallback):
    """Appends one CSV row per finished training episode."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.file = self.writer = None

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if not done or 'outcome' not in info:
                continue
            row = {'timesteps': self.num_timesteps}
            row.update({k: v for k, v in info.items()
                        if k not in ('terminal_observation', 'TimeLimit.truncated', 'episode')})
            if self.writer is None:
                self.file = open(self.path, 'w', newline='')
                self.writer = csv.DictWriter(self.file, fieldnames=list(row), extrasaction='ignore')
                self.writer.writeheader()
            self.writer.writerow(row)
        return True

    def _on_training_end(self):
        if self.file:
            self.file.close()


class PeriodicEval(BaseCallback):
    """Every `every` steps: deterministic evaluation on fixed seeds; keeps best_model.zip."""

    def __init__(self, env_cfg, run_dir, every, episodes, n_envs=8):
        super().__init__()
        self.env_cfg, self.run_dir = env_cfg, run_dir
        self.every, self.seeds, self.n_envs = every, eval_seeds(episodes), n_envs
        self.next_eval, self.best = every, -1.0

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.every
        s = summarize(run_episodes(self.model, self.env_cfg, self.seeds, self.n_envs))
        line = {'timesteps': self.num_timesteps, 'success': s['success'],
                'collision': s['collision'], 'timeout': s['timeout'],
                'stalled': s['stalled'], 'spl': s['spl']}
        path = self.run_dir / 'evals.csv'
        new = not path.exists()
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(line))
            if new:
                w.writeheader()
            w.writerow(line)
        for k, v in line.items():
            if k != 'timesteps':
                self.logger.record(f'eval/{k}', v)
        if s['success'] > self.best:
            self.best = s['success']
            self.model.save(self.run_dir / 'best_model.zip')
        print(f'[eval] {self.num_timesteps} steps: success={s["success"]:.3f} '
              f'collision={s["collision"]:.3f} spl={s["spl"]:.3f} (best {self.best:.3f})', flush=True)
        return True


def build_config(preset, collision=None, stalled=None, lidar_encoding='inverse'):
    """EnvConfig for a preset; the optional arguments are experiment overrides."""
    p = PRESETS[preset]
    cfg = EnvConfig()
    reward = cfg.reward
    if collision is not None:
        reward = replace(reward, collision=collision)
    if stalled is not None:
        reward = replace(reward, stalled=stalled)
    return replace(cfg, reward=reward, lidar_encoding=lidar_encoding,
                   scenario=replace(cfg.scenario, sources=tuple(p['sources']),
                                    obstacle_mode=p['obstacle_mode']))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Train the PPO local planner.')
    ap.add_argument('--preset', choices=list(PRESETS), default='gate')
    ap.add_argument('--arch', choices=['cnn', 'mlp'], default='cnn')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None, help='override the preset budget')
    ap.add_argument('--n-envs', type=int, default=16)
    ap.add_argument('--eval-every', type=int, default=250_000)
    ap.add_argument('--eval-episodes', type=int, default=200)
    ap.add_argument('--device', default='auto', help="'auto' uses CUDA when available (~40%% faster)")
    ap.add_argument('--runs-dir', default=str(RUNS_DIR))
    ap.add_argument('--name', default=None)
    ap.add_argument('--reward-collision', type=float, default=None, help='override RewardConfig.collision')
    ap.add_argument('--reward-stalled', type=float, default=None,
                    help='override RewardConfig.stalled (non-zero makes stalls terminal)')
    ap.add_argument('--lidar-encoding', choices=['linear', 'inverse'], default='inverse')
    args = ap.parse_args(argv)

    steps = args.steps or PRESETS[args.preset]['steps']
    name = args.name or f'{args.preset}_{args.arch}_s{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir = Path(args.runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    env_cfg = build_config(args.preset, args.reward_collision, args.reward_stalled, args.lidar_encoding)
    config = {'preset': args.preset, 'arch': args.arch, 'seed': args.seed, 'steps': steps,
              'n_envs': args.n_envs, 'learning_rate': LEARNING_RATE, 'ppo': PPO_PARAMS,
              'eval_every': args.eval_every, 'eval_episodes': args.eval_episodes,
              'env': json.loads(json.dumps(asdict(env_cfg)))}
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    torch.set_num_threads(4)
    venv = SubprocVecEnv([lambda: NavEnv(env_cfg) for _ in range(args.n_envs)])
    venv.seed(args.seed)
    venv = VecNormalize(VecMonitor(venv), norm_obs=False, norm_reward=True,
                        gamma=PPO_PARAMS['gamma'])
    model = PPO('MlpPolicy', venv, learning_rate=lambda f: LEARNING_RATE * f,
                policy_kwargs=policy_kwargs(args.arch), seed=args.seed, device=args.device,
                tensorboard_log=str(run_dir / 'tb'), verbose=0, **PPO_PARAMS)
    callbacks = [EpisodeLogger(run_dir / 'episodes.csv'),
                 PeriodicEval(env_cfg, run_dir, args.eval_every, args.eval_episodes)]
    start = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name='ppo')
    model.save(run_dir / 'last_model.zip')
    venv.save(str(run_dir / 'vecnormalize.pkl'))
    venv.close()
    minutes = (time.time() - start) / 60
    print(f'done: {steps} steps in {minutes:.1f} min ({steps / (minutes * 60):.0f} steps/s) -> {run_dir}')
    return run_dir


if __name__ == '__main__':
    main()
