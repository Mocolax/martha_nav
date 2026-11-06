"""Score the PID baseline on TRAINING seeds (never the evaluation ones), clean and with obstacles.

./tools/ct_ros python3 tools/pid_tune.py K_SIDE=12 SIDE=0.35
Each NAME=value overrides a constant of martha_nav/learning/pid.py for this run.
"""
import sys
from dataclasses import replace

from martha_nav.learning import pid
from martha_nav.learning.evaluate import load_model, run_episodes, summarize, trained_env_config
from martha_nav.sim2d.scenarios import CONDITIONS

MODEL = 'runs/pid_martha_tower/best_model.zip'   # config.yaml only: `controller: pid`
SEEDS = list(range(5000, 5300))                  # training seeds, below TRAIN_SEED_LIMIT

if __name__ == '__main__':
    for kv in sys.argv[1:]:
        name, value = kv.split('=')
        setattr(pid, name, float(value))
    cfg, ctl = trained_env_config(MODEL), load_model(MODEL)
    out = []
    for cond in ('clean', 'obstacles'):
        c = replace(cfg, scenario=replace(cfg.scenario, obstacle_mode=CONDITIONS[cond]))
        s = summarize(run_episodes(ctl, c, SEEDS, n_envs=12))
        out.append(f"{cond}: success={s['success']:.3f} collision={s['collision']:.3f} "
                   f"stalled={s['stalled']:.3f} timeout={s['timeout']:.3f}")
    print(' '.join(sys.argv[1:]) or 'default', '|', ' | '.join(out))
