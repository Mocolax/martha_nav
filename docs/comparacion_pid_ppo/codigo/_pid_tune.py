"""Score the PID baseline on TRAINING seeds (never the evaluation ones): clean + obstacles."""
import sys
from dataclasses import replace
from martha_nav.learning.evaluate import load_model, run_episodes, summarize, trained_env_config
from martha_nav.learning import pid
from martha_nav.sim2d.scenarios import CONDITIONS
if __name__ == '__main__':
    for kv in sys.argv[1:]:
        k, v = kv.split('='); setattr(pid, k, float(v))
    m = 'runs/pid_martha_tower/best_model.zip'
    cfg = trained_env_config(m); ctl = load_model(m)
    out = []
    for cond in ('clean', 'obstacles'):
        c = replace(cfg, scenario=replace(cfg.scenario, obstacle_mode=CONDITIONS[cond]))
        s = summarize(run_episodes(ctl, c, list(range(5000, 5300)), n_envs=12))
        out.append(f"{cond}: succ={s['success']:.3f} coll={s['collision']:.3f} stall={s['stalled']:.3f} tout={s['timeout']:.3f}")
    print(' '.join(sys.argv[1:]) or 'default', '|', ' | '.join(out))
