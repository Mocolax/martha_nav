"""Compare runs side by side.

python3 tools/compare_runs.py --out docs/figures/comparacion.png \
    final=runs/wide_dyn_s0 base=runs/long_c_kl_s0 H=runs/armH_holonomic_s0

A: periodic-evaluation success.  B: periodic-evaluation collision.  C: deterministic
evaluation of each best model (success) in three conditions.
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 of the reference palette, assigned to runs in the given order.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
CONDITIONS = [('eval_clean_train.csv', 'limpio'), ('eval_obstacles_train.csv', 'obstáculos'),
              ('eval_obstacles_lab.csv', 'obstáculos, lab\n(no visto)')]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def curve(ax, runs, key, title, max_steps, end_labels=True):
    for (label, run), color in zip(runs, SLOTS):
        ev = pd.read_csv(run / 'evals.csv')
        ev = ev[ev.timesteps <= max_steps]
        x = ev.timesteps / 1e6
        ax.plot(x, ev[key], color=color, linewidth=2, marker='o', markersize=4, label=label)
        if end_labels:
            ax.annotate(label, (x.iloc[-1], ev[key].iloc[-1]), xytext=(6, 0), textcoords='offset points',
                        va='center', color=INK, fontsize=9)
    ax.set_xlim(0, max_steps / 1e6 * (1.45 if end_labels else 1.05))
    ax.set_xlabel('millones de pasos', color=MUTED, fontsize=9)
    style(ax, title)


def finals(ax, runs):
    w = 0.8 / len(runs)
    for i, ((label, run), color) in enumerate(zip(runs, SLOTS)):
        vals = []
        for fname, _ in CONDITIONS:
            f = run / fname
            vals.append((pd.read_csv(f).outcome == 'success').mean() if f.exists() else 0.0)
        xs = [j + (i - (len(runs) - 1) / 2) * w for j in range(len(CONDITIONS))]
        ax.bar(xs, vals, width=w - 0.03, color=color, label=label)
        for xv, v in zip(xs, vals):
            ax.annotate(f'{v:.0%}', (xv, v), xytext=(0, 3), textcoords='offset points', ha='center',
                        color=INK, fontsize=8)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c[1] for c in CONDITIONS], fontsize=9)
    style(ax, 'C · Éxito del mejor modelo (500 / 200 episodios)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='label=run_dir, in legend order (max 4)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-steps', type=int, default=2_000_000)
    args = ap.parse_args()
    runs = [(s.split('=', 1)[0], Path(s.split('=', 1)[1])) for s in args.runs][:4]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1, 1, 1.2]})
    curve(axes[0], runs, 'success', 'A · Evaluación periódica: éxito', args.max_steps)
    curve(axes[1], runs, 'collision', 'B · Evaluación periódica: colisión', args.max_steps, end_labels=False)
    finals(axes[2], runs)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(runs), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(args.out)


if __name__ == '__main__':
    main()
