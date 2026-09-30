"""Summary figure for a training run.

python3 tools/experiments/plot_run.py runs/full_cnn_s0 [--out docs/figures/full_cnn_s0.png]

A: periodic evaluation over training.  B: how training episodes with obstacles
end, per 500k-step window.  C: deterministic evaluations (eval_*.csv).
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 of the reference palette, fixed order per outcome.
COLORS = {'success': '#2a78d6', 'collision': '#eb6834', 'stalled': '#1baf7a', 'timeout': '#eda100'}
LABELS = {'success': 'éxito', 'collision': 'colisión', 'stalled': 'estancado', 'timeout': 'timeout'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
EVAL_NAMES = {
    'eval_clean_train.csv': 'limpio',
    'eval_obstacles_train.csv': 'obstáculos',
    'eval_obstacles_lab.csv': 'obstáculos\nlab (no visto)',
    'eval_obstacles_train_last.csv': 'obstáculos\nmodelo final',
}


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


def panel_evals(ax, run):
    ev = pd.read_csv(run / 'evals.csv')
    x = ev.timesteps / 1e6
    for key in ('success', 'collision'):
        ax.plot(x, ev[key], color=COLORS[key], linewidth=2, marker='o', markersize=4)
        ax.annotate(LABELS[key], (x.iloc[-1], ev[key].iloc[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', color=INK, fontsize=9)
    best = ev.success.idxmax()
    ax.plot(x[best], ev.success[best], 'o', markersize=9, markerfacecolor='none',
            markeredgecolor=INK, markeredgewidth=1.5)
    ax.annotate(f'mejor: {ev.success[best]:.1%}\n({x[best]:.2f}M pasos)', (x[best], ev.success[best]),
                xytext=(0, 14), textcoords='offset points', ha='center', color=INK, fontsize=9)
    ax.set_xlabel('millones de pasos', color=MUTED, fontsize=9)
    ax.set_xlim(0, x.max() * 1.12)
    style(ax, 'A · Evaluación periódica (200 episodios)')


def panel_windows(ax, run):
    d = pd.read_csv(run / 'episodes.csv')
    d = d[d.n_obstacles > 0]
    d['window'] = d.timesteps // 500_000
    share = d.groupby('window').outcome.value_counts(normalize=True).unstack().fillna(0)
    share = share[share.index < share.index.max()]          # drop the partial last window
    bottom = 0
    for key in ('success', 'collision', 'stalled', 'timeout'):
        ax.bar(share.index * 0.5 + 0.25, share[key], width=0.46, bottom=bottom,
               color=COLORS[key], edgecolor=SURFACE, linewidth=1, label=LABELS[key])
        bottom = bottom + share[key]
    ax.set_xticks(share.index * 0.5 + 0.25)
    ax.set_xticklabels([f'{w * 0.5:g}' for w in share.index])
    ax.set_xlabel('inicio del tramo (millones de pasos, tramos de 0.5M)', color=MUTED, fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=9)
    style(ax, 'B · Episodios de entrenamiento con obstáculos')


def panel_final(ax, run):
    rows = []
    for fname, name in EVAL_NAMES.items():
        f = run / fname
        if f.exists():
            e = pd.read_csv(f).outcome.value_counts(normalize=True)
            rows.append((name, [e.get(k, 0.0) for k in ('success', 'collision', 'stalled')]))
    keys = ('success', 'collision', 'stalled')
    w = 0.26
    for i, key in enumerate(keys):
        xs = [j + (i - 1) * (w + 0.02) for j in range(len(rows))]
        vals = [r[1][i] for r in rows]
        ax.bar(xs, vals, width=w, color=COLORS[key], label=LABELS[key])
        if key == 'success':
            for xv, v in zip(xs, vals):
                ax.annotate(f'{v:.0%}', (xv, v), xytext=(0, 3), textcoords='offset points',
                            ha='center', color=INK, fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    style(ax, 'C · Evaluación determinista (500 / 200 episodios)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run = Path(args.run)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1, 1.1, 1.2]})
    panel_evals(axes[0], run)
    panel_windows(axes[1], run)
    panel_final(axes[2], run)
    fig.suptitle(f'Run {run.name}', x=0.01, ha='left', color=INK, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out) if args.out else run / 'report.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main()
