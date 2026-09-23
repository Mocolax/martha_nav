"""E1 figure: CNN vs MLP over three seeds each.

python3 tools/plot_e1.py --out docs/figures/e1_cnn_vs_mlp.png
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ARCHS = {'cnn': '#2a78d6', 'mlp': '#eb6834'}
SEEDS = (0, 1, 2)
CONDITIONS = [('eval_clean_train.csv', 'limpio'), ('eval_obstacles_train.csv', 'obstáculos'),
              ('eval_obstacles_lab.csv', 'obstáculos, lab\n(no visto)')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax, title, percent=True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def success(runs_dir, arch, seed, filename):
    path = runs_dir / f'e1_{arch}_s{seed}' / filename
    return float((pd.read_csv(path).outcome == 'success').mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default='runs')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = Path(args.runs_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [1.15, 1]})

    width = 0.36
    for i, (arch, color) in enumerate(ARCHS.items()):
        for j, (filename, _) in enumerate(CONDITIONS):
            values = [success(runs, arch, seed, filename) for seed in SEEDS]
            x = j + (i - 0.5) * width
            mean = float(np.mean(values))
            axes[0].bar(x, mean, width=width - 0.03, color=color,
                        label=arch.upper() if j == 0 else None)
            axes[0].errorbar(x, mean, yerr=1.96 * np.std(values, ddof=1) / np.sqrt(len(values)),
                             color=INK, capsize=4, linewidth=1.2)
            axes[0].plot([x] * len(values), values, 'o', color=SURFACE, markersize=5,
                         markeredgecolor=INK, markeredgewidth=1)
            axes[0].annotate(f'{mean:.0%}', (x, mean), xytext=(0, 12), textcoords='offset points',
                             ha='center', color=INK, fontsize=9)
    axes[0].set_xticks(range(len(CONDITIONS)))
    axes[0].set_xticklabels([c[1] for c in CONDITIONS], fontsize=9)
    axes[0].legend(loc='lower left', frameon=False, fontsize=10)
    style(axes[0], 'Éxito del mejor modelo · media de 3 semillas (puntos: cada semilla)')

    for arch, color in ARCHS.items():
        for seed, dash in zip(SEEDS, ['-', '--', ':']):
            ev = pd.read_csv(runs / f'e1_{arch}_s{seed}' / 'evals.csv')
            axes[1].plot(ev.timesteps / 1e6, ev.success, dash, color=color, linewidth=1.6,
                         label=f'{arch.upper()} s{seed}')
    axes[1].set_xlabel('millones de pasos', color=MUTED, fontsize=9)
    axes[1].legend(loc='lower right', frameon=False, fontsize=8, ncol=2)
    style(axes[1], 'Evaluación periódica de cada run')

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(args.out)


if __name__ == '__main__':
    main()
