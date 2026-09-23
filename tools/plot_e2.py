"""E2 figure: the same episodes in the 2D simulator and in Gazebo.

python3 tools/plot_e2.py --run runs/long_c_kl_s0 --out docs/figures/e2_2d_vs_gazebo.png
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUTCOMES = [('success', 'éxito', '#2a78d6'), ('collision', 'colisión', '#eb6834'),
            ('stuck', 'estancado / timeout', '#1baf7a'), ('failed', 'sin ruta', '#eda100')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
CONDITIONS = [('semillas generadas', 'eval_obstacles_lab.csv', 'eval_gazebo_lab_v3.csv'),
              ('puntos fijos del\npaquete anterior', 'eval_obstacles_lab-points.csv',
               'eval_gazebo_lab_points_v3.csv')]


def shares(path, seeds=None):
    d = pd.read_csv(path).set_index('episode_seed')
    if seeds is not None:
        d = d.loc[sorted(seeds)]
    outcome = d.outcome.replace({'stalled': 'stuck', 'timeout': 'stuck'})
    return outcome.value_counts(normalize=True), set(d.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/long_c_kl_s0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run = Path(args.run)

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    labels, positions = [], []
    for i, (name, f2d, fgz) in enumerate(CONDITIONS):
        _, s2d = shares(run / f2d)
        _, sgz = shares(run / fgz)
        common = s2d & sgz
        for j, (source, path) in enumerate((('2D', f2d), ('Gazebo', fgz))):
            share, _ = shares(run / path, common)
            x = i * 2.6 + j
            bottom = 0.0
            for key, label, color in OUTCOMES:
                value = float(share.get(key, 0.0))
                if value <= 0:
                    continue
                ax.bar(x, value, width=0.7, bottom=bottom, color=color, edgecolor=SURFACE,
                       linewidth=1.5, label=label if (i == 0 and j == 0) else None)
                if value > 0.05:
                    ax.annotate(f'{value:.0%}', (x, bottom + value / 2), ha='center', va='center',
                                color='white', fontsize=10, fontweight='bold')
                bottom += value
            labels.append(f'{source}\n({len(common)} episodios)')
            positions.append(x)
        ax.annotate(name.replace('\n', ' '), (i * 2.6 + 0.5, 1.04), ha='center', va='bottom',
                    color=INK, fontsize=11, fontweight='bold', annotation_clip=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    fig.suptitle('E2 · los mismos episodios en el simulador 2D y en Gazebo (lab.world)',
                 x=0.01, ha='left', color=INK, fontsize=12, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(args.out)


if __name__ == '__main__':
    main()
