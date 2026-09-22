"""Per-run report, in the spirit of ppo_plot from the previous package.

python3 tools/plot_report.py runs/e1_cnn_s0

Writes <run>/learning_report.png (3x2: reward, outcomes, episode length, SPL,
reward terms, deterministic evaluation) and <run>/ppo_diagnostics.png (2x2:
losses, exploration, update size, critic).
"""
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

WINDOW = 200            # episodes in the rolling mean
OUTCOMES = ('success', 'collision', 'stalled', 'timeout')
COLORS = {'success': '#2a78d6', 'collision': '#eb6834', 'stalled': '#1baf7a', 'timeout': '#eda100'}
LABELS = {'success': 'éxito', 'collision': 'colisión', 'stalled': 'estancado', 'timeout': 'timeout'}
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax, title, ylabel=None, percent=False):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel('millones de pasos', color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def load(run):
    episodes = pd.read_csv(run / 'episodes.csv')
    episodes['x'] = episodes.timesteps / 1e6
    terms = [c for c in episodes.columns if c.startswith('r_')]
    episodes['reward'] = episodes[terms].sum(axis=1)
    evals = pd.read_csv(run / 'evals.csv') if (run / 'evals.csv').exists() else None
    return episodes, terms, evals


def scalars(run):
    """TensorBoard scalars as {tag: (steps, values)}; empty when there is no log."""
    logs = glob.glob(str(run / 'tb' / '*'))
    if not logs:
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logs[0])
    ea.Reload()
    out = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]) / 1e6, np.array([e.value for e in events]))
    return out


def rolling(series, window=WINDOW):
    return series.rolling(window, min_periods=max(5, window // 10)).mean()


def learning_report(run, episodes, terms, evals, out):
    fig, ax = plt.subplots(3, 2, figsize=(15, 13), facecolor=SURFACE, constrained_layout=True)

    ax[0, 0].plot(episodes.x, rolling(episodes.reward), color=SLOTS[0], linewidth=2)
    style(ax[0, 0], 'Recompensa de entrenamiento', 'recompensa por episodio')

    share = pd.DataFrame({o: rolling((episodes.outcome == o).astype(float)) for o in OUTCOMES})
    bottom = np.zeros(len(episodes))
    for o in OUTCOMES:
        ax[0, 1].fill_between(episodes.x, bottom, bottom + share[o].fillna(0), color=COLORS[o],
                              label=LABELS[o], linewidth=0)
        bottom = bottom + share[o].fillna(0).to_numpy()
    ax[0, 1].legend(loc='lower right', frameon=False, fontsize=9, ncol=4)
    style(ax[0, 1], f'Resultados por episodio (media móvil de {WINDOW})', percent=True)

    ax[1, 0].plot(episodes.x, rolling(episodes.steps), color=SLOTS[0], linewidth=2)
    style(ax[1, 0], 'Duración de los episodios', 'pasos')

    ax[1, 1].plot(episodes.x, rolling(episodes.spl), color=SLOTS[2], linewidth=2)
    style(ax[1, 1], 'Eficiencia de ruta (SPL)', percent=True)

    for term, color in zip(terms, SLOTS):
        ax[2, 0].plot(episodes.x, rolling(episodes[term]), color=color, linewidth=2,
                      label=term.replace('r_', ''))
    ax[2, 0].legend(loc='upper left', frameon=False, fontsize=9, ncol=3)
    ax[2, 0].axhline(0, color=MUTED, linewidth=0.8)
    style(ax[2, 0], 'Contribución de cada término de recompensa', 'por episodio')

    if evals is not None:
        for key in ('success', 'collision'):
            ax[2, 1].plot(evals.timesteps / 1e6, evals[key], color=COLORS[key], marker='o',
                          markersize=4, linewidth=2, label=LABELS[key])
        best = evals.success.idxmax()
        ax[2, 1].plot(evals.timesteps[best] / 1e6, evals.success[best], 'o', markersize=9,
                      markerfacecolor='none', markeredgecolor=INK, markeredgewidth=1.5)
        ax[2, 1].legend(loc='center right', frameon=False, fontsize=9)
    style(ax[2, 1], 'Evaluación determinista (semillas fijas)', percent=True)

    fig.suptitle(f'Run {run.name}', x=0.01, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def diagnostics_report(run, tb, out):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURFACE, constrained_layout=True)
    groups = [
        ((0, 0), 'Pérdidas', [('train/policy_gradient_loss', 'actor'), ('train/value_loss', 'crítico'),
                              ('train/entropy_loss', 'entropía')], None),
        ((0, 1), 'Exploración', [('train/std', 'std de la política')], None),
        ((1, 0), 'Magnitud de las actualizaciones', [('train/approx_kl', 'approx_kl'),
                                                     ('train/clip_fraction', 'clip_fraction')], 0.02),
        ((1, 1), 'Crítico', [('train/explained_variance', 'explained_variance')], None),
    ]
    for (row, col), title, tags, reference in groups:
        axis = ax[row, col]
        for (tag, label), color in zip(tags, SLOTS):
            if tag not in tb:
                continue
            steps, values = tb[tag]
            axis.plot(steps, values, color=color, linewidth=2, label=label)
        if reference is not None:
            axis.axhline(reference, color=MUTED, linewidth=1, linestyle='--')
            axis.annotate(f'objetivo {reference}', (steps[-1], reference), xytext=(-70, 4),
                          textcoords='offset points', color=MUTED, fontsize=8)
        axis.legend(loc='best', frameon=False, fontsize=9)
        style(axis, title)
    fig.suptitle(f'PPO · {run.name}', x=0.01, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Learning and PPO reports for a run.')
    ap.add_argument('run')
    args = ap.parse_args()
    run = Path(args.run)
    episodes, terms, evals = load(run)
    learning_report(run, episodes, terms, evals, run / 'learning_report.png')
    diagnostics_report(run, scalars(run), run / 'ppo_diagnostics.png')
    print(run / 'learning_report.png')
    print(run / 'ppo_diagnostics.png')


if __name__ == '__main__':
    main()
