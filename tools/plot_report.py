"""Per-run report: the panels of the old ppo_plot, drawn for print.

python3 tools/plot_report.py runs/e1_cnn_s0
python3 tools/plot_report.py --all

Writes <run>/learning_report.png (3x2: reward, outcomes, episode length, SPL,
reward terms, deterministic evaluation) and <run>/ppo_diagnostics.png (2x2:
losses, exploration, update size, critic).

The x axis is the episode, with a second axis in millions of steps on top. Rates
are lines, one per outcome; spread is a percentile band, never a raw cloud.
For the exact look of the previous package, use tools/plot_report_legacy.py.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

WINDOW = 50             # episodes in the rolling mean, like REPORT_WINDOW in the old package
OUTCOMES = ('success', 'collision', 'stalled', 'timeout')
COLORS = {'success': '#2a78d6', 'collision': '#eb6834', 'stalled': '#1baf7a', 'timeout': '#eda100'}
LABELS = {'success': 'éxito', 'collision': 'colisión', 'stalled': 'estancado', 'timeout': 'timeout'}
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax, title, ylabel=None, percent=False, steps_axis=None, xlabel='episodio'):
    """steps_axis: (episodes, timesteps) to add a second x axis in millions of steps."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=INK, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if steps_axis is not None:
        episodes, timesteps = steps_axis
        top = ax.secondary_xaxis(
            'top',
            functions=(lambda e: np.interp(e, episodes, timesteps / 1e6),
                       lambda m: np.interp(m, timesteps / 1e6, episodes)))
        top.set_xlabel('millones de pasos', color=MUTED, fontsize=8)
        top.tick_params(colors=MUTED, labelsize=8, length=0)


def trace(ax, x, series, color, label=None, window=WINDOW):
    """Rolling mean, over a light band with the 10th to 90th percentile of the window.

    With tens of thousands of episodes the raw per-episode line is an unreadable
    blur in print, so the spread is shown as a band instead.
    """
    # Band: 10th to 90th percentile inside bins, so it stays smooth with 20k episodes.
    bins = max(1, len(series) // max(window, len(series) // 300))
    group = pd.Series(series.to_numpy()).groupby(np.minimum(np.arange(len(series)) // bins,
                                                            len(series) // bins))
    centre = group.apply(lambda g: g.index.to_numpy().mean() + 1)
    ax.fill_between(centre, group.quantile(0.1), group.quantile(0.9), color=color, alpha=0.16,
                    linewidth=0)
    ax.plot(x, series.rolling(window, min_periods=max(3, window // 5)).mean(), color=color,
            linewidth=1.8, label=label)


def load(run):
    episodes = pd.read_csv(run / 'episodes.csv')
    episodes['x'] = np.arange(1, len(episodes) + 1)
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
    if 'train/entropy_loss' in out:      # SB3 logs the loss, which is minus the entropy
        steps, values = out['train/entropy_loss']
        out['train/entropy'] = (steps, -values)
    return out


def rolling(series, window=WINDOW):
    return series.rolling(window, min_periods=max(5, window // 10)).mean()


def learning_report(run, episodes, terms, evals, out):
    fig, ax = plt.subplots(3, 2, figsize=(15, 13), facecolor=SURFACE, constrained_layout=True)
    steps_axis = (episodes.x.to_numpy(), episodes.timesteps.to_numpy())

    trace(ax[0, 0], episodes.x, episodes.reward, SLOTS[0])
    style(ax[0, 0], f'Recompensa por episodio (media móvil de {WINDOW})',
          'recompensa', steps_axis=steps_axis)

    # One line per outcome, no shading: a stacked area hides the small rates.
    for outcome in OUTCOMES:
        ax[0, 1].plot(episodes.x, rolling((episodes.outcome == outcome).astype(float)),
                      color=COLORS[outcome], linewidth=1.8, label=LABELS[outcome])
    ax[0, 1].legend(loc='center right', frameon=False, fontsize=9, ncol=2)
    style(ax[0, 1], f'Resultados por episodio (media móvil de {WINDOW})', 'tasa', percent=True,
          steps_axis=steps_axis)

    trace(ax[1, 0], episodes.x, episodes.steps, SLOTS[0])
    style(ax[1, 0], 'Duración de los episodios', 'pasos', steps_axis=steps_axis)

    # SPL is 0 on every failure, so averaging it over all episodes hides the shape.
    spl = episodes.spl.where(episodes.outcome == 'success')
    trace(ax[1, 1], episodes.x, spl, SLOTS[2])
    style(ax[1, 1], 'Eficiencia de ruta (SPL entre los éxitos)', percent=True,
          steps_axis=steps_axis)

    for term, color in zip(terms, SLOTS):
        ax[2, 0].plot(episodes.x, rolling(episodes[term]), color=color, linewidth=2,
                      label=term.replace('r_', ''))
    ax[2, 0].legend(loc='upper left', frameon=False, fontsize=9, ncol=3)
    ax[2, 0].axhline(0, color=MUTED, linewidth=0.8)
    style(ax[2, 0], 'Contribución de cada término de recompensa', 'por episodio',
          steps_axis=steps_axis)

    if evals is not None:
        at_episode = np.interp(evals.timesteps, episodes.timesteps, episodes.x)
        for key in ('success', 'collision'):
            ax[2, 1].plot(at_episode, evals[key], color=COLORS[key], marker='o',
                          markersize=4, linewidth=2, label=LABELS[key])
        best = evals.success.idxmax()
        ax[2, 1].plot(at_episode[best], evals.success[best], 'o', markersize=9,
                      markerfacecolor='none', markeredgecolor=INK, markeredgewidth=1.5)
        ax[2, 1].legend(loc='center right', frameon=False, fontsize=9)
    style(ax[2, 1], 'Evaluación determinista (semillas fijas)', percent=True,
          steps_axis=steps_axis)

    fig.suptitle(f'Run {run.name}', x=0.01, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def diagnostics_report(run, tb, out):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURFACE, constrained_layout=True)
    groups = [
        ((0, 0), 'Pérdidas', [('train/policy_gradient_loss', 'actor'), ('train/value_loss', 'crítico'),
                              ('train/entropy', 'entropía')], None),
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
        style(axis, title, xlabel='millones de pasos')
    fig.suptitle(f'PPO · {run.name}', x=0.01, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Learning and PPO reports for a run.')
    ap.add_argument('run', nargs='?')
    ap.add_argument('--all', action='store_true', help='every run under runs/')
    args = ap.parse_args()
    runs = ([p.parent for p in sorted(Path('runs').glob('*/episodes.csv'))] if args.all
            else [Path(args.run)])
    for run in runs:
        episodes, terms, evals = load(run)
        learning_report(run, episodes, terms, evals, run / 'learning_report.png')
        diagnostics_report(run, scalars(run), run / 'ppo_diagnostics.png')
        print(run / 'learning_report.png')


if __name__ == '__main__':
    main()
