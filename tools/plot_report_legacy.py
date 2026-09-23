"""Learning and PPO reports, in the exact format of the previous package.

python3 tools/plot_report.py runs/armH_holonomic_s0        # one run
python3 tools/plot_report.py --all                          # every run with episodes.csv

Replicates martha/martha/PPO/analytics.py: same ggplot style, same figure sizes,
same panel order, same titles and the same smoothing (rolling mean over 50
episodes, with the raw series behind it where the original drew it).

Where this package logs something different from the old one, the panel says so
instead of inventing a value:
  - "Cerca de obstáculo" does not exist here; the reward has no proximity term on.
  - ReLU inactivity is not logged by Stable-Baselines3.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPORT_WINDOW = 50          # same default as the old package

OUTCOMES = (('success', 'Éxito'), ('collision', 'Colisión'), ('timeout', 'Truncado'),
            ('stalled', 'Estancado'))
REWARD_COMPONENTS = (('r_step', 'Costo temporal'), ('r_progress', 'Progreso'),
                     ('r_goal', 'Meta'), ('r_collision', 'Colisión'),
                     ('r_proximity', 'Proximidad'), ('r_turn', 'Giro'),
                     ('r_stalled', 'Estancamiento'))
EVALUATIONS = (('success', 'Éxito eval'), ('collision', 'Colisión eval'), ('spl', 'SPL eval'))
LOSSES = (('train/policy_gradient_loss', 'Actor loss'), ('train/value_loss', 'Critic loss'),
          ('train/loss', 'Loss total'))
EXPLORATION = (('train/entropy', 'Entropía'), ('train/std', 'Policy std'))
UPDATES = (('train/approx_kl', 'Approx KL'), ('train/clip_fraction', 'Clip fraction'))
CRITIC = (('train/explained_variance', 'Explained variance'),)


def rolling(values, window):
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window, min_periods=1).mean().to_numpy()


def plot_smoothed(axis, x, values, label, window, raw=False, band=False):
    """Smoothed line; `raw` draws the per-episode series behind it, as the old
    package did, and `band` replaces that cloud with the 10th-90th percentile of
    each window, which is what shows the spread actually shrinking."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return False
    if band:
        series = pd.Series(values)
        window_view = series.rolling(window, min_periods=max(3, window // 5))
        axis.fill_between(x, window_view.quantile(0.1), window_view.quantile(0.9),
                          alpha=0.18, linewidth=0)
    elif raw:
        axis.plot(x, values, alpha=0.16, linewidth=0.7)
    axis.plot(x, rolling(values, window), linewidth=2.0, label=label)
    return True


def missing(axis, text):
    axis.text(0.5, 0.5, text, ha='center', va='center', transform=axis.transAxes)


def load_episodes(run):
    episodes = pd.read_csv(run / 'episodes.csv')
    episodes['episode'] = np.arange(1, len(episodes) + 1)
    terms = [c for c in episodes.columns if c.startswith('r_')]
    episodes['reward'] = episodes[terms].sum(axis=1)
    return episodes


def load_evaluations(run, episodes):
    """Periodic evaluations, placed on the episode axis through their timesteps."""
    path = run / 'evals.csv'
    if not path.exists():
        return None
    evals = pd.read_csv(path)
    evals['episode'] = np.interp(evals.timesteps, episodes.timesteps, episodes.episode)
    return evals


def load_scalars(run, episodes):
    """TensorBoard scalars, also placed on the episode axis."""
    logs = glob.glob(str(run / 'tb' / '*'))
    if not logs:
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(logs[0])
    accumulator.Reload()
    out = {}
    for tag in accumulator.Tags()['scalars']:
        events = accumulator.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=float)
        out[tag] = (np.interp(steps, episodes.timesteps, episodes.episode),
                    np.array([e.value for e in events], dtype=float))
    if 'train/entropy_loss' in out:            # SB3 logs the loss, which is minus the entropy
        x, y = out['train/entropy_loss']
        out['train/entropy'] = (x, -y)
    return out


def learning_report(run, episodes, evals, window, band=False):
    plt.style.use('ggplot')
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    spread = 'percentil 10-90' if band else 'valores por episodio al fondo'
    figure.suptitle(f'Aprendizaje PPO Martha — media móvil de {window} episodios ({spread})',
                    fontsize=16)
    x = episodes.episode

    plot_smoothed(axes[0, 0], x, episodes.reward, 'Recompensa', window, raw=True, band=band)
    axes[0, 0].set_title('Recompensa de entrenamiento')
    axes[0, 0].set_ylabel('Recompensa original')

    for name, label in OUTCOMES:
        plot_smoothed(axes[0, 1], x, (episodes.outcome == name).astype(float), label, window)
    axes[0, 1].set_title('Resultados por episodio')
    axes[0, 1].set_ylabel('Tasa')
    axes[0, 1].set_ylim(-0.03, 1.03)

    plot_smoothed(axes[1, 0], x, episodes.steps, 'Steps', window, raw=True, band=band)
    axes[1, 0].set_title('Duración de los episodios')
    axes[1, 0].set_ylabel('Steps')

    plot_smoothed(axes[1, 1], x, episodes.spl, 'SPL', window)
    axes[1, 1].set_title('Eficiencia de ruta (SPL)')
    axes[1, 1].set_ylabel('SPL')
    axes[1, 1].set_ylim(-0.03, 1.03)

    drawn = False
    for name, label in REWARD_COMPONENTS:
        if name in episodes and episodes[name].abs().sum() > 0:
            drawn |= plot_smoothed(axes[2, 0], x, episodes[name], label, window)
    axes[2, 0].set_title('Contribución de cada término de recompensa')
    axes[2, 0].set_ylabel('Suma por episodio')
    if not drawn:
        missing(axes[2, 0], 'Este run no registró componentes')

    if evals is not None:
        for name, label in EVALUATIONS:
            axes[2, 1].plot(evals.episode, evals[name], marker='o', label=label)
    axes[2, 1].set_title('Evaluación determinista')
    axes[2, 1].set_ylabel('Tasa / SPL')
    axes[2, 1].set_ylim(-0.03, 1.03)
    if evals is None:
        missing(axes[2, 1], 'Aún no hay evaluaciones periódicas')

    for axis in axes.flat:
        axis.set_xlabel('Episodio')
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc='best')
    path = run / 'learning_report.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def diagnostics_report(run, scalars, window):
    plt.style.use('ggplot')
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle('Diagnóstico interno de PPO', fontsize=16)
    panels = (((0, 0), 'Pérdidas', LOSSES), ((0, 1), 'Exploración', EXPLORATION),
              ((1, 0), 'Magnitud de las actualizaciones', UPDATES),
              ((1, 1), 'Crítico y activaciones', CRITIC))
    for (row, col), title, series in panels:
        axis = axes[row, col]
        for tag, label in series:
            if tag in scalars:
                x, values = scalars[tag]
                plot_smoothed(axis, x, values, label, window)
        axis.set_title(title)
    axes[1, 1].text(0.5, 0.08, 'Stable-Baselines3 no registra ReLU inactivas',
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=9)
    for axis in axes.flat:
        axis.set_xlabel('Episodio')
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc='best')
    path = run / 'ppo_diagnostics.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def report(run, window=REPORT_WINDOW, band=False):
    episodes = load_episodes(run)
    evals = load_evaluations(run, episodes)
    scalars = load_scalars(run, episodes)
    return (learning_report(run, episodes, evals, window, band),
            diagnostics_report(run, scalars, window))


def main():
    ap = argparse.ArgumentParser(description='Learning and PPO reports for a run.')
    ap.add_argument('run', nargs='?', help='run directory')
    ap.add_argument('--all', action='store_true', help='every run under runs/')
    ap.add_argument('--window', type=int, default=REPORT_WINDOW)
    ap.add_argument('--band', action='store_true',
                    help='show the 10-90 percentile band instead of the raw episode cloud')
    args = ap.parse_args()
    runs = ([p.parent for p in sorted(Path('runs').glob('*/episodes.csv'))] if args.all
            else [Path(args.run)])
    for run in runs:
        for path in report(run, args.window, args.band):
            print(path)


if __name__ == '__main__':
    main()
