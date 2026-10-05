"""Build entrega_tesis/: per configuration its data, plus summary, paired and verification tables.

    ./tools/ct_ros python3 tools/build_entrega.py [--out entrega_tesis]

Reads runs/<run>/{config.yaml, best_model.zip, vecnormalize.pkl, evals.csv, episodes.csv} and the
evaluations in runs/<run>/v5/ (tools/evaluate_gazebo_fast.sh, evaluate_2d --out).
"""
import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from martha_nav.learning.evaluate import summarize  # noqa: E402
from martha_nav.sim2d.dynamics import DT  # noqa: E402
from martha_nav.sim2d.scenarios import CONDITIONS, ScenarioConfig, generate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = 'wide_dyn_s0'
# run, what it is, where the document discusses it, evaluated in Gazebo
CONFIGS = [
    ('wide_dyn_s0', 'Modelo final: C + target_kl 0.02 + dinámica amplia (retardo 0-2 periodos)',
     'resultados.md, "Corrección: entrenar con dinámica amplia"', True),
    ('long_c_kl_s0', 'Línea base: C + target_kl 0.02, 5M pasos',
     'resultados.md, "Verificación: long_c_kl_s0"', True),
    ('armG_goal_s0', 'Brazo G: holonómico, la política ve la meta en vez de la zanahoria',
     'resultados-atasco.md, "Brazo G"', True),
    ('armH_holonomic_s0', 'Brazo H: acción holonómica (vx, vy, w)',
     'resultados-noche.md, "Brazo H"', True),
    ('armL_lstm_s0', 'Brazo L: política recurrente (LSTM)', 'resultados-noche.md, "Brazo L"', True),
    ('e1_cnn_s0', 'E1: CNN, semilla 0 (sin target_kl)', 'ppo-local-planner-design.md, E1', False),
    ('e1_cnn_s1', 'E1: CNN, semilla 1 (sin target_kl)', 'ppo-local-planner-design.md, E1', False),
    ('e1_cnn_s2', 'E1: CNN, semilla 2 (sin target_kl)', 'ppo-local-planner-design.md, E1', False),
    ('full_cnn_s0', 'Primer run completo: LiDAR lineal, choque -10, sin target_kl',
     'resultados.md, "Primer run completo"', False),
    ('expA_col20', 'A: primer run + choque -20 (2M pasos)', 'resultados.md, "Experimentos A / B / C"',
     False),
    ('expB_col20_stall5', 'B: A + atasco terminal -5 (2M pasos)',
     'resultados.md, "Experimentos A / B / C"', False),
    ('expC_col20_inverse', 'C: A + LiDAR d/(d+1) (2M pasos)',
     'resultados.md, "Experimentos A / B / C"', False),
]
# evaluation set -> (file in v5/, scenario of its episodes)
SETS = {
    '2d_limpio': ('eval_clean_train.csv', None),
    '2d_obstaculos': ('eval_obstacles_train.csv', None),
    '2d_lab': ('eval_obstacles_lab.csv', None),
    '2d_lab_puntos': ('eval_obstacles_lab-points.csv', None),
    'gazebo_lab': ('eval_gazebo_lab.csv', None),
    'gazebo_lab_puntos': ('eval_gazebo_lab_points.csv', None),
}
PAIRS_2D_GAZEBO = [('2d_lab', 'gazebo_lab'), ('2d_lab_puntos', 'gazebo_lab_puntos')]
SIDE = ('_traj', '_route', '_obstacles')


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---- statistics, written out again here so the verification does not trust the package ----
def wilson(k, n, z=1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(b, c):
    """Exact two-sided McNemar: binomial test of b discordant pairs out of b + c, p = 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def seconds(row):
    return float(row['seconds']) if 'seconds' in row else int(row['steps']) * DT


def metrics(rows):
    n = len(rows)
    count = lambda o: sum(r['outcome'] == o for r in rows)  # noqa: E731
    ok = [r for r in rows if r['outcome'] == 'success']
    lo, hi = wilson(len(ok), n)
    out = {'episodios': n, 'exito': len(ok) / n, 'exito_ic95_inf': lo, 'exito_ic95_sup': hi}
    for name, o in (('colision', 'collision'), ('estancado', 'stalled'), ('timeout', 'timeout'),
                    ('planificador_fallo', 'failed'), ('perdido_localizacion', 'lost')):
        out[name] = count(o) / n
    out['spl'] = sum(float(r['spl']) for r in rows) / n
    out['tiempo_medio_exito_s'] = sum(seconds(r) for r in ok) / len(ok) if ok else None
    out['distancia_media_exito_m'] = sum(float(r['travelled']) for r in ok) / len(ok) if ok else None
    return out


def rounded(d):
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


# ---- per configuration ----
def commit_for(run_dir):
    """The last commit before the training started (config.yaml is written at the start)."""
    when = int((run_dir / 'config.yaml').stat().st_mtime)
    return subprocess.run(['git', 'log', '-1', f'--before=@{when}', '--format=%H %ad %s',
                           '--date=iso'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def commands(run, gazebo, commit):
    lines = [
        '#!/usr/bin/env bash',
        f'# {run}. Desde martha_nav/, con el contenedor ros2_humble levantado.',
        '# Entrenamiento: el código de la época (commit aproximado, por la fecha de inicio);',
        '# config.yaml tiene todos los parámetros con que corrió.',
        f'git checkout {commit.split()[0]}',
        f'#   y el comando de entrenamiento de tools/experiments/ o docs/comandos.md con --name {run}',
        'git checkout -',
        '# Evaluación 2D (código actual; determinista: repite los mismos números):',
        f'm=runs/{run}/best_model.zip; o=runs/{run}/v5',
        './tools/ct_ros ros2 run martha_nav evaluate_2d --model $m --episodes 500 --condition clean '
        '--out $o/eval_clean_train.csv',
        './tools/ct_ros ros2 run martha_nav evaluate_2d --model $m --episodes 500 --condition obstacles '
        '--out $o/eval_obstacles_train.csv',
        './tools/ct_ros ros2 run martha_nav evaluate_2d --model $m --episodes 200 --condition obstacles '
        '--sources lab --out $o/eval_obstacles_lab.csv',
        './tools/ct_ros ros2 run martha_nav evaluate_2d --model $m --condition obstacles --points lab '
        '--out $o/eval_obstacles_lab-points.csv',
    ]
    if gazebo:
        lines += ['# Gazebo (3 simulaciones en paralelo; Gazebo no es determinista: ~±4 puntos):',
                  f'./tools/evaluate_gazebo_fast.sh runs/{run} "" 3 runs/{run}/v5']
    return '\n'.join(lines) + '\n'


def copy_config(run, gazebo, dest):
    src = ROOT / 'runs' / run
    d = dest / 'configuraciones' / run
    (d / 'entrenamiento').mkdir(parents=True, exist_ok=True)
    for name in ('config.yaml', 'best_model.zip', 'vecnormalize.pkl'):
        shutil.copy2(src / name, d / name)
    for name in ('evals.csv', 'episodes.csv'):
        shutil.copy2(src / name, d / 'entrenamiento' / name)
    commit = commit_for(src)
    (d / 'commit.txt').write_text(commit + '\n')
    (d / 'comandos.sh').write_text(commands(run, gazebo, commit))
    for kind in ('2d', 'gazebo'):
        (d / f'eval_{kind}').mkdir(exist_ok=True)
    for key, (name, _) in SETS.items():
        if key.startswith('gazebo') and not gazebo:
            continue
        kind = key.split('_')[0]
        stem = name[:-4]
        for side in ('',) + SIDE:
            shutil.copy2(src / 'v5' / f'{stem}{side}.csv', d / f'eval_{kind}' / f'{stem}{side}.csv')
    return commit


def load(run, key):
    path = ROOT / 'runs' / run / 'v5' / SETS[key][0]
    return read(path) if path.exists() else None


# ---- tables ----
def summary_table(dest):
    rows = []
    for run, desc, doc, gazebo in CONFIGS:
        for key in SETS:
            data = load(run, key)
            if data:
                rows.append({'configuracion': run, 'conjunto': key, **rounded(metrics(data))})
    write(rows, dest / 'resumen_modelos.csv')
    return rows


def paired(a, b):
    """a, b: rows of the same episodes. Returns (n, success a, success b, a-only, b-only, p)."""
    sa = {r['episode_seed']: r['outcome'] == 'success' for r in a}
    sb = {r['episode_seed']: r['outcome'] == 'success' for r in b}
    seeds = [s for s in sb if s in sa]
    only_a = sum(sa[s] and not sb[s] for s in seeds)
    only_b = sum(sb[s] and not sa[s] for s in seeds)
    return (len(seeds), sum(sa[s] for s in seeds) / len(seeds), sum(sb[s] for s in seeds) / len(seeds),
            only_a, only_b, mcnemar(only_a, only_b))


def paired_table(dest):
    rows = []
    for run, _, _, gazebo in CONFIGS:
        if gazebo:
            for k2d, kgz in PAIRS_2D_GAZEBO:
                n, a, b, oa, ob, p = paired(load(run, k2d), load(run, kgz))
                rows.append({'comparacion': '2D vs Gazebo', 'configuracion': run, 'conjunto': kgz,
                             'a': f'{run} 2D', 'b': f'{run} Gazebo', 'episodios': n,
                             'exito_a': round(a, 4), 'exito_b': round(b, 4), 'solo_a': oa,
                             'solo_b': ob, 'p_mcnemar': round(p, 4)})
    for run, _, _, _ in CONFIGS:
        if run == REFERENCE:
            continue
        for key in SETS:
            ref, other = load(REFERENCE, key), load(run, key)
            if ref and other:
                n, a, b, oa, ob, p = paired(ref, other)
                rows.append({'comparacion': 'frente al modelo final', 'configuracion': run,
                             'conjunto': key, 'a': REFERENCE, 'b': run, 'episodios': n,
                             'exito_a': round(a, 4), 'exito_b': round(b, 4), 'solo_a': oa,
                             'solo_b': ob, 'p_mcnemar': round(p, 4)})
    write(rows, dest / 'comparaciones_pareadas.csv')


# ---- verification ----
def verification(dest):
    """Each number recomputed from the per-episode rows, against the package's own code."""
    checks = []

    def check(what, ok, detail=''):
        checks.append((what, ok, detail))

    for run, _, _, gazebo in CONFIGS:
        for key in SETS:
            data = load(run, key)
            if not data:
                continue
            tag = f'{run} / {key}'
            mine = metrics(data)
            theirs = summarize([{**r, 'spl': float(r['spl'])} for r in data])
            same = all(abs(mine[a] - theirs[b]) < 1e-12 for a, b in (
                ('exito', 'success'), ('colision', 'collision'), ('estancado', 'stalled'),
                ('timeout', 'timeout'), ('spl', 'spl')))
            ci = all(abs(x - y) < 1e-4 for x, y in zip(
                (mine['exito_ic95_inf'], mine['exito_ic95_sup']), theirs['success_ci']))
            check(f'{tag}: tasas, SPL e IC95 iguales al código', same and ci)
            bad = [r['episode_seed'] for r in data if abs(float(r['spl']) - (
                float(r['shortest']) / max(float(r['shortest']), float(r['travelled']))
                if r['outcome'] == 'success' else 0.0)) > 2e-3]
            check(f'{tag}: SPL = éxito x más_corta / max(más_corta, recorrida) en cada episodio',
                  not bad, ', '.join(bad[:5]))
            seeds = [int(r['episode_seed']) for r in data]
            check(f'{tag}: semillas de evaluación únicas y >= 1 000 000 (disjuntas del entrenamiento)',
                  len(set(seeds)) == len(seeds) and min(seeds) >= 1_000_000)
            traj = read(ROOT / 'runs' / run / 'v5' / (SETS[key][0][:-4] + '_traj.csv'))
            walked = {}
            last = {}
            for p in traj:
                s = p['episode_seed']
                xy = (float(p['x']), float(p['y']))
                if s in last:
                    walked[s] = walked.get(s, 0.0) + math.dist(last[s], xy)
                last[s] = xy
            off = [r['episode_seed'] for r in data
                   if abs(walked.get(r['episode_seed'], 0.0) - float(r['travelled'])) > 0.01]
            check(f'{tag}: distancia recorrida = suma de la trayectoria registrada', not off,
                  ', '.join(off[:5]))
    # The scenario of every episode is the same in 2D and Gazebo: same route, same obstacles.
    for run, _, _, gazebo in CONFIGS:
        if not gazebo:
            continue
        for k2d, kgz in PAIRS_2D_GAZEBO:
            r2d = read(ROOT / 'runs' / run / 'v5' / (SETS[k2d][0][:-4] + '_route.csv'))
            rgz = read(ROOT / 'runs' / run / 'v5' / (SETS[kgz][0][:-4] + '_route.csv'))
            seeds = {r['episode_seed'] for r in rgz}
            a = [(r['episode_seed'], r['x'], r['y']) for r in r2d if r['episode_seed'] in seeds]
            b = [(r['episode_seed'], r['x'], r['y']) for r in rgz]
            check(f'{run}: rutas de {kgz} idénticas a las de {k2d} (episodios pareados)', a == b)
    lines = ['# Verificación de los cálculos', '',
             'Generado por `tools/build_entrega.py`. Cada número se recalcula aquí desde las filas',
             'por episodio, con fórmulas escritas aparte del código del paquete, y se compara con lo',
             'que entrega el código (`summarize` de `evaluate_2d`).', '',
             '- Éxito, colisión, estancamiento, timeout: proporción de episodios con ese resultado.',
             '- IC95: intervalo de Wilson, z = 1.96.',
             '- SPL (Anderson et al., 2018): por episodio, éxito x L / max(L, P), con L la ruta más',
             '  corta con obstáculos y P la distancia recorrida; se promedia sobre todos los episodios.',
             '- Comparaciones pareadas: McNemar exacto bilateral (binomial de los pares discordantes).',
             '', f'**{sum(ok for _, ok, _ in checks)} de {len(checks)} comprobaciones correctas.**', '',
             '| comprobación | resultado | detalle |', '|---|---|---|']
    lines += [f'| {w} | {"ok" if ok else "FALLA"} | {d} |' for w, ok, d in checks]
    (dest / 'verificacion.md').write_text('\n'.join(lines) + '\n')
    return checks


# ---- trajectory figures ----
def trajectory_figure(dest, run=REFERENCE, n=6):
    """Route, obstacles and the 2D and Gazebo trajectories of the same episodes."""
    base = ROOT / 'runs' / run / 'v5'
    outcome2d = {r['episode_seed']: r['outcome'] for r in read(base / 'eval_obstacles_lab.csv')}
    outcomegz = {r['episode_seed']: r['outcome'] for r in read(base / 'eval_gazebo_lab.csv')}
    seeds = sorted(outcomegz, key=int)
    same = [s for s in seeds if outcome2d[s] == outcomegz[s] == 'success']
    differ = [s for s in seeds if outcome2d[s] != outcomegz[s]]
    chosen = same[:n - min(2, len(differ))] + differ[:2]
    t2d, tgz = read(base / 'eval_obstacles_lab_traj.csv'), read(base / 'eval_gazebo_lab_traj.csv')
    route = read(base / 'eval_gazebo_lab_route.csv')
    cfg = ScenarioConfig(sources=('lab',), obstacle_mode=CONDITIONS['obstacles'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, seed in zip(axes.flat, chosen):
        sc = generate(int(seed), cfg)
        g = sc.full
        h, w = g.occ.shape
        ax.imshow(g.occ, origin='lower', cmap='Greys', alpha=0.6, extent=(
            g.origin[0], g.origin[0] + w * g.resolution, g.origin[1], g.origin[1] + h * g.resolution))
        pts = lambda rows: ([float(p['x']) for p in rows if p['episode_seed'] == seed],  # noqa: E731
                            [float(p['y']) for p in rows if p['episode_seed'] == seed])
        ax.plot(*pts(route), '--', color='0.4', lw=1.5, label='ruta A* (referencia)')
        ax.plot(*pts(t2d), color='tab:blue', lw=2, label=f'2D: {outcome2d[seed]}')
        ax.plot(*pts(tgz), color='tab:orange', lw=2, label=f'Gazebo: {outcomegz[seed]}')
        ax.plot(*sc.start[:2], 'o', color='k')
        ax.plot(*sc.goal, '*', color='tab:green', ms=14)
        xs, ys = pts(route)
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'semilla {seed}, {len(sc.obstacles)} obstáculos')
        ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'{run}: el mismo episodio en 2D y en Gazebo (lab.world, obstáculos fuera del mapa)')
    fig.tight_layout()
    (dest / 'figuras').mkdir(exist_ok=True)
    fig.savefig(dest / 'figuras' / f'trayectorias_2d_vs_gazebo_{run}.png', dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(ROOT / 'entrega_tesis'))
    dest = Path(ap.parse_args().out)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run, desc, doc, gazebo in CONFIGS:
        commit = copy_config(run, gazebo, dest)
        cfg = (ROOT / 'runs' / run / 'config.yaml').read_text()
        manifest.append({'configuracion': run, 'descripcion': desc, 'documento': doc,
                         'gazebo': 'sí' if gazebo else 'no', 'commit_aproximado': commit,
                         'pasos': next((ln.split()[1] for ln in cfg.splitlines()
                                        if ln.startswith('steps:')), ''),
                         'semilla_entrenamiento': next((ln.split()[1] for ln in cfg.splitlines()
                                                        if ln.startswith('seed:')), '')})
    write(manifest, dest / 'MANIFEST.csv')
    for doc in (ROOT / 'docs' / 'entrega').glob('*.md'):      # README, recompensa, datos faltantes
        shutil.copy2(doc, dest / doc.name)
    summary_table(dest)
    paired_table(dest)
    checks = verification(dest)
    trajectory_figure(dest)
    print(f'{dest}: {len(CONFIGS)} configuraciones; verificación {sum(c[1] for c in checks)}/'
          f'{len(checks)}')


if __name__ == '__main__':
    main()
