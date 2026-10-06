# TurtleBot3 Burger en martha_nav — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que `martha_nav` entrene, evalúe y despliegue la política tanto en Martha como en un TurtleBot3 Burger, con un perfil de robot en vez de constantes fijas, y que el Burger haga la demo con todo corriendo a bordo.

**Architecture:** un módulo `martha_nav/robots.py` con un `Robot` inmutable por robot. El simulador 2D, el entrenamiento, los nodos ROS y los launch leen el perfil (el del `config.yaml` del modelo). Martha conserva exactamente sus números (huella de episodios idéntica). Para el robot, la política se exporta a `.npz` y corre en numpy, sin PyTorch.

**Tech Stack:** Python 3.10, ROS 2 Humble (rclpy, launch), Gazebo Classic 11, Stable-Baselines3 2.3.2, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-burger-support-design.md`

## Global Constraints

- Martha no cambia: `runs/_fingerprint.py` (con las variantes que fija la Tarea 3) da siempre el mismo hash, y la evaluación 2D de `wide_dyn_s0` repite sus CSV (salvo las columnas `r_proximity`, `r_turn` y `r_stalled`, que desaparecen en la Tarea 3).
- Los runs sin campo `robot` en su `config.yaml` son `martha`.
- El lado del robot (`ppo_local_planner` con `.npz`, `policy_core`, `numpy_policy`, `robots`) no importa `torch`, `stable_baselines3` ni `gymnasium`.
- Código en inglés con el estilo del repo (docstrings cortos, sin comentarios obvios); documentación en español.
- El código no guarda historia: nada de guiones de experimentos ya corridos, código legado ni comentarios sobre runs viejos; eso va en `docs/*.md`.
- Todo se corre desde `martha_nav/` con el contenedor levantado: tests con `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null`.
- Commits pequeños por tarea, terminados en `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## Mapa de archivos

| archivo | qué hace | tarea |
|---|---|---|
| `martha_nav/learning/policy.py`, `train.py` | solo CNN | 1 |
| `tools/experiments/`, `tools/plot_report_legacy.py` (borrados), `tools/plot_2d_vs_gazebo.py`, comentarios, `docs/*.md` | código sin historia ni E1/E2 | 2 |
| `sim2d/observation.py`, `reward.py`, `env.py`, `learning/train.py`, `evaluate.py`, `ros/policy_core.py`, `ppo_local_planner.py` | sin LiDAR `linear`, señal ni castigo de atasco, proximidad ni giro | 3 |
| `martha_nav/robots.py` (nuevo) | `Robot`, `ROBOTS`, `checkpoint_robot()` | 4 |
| `martha_nav/sim2d/geometry.py`, `observation.py` | contorno, scan y escalas del perfil | 5 |
| `martha_nav/sim2d/env.py`, `learning/train.py`, `learning/evaluate.py` | `EnvConfig.robot`, LiDAR a 5 Hz, `--robot` | 6 |
| `martha_nav/ros/policy_core.py`, `global_planner.py`, `slam.py` | nodos con perfil | 7 |
| `martha_nav/learning/__init__.py`, `ros/numpy_policy.py` (nuevo), `learning/export.py` (nuevo) | política en numpy | 8 |
| `martha_nav/ros/ppo_local_planner.py` | `.npz`, chequeo de robot, `speed_scale`, latencia | 9 |
| `urdf/burger.urdf.xacro` (nuevo), `launch/sim.launch.py`, `ros/evaluate_gazebo.py`, `ros/gazebo_ground_truth_tf.py`, `tools/evaluate_run_gazebo.sh` | Burger en Gazebo | 10 |
| `launch/burger.launch.py` (nuevo), `tools/deploy_burger.sh` (nuevo), `docs/comandos.md` | Burger real | 11 |
| `runs/burger_s0/`, `docs/resultados.md` | entrenar y evaluar | 12 |

---

### Task 1: Solo la arquitectura CNN

**Files:**
- Modify: `martha_nav/learning/policy.py:36-44`
- Modify: `martha_nav/learning/train.py:1-5,110-117,143,171,177,186`
- Modify: `docs/comandos.md:31,37,45,57`
- Test: `test/test_policy.py:33-37`, `test/test_train.py:9,49`

**Interfaces:**
- Produces: `policy_kwargs() -> dict` (sin argumentos); `build_model(venv, recurrent, seed, device, tensorboard_log)`.

- [ ] **Step 1: Cambiar los tests al API sin `arch`**

En `test/test_policy.py` reemplazar `test_policy_kwargs` por:

```python
def test_policy_kwargs_use_the_cnn():
    kwargs = policy_kwargs()
    assert kwargs['features_extractor_class'] is LidarCnnExtractor
    assert kwargs['share_features_extractor'] is False
```

y borrar `import pytest` si queda sin uso. En `test/test_train.py`: en la línea 9 quitar `'--arch', 'cnn', ` de la lista, y en la línea 49 cambiar `build_model(venv, arch='cnn', recurrent=True, ...` por `build_model(venv, recurrent=True, ...`.

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_policy.py test/test_train.py </dev/null`
Expected: FAIL (`policy_kwargs() missing 1 required positional argument: 'arch'`).

- [ ] **Step 3: Implementar**

`martha_nav/learning/policy.py`: el docstring del módulo pasa a `"""Feature extractor and SB3 policy kwargs: a circular 1D CNN over the LiDAR."""` y la función:

```python
def policy_kwargs():
    """SB3 policy_kwargs: the CNN extractor, separate for the actor and the critic."""
    return dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), share_features_extractor=False,
                log_std_init=-0.5, features_extractor_class=LidarCnnExtractor)
```

`martha_nav/learning/train.py`:
- docstring de uso (líneas 3-4): quitar `--arch cnn `.
- `def build_model(venv, recurrent, seed, device, tensorboard_log):` y `kwargs = dict(policy_kwargs=policy_kwargs(), ...`.
- borrar `ap.add_argument('--arch', ...)`.
- `name = args.name or f'{args.preset}_s{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'`
- en `config` quitar `'arch': args.arch, `.
- `model = build_model(venv, args.recurrent, args.seed, args.device, str(run_dir / 'tb'))`.

`docs/comandos.md`: quitar `--arch cnn ` de las líneas 31, 37 y 57, y borrar la fila de la tabla `| `--arch cnn\|mlp` | arquitectura del extractor |`.

- [ ] **Step 4: Ver pasar y que nada más se rompa**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null`
Expected: todo PASS. `grep -rn "arch" martha_nav/learning test docs/comandos.md | grep -iv "search\|archiv"` no muestra `--arch` ni `mlp`.

- [ ] **Step 5: Commit**

```bash
git add martha_nav/learning test/test_policy.py test/test_train.py docs/comandos.md
git commit -m "Keep only the CNN architecture

The MLP existed only for the E1 comparison and will not be used again;
runs trained with it still load (the zip stores its architecture).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Código limpio: sin E1/E2 ni registros históricos

Los guiones de experimentos ya corridos y las menciones a runs viejos o al "paquete anterior" salen
del código. Lo que haga falta recordar queda en `docs/comandos.md`.

**Files:**
- Delete: `tools/experiments/` (todo), `tools/plot_report_legacy.py`
- Rename: `tools/plot_e2.py` → `tools/plot_2d_vs_gazebo.py`
- Modify: `martha_nav/sim2d/env.py:43`, `martha_nav/sim2d/reward.py:13`, `martha_nav/learning/evaluate.py:109`, `martha_nav/ros/mecanum_cmd_vel_bridge.py:4`, `martha_nav/ros/evaluate_gazebo.py:8`, `martha_nav/sim2d/scenarios.py:120`, `urdf/martha.urdf.xacro:143-144`, `tools/plot_report.py:1-12`, `tools/compare_runs.py:3-4`, `tools/run_demo_worlds.sh:15`, `tools/evaluate_run_gazebo.sh:2,7`, `tools/build_entrega.py:120`
- Modify: `docs/comandos.md` (líneas ~110-126, 176, sección 7), `docs/contexto.md:39-40`, `docs/limitaciones-datos-tesis.md:21`, `docs/resultados.md` (títulos)

- [ ] **Step 1: Borrar lo histórico**

```bash
git rm -r -q tools/experiments tools/plot_report_legacy.py
git mv tools/plot_e2.py tools/plot_2d_vs_gazebo.py
```

- [ ] **Step 2: `tools/plot_2d_vs_gazebo.py` lee una carpeta de evaluaciones, sin `--tag`**

Docstring del módulo:

```python
"""The same episodes in the 2D simulator and in Gazebo: outcome shares side by side.

python3 tools/plot_2d_vs_gazebo.py --dir runs/wide_dyn_s0/v5 --out docs/figures/2d_vs_gazebo.png

--dir holds a run's evaluations: eval_obstacles_lab{,-points}.csv and eval_gazebo_lab{,_points}.csv.
"""
```

```python
CONDITIONS = [('semillas generadas', 'eval_obstacles_lab.csv', 'eval_gazebo_lab.csv'),
              ('puntos fijos', 'eval_obstacles_lab-points.csv', 'eval_gazebo_lab_points.csv')]
```

En `main`: `ap.add_argument('--dir', default='runs/wide_dyn_s0/v5')` en vez de `--run` y `--tag`;
`run = Path(args.dir)`; borrar `fgz = fgz.format(tag=args.tag)`; el título pasa a
`'Los mismos episodios en el simulador 2D y en Gazebo (lab.world)'`. Después,
`grep -n "args.run\|args.tag\|tag" tools/plot_2d_vs_gazebo.py` no debe mostrar nada.

- [ ] **Step 3: Comentarios y docstrings sin historia** (reemplazos exactos)

| archivo | antes | después |
|---|---|---|
| `sim2d/env.py:43` | `lidar_encoding: str = 'inverse'  # see observation.encode_lidar; 'linear' in full_cnn_s0` | `lidar_encoding: str = 'inverse'  # see observation.encode_lidar` |
| `sim2d/reward.py:13` | `collision: float = -20.0     # -10 in full_cnn_s0; see docs/resultados.md (A/B/C)` | `collision: float = -20.0` |
| `learning/evaluate.py:109` | `# Runs trained before these options existed used the linear encoding and (v, w).` | `# Defaults for configs that do not record an option.` |
| `ros/mecanum_cmd_vel_bridge.py:4` | `TwistStamped on <controller>/reference. Reused from the previous package.` | `TwistStamped on <controller>/reference.` |
| `ros/evaluate_gazebo.py:8` | `"points" plays the hand-placed start/goal pairs of the previous package. Both build` | `"points" plays the hand-placed start/goal pairs of config/training_points.yaml. Both build` |
| `sim2d/scenarios.py:120` | `"""The hand-placed start/goal points of a world, from the previous package."""` | `"""The hand-placed start/goal points of a world (config/training_points.yaml)."""` |
| `urdf/martha.urdf.xacro:143-144` | `<!-- Mecanum roller. The contact parameters come from the previous package:` / `without maxVel and minDepth the rollers chatter and the robot hops. -->` | `<!-- Mecanum roller. Without maxVel and minDepth the rollers chatter and the robot hops. -->` |
| `tools/plot_report.py:1` | `"""Per-run report: the panels of the old ppo_plot, drawn for print.` | `"""Per-run report: learning curves and PPO diagnostics, drawn for print.` |
| `tools/plot_report.py:3` | `python3 tools/plot_report.py runs/e1_cnn_s0` | `python3 tools/plot_report.py runs/wide_dyn_s0` |
| `tools/plot_report.py:12` | `For the exact look of the previous package, use tools/plot_report_legacy.py.` | (borrar la línea) |
| `tools/compare_runs.py:3-4` | `python3 tools/compare_runs.py --out docs/figures/exp_abc.png \` / `    base=runs/full_cnn_s0 A=runs/expA_col20 B=runs/expB_col20_stall5 C=runs/expC_col20_inverse` | `python3 tools/compare_runs.py --out docs/figures/comparacion.png \` / `    final=runs/wide_dyn_s0 base=runs/long_c_kl_s0 H=runs/armH_holonomic_s0` |
| `tools/run_demo_worlds.sh:15` | `MODEL=${MODEL:-runs/long_c_kl_s0/best_model.zip}` | `MODEL=${MODEL:-runs/wide_dyn_s0/best_model.zip}` |
| `tools/evaluate_run_gazebo.sh:2` | `# The standard Gazebo evaluation (E2) of a run's best model, as evaluate_run.sh is for 2D:` | `# The standard Gazebo evaluation of a run's best model, as evaluate_run.sh is for 2D:` |
| `tools/evaluate_run_gazebo.sh:7` | `#   [EPISODES=100] ./tools/evaluate_run_gazebo.sh runs/wide_dyn_s0 _v5 [instances] [out_dir]` | `#   [EPISODES=100] ./tools/evaluate_run_gazebo.sh runs/wide_dyn_s0 "" [instances] [out_dir]` |
| `tools/build_entrega.py:120` | `f'#   y el comando de entrenamiento de tools/experiments/ o docs/comandos.md con --name {run}',` | `f'#   train_policy con los flags de su config.yaml y --name {run} (docs/comandos.md)',` |

- [ ] **Step 4: Documentación**

`docs/comandos.md`:
- Borrar el bloque "Mismo contenido con el formato exacto del `ppo_plot` anterior" con su comando `plot_report_legacy.py`, y el bloque con `tools/experiments/plot_e1.py`.
- El comando de `tools/plot_e2.py` pasa a `./tools/ct python3 tools/plot_2d_vs_gazebo.py --dir runs/wide_dyn_s0/v5 --out docs/figures/2d_vs_gazebo.png`.
- `## 6. Evaluación en Gazebo (E2)` → `## 6. Evaluación en Gazebo`.
- La sección `## 7. Guiones de experimentos ya hechos` entera (título, párrafo y el bloque con `run_e1.sh`) se reemplaza por:

````markdown
## 7. Experimentos ya corridos

Los guiones que lanzaron los experimentos de la bitácora (A/B/C, CNN contra MLP, los brazos y la
primera evaluación en Gazebo) ya no están en el repositorio. Los parámetros completos de cada run
están en su `config.yaml`, y los guiones en el historial de git:

```bash
git log --diff-filter=D --name-only -- tools/experiments
```
````

`docs/contexto.md`: en la fila de `tools/` quitar `; \`experiments/\` guarda las colas ya corridas`;
en la fila de `docs/resultados.md` cambiar `E1, E2` por `CNN contra MLP, brecha 2D → Gazebo`.

`docs/limitaciones-datos-tesis.md:21`: `exactos están en \`tools/experiments/\` del repositorio, para E1, A/B/C y los brazos.` →
`exactos están en el historial de git (\`git log --diff-filter=D -- tools/experiments\`).`

`docs/resultados.md`, títulos:
- `## E2: brecha 2D → Gazebo en \`lab.world\`` → `## Brecha 2D → Gazebo en \`lab.world\``
- `### Corrección de E2: la brecha era` → `### Corrección: la brecha era`
- `### E2 final, con la regla` → `### Brecha 2D → Gazebo, con la regla`
- `### E2 v4 (2026-09-29):` → `### Brecha 2D → Gazebo v4 (2026-09-29):`
- `## Causa de la brecha de E2:` → `## Causa de la brecha 2D → Gazebo:`

- [ ] **Step 5: Verificar**

Run: `grep -rn "tools/experiments\|plot_report_legacy\|plot_e2\|previous package\|full_cnn\|expA\|(E2)" martha_nav tools launch urdf config --include=*.py --include=*.sh --include=*.xacro --include=*.yaml`
Expected: sin resultados.
Run: `./tools/ct python3 tools/plot_2d_vs_gazebo.py --out runs/_plot_check.png && ls runs/_plot_check.png && rm runs/_plot_check.png` → genera la figura.
Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null` → todo PASS.

- [ ] **Step 6: Commit**

```bash
git add -A tools docs martha_nav urdf
git commit -m "Remove the scripts of experiments already run and the history notes in the code

The experiment queues and the legacy report live in git history; docs/comandos.md
says how to find them. E2 is now the 2D -> Gazebo gap in names and headings.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Quitar las opciones que ya no se usan

Salen la codificación `linear` del LiDAR, la señal de atasco en la observación, el castigo de
atasco y los términos de proximidad y giro brusco de la recompensa (siempre en 0). El progreso
geodésico se queda. Los runs que usaron lo que sale (el primer run y A/B, los brazos S y SL) dejan
de poder evaluarse; ninguno está en la tesis.

**Files:**
- Modify: `martha_nav/sim2d/observation.py` (`obs_dim`, `encode_lidar`, `build_observation`, `LIDAR_ENCODINGS`)
- Modify: `martha_nav/sim2d/reward.py` (todo el archivo)
- Modify: `martha_nav/sim2d/env.py` (`EnvConfig`, espacio de observación, `step`, `_obs`)
- Modify: `martha_nav/sim2d/planner.py:50` (docstring de `RouteProgress`)
- Modify: `martha_nav/learning/train.py` (`build_config`, flags), `martha_nav/learning/evaluate.py` (`trained_env_config`)
- Modify: `martha_nav/ros/policy_core.py`, `martha_nav/ros/ppo_local_planner.py`
- Modify: `docs/comandos.md:49,51` (filas de flags)
- Modify (local, fuera de git): `runs/_fingerprint.py`
- Test: `test/test_observation.py`, `test/test_reward.py`, `test/test_env.py`, `test/test_train.py`, `test/test_evaluate.py`, `test/test_policy_core.py`

**Interfaces:**
- Produces:
  - `obs_dim(action_dim=2)`; `encode_lidar(sectors)` = `d / (d + 1)`.
  - `build_observation(ranges, angles, velocity, waypoint_rel, prev_action, waypoint_max=WAYPOINT_MAX)`.
  - `RewardConfig(progress, progress_mode, goal, collision, step)`; `compute_reward(progress_gain, reached, collided, cfg=RewardConfig())` → términos `progress`, `goal`, `collision`, `step`.
  - `EnvConfig` sin `lidar_encoding` ni `stuck_signal`; un estancamiento siempre trunca el episodio.
  - `build_config(preset, collision=None, action_dim=2, target='carrot', progress_mode='route', wide_dynamics=False)`.
  - `trained_env_config()` lanza `ValueError` ("no longer") con `lidar_encoding: linear` o `stuck_signal: true`.
  - `PolicyCore(model, lookahead=1.5, carrot_clearance=0.4, action_dim=2, target='carrot', action_delay=0)`; su `info` ya no trae `'stuck'`.

- [ ] **Step 1: Fijar la huella nueva con el código de hoy (antes de tocar nada)**

La huella vieja usaba variantes con señal y castigo de atasco. Reemplazar `runs/_fingerprint.py`
completo por:

```python
import hashlib
import numpy as np
from martha_nav.sim2d.env import EnvConfig, NavEnv
from martha_nav.sim2d.reward import RewardConfig
# Only options that stay; the reward terms removed in Task 3 are left out of the hash.
VARIANTS = [EnvConfig(), EnvConfig(action_dim=3),
            EnvConfig(target='goal', reward=RewardConfig(progress_mode='geodesic'))]
GONE = ('r_proximity', 'r_turn', 'r_stalled')
h = hashlib.sha256()
for cfg in VARIANTS:
    env = NavEnv(cfg)
    rng = np.random.default_rng(0)
    for ep in range(6):
        obs, _ = env.reset(seed=100 + ep)
        h.update(obs.tobytes())
        for t in range(400):
            a = rng.uniform(-1, 1, cfg.action_dim) * (0.2 if t > 150 else 1.0)
            obs, r, term, trunc, info = env.step(a)
            h.update(obs.tobytes()); h.update(np.float64(r).tobytes())
            h.update(bytes([term, trunc]))
            if term or trunc:
                h.update(repr(sorted((k, v) for k, v in info.items() if k not in GONE)).encode())
                break
print(h.hexdigest())
```

Run: `./tools/ct_ros python3 runs/_fingerprint.py </dev/null` y agregar el hash impreso como
última línea del archivo: `# expected: <hash>`. Ese es el hash que todas las tareas siguientes
deben reproducir.

- [ ] **Step 2: Tests**

`test/test_reward.py` completo:

```python
from dataclasses import replace

from martha_nav.sim2d.reward import RewardConfig, compute_reward


def test_default_terms():
    total, t = compute_reward(0.1, False, False)
    assert t == {'progress': 0.1, 'goal': 0.0, 'collision': 0.0, 'step': -0.005}
    assert abs(total - 0.095) < 1e-12


def test_terminal_values():
    assert compute_reward(0.0, True, False)[1]['goal'] == 20.0
    assert compute_reward(0.0, False, True)[1]['collision'] == -20.0


def test_negative_progress_is_not_paid():
    assert compute_reward(-0.5, False, False)[1]['progress'] == 0.0


def test_geodesic_progress_can_be_negative_and_route_progress_cannot():
    geo = replace(RewardConfig(), progress_mode='geodesic')
    assert compute_reward(-0.3, False, False, geo)[1]['progress'] == -0.3
    assert compute_reward(-0.3, False, False, RewardConfig())[1]['progress'] == 0.0
```

`test/test_observation.py`:
- En `test_observation_layout_and_bounds`, borrar las tres líneas de `linear = build_observation(...)` (dos líneas) y `assert np.allclose(linear[:90], 0.5)`.
- En `test_inverse_lidar_encoding_gives_more_resolution_up_close`, quitar los argumentos `, lidar_encoding='inverse'` (tres llamadas; la tercera queda `empty = build_observation(np.array([np.inf]), np.zeros(1), (0, 0), (1, 0), (0, 0))`) y el comentario `# linear /8 m gives only 0.025`.
- Borrar `test_unknown_lidar_encoding_is_rejected` y `test_stuck_signal_adds_one_value_at_the_end`.

`test/test_env.py`: borrar `test_stall_penalty_turns_stalls_into_terminal_episodes`,
`test_lidar_encoding_reaches_the_observation` y `test_stuck_signal_grows_while_the_robot_does_not_advance`.

`test/test_train.py`: `test_experiment_flags_reach_the_env_config` pasa a:

```python
def test_experiment_flags_reach_the_env_config():
    from martha_nav.learning.train import build_config
    assert build_config('full').reward.collision == -20.0
    cfg = build_config('full', collision=-10.0, progress_mode='geodesic')
    assert cfg.reward.collision == -10.0 and cfg.reward.progress_mode == 'geodesic'
```

`test/test_evaluate.py`: `test_evaluation_uses_the_env_the_model_was_trained_with` pasa a estos dos:

```python
def test_evaluation_uses_the_env_the_model_was_trained_with(tmp_path):
    from martha_nav.learning.evaluate import trained_env_config
    model = tmp_path / 'best_model.zip'
    assert trained_env_config(model) == EnvConfig()                             # no config: defaults
    (tmp_path / 'config.yaml').write_text('env:\n  action_dim: 3\n  target: goal\n')
    cfg = trained_env_config(model)
    assert (cfg.action_dim, cfg.target) == (3, 'goal')


def test_a_model_trained_with_a_removed_option_is_refused(tmp_path):
    import pytest

    from martha_nav.learning.evaluate import trained_env_config
    model = tmp_path / 'best_model.zip'
    for env in ('lidar_encoding: linear', 'lidar_encoding: inverse\n  stuck_signal: true'):
        (tmp_path / 'config.yaml').write_text(f'env:\n  {env}\n')
        with pytest.raises(ValueError, match='no longer'):
            trained_env_config(model)
```

`test/test_policy_core.py`:
- `test_progress_along_a_replanned_route_counts`: `PolicyCore(FakeModel(), stuck_signal=True)` → `PolicyCore(FakeModel())`, y la última línea `assert info['stuck'] == 0.0` → `assert np.isclose(info['s'], 0.5)` (medio metro sobre el desvío).
- Borrar `test_stuck_time_survives_a_replan_to_the_same_goal` (el reloj de atasco ya no lo usa el núcleo; `test_planner.py` cubre `RouteProgress.reroute`).

- [ ] **Step 3: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_reward.py test/test_train.py test/test_evaluate.py test/test_policy_core.py </dev/null`
Expected: FAIL (`compute_reward() missing 2 required positional arguments`, `unexpected keyword 'progress_mode'`, no se lanza `ValueError`, `info['s']`).

- [ ] **Step 4: Implementar**

`martha_nav/sim2d/reward.py` completo:

```python
"""Reward terms. Every term is logged apart."""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    progress: float = 1.0        # per metre of progress, measured as progress_mode says
    # 'route': new-record arc length along the A* route (never negative).
    # 'geodesic': drop in geodesic distance to the goal on the map with the obstacles,
    # potential-based shaping (Ng et al. 1999), so moving away costs what coming back pays.
    progress_mode: str = 'route'
    goal: float = 20.0
    collision: float = -20.0
    step: float = -0.005


def compute_reward(progress_gain, reached, collided, cfg=RewardConfig()):
    """Return (total, terms). progress_gain is in metres; see RewardConfig.progress_mode."""
    gain = progress_gain if cfg.progress_mode == 'geodesic' else max(progress_gain, 0.0)
    terms = {
        'progress': cfg.progress * gain,
        'goal': cfg.goal if reached else 0.0,
        'collision': cfg.collision if collided else 0.0,
        'step': cfg.step,
    }
    return sum(terms.values()), terms
```

`martha_nav/sim2d/observation.py`:

```python
def obs_dim(action_dim=2):
    """90 LiDAR sectors, the waypoint, the measured velocity and the previous action."""
    return N_SECTORS + 2 + 2 * action_dim
```

Borrar `LIDAR_ENCODINGS`;

```python
def encode_lidar(sectors):
    """Metres -> [0, 1) as d / (d + 1 m): finer up close, where it matters."""
    return sectors / (sectors + 1.0)
```

En `build_observation`: firma `def build_observation(ranges, angles, velocity, waypoint_rel, prev_action, waypoint_max=WAYPOINT_MAX):`; el docstring pierde las líneas de `stuck` y dice `96-value observation in [-1, 1] (98 with the holonomic action space).`; en el cuerpo `lidar = encode_lidar(reduce_scan(ranges, angles))`, y se borran `if stuck is not None: parts.append([float(stuck)])`. El comentario de `LIDAR_MAX` queda `# m, RPLIDAR A2M8 range`.

`martha_nav/sim2d/env.py`:
- En `EnvConfig`, borrar `lidar_encoding` y `stuck_signal` con sus comentarios.
- `spaces.Box(-1.0, 1.0, (obs_dim(self.cfg.action_dim),), np.float32)`.
- En `step`:

```python
        reward, terms = compute_reward(self._reward_progress(gain, x, y), reached, collided,
                                       self.cfg.reward)
        for k, v in terms.items():
            self.terms[k] += v
        self.prev_action = action
        # A stall truncates the episode: its value is bootstrapped, not punished.
        terminated = collided or reached
        truncated = not terminated and (self.steps >= self.max_steps or stalled)
```

- En `_obs`, borrar las dos líneas de `stuck = ...` y llamar
  `build_observation(self.ranges, self.ray_angles, vel, rel, self.prev_action, scale)`.

`martha_nav/sim2d/planner.py:50`: `One rule for the 2D environment's stall, the policy's stuck signal and the end` →
`One rule for the 2D environment's stall and the end`, y la línea siguiente `of a Gazebo episode, so the three measure the same thing.` → `of a Gazebo episode, so the two measure the same thing.`

`martha_nav/learning/train.py`:

```python
def build_config(preset, collision=None, action_dim=2, target='carrot', progress_mode='route',
                 wide_dynamics=False):
    """EnvConfig for a preset; the optional arguments are experiment overrides."""
    p = PRESETS[preset]
    cfg = EnvConfig()
    reward = replace(cfg.reward, progress_mode=progress_mode)
    if collision is not None:
        reward = replace(reward, collision=collision)
    if wide_dynamics:
        cfg = replace(cfg, dynamics=WIDE_DYNAMICS)
    return replace(cfg, reward=reward, action_dim=action_dim, target=target,
                   scenario=replace(cfg.scenario, sources=tuple(p['sources']),
                                    obstacle_mode=p['obstacle_mode']))
```

En `main`: borrar `--reward-stalled`, `--lidar-encoding` y `--stuck-signal`, y
`env_cfg = build_config(args.preset, args.reward_collision, args.action_dim, args.target, args.reward_progress, args.wide_dynamics)`.

`martha_nav/learning/evaluate.py`:

```python
def trained_env_config(model_path):
    """The EnvConfig the model was trained with: its observation must match."""
    config = _run_config(model_path)
    if not config:
        return EnvConfig()
    saved = config['env']
    if saved.get('lidar_encoding') == 'linear' or saved.get('stuck_signal'):
        raise ValueError(f'{model_path} was trained with options this code no longer has')
    dynamics = DynamicsRanges(**{k: tuple(v) for k, v in saved.get('dynamics', {}).items()})
    # Defaults for configs that do not record an option.
    return EnvConfig(dynamics=dynamics, action_dim=saved.get('action_dim', 2),
                     target=saved.get('target', 'carrot'))
```

`martha_nav/ros/policy_core.py`:
- `def __init__(self, model, lookahead=1.5, carrot_clearance=0.4, action_dim=2, target='carrot', action_delay=0):` y borrar `self.lidar_encoding`, `self.stuck_signal`, `self.no_progress_time`.
- En `compute`: el comentario `# Same rule as the 2D environment, so the stuck signal means the same thing.` pasa a `# The progress along the route places the carrot.`; borrar las dos líneas de `stuck = ...`; `obs = build_observation(ranges, angles, velocity, rel, self.prev_action, scale)`; en el `return`, el diccionario queda `{'blocked': sorted(blocked), 's': self.progress.s, 'carrot': point}`.

`martha_nav/ros/ppo_local_planner.py`: el `PolicyCore(...)` pierde `lidar_encoding=...`,
`stuck_signal=...` y `no_progress_time=...`; el log queda
`self.get_logger().info(f'action space: {self.action_dim}D, action delay {self.core.action_delay}')`.

`docs/comandos.md`: borrar las filas de `--lidar-encoding inverse\|linear` y `--reward-stalled -5`
de la tabla de opciones de `train_policy`.

- [ ] **Step 5: Ver pasar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null` → todo PASS.
Run: `grep -rn "stuck_signal\|lidar_encoding\|LIDAR_ENCODINGS\|proximity\|'turn'\|reward_stalled\|stuck=" martha_nav test` → sin resultados.

- [ ] **Step 6: La huella no cambió**

Run: `./tools/ct_ros python3 runs/_fingerprint.py </dev/null` → el mismo hash del Step 1.

- [ ] **Step 7: La evaluación 2D de `wide_dyn_s0` no cambió**

```bash
./tools/ct_ros ros2 run martha_nav evaluate_2d --model runs/wide_dyn_s0/best_model.zip --episodes 200 --condition obstacles --sources lab --out runs/_check/eval_obstacles_lab.csv </dev/null
./tools/ct python3 -c "
import pandas as pd
a = pd.read_csv('runs/wide_dyn_s0/v5/eval_obstacles_lab.csv')
b = pd.read_csv('runs/_check/eval_obstacles_lab.csv')
pd.testing.assert_frame_equal(a[b.columns], b)
print('igual')"
```

Expected: `igual`.

- [ ] **Step 8: Commit**

```bash
git add martha_nav test docs/comandos.md
git commit -m "Remove the options no run in the thesis uses

The linear LiDAR encoding, the stuck signal, the stall penalty and the
proximity and turn reward terms go; geodesic progress stays. A model trained
with a removed option is refused instead of being misread. Martha's episodes
are unchanged (same fingerprint, same wide_dyn_s0 evaluation).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: El perfil del robot

**Files:**
- Create: `martha_nav/robots.py`
- Test: `test/test_robots.py`

**Interfaces:**
- Produces: `Robot` (frozen dataclass: `name, length, width, footprint_offset_x, lidar_offset_x, lidar_range, lidar_min, lidar_rate, v_max, v_reverse, v_lateral, w_max, inflation`; propiedad `holonomic`), `ROBOTS: dict[str, Robot]` con `'martha'` y `'burger'`, `checkpoint_robot(path) -> str`.

- [ ] **Step 1: Test**

`test/test_robots.py`:

```python
import json

import numpy as np
import yaml

from martha_nav.robots import ROBOTS, checkpoint_robot


def test_martha_keeps_the_numbers_the_code_used_before_the_profiles():
    m = ROBOTS['martha']
    assert (m.length, m.width, m.footprint_offset_x, m.lidar_offset_x) == (0.56, 0.41, 0.0, 0.2325)
    assert (m.lidar_range, m.lidar_min, m.lidar_rate) == (8.0, 0.0, 10.0)
    assert (m.v_max, m.v_reverse, m.v_lateral, m.w_max, m.inflation) == (0.35, 0.15, 0.25, 0.8, 0.40)
    assert m.holonomic


def test_burger_is_a_small_differential_robot():
    b = ROBOTS['burger']
    assert (b.length, b.width, b.footprint_offset_x, b.lidar_offset_x) == (0.14, 0.178, -0.032, -0.032)
    assert (b.v_max, b.w_max, b.inflation, b.lidar_rate) == (0.22, 1.5, 0.20, 5.0)
    assert not b.holonomic


def test_a_checkpoint_says_which_robot_it_drives(tmp_path):
    model = tmp_path / 'best_model.zip'
    assert checkpoint_robot(model) == 'martha'                          # no config.yaml
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'n_rays': 180}}))
    assert checkpoint_robot(model) == 'martha'                          # trained before profiles
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'burger'}}))
    assert checkpoint_robot(model) == 'burger'
    exported = tmp_path / 'policy.npz'
    np.savez(exported, settings=json.dumps({'robot': 'burger'}))
    assert checkpoint_robot(exported) == 'burger'
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_robots.py </dev/null`
Expected: FAIL (`No module named 'martha_nav.robots'`).

- [ ] **Step 3: Implementar** `martha_nav/robots.py`:

```python
"""The robots the policy can drive: everything the code assumes about the body and the LiDAR."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Robot:
    name: str
    length: float              # m, contact rectangle along x of base_link
    width: float               # m, along y
    footprint_offset_x: float  # m, centre of that rectangle in base_link
    lidar_offset_x: float      # m, LiDAR position in base_link
    lidar_range: float         # m, farthest reading; beyond it the scan reads "nothing"
    lidar_min: float           # m, nearest reading; closer also reads "nothing" (0: no limit)
    lidar_rate: float          # Hz, scans per second
    v_max: float               # m/s forward
    v_reverse: float           # m/s backward
    v_lateral: float           # m/s sideways; 0 for a robot that cannot slide
    w_max: float               # rad/s
    inflation: float           # m, obstacle inflation of the global planner

    @property
    def holonomic(self):
        return self.v_lateral > 0


ROBOTS = {
    # Mecanum wheels, RPLIDAR A2M8. Reverse and sideways are capped lower than forward.
    'martha': Robot('martha', length=0.56, width=0.41, footprint_offset_x=0.0,
                    lidar_offset_x=0.2325, lidar_range=8.0, lidar_min=0.0, lidar_rate=10.0,
                    v_max=0.35, v_reverse=0.15, v_lateral=0.25, w_max=0.8, inflation=0.40),
    # TurtleBot3 Burger: turtlebot3_burger.urdf (body 0.140 m centred at x = -0.032, wheels
    # 0.178 m across, base_scan at x = -0.032) and its spec sheet (0.22 m/s; 2.84 rad/s, capped
    # at 1.5 so the action keeps its resolution). LDS-01: 3.5 m, 0.12 m, 5 Hz (an LDS-02 is
    # 8.0 m and 0.16 m).
    'burger': Robot('burger', length=0.14, width=0.178, footprint_offset_x=-0.032,
                    lidar_offset_x=-0.032, lidar_range=3.5, lidar_min=0.12, lidar_rate=5.0,
                    v_max=0.22, v_reverse=0.22, v_lateral=0.0, w_max=1.5, inflation=0.20),
}


def checkpoint_robot(path):
    """The robot a checkpoint was trained for: its config.yaml (.zip) or its settings (.npz)."""
    path = Path(path)
    if path.suffix == '.npz':
        return json.loads(str(np.load(path)['settings']))['robot']
    config = path.with_name('config.yaml')
    if not config.exists():
        return 'martha'
    return yaml.safe_load(config.read_text())['env'].get('robot', 'martha')
```

- [ ] **Step 4: Ver pasar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_robots.py </dev/null` → PASS.

- [ ] **Step 5: Commit**

```bash
git add martha_nav/robots.py test/test_robots.py
git commit -m "Add robot profiles for Martha and the TurtleBot3 Burger

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Geometría y observación con el perfil

**Files:**
- Modify: `martha_nav/sim2d/geometry.py:6-9,105-125`
- Modify: `martha_nav/sim2d/observation.py` (constantes, `reduce_scan`, `build_observation`, `action_to_cmd`)
- Test: `test/test_geometry.py`, `test/test_observation.py`, `test/test_scan_adapter.py`, `test/test_urdf.py:5`

**Interfaces:**
- Consumes: `ROBOTS`, `Robot` (Task 4).
- Produces:
  - `footprint_points(robot, spacing=RESOLUTION / 2) -> np.ndarray (n, 2)` en `base_link`.
  - `footprint_collides(grid, x, y, theta, footprint) -> bool` (`footprint` obligatorio).
  - `reduce_scan(ranges, angles, max_range=8.0)`.
  - `build_observation(ranges, angles, velocity, waypoint_rel, prev_action, waypoint_max=WAYPOINT_MAX, robot=ROBOTS['martha'])`.
  - `action_to_cmd(action, robot=ROBOTS['martha'])`.
  - Desaparecen: `ROBOT_LENGTH`, `ROBOT_WIDTH`, `LIDAR_OFFSET_X`, `FOOTPRINT`, `LIDAR_MAX`, `V_MAX`, `V_REVERSE`, `V_LATERAL`, `W_MAX`.

- [ ] **Step 1: Tests nuevos y alias en los viejos**

En `test/test_observation.py`, reemplazar el import de las líneas 4-5 por:

```python
from martha_nav.robots import ROBOTS
from martha_nav.sim2d.observation import OBS_DIM, action_to_cmd, build_observation, reduce_scan

MARTHA, BURGER = ROBOTS['martha'], ROBOTS['burger']
LIDAR_MAX, V_MAX, W_MAX, V_LATERAL = MARTHA.lidar_range, MARTHA.v_max, MARTHA.w_max, MARTHA.v_lateral
```

borrar la línea `from martha_nav.sim2d.observation import V_LATERAL, obs_dim` (dejar `from martha_nav.sim2d.observation import obs_dim`), y agregar al final:

```python
def test_the_burger_scales_actions_and_velocities_to_its_own_limits():
    assert action_to_cmd([1.0, -1.0], BURGER) == (0.22, -1.5)
    assert action_to_cmd([-1.0, 0.0], BURGER) == (-0.22, 0.0)
    obs = build_observation(np.full(4, 2.0), np.zeros(4), (0.11, 0.75), (1.0, 0.0), (0, 0),
                            robot=BURGER)
    assert np.allclose(obs[92:94], [0.5, 0.5])


def test_an_empty_sector_reads_the_robots_range():
    assert reduce_scan(np.array([np.inf]), np.zeros(1), max_range=3.5)[0] == 3.5
```

En `test/test_scan_adapter.py` línea 6: `from martha_nav.sim2d.observation import LIDAR_MAX, build_observation` →

```python
from martha_nav.robots import ROBOTS
from martha_nav.sim2d.observation import build_observation

LIDAR_MAX = ROBOTS['martha'].lidar_range
```

En `test/test_urdf.py` línea 5: `from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH` →

```python
from martha_nav.robots import ROBOTS

MARTHA = ROBOTS['martha']
LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH = MARTHA.lidar_offset_x, MARTHA.length, MARTHA.width
```

En `test/test_geometry.py`, la línea 3 importa también `footprint_points` y `ROBOTS` (`from martha_nav.robots import ROBOTS`); en `test_footprint_collision_respects_orientation` las dos llamadas pasan `footprint_points(ROBOTS['martha'])` como quinto argumento. Agregar:

```python
def test_the_burger_footprint_sits_behind_its_axle():
    pts = footprint_points(ROBOTS['burger'])
    assert np.isclose(pts[:, 0].min(), -0.032 - 0.07) and np.isclose(pts[:, 0].max(), -0.032 + 0.07)
    assert np.isclose(pts[:, 1].max(), 0.089)
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_geometry.py test/test_observation.py test/test_scan_adapter.py test/test_urdf.py </dev/null`
Expected: FAIL (`cannot import name 'footprint_points'`, `unexpected keyword argument 'robot'`).

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/geometry.py`: borrar `ROBOT_LENGTH`, `ROBOT_WIDTH`, `LIDAR_OFFSET_X` (líneas 7-9) y reemplazar `_footprint_points`, `FOOTPRINT` y `footprint_collides` por:

```python
def footprint_points(robot, spacing=RESOLUTION / 2):
    """Points on the perimeter of the robot's contact rectangle, in base_link."""
    hx, hy = robot.length / 2, robot.width / 2
    xs = np.linspace(-hx, hx, int(np.ceil(robot.length / spacing)) + 1)
    ys = np.linspace(-hy, hy, int(np.ceil(robot.width / spacing)) + 1)
    top = np.stack([xs, np.full_like(xs, hy)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, -hy)], axis=1)
    left = np.stack([np.full_like(ys, -hx), ys], axis=1)
    right = np.stack([np.full_like(ys, hx), ys], axis=1)
    points = np.concatenate([top, bottom, left, right])
    points[:, 0] += robot.footprint_offset_x
    return points


def footprint_collides(grid, x, y, theta, footprint):
    """True when the robot rectangle's perimeter touches an occupied cell."""
    c, s = np.cos(theta), np.sin(theta)
    px = x + c * footprint[:, 0] - s * footprint[:, 1]
    py = y + s * footprint[:, 0] + c * footprint[:, 1]
    return bool(grid.occupied(px, py).any())
```

`martha_nav/sim2d/observation.py`: arriba, `from martha_nav.robots import ROBOTS`; las constantes quedan:

```python
N_SECTORS = 90
WAYPOINT_MAX = 3.0  # m, carrot distance scale
GOAL_MAX = 12.0     # m, goal distance scale when the policy sees the goal instead (longest route)
OBS_DIM = N_SECTORS + 6          # the default (vx, w) action space
```

`reduce_scan`: firma `def reduce_scan(ranges, angles, max_range=8.0):`, docstring `... Invalid readings (inf, NaN, <= 0) and empty sectors count as max_range.` y en el cuerpo `LIDAR_MAX` → `max_range` (tres veces).

`build_observation`: firma

```python
def build_observation(ranges, angles, velocity, waypoint_rel, prev_action,
                      waypoint_max=WAYPOINT_MAX, robot=ROBOTS['martha']):
```

docstring: agregar `robot: the profile whose LiDAR range and velocity limits scale the values.`; cuerpo:

```python
    lidar = encode_lidar(reduce_scan(ranges, angles, robot.lidar_range))
    ...
    vel = ([velocity[0] / robot.v_max, velocity[1] / robot.v_lateral, velocity[2] / robot.w_max]
           if len(velocity) == 3 else [velocity[0] / robot.v_max, velocity[1] / robot.w_max])
```

`action_to_cmd`:

```python
def action_to_cmd(action, robot=ROBOTS['martha']):
    """[-1, 1]^n -> velocities: (v, w), or (vx, vy, w) with the holonomic space."""
    values = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    a_v = values[0]
    v = float(a_v * (robot.v_max if a_v >= 0 else robot.v_reverse))
    if len(values) == 2:
        return v, float(values[1] * robot.w_max)
    return v, float(values[1] * robot.v_lateral), float(values[2] * robot.w_max)
```

- [ ] **Step 4: Ver pasar estos tests** (env.py y policy_core.py todavía importan los nombres viejos; eso lo arreglan las tareas 5 y 6)

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_geometry.py test/test_observation.py test/test_urdf.py </dev/null` → PASS.

- [ ] **Step 5: Commit** (junto con la Tarea 6, que deja el paquete importable de nuevo; no commitear aquí solo)

---

### Task 6: Simulador 2D y entrenamiento con el perfil

**Files:**
- Modify: `martha_nav/sim2d/env.py` (imports 9-15; `episode_steps` 25-27; `EnvConfig`; `__init__`; `reset`; `step`; `_scan`; `_obs`)
- Modify: `martha_nav/learning/train.py` (`build_config`, `main`)
- Modify: `martha_nav/learning/evaluate.py` (`trained_env_config`)
- Test: `test/test_env.py`, `test/test_train.py`, `test/test_evaluate.py`

**Interfaces:**
- Consumes: Task 4 y 5.
- Produces: `EnvConfig.robot: str = 'martha'`; `episode_steps(route_length, robot=ROBOTS['martha'])`; `build_config(..., robot='martha')`; `train_policy --robot {martha,burger}`; `trained_env_config()` devuelve `robot` y `scenario.inflation` del perfil.

- [ ] **Step 1: Tests**

Agregar a `test/test_env.py`:

```python
def test_the_burger_episode_uses_its_profile():
    from martha_nav.robots import ROBOTS
    from martha_nav.sim2d.env import episode_steps
    env = NavEnv(EnvConfig(robot='burger',
                           scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none')))
    env.reset(seed=3)
    assert env.robot is ROBOTS['burger']
    assert env.max_steps == episode_steps(env.sc.path.length, ROBOTS['burger'])
    assert episode_steps(10.0, ROBOTS['burger']) > episode_steps(10.0)      # slower robot, more time
    # Its LiDAR scans at 5 Hz: the scan changes every other control step.
    scans = []
    for _ in range(6):
        env.step(np.array([1.0, 0.0]))
        scans.append(env.ranges.copy())
    changed = [not np.array_equal(a, b) for a, b in zip(scans, scans[1:])]
    assert changed in ([True, False, True, False, True], [False, True, False, True, False])
    assert env.ranges.max() <= 3.5 and (env.ranges >= 0.12).all()


def test_a_holonomic_action_space_needs_a_holonomic_robot():
    with pytest.raises(ValueError, match='burger'):
        NavEnv(EnvConfig(robot='burger', action_dim=3))
```

Agregar a `test/test_train.py`:

```python
def test_build_config_takes_the_robot_and_its_inflation():
    from martha_nav.learning.train import build_config
    cfg = build_config('full', robot='burger')
    assert cfg.robot == 'burger' and cfg.scenario.inflation == 0.20
    assert build_config('full').scenario.inflation == 0.40
```

Agregar a `test/test_evaluate.py`:

```python
def test_evaluation_drives_the_robot_the_model_was_trained_for(tmp_path):
    from martha_nav.learning.evaluate import trained_env_config
    model = tmp_path / 'best_model.zip'
    (tmp_path / 'config.yaml').write_text('env:\n  n_rays: 180\n')
    assert trained_env_config(model).robot == 'martha'
    (tmp_path / 'config.yaml').write_text('env:\n  robot: burger\n')
    cfg = trained_env_config(model)
    assert cfg.robot == 'burger' and cfg.scenario.inflation == 0.20
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_env.py test/test_train.py test/test_evaluate.py </dev/null`
Expected: FAIL (`ImportError: cannot import name 'LIDAR_OFFSET_X'` desde env.py).

- [ ] **Step 3: Implementar `env.py`**

Imports:

```python
from martha_nav.robots import ROBOTS
from martha_nav.sim2d.dynamics import DT, Dynamics, DynamicsRanges, sample_params
from martha_nav.sim2d.geometry import footprint_collides, footprint_points, raycast
from martha_nav.sim2d.observation import (GOAL_MAX, WAYPOINT_MAX, action_to_cmd, build_observation,
                                          obs_dim)
```

(conservar los demás imports tal cual).

```python
def episode_steps(route_length, robot=ROBOTS['martha']):
    """Control steps before a timeout: three times the route at full speed, plus 10 s."""
    return int(np.ceil((3 * route_length / robot.v_max + 10) / DT))
```

En `EnvConfig`, después de `target`:

```python
    robot: str = 'martha'        # a key of martha_nav.robots.ROBOTS
```

En `NavEnv.__init__`, después de `self.cfg = cfg or EnvConfig()`:

```python
        self.robot = ROBOTS[self.cfg.robot]
        if self.cfg.action_dim == 3 and not self.robot.holonomic:
            raise ValueError(f'{self.robot.name} cannot slide sideways: use action_dim 2')
        self.footprint = footprint_points(self.robot)
```

En `reset`, después de `self.lidar_sigma = ...`:

```python
        # A LiDAR slower than the control loop repeats its last scan in between (random phase).
        self.scan_every = max(1, round(1 / (self.robot.lidar_rate * DT)))
        self.scan_phase = int(self.rng.integers(self.scan_every)) if self.scan_every > 1 else 0
```

y `self.max_steps = episode_steps(self.sc.path.length, self.robot)`.

En `step`:

```python
        commands = action_to_cmd(action, self.robot)
        step = self.dyn.step_holonomic if len(commands) == 3 else self.dyn.step
        collided = step(*commands,
                        lambda x, y, t: footprint_collides(self.sc.full, x, y, t, self.footprint))
```

y reemplazar el `self._scan()` de `step` por:

```python
        if (self.steps + self.scan_phase) % self.scan_every == 0:
            self._scan()
```

`_scan`:

```python
    def _scan(self):
        x, y, th = self.dyn.pose
        reach, offset = self.robot.lidar_range, self.robot.lidar_offset_x
        ox, oy = x + offset * np.cos(th), y + offset * np.sin(th)
        r = raycast(self.sc.full, ox, oy, th + self.ray_angles, reach)
        r = r + self.rng.normal(0.0, self.lidar_sigma, r.shape)
        r[self.rng.random(r.shape) < self.cfg.lidar_dropout] = reach
        if self.robot.lidar_min > 0:
            r[r < self.robot.lidar_min] = reach        # too close to measure, as the real sensor
        self.ranges = np.clip(r, 0.0, reach)
        hit = self.ranges < reach
        ...  # el resto igual
```

`_obs`: la llamada pasa el perfil:

```python
        return build_observation(self.ranges, self.ray_angles, vel, rel, self.prev_action,
                                 scale, robot=self.robot)
```

- [ ] **Step 4: Implementar `train.py` y `evaluate.py`**

`train.py`: `from martha_nav.robots import ROBOTS`; `build_config(preset, collision=None, action_dim=2, target='carrot', progress_mode='route', wide_dynamics=False, robot='martha')` y su `return`:

```python
    return replace(cfg, reward=reward, action_dim=action_dim, target=target, robot=robot,
                   scenario=replace(cfg.scenario, sources=tuple(p['sources']),
                                    obstacle_mode=p['obstacle_mode'],
                                    inflation=ROBOTS[robot].inflation))
```

En `main`: `ap.add_argument('--robot', choices=list(ROBOTS), default='martha', help='the robot to train for (martha_nav/robots.py)')` y pasar `args.robot` como último argumento de `build_config(...)`.

`evaluate.py`, en `trained_env_config` (importar `from martha_nav.robots import ROBOTS` y `from martha_nav.sim2d.scenarios import ScenarioConfig` junto a los imports existentes):

```python
    robot = saved.get('robot', 'martha')
    return EnvConfig(dynamics=dynamics, action_dim=saved.get('action_dim', 2),
                     target=saved.get('target', 'carrot'), robot=robot,
                     scenario=ScenarioConfig(inflation=ROBOTS[robot].inflation))
```

`evaluate_gazebo.py` no cambia en esta tarea: `episode_steps(length)` sin perfil sigue siendo el de Martha (la Tarea 10 le pasa el robot).

- [ ] **Step 5: Ver pasar y verificar que Martha no cambió**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_env.py test/test_train.py test/test_evaluate.py test/test_geometry.py test/test_observation.py test/test_urdf.py </dev/null` → PASS.
Run: `./tools/ct_ros python3 runs/_fingerprint.py </dev/null`
Expected: el hash que la Tarea 3 anotó como comentario en `runs/_fingerprint.py`. Si difiere, algo cambió para Martha: buscarlo antes de seguir. Repetir también la comparación de `eval_obstacles_lab.csv` de la Tarea 3 (Step 7): debe imprimir `igual`.

- [ ] **Step 6: Commit (Tareas 5 y 6)**

```bash
git add martha_nav/sim2d martha_nav/learning test
git commit -m "Read the body, LiDAR and limits from the robot profile in the 2D simulator

train_policy --robot burger trains for the TurtleBot3 Burger: its contact
rectangle behind the axle, 3.5 m LiDAR at 5 Hz (the scan repeats every other
control step) with a 0.12 m blind zone, 0.22 m/s and 1.5 rad/s. Martha's
episodes are bit-identical (same fingerprint).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Si `test_policy_core.py` falla en este punto porque `policy_core.py` importa los nombres viejos, es la Tarea 7: hacerla antes de correr la suite completa.)

---

### Task 7: Nodos ROS con el perfil

**Files:**
- Modify: `martha_nav/ros/policy_core.py` (imports 1-8, `footprint_blocked`, `PolicyCore.__init__`, `compute`, `_scan_points`)
- Modify: `martha_nav/ros/global_planner.py:20-25`
- Modify: `martha_nav/ros/slam.py:11-16`
- Test: `test/test_policy_core.py`, `test/test_global_planner.py`

**Interfaces:**
- Consumes: Tasks 4-5.
- Produces: `footprint_blocked(ranges, angles, robot=ROBOTS['martha'], margin=0.05)`; `PolicyCore(model, ..., robot=ROBOTS['martha'])` (atributo `core.robot`); parámetro `robot` de `global_planner` (inflado por defecto del perfil); `slam_toolbox(..., max_range=8.0)`.

- [ ] **Step 1: Tests**

`test/test_policy_core.py` línea 4 →

```python
from martha_nav.robots import ROBOTS

LIDAR_MAX, V_MAX, W_MAX = (ROBOTS['martha'].lidar_range, ROBOTS['martha'].v_max,
                           ROBOTS['martha'].w_max)
```

y en `test_holonomic_core_returns_three_velocities_and_guards_the_sides` reemplazar `from martha_nav.sim2d.observation import V_LATERAL` por `V_LATERAL = ROBOTS['martha'].v_lateral`. Agregar:

```python
def test_the_burger_guard_uses_its_footprint_and_lidar():
    burger = ROBOTS['burger']
    angles = np.array([0.0, np.pi])
    # LiDAR and footprint centre coincide: 0.07 m to the front and to the back.
    assert 'front' in footprint_blocked(np.array([0.10, 3.5]), angles, burger)
    assert footprint_blocked(np.array([0.15, 3.5]), angles, burger) == set()
    assert 'rear' in footprint_blocked(np.array([3.5, 0.10]), angles, burger)


def test_the_burger_core_commands_its_own_speeds():
    burger = ROBOTS['burger']
    core = PolicyCore(FakeModel((1.0, -1.0)), robot=burger)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    v, w, _ = core.compute(straight_path(), (0.0, 0.0, 0.0), np.full(360, 3.5), angles, (0.0, 0.0))
    assert (v, w) == (burger.v_max, -burger.w_max)
```

`test/test_global_planner.py`, agregar (usa el mismo patrón de `rclpy.init` que el resto del archivo):

```python
def test_the_inflation_comes_from_the_robot():
    import rclpy

    from martha_nav.ros.global_planner import GlobalPlanner
    rclpy.init(args=['--ros-args', '-p', 'robot:=burger'])
    try:
        node = GlobalPlanner()
        assert node.core.inflation == 0.20
        node.destroy_node()
    finally:
        rclpy.try_shutdown()
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_policy_core.py test/test_global_planner.py </dev/null`
Expected: FAIL (import de `LIDAR_OFFSET_X` en `policy_core`, `inflation == 0.4`).

- [ ] **Step 3: Implementar**

`policy_core.py` imports:

```python
from martha_nav.learning.policy import is_recurrent
from martha_nav.robots import ROBOTS
from martha_nav.sim2d.observation import GOAL_MAX, WAYPOINT_MAX, action_to_cmd, build_observation
from martha_nav.sim2d.planner import RouteProgress, carrot
```

(`is_recurrent` cambia de origen en la Tarea 8.)

```python
def footprint_blocked(ranges, angles, robot=ROBOTS['martha'], margin=0.05):
    """Which sides of the footprint a scan point has entered.

    Returns a set from {'front', 'rear', 'left', 'right'}, empty when clear. Ranges are
    measured from the LiDAR, so the points are moved into the footprint's frame first.
    The side matters because a guard that blocks every motion leaves the robot frozen
    against the obstacle.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    valid = np.isfinite(ranges) & (ranges > 0)
    x = (ranges[valid] * np.cos(angles[valid]) + robot.lidar_offset_x
         - robot.footprint_offset_x)
    y = ranges[valid] * np.sin(angles[valid])
    half_x, half_y = robot.length / 2 + margin, robot.width / 2 + margin
    ...  # el resto igual
```

`PolicyCore.__init__`: agregar el parámetro final `robot=ROBOTS['martha']` y `self.robot = robot`.
En `compute`: `blocked = footprint_blocked(ranges, angles, self.robot)`; `build_observation(ranges, angles, velocity, rel, self.prev_action, scale, robot=self.robot)`; `self.pending.append(action_to_cmd(action, self.robot))`.
En `_scan_points`: `hit = np.isfinite(ranges) & (ranges > 0) & (ranges < self.robot.lidar_range)` y `ox = x + self.robot.lidar_offset_x * np.cos(yaw)`, `oy = y + self.robot.lidar_offset_x * np.sin(yaw)`.

`global_planner.py` (importar `from martha_nav.robots import ROBOTS`):

```python
        robot = ROBOTS[self.declare_parameter('robot', 'martha').value]
        self.core = PlannerCore(
            inflation=self.declare_parameter('inflation', robot.inflation).value,
            ...
```

`slam.py`:

```python
def slam_toolbox(mode, scan_topic, use_sim_time, publish_tf=True, map_file='',
                 start_pose=(0.0, 0.0, 0.0), max_range=8.0):
    """mode 'mapping' or 'localization'; map_file is the saved posegraph without extension;
    max_range is the LiDAR's (ROBOTS[...].lidar_range)."""
    ...
    params = {'use_sim_time': use_sim_time, 'mode': mode, 'base_frame': 'base_link',
              'scan_topic': scan_topic, 'max_laser_range': max_range}
```

- [ ] **Step 4: Suite completa**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null` → todo PASS.

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros test
git commit -m "Take the footprint, LiDAR, speeds and inflation of the ROS nodes from the profile

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: La política en numpy

**Files:**
- Modify: `martha_nav/learning/__init__.py`, `martha_nav/learning/policy.py:31-33`, `martha_nav/learning/evaluate.py:11`, `martha_nav/ros/policy_core.py` (import de `is_recurrent`)
- Create: `martha_nav/ros/numpy_policy.py`, `martha_nav/learning/export.py`
- Modify: `setup.py` (console script `export_policy`)
- Test: `test/test_export.py`

**Interfaces:**
- Consumes: `trained_env_config`, `load_model` (evaluate.py), `policy_kwargs()` (Task 1).
- Produces:
  - `martha_nav.learning.is_recurrent(model) -> bool` (sin importar torch).
  - `SETTINGS = ('robot', 'action_dim', 'target')` y `settings_of(cfg) -> dict` en `martha_nav/ros/numpy_policy.py`.
  - `NumpyPolicy(path)`: `.settings: dict`, `.predict(obs, deterministic=True) -> (np.ndarray, None)`.
  - `export(model_path, out=None) -> Path` y CLI `export_policy --model ... [--out ...]`, por defecto `policy.npz` junto al modelo.

- [ ] **Step 1: Tests** `test/test_export.py`:

```python
"""The exported numpy policy must act exactly as the PyTorch one, without importing it."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from martha_nav.sim2d.env import EnvConfig, NavEnv

WIDE = Path(__file__).resolve().parents[1] / 'runs' / 'wide_dyn_s0' / 'best_model.zip'


def same_actions(model_path, env, steps=40):
    from stable_baselines3 import PPO

    from martha_nav.learning.export import export
    from martha_nav.ros.numpy_policy import NumpyPolicy
    model = PPO.load(model_path, device='cpu')
    policy = NumpyPolicy(export(model_path))
    obs, _ = env.reset(seed=7)
    for _ in range(steps):
        expected, _ = model.predict(obs, deterministic=True)
        got, _ = policy.predict(obs)
        assert np.allclose(got, expected, atol=1e-5)
        obs, _, done, truncated, _ = env.step(expected)
        if done or truncated:
            obs, _ = env.reset()
    return policy


def test_a_fresh_burger_policy_exports_exactly(tmp_path):
    from stable_baselines3 import PPO

    from martha_nav.learning.policy import policy_kwargs
    env = NavEnv(EnvConfig(robot='burger'))
    PPO('MlpPolicy', env, policy_kwargs=policy_kwargs(), seed=0,
        device='cpu').save(tmp_path / 'best_model.zip')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'burger'}}))
    policy = same_actions(tmp_path / 'best_model.zip', env)
    assert policy.settings['robot'] == 'burger' and policy.settings['action_dim'] == 2


@pytest.mark.skipif(not WIDE.exists(), reason='needs runs/wide_dyn_s0')
def test_the_final_model_exports_exactly(tmp_path):
    import shutil
    for name in ('best_model.zip', 'config.yaml'):
        shutil.copy(WIDE.with_name(name), tmp_path / name)
    from martha_nav.learning.evaluate import trained_env_config
    same_actions(tmp_path / 'best_model.zip', NavEnv(trained_env_config(WIDE)))


def test_the_robot_side_never_imports_pytorch():
    code = ('import sys\n'
            'from martha_nav.ros import ppo_local_planner, numpy_policy\n'
            "heavy = [m for m in ('torch', 'stable_baselines3', 'gymnasium') if m in sys.modules]\n"
            'assert not heavy, heavy\n')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_export.py </dev/null`
Expected: FAIL (`No module named 'martha_nav.learning.export'`; el tercer test falla porque `ppo_local_planner` importa torch; ese lo termina de arreglar la Tarea 9).

- [ ] **Step 3: Implementar**

`martha_nav/learning/__init__.py`:

```python
def is_recurrent(model):
    """RecurrentPPO's policy has an LSTM; its predict() takes and returns the state.

    Here, not in learning.policy, so the robot can ask without importing PyTorch.
    """
    return hasattr(getattr(model, 'policy', None), 'lstm_actor')
```

Borrar `is_recurrent` de `learning/policy.py`; en `learning/evaluate.py` y `ros/policy_core.py`: `from martha_nav.learning import is_recurrent`.

`martha_nav/ros/numpy_policy.py`:

```python
"""The trained policy's deterministic action in numpy, so the robot needs no PyTorch.

export_policy (martha_nav/learning/export.py) writes the .npz read here: the weights of the
actor (CNN extractor, policy MLP, action layer) and the settings the observation needs.
"""
import json

import numpy as np

from martha_nav.sim2d.observation import N_SECTORS

SETTINGS = ('robot', 'action_dim', 'target')
CONVS = ((0, 1, 2), (2, 2, 2), (4, 2, 1))    # LidarCnnExtractor.cnn: (index, stride, padding)


def settings_of(cfg):
    """What the local planner needs from the EnvConfig a policy was trained with."""
    return {k: getattr(cfg, k) for k in SETTINGS}


def conv1d_circular(x, w, b, stride, pad):
    """torch's Conv1d(padding_mode='circular') on one sample: x (channels, length)."""
    xp = np.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)
    k = w.shape[2]
    n = (xp.shape[1] - k) // stride + 1
    windows = xp[:, np.arange(n)[:, None] * stride + np.arange(k)[None, :]]
    return np.einsum('ock,cnk->on', w, windows) + b[:, None]


class NumpyPolicy:
    """Same predict() as SB3's PPO, always deterministic, for one observation."""

    def __init__(self, path):
        data = np.load(path)
        self.settings = json.loads(str(data['settings']))
        self.w = {k: data[k].astype(np.float64) for k in data.files if k != 'settings'}

    def _linear(self, name, x):
        return self.w[f'{name}.weight'] @ x + self.w[f'{name}.bias']

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        x = obs[None, :N_SECTORS]
        for i, stride, pad in CONVS:
            name = f'pi_features_extractor.cnn.{i}'
            x = np.maximum(conv1d_circular(x, self.w[f'{name}.weight'], self.w[f'{name}.bias'],
                                           stride, pad), 0.0)
        lidar = np.maximum(self._linear('pi_features_extractor.lidar_head.0', x.reshape(-1)), 0.0)
        rest = np.maximum(self._linear('pi_features_extractor.rest_head.0', obs[N_SECTORS:]), 0.0)
        h = np.concatenate([lidar, rest])
        for i in (0, 2):                                   # Linear, Tanh, Linear, Tanh
            h = np.tanh(self._linear(f'mlp_extractor.policy_net.{i}', h))
        return np.clip(self._linear('action_net', h), -1.0, 1.0), None
```

`martha_nav/learning/export.py`:

```python
"""export_policy: a trained policy's actor and settings in one .npz, for the robot.

    ros2 run martha_nav export_policy --model runs/burger_s0/best_model.zip
-> runs/burger_s0/policy.npz, run with ppo_local_planner checkpoint:=.../policy.npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from martha_nav.learning import is_recurrent
from martha_nav.learning.evaluate import load_model, trained_env_config
from martha_nav.ros.numpy_policy import settings_of

ACTOR = ('pi_features_extractor.', 'mlp_extractor.policy_net.', 'action_net.')


def export(model_path, out=None):
    model = load_model(model_path)
    if is_recurrent(model):
        raise ValueError('only policies without an LSTM can be exported')
    weights = {k: v.cpu().numpy() for k, v in model.policy.state_dict().items()
               if k.startswith(ACTOR)}
    out = Path(out) if out else Path(model_path).with_name('policy.npz')
    np.savez(out, settings=json.dumps(settings_of(trained_env_config(model_path))), **weights)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Export a trained policy to numpy, for the robot.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=None, help='default: policy.npz next to the model')
    args = ap.parse_args(argv)
    print(f'policy -> {export(args.model, args.out)}')
```

`setup.py`, en `console_scripts`: `'export_policy = martha_nav.learning.export:main',`.

- [ ] **Step 4: Ver pasar los dos primeros tests**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_export.py -k exports </dev/null` → 2 PASS (o 1 PASS + 1 SKIP si no está `runs/wide_dyn_s0`).

- [ ] **Step 5: Commit**

```bash
git add martha_nav/learning martha_nav/ros/numpy_policy.py martha_nav/ros/policy_core.py setup.py test/test_export.py
git commit -m "Export a trained policy to numpy, so the robot runs it without PyTorch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `ppo_local_planner`: `.npz`, robot correcto, `speed_scale`, latencia

**Files:**
- Modify: `martha_nav/ros/ppo_local_planner.py`
- Test: `test/test_ppo_local_planner.py`

**Interfaces:**
- Consumes: `NumpyPolicy`, `settings_of` (Task 8), `ROBOTS` (Task 4), `PolicyCore(robot=...)` (Task 7).
- Produces: `load_policy(checkpoint) -> (model, settings)`; `to_twist(velocities, scale=1.0) -> Twist`; parámetros `robot` (`''` = el del modelo) y `speed_scale` (1.0).

- [ ] **Step 1: Tests**

En `test/test_ppo_local_planner.py`, el fixture `node` pasa a:

```python
@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig())))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/best_model.zip'])
    node = ppo_local_planner.PpoLocalPlanner()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()
```

con `from martha_nav.ros.numpy_policy import settings_of` arriba. Agregar:

```python
def test_a_checkpoint_for_another_robot_is_refused(monkeypatch):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig(robot='burger'))))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/policy.npz', '-p', 'robot:=martha'])
    try:
        with pytest.raises(RuntimeError, match='burger'):
            ppo_local_planner.PpoLocalPlanner()
    finally:
        rclpy.try_shutdown()


def test_the_core_drives_the_models_robot(monkeypatch):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig(robot='burger'))))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/policy.npz'])
    try:
        node = ppo_local_planner.PpoLocalPlanner()
        assert node.core.robot.name == 'burger'
        node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_speed_scale_slows_every_velocity():
    cmd = ppo_local_planner.to_twist((0.2, 1.0), 0.5)
    assert (cmd.linear.x, cmd.linear.y, cmd.angular.z) == (0.1, 0.0, 0.5)
    cmd = ppo_local_planner.to_twist((0.2, -0.1, 1.0))
    assert (cmd.linear.x, cmd.linear.y, cmd.angular.z) == (0.2, -0.1, 1.0)
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_ppo_local_planner.py test/test_export.py </dev/null`
Expected: FAIL (`has no attribute 'load_policy'`, `to_twist`, y el test de imports por `torch`).

- [ ] **Step 3: Implementar** en `ppo_local_planner.py`:

Imports: quitar `import torch` y `from martha_nav.learning.evaluate import load_model, trained_env_config`; agregar `from martha_nav.robots import ROBOTS` y `from martha_nav.ros.numpy_policy import NumpyPolicy, settings_of`. Funciones de módulo:

```python
LATENCY_REPORT = 10.0   # s between latency log lines


def load_policy(checkpoint):
    """(model, settings): an exported .npz runs on numpy alone; a .zip needs PyTorch."""
    if checkpoint.endswith('.npz'):
        policy = NumpyPolicy(checkpoint)
        return policy, policy.settings
    import torch

    from martha_nav.learning.evaluate import load_model, trained_env_config
    torch.set_num_threads(1)
    return load_model(checkpoint), settings_of(trained_env_config(checkpoint))


def to_twist(velocities, scale=1.0):
    """(v, w) or (vx, vy, w) -> Twist, every component times scale."""
    cmd = Twist()
    cmd.linear.x = scale * float(velocities[0])
    if len(velocities) == 3:
        cmd.linear.y = scale * float(velocities[1])
    cmd.angular.z = scale * float(velocities[-1])
    return cmd
```

En `__init__`, reemplazar desde `torch.set_num_threads(1)` hasta el primer `get_logger().info(...)` por:

```python
        model, settings = load_policy(checkpoint)
        wanted = self.declare_parameter('robot', '').value
        if wanted and wanted != settings['robot']:
            raise RuntimeError(f"{checkpoint} drives {settings['robot']}, not {wanted}")
        self.core = PolicyCore(
            model,
            lookahead=self.declare_parameter('lookahead', 1.5).value,
            action_dim=settings['action_dim'],
            target=settings['target'],
            action_delay=self.declare_parameter('action_delay', 0).value,
            robot=ROBOTS[settings['robot']])
        self.speed_scale = self.declare_parameter('speed_scale', 1.0).value
        self.action_dim = self.core.action_dim
        self.get_logger().info(f"robot {settings['robot']}, action space {self.action_dim}D, "
                               f'action delay {self.core.action_delay}, speed x{self.speed_scale}')
```

Latencia: en `__init__`, `self.scan_stamp = None`, `self.latencies = []` y `self.create_timer(LATENCY_REPORT, self.report_latency)`. En `on_scan`: `self.scan_stamp = rclpy.time.Time.from_msg(msg.header.stamp)`. En `tick`, reemplazar la construcción del `Twist` por:

```python
        self.cmd_pub.publish(to_twist(velocities, self.speed_scale))
        self.latencies.append((self.get_clock().now() - self.scan_stamp).nanoseconds * 1e-9)
```

y el método:

```python
    def report_latency(self):
        """Scan stamp -> /cmd_vel: the delay the policy acts with on this machine and network."""
        if self.latencies:
            self.get_logger().info(f'latency scan -> cmd_vel: mean {np.mean(self.latencies):.3f} s, '
                                   f'max {np.max(self.latencies):.3f} s')
            self.latencies = []
```

- [ ] **Step 4: Suite completa**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null` → todo PASS (incluye `test_the_robot_side_never_imports_pytorch`).

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros/ppo_local_planner.py test/test_ppo_local_planner.py
git commit -m "Run an exported policy without PyTorch, refuse another robot's model, log latency

speed_scale slows every command for the first real-robot runs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: El Burger en Gazebo

**Files:**
- Create: `urdf/burger.urdf.xacro`
- Modify: `launch/sim.launch.py`, `martha_nav/ros/evaluate_gazebo.py` (entidad, `robot`), `martha_nav/ros/gazebo_ground_truth_tf.py:21`, `tools/evaluate_run_gazebo.sh`
- Test: `test/test_urdf.py`, `test/test_evaluate_gazebo.py`

**Interfaces:**
- Consumes: `ROBOTS`, `checkpoint_robot` (Task 4), `episode_steps(length, robot)` (Task 6), `slam_toolbox(max_range=...)` (Task 7).
- Produces: entidad de Gazebo `robot` para los dos robots; `sim.launch.py robot:=martha|burger` (por defecto el del checkpoint); parámetro `robot` de `evaluate_gazebo`.

- [ ] **Step 1: Tests**

Agregar a `test/test_urdf.py` (con `import shutil`, `import subprocess` y `import pytest` arriba):

```python
BURGER_URDF = Path(__file__).resolve().parents[1] / 'urdf' / 'burger.urdf.xacro'
BURGER = ROBOTS['burger']


def burger():
    return xacro.process_file(str(BURGER_URDF)).toprettyxml()


def test_burger_contact_shell_is_the_2d_footprint():
    doc = burger()
    size = doc.split('name="contact_shell_collision"', 1)[1].split('size="', 1)[1].split('"', 1)[0]
    sx, sy, _ = (float(v) for v in size.split())
    assert (sx, sy) == (BURGER.length, BURGER.width)
    assert joint_origin(doc, 'contact_shell_joint')[0] == BURGER.footprint_offset_x


def test_burger_lidar_matches_the_profile():
    doc = burger()
    assert joint_origin(doc, 'scan_joint')[0] == BURGER.lidar_offset_x
    ray = doc.split('<range>', 1)[1]
    assert float(ray.split('<min>', 1)[1].split('<', 1)[0]) == BURGER.lidar_min
    assert float(ray.split('<max>', 1)[1].split('<', 1)[0]) == BURGER.lidar_range
    rate = doc.split('name="lidar_sensor"', 1)[1].split('<update_rate>', 1)[1].split('<', 1)[0]
    assert float(rate) == BURGER.lidar_rate


@pytest.mark.skipif(shutil.which('gz') is None, reason='needs Gazebo')
def test_burger_contact_sensor_watches_a_collision_gazebo_keeps(tmp_path):
    """Gazebo renames collisions when it lumps fixed joints; the sensor must use the new name."""
    path = tmp_path / 'burger.urdf'
    path.write_text(burger())
    sdf = subprocess.run(['gz', 'sdf', '-p', str(path)], capture_output=True, text=True,
                         check=True).stdout
    watched = sdf.split('<contact>', 1)[1].split('<collision>', 1)[1].split('</collision>', 1)[0]
    assert f"<collision name='{watched}'>" in sdf or f'<collision name="{watched}">' in sdf
```

En `test/test_evaluate_gazebo.py` agregar:

```python
def test_the_burger_episode_gets_its_time_and_inflation():
    import rclpy

    from martha_nav.ros.evaluate_gazebo import EvaluateGazebo
    rclpy.init(args=['--ros-args', '-p', 'robot:=burger'])
    try:
        node = EvaluateGazebo()
        assert node.robot.name == 'burger' and node.cfg.inflation == 0.20
        node.destroy_node()
    finally:
        rclpy.try_shutdown()
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_urdf.py test/test_evaluate_gazebo.py </dev/null`
Expected: FAIL (no existe `burger.urdf.xacro`; `EvaluateGazebo` sin `robot`).

- [ ] **Step 3: `urdf/burger.urdf.xacro`**

```xml
<?xml version="1.0"?>
<!-- TurtleBot3 Burger for Gazebo: the measures of turtlebot3_burger.urdf (ROBOTIS) in simple
     shapes, plus the contact sensor evaluate_gazebo needs. The 2D simulator uses the same
     numbers, ROBOTS['burger'] in martha_nav/robots.py; test/test_urdf.py checks they agree.
     No ":=" anywhere: see the note at the top of martha.urdf.xacro. -->
<robot name="burger" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="lidar_samples" default="360"/>
  <xacro:arg name="lidar_range" default="3.5"/>
  <xacro:arg name="lidar_min" default="0.12"/>
  <xacro:arg name="lidar_rate" default="5"/>

  <link name="base_footprint"/>
  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.010"/>
  </joint>
  <link name="base_link">
    <visual>
      <origin xyz="-0.032 0 0.070"/>
      <geometry><box size="0.140 0.140 0.143"/></geometry>
    </visual>
    <inertial>
      <origin xyz="-0.032 0 0.070"/>
      <mass value="0.825"/>
      <inertia ixx="2.2e-03" ixy="0" ixz="0" iyy="2.2e-03" iyz="0" izz="2.0e-03"/>
    </inertial>
  </link>

  <!-- Contact envelope = the 2D footprint (body and wheels), held above the floor so the
       ground is never a contact. -->
  <joint name="contact_shell_joint" type="fixed">
    <parent link="base_link"/>
    <child link="contact_shell"/>
    <origin xyz="-0.032 0 0.080"/>
  </joint>
  <link name="contact_shell">
    <collision name="contact_shell_collision">
      <geometry><box size="0.14 0.178 0.100"/></geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <xacro:macro name="wheel" params="side y">
    <joint name="wheel_${side}_joint" type="continuous">
      <parent link="base_link"/>
      <child link="wheel_${side}_link"/>
      <origin xyz="0 ${y} 0.023" rpy="-1.5708 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
    <link name="wheel_${side}_link">
      <visual><geometry><cylinder radius="0.033" length="0.018"/></geometry></visual>
      <collision><geometry><cylinder radius="0.033" length="0.018"/></geometry></collision>
      <inertial>
        <mass value="0.0285"/>
        <inertia ixx="1.1e-05" ixy="0" ixz="0" iyy="1.1e-05" iyz="0" izz="2.0e-05"/>
      </inertial>
    </link>
    <gazebo reference="wheel_${side}_link">
      <mu1>100000</mu1>
      <mu2>100000</mu2>
      <kp>500000</kp>
      <kd>10</kd>
      <minDepth>0.001</minDepth>
      <maxVel>0.1</maxVel>
      <fdir1>1 0 0</fdir1>
    </gazebo>
  </xacro:macro>
  <xacro:wheel side="left" y="0.080"/>
  <xacro:wheel side="right" y="-0.080"/>

  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_link"/>
    <origin xyz="-0.081 0 -0.004"/>
  </joint>
  <link name="caster_link">
    <collision><geometry><sphere radius="0.005"/></geometry></collision>
    <inertial>
      <mass value="0.005"/>
      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>
    </inertial>
  </link>
  <gazebo reference="caster_link">
    <mu1>0</mu1>
    <mu2>0</mu2>
  </gazebo>

  <joint name="scan_joint" type="fixed">
    <parent link="base_link"/>
    <child link="base_scan"/>
    <origin xyz="-0.032 0 0.172"/>
  </joint>
  <link name="base_scan">
    <visual><geometry><cylinder radius="0.035" length="0.030"/></geometry></visual>
    <inertial>
      <mass value="0.114"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>

  <gazebo>
    <!-- turtlebot3_gazebo's burger values. -->
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <update_rate>30</update_rate>
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.160</wheel_separation>
      <wheel_diameter>0.066</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>false</publish_wheel_tf>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_footprint</robot_base_frame>
    </plugin>
    <plugin name="joint_states" filename="libgazebo_ros_joint_state_publisher.so">
      <update_rate>30</update_rate>
      <joint_name>wheel_left_joint</joint_name>
      <joint_name>wheel_right_joint</joint_name>
    </plugin>
  </gazebo>

  <gazebo reference="base_footprint">
    <sensor name="contact_sensor" type="contact">
      <always_on>true</always_on>
      <update_rate>50</update_rate>
      <contact>
        <collision>base_footprint_fixed_joint_lump__contact_shell_collision_collision</collision>
      </contact>
      <!-- Publishes on /bumper_states, where evaluate_gazebo listens. -->
      <plugin name="gazebo_ros_bumper" filename="libgazebo_ros_bumper.so">
        <frame_name>base_footprint</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="base_scan">
    <sensor name="lidar_sensor" type="ray">
      <always_on>true</always_on>
      <visualize>false</visualize>
      <update_rate>$(arg lidar_rate)</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>$(arg lidar_samples)</samples>
            <resolution>1</resolution>
            <min_angle>${-pi}</min_angle>
            <max_angle>${pi}</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>$(arg lidar_min)</min>
          <max>$(arg lidar_range)</max>
          <resolution>0.015</resolution>
        </range>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </ray>
      <!-- Publishes on /gazebo_ros_lidar/out, the topic sim.launch.py remaps to /scan. -->
      <plugin name="gazebo_ros_lidar" filename="libgazebo_ros_ray_sensor.so">
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>base_scan</frame_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

Si `test_burger_contact_sensor_watches_a_collision_gazebo_keeps` falla, el mensaje del assert muestra el nombre buscado; copiar en `<contact><collision>` el nombre exacto que aparece en `gz sdf -p` para `contact_shell_collision`.

- [ ] **Step 4: `evaluate_gazebo.py` y `gazebo_ground_truth_tf.py`**

`evaluate_gazebo.py`: `from martha_nav.robots import ROBOTS`; en `__init__`, antes de construir `self.cfg`:

```python
        self.robot = ROBOTS[self.declare_parameter('robot', 'martha').value]
        self.cfg = ScenarioConfig(sources=(self.world,), obstacle_mode=CONDITIONS[self.condition],
                                  inflation=self.robot.inflation)
```

y en `run_episode`: `time_limit = episode_steps(scenario.path.length, self.robot) * DT`. Cambiar la entidad `'martha'` por una constante `ENTITY = 'robot'   # sim.launch.py spawns either robot under this name` en las tres apariciones (`on_states` dos veces, `teleport` una).

`gazebo_ground_truth_tf.py` línea 21: `self.declare_parameter('model_name', 'robot')`.

- [ ] **Step 5: `sim.launch.py`**

- Docstring: agregar `robot:=burger simulates the TurtleBot3 Burger (urdf/burger.urdf.xacro); by default the robot is the one the checkpoint was trained for.`
- `ARGUMENTS`: `DeclareLaunchArgument('robot', default_value='', description="martha or burger; empty: the checkpoint's")`.
- Imports: `from martha_nav.robots import ROBOTS, checkpoint_robot`.
- Al inicio de `launch_setup`:

```python
    checkpoint = LaunchConfiguration('checkpoint').perform(context)
    trained = checkpoint_robot(checkpoint) if checkpoint else 'martha'
    name = LaunchConfiguration('robot').perform(context) or trained
    if name != trained:
        raise RuntimeError(f'{checkpoint} drives {trained}, not {name}')
    robot = ROBOTS[name]
    drive = LaunchConfiguration('drive').perform(context) if name == 'martha' else 'diff'
```

- `odom_remap = [('/odom', MECANUM_ODOM)] if drive == 'mecanum' else []` (igual que hoy).
- `urdf`: con `burger`,

```python
    if name == 'burger':
        urdf = ParameterValue(Command([
            'xacro ', str(share / 'urdf' / 'burger.urdf.xacro'),
            ' lidar_samples:=', LaunchConfiguration('lidar_samples'),
            f' lidar_range:={robot.lidar_range} lidar_min:={robot.lidar_min}',
            f' lidar_rate:={robot.lidar_rate:g}',
        ]), value_type=str)
    else:
        urdf = ...   # el Command de Martha, tal cual
```

- `spawn`: `'-entity', 'robot'`.
- `global_planner`: `parameters=[{'use_sim_time': True, 'robot': name}]`.
- `ppo_local_planner`: agregar `'robot': name` a sus parámetros.
- `slam_toolbox(..., max_range=robot.lidar_range)`.
- El bloque `if drive == 'mecanum':` queda igual (con `burger`, `drive` es `'diff'` y no entra).

- [ ] **Step 6: `tools/evaluate_run_gazebo.sh`**

Después de `ros=...`:

```bash
robot=$(./tools/ct_ros python3 -c "from martha_nav.robots import checkpoint_robot; print(checkpoint_robot('$run/best_model.zip'))" </dev/null)
```

y en la llamada a `evaluate_gazebo` agregar `-p robot:=$robot`.

- [ ] **Step 7: Tests y prueba en Gazebo**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null` → todo PASS.

Prueba de humo con Martha (que nada se rompió): `EPISODES=6 ./tools/evaluate_run_gazebo.sh runs/wide_dyn_s0 _smoke 3`. Expected: termina con 6 semillas y 90 puntos y las líneas `done:`; borrar después `rm runs/wide_dyn_s0/*_smoke*`.

Prueba con el Burger, con un modelo cualquiera de Burger (el de la Tarea 12 o uno de 50k pasos: `./tools/ct_ros ros2 run martha_nav train_policy --preset full --robot burger --wide-dynamics --steps 50000 --name burger_smoke </dev/null`):

```bash
./tools/ct_ros ros2 launch martha_nav sim.launch.py world:=lab gui:=false x:=0.95 y:=1.35 checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/burger_smoke/best_model.zip
```

Expected en el log: `robot burger, action space 2D`; en otra terminal, `./tools/ct_ros ros2 topic hz /gazebo_ros_lidar/out` da ~5 Hz, y `./tools/ct_ros ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"` mueve el robot (ver `/odom`). Con `robot:=martha` en ese mismo comando, el launch debe fallar con `drives burger, not martha`.

- [ ] **Step 8: Commit**

```bash
git add urdf/burger.urdf.xacro launch/sim.launch.py martha_nav/ros/evaluate_gazebo.py martha_nav/ros/gazebo_ground_truth_tf.py tools/evaluate_run_gazebo.sh test
git commit -m "Simulate the TurtleBot3 Burger in Gazebo and evaluate it like Martha

Both robots spawn as the entity 'robot'; sim.launch.py takes the robot from
the checkpoint and refuses a mismatch.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: El Burger real: launch, despliegue y documentación

**Files:**
- Create: `launch/burger.launch.py`, `tools/deploy_burger.sh`
- Modify: `docs/comandos.md` (sección nueva al final)
- Test: `test/test_package.py` (o `test/test_burger_launch.py` si `test_package.py` no carga launch files)

**Interfaces:**
- Consumes: `slam_toolbox(max_range=...)`, parámetros `robot`/`speed_scale`/`checkpoint` de los nodos.

- [ ] **Step 1: Test** `test/test_burger_launch.py`:

```python
"""burger.launch.py builds its nodes: mapping without a map, navigation with one."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch_ros.actions import Node

LAUNCH = Path(__file__).resolve().parents[1] / 'launch' / 'burger.launch.py'


def actions(**args):
    spec = importlib.util.spec_from_file_location('burger_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update({'map': '', 'checkpoint': '', 'speed_scale': '1.0', **args})
    return module.launch_setup(context)


def executables(nodes):
    return sorted(n.node_executable for n in nodes if isinstance(n, Node))


def test_without_a_map_it_only_maps():
    assert executables(actions()) == ['sync_slam_toolbox_node']


def test_with_a_map_it_navigates_as_the_burger():
    nodes = actions(map='/maps/sala', checkpoint='/policies/burger.npz')
    assert executables(nodes) == ['global_planner', 'localization_slam_toolbox_node',
                                  'ppo_local_planner', 'world_map_publisher']
```

- [ ] **Step 2: Ver fallar**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test/test_burger_launch.py </dev/null` → FAIL (no existe el launch).

- [ ] **Step 3: `launch/burger.launch.py`**

```python
"""The TurtleBot3 Burger's navigation: on its Raspberry Pi for the demo, or on the PC to debug.

The robot's own bringup runs apart, on the Pi: ros2 launch turtlebot3_bringup robot.launch.py
Map a place (drive it with teleop_twist_keyboard), then save it with tools/save_map.sh:
    ros2 launch martha_nav burger.launch.py
Navigate in it, starting where the mapping began (or use 2D Pose Estimate in RViz):
    ros2 launch martha_nav burger.launch.py map:=/abs/maps/sala checkpoint:=/abs/policy.npz [speed_scale:=0.5]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from martha_nav.robots import ROBOTS
from martha_nav.ros.slam import slam_toolbox

ARGUMENTS = [
    DeclareLaunchArgument('map', default_value='',
                          description='saved map without extension; empty maps the place'),
    DeclareLaunchArgument('checkpoint', default_value='',
                          description='policy.npz from export_policy (or a .zip, with PyTorch)'),
    DeclareLaunchArgument('speed_scale', default_value='1.0'),
]


def launch_setup(context, *args, **kwargs):
    reach = ROBOTS['burger'].lidar_range
    saved_map = LaunchConfiguration('map').perform(context)
    if not saved_map:
        return [slam_toolbox('mapping', '/scan', use_sim_time=False, max_range=reach)]
    return [
        slam_toolbox('localization', '/scan', use_sim_time=False, map_file=saved_map,
                     max_range=reach),
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'map_yaml': saved_map + '.yaml'}]),
        Node(package='martha_nav', executable='global_planner', output='screen',
             parameters=[{'robot': 'burger'}]),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint'), 'robot': 'burger',
                          'speed_scale': ParameterValue(LaunchConfiguration('speed_scale'),
                                                        value_type=float)}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
```

- [ ] **Step 4: `tools/deploy_burger.sh`** (`chmod +x`):

```bash
#!/usr/bin/env bash
# Copy martha_nav and an exported policy to the Burger's Raspberry Pi and build it there.
#   ./tools/deploy_burger.sh ubuntu@192.168.1.50 runs/burger_s0/policy.npz
# On the Pi, once: sudo apt install ros-humble-slam-toolbox python3-scipy
set -e
cd "$(dirname "$0")/.."
usage="usage: $0 user@host policy.npz"
host=${1:?$usage}
policy=${2:?$usage}
ws='~/martha_ws'
ssh "$host" "mkdir -p $ws/src/martha_nav/policies"
rsync -a --delete --exclude runs/ --exclude maps/ --exclude policies/ --exclude 'entrega_tesis*' \
  --exclude graphify-out/ --exclude __pycache__/ --exclude .git/ ./ "$host:$ws/src/martha_nav/"
rsync -a "$policy" "$host:$ws/src/martha_nav/policies/"
ssh "$host" "source /opt/ros/humble/setup.bash && cd $ws && colcon build --symlink-install --packages-select martha_nav"
echo "en el robot: source $ws/install/setup.bash && ros2 launch martha_nav burger.launch.py map:=... checkpoint:=$ws/src/martha_nav/policies/$(basename "$policy")"
```

- [ ] **Step 5: `docs/comandos.md`**, sección nueva al final:

````markdown
## TurtleBot3 Burger

El Burger usa el mismo paquete con su perfil (`martha_nav/robots.py`). Su bringup oficial corre
en la Raspberry Pi; la navegación (slam_toolbox, planificadores y política) también, con
`burger.launch.py`. El PC solo mira y manda metas con RViz.

**Antes de entrenar, una vez:** con el bringup corriendo, medir el LiDAR.

```bash
ros2 topic echo /scan --once --field range_max
```

`3.5` es un LDS-01 (el perfil ya lo tiene). `8.0` es un LDS-02: cambiar en `ROBOTS['burger']`
`lidar_range=8.0, lidar_min=0.16`, y en `urdf/burger.urdf.xacro` los `default` de `lidar_range` y
`lidar_min` (`test/test_urdf.py` exige que coincidan).

**Entrenar y evaluar** (como cualquier run):

```bash
./tools/ct_ros ros2 run martha_nav train_policy --preset full --seed 0 --robot burger --wide-dynamics --name burger_s0
./tools/evaluate_run.sh runs/burger_s0
./tools/evaluate_run_gazebo.sh runs/burger_s0 ""
./tools/ct_ros ros2 run martha_nav export_policy --model runs/burger_s0/best_model.zip
```

**Red:** el PC y el robot con el mismo `ROS_DOMAIN_ID` (el del TurtleBot, por defecto 30):
`export ROS_DOMAIN_ID=30` en las dos máquinas antes de lanzar nada.

**Desplegar** (en la Pi, una vez: `sudo apt install ros-humble-slam-toolbox python3-scipy`):

```bash
./tools/deploy_burger.sh ubuntu@<ip-del-robot> runs/burger_s0/policy.npz
```

**Demo, en la Pi** (dos terminales SSH):

```bash
ros2 launch turtlebot3_bringup robot.launch.py
source ~/martha_ws/install/setup.bash && ros2 launch martha_nav burger.launch.py
```

Mapear manejando con `ros2 run teleop_twist_keyboard teleop_twist_keyboard`, guardar con
`tools/save_map.sh ~/martha_ws/src/martha_nav/maps/sala`, y relanzar navegando:

```bash
ros2 launch martha_nav burger.launch.py map:=$HOME/martha_ws/src/martha_nav/maps/sala checkpoint:=$HOME/martha_ws/src/martha_nav/policies/policy.npz speed_scale:=0.5
```

En el PC, `rviz2 -d rviz/nav.rviz` para la pose inicial y las metas. Cada 10 s el planificador
local registra la latencia scan → `/cmd_vel`. Para depurar desde el PC, el mismo
`burger.launch.py` corre allí sin cambios.

**Primera prueba real:** `speed_scale:=0.5`, y comprobar a mano que
`ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"` avanza y
`"{angular: {z: 0.5}}"` gira a la izquierda.
````

- [ ] **Step 6: Ver pasar y commit**

Run: `./tools/ct_ros python3 -m pytest -q -p no:cacheprovider test </dev/null` → todo PASS.

```bash
git add launch/burger.launch.py tools/deploy_burger.sh docs/comandos.md test/test_burger_launch.py
git commit -m "Add the Burger's onboard launch, its deploy script and the demo steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Entrenar y evaluar el Burger

Operativa; requiere la Tarea 10 completa y el `range_max` del LiDAR real.

- [ ] **Step 1: Confirmar el LiDAR.** Pedir al usuario el resultado de `ros2 topic echo /scan --once --field range_max`. Si es `8.0`, aplicar el cambio de perfil y de URDF que describe `docs/comandos.md`, correr `test/test_urdf.py` y commitear (`Use the LDS-02 numbers for the Burger's LiDAR`).

- [ ] **Step 2: Entrenar** (~1 h en GPU, sin Gazebo abierto):

```bash
nohup ./tools/ct_ros ros2 run martha_nav train_policy --preset full --seed 0 --robot burger --wide-dynamics --name burger_s0 > runs/burger_s0.log 2>&1 &
```

Expected: `runs/burger_s0/evals.csv` sube; el mejor éxito de la evaluación periódica ≥ 0.85. Si queda claramente por debajo, detenerse y diagnosticar con el usuario antes de evaluar.

- [ ] **Step 3: Evaluar en 2D y en Gazebo**

```bash
./tools/evaluate_run.sh runs/burger_s0
./tools/evaluate_run_gazebo.sh runs/burger_s0 ""
```

- [ ] **Step 4: Comparar 2D y Gazebo** con McNemar exacto en las mismas semillas (`eval_obstacles_lab.csv` frente a `eval_gazebo_lab.csv`, y los puntos). Criterio del spec: p > 0.05.

- [ ] **Step 5: Exportar** y comprobar que el `.npz` actúa igual:

```bash
./tools/ct_ros ros2 run martha_nav export_policy --model runs/burger_s0/best_model.zip
```

- [ ] **Step 6: Documentar** en `docs/resultados.md` una sección `## TurtleBot3 Burger (burger_s0)` con la tabla de los cuatro conjuntos 2D, los dos de Gazebo y la comparación pareada, y commitear.

---

## Self-review

- Cobertura del spec: Parte 0 (solo CNN, código sin historia, opciones sin uso) → Tareas 1-3; Parte 1 → 4-7; Parte 2 → 6, 10, 12; Parte 3 → 8, 9, 11; pruebas del spec → tests de cada tarea más la huella (Tareas 3 y 6).
- Nombres consistentes: `ROBOTS`, `Robot`, `checkpoint_robot`, `footprint_points`, `settings_of`, `SETTINGS`, `NumpyPolicy`, `load_policy`, `to_twist`, `export`, entidad `robot`.
- Desviación respecto del spec, ya reflejada en él: el URDF del Burger usa geometría propia y no los paquetes `turtlebot3_*`.
