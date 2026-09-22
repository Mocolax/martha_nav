# Plan de implementación 1: simulador 2D y entrenamiento PPO

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el simulador 2D (`sim2d/`) y el entrenamiento/evaluación con Stable-Baselines3 (`learning/`) hasta pasar la puerta de convergencia y completar un primer run completo con obstáculos.

**Architecture:** `sim2d/` es Python puro + numpy/scipy, sin ROS: grid de ocupación, raycasting, planificador (Dijkstra sobre grid inflado), generador de escenarios, dinámica, observación, recompensa y un entorno Gymnasium. `learning/` envuelve PPO de SB3 con dos extractores (CNN-1D circular y MLP), evaluación determinista con semillas fijas y registro por episodio. Todo corre dentro del contenedor Docker `ros2_humble`.

**Tech Stack:** Python 3.10, numpy 1.21, scipy 1.8, gymnasium 0.29.1, torch 2.2.2, stable-baselines3 2.3.2, pytest.

**Spec:** `docs/ppo-local-planner-design.md`

**Alcance de este plan:** fases 1 y 2 de la spec (secciones 3–6, 8 y el primer run de la 9). La integración con ROS 2 y Gazebo (sección 7) va en un **plan 2**, que se escribe después de ver los resultados de este.

**Estado de verificación:** todo el código de este plan se ejecutó como prototipo dentro del contenedor antes de escribirlo: 69 pruebas pasan, y la puerta de convergencia con la CNN dio 99.5% de éxito en 200 episodios nuevos (tarea 13). El rendimiento medido está en las tareas 9 y 12.

## Restricciones globales

- Todo comando de Python o pytest se ejecuta **dentro del contenedor** con `./tools/ct <comando>` desde la raíz del repo (`~/ros2_ws/src/martha_nav`). `git` se ejecuta en el host.
- `sim2d/` **no importa nada de ROS** ni de `learning/`.
- Observación: **96 valores** en [−1, 1] = 90 sectores LiDAR (`min(d, 8 m)/8 m`) + distancia al waypoint (`min(d, 3 m)/3 m`) + ángulo al waypoint (`atan2/π`, **sin seno/coseno**) + velocidad medida (`v/0.35`, `ω/0.8`) + acción anterior (2).
- Acción: `[a_v, a_ω] ∈ [−1, 1]²` → `v ∈ [−0.15, 0.35]` m/s (asimétrica), `ω ∈ [−0.8, 0.8]` rad/s.
- Sin LSTM y sin apilar frames.
- Recompensa: progreso +1.0 por metro de **nuevo récord**, llegada +20, choque −10, tiempo −0.005 por paso; proximidad y giro en 0.
- Huella de Martha 0.56 × 0.41 m; LiDAR en +0.2325 m en x; inflado del planificador 0.40 m (hueco mínimo 0.8 m).
- `lab.world` **nunca** entra al entrenamiento.
- Semillas de entrenamiento `< 1_000_000`; semillas de evaluación `≥ 1_000_000`.
- Mensajes de commit terminan con la línea `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Estilo: flake8 con `--max-line-length 110`.

## Mapa de archivos

| archivo | responsabilidad |
|---|---|
| `package.xml`, `setup.py`, `setup.cfg`, `resource/martha_nav` | paquete ROS 2 ament_python (se compila en el plan 2) |
| `requirements.txt` | dependencias pip sobre la imagen de ROS |
| `tools/ct` | ejecutar un comando en el contenedor, en este repo |
| `worlds/*.world` | copia de los 7 mundos de `martha/worlds` |
| `martha_nav/sim2d/geometry.py` | `Grid`, dibujo de cajas y cilindros, `raycast`, `footprint_collides` |
| `martha_nav/sim2d/planner.py` | `Path` (longitud de arco, proyección), `carrot`, `PlanningGrid` (Dijkstra sobre grid inflado) |
| `martha_nav/sim2d/worlds.py` | rasterizar las cajas y cilindros de un `.world` (**archivo nuevo respecto a la spec**: se separó de `scenarios.py` para mantenerlo enfocado) |
| `martha_nav/sim2d/scenarios.py` | plantillas, fuentes, `ScenarioConfig`, `generate(seed)` con obstáculos sorpresa |
| `martha_nav/sim2d/dynamics.py` | modelo `(v, ω)` con retardo, aceleración limitada y aleatorización |
| `martha_nav/sim2d/observation.py` | `reduce_scan`, `build_observation`, `action_to_cmd` (contrato compartido con ROS) |
| `martha_nav/sim2d/reward.py` | `RewardConfig`, `compute_reward` |
| `martha_nav/sim2d/env.py` | `EnvConfig`, `NavEnv` (Gymnasium) |
| `martha_nav/learning/policy.py` | `LidarCnnExtractor`, `policy_kwargs(arch)` |
| `martha_nav/learning/evaluate.py` | `eval_seeds`, `run_episodes`, `summarize`, `wilson`, CLI |
| `martha_nav/learning/train.py` | presets `gate`/`full`, callbacks de registro y evaluación, CLI |
| `docs/resultados.md` | bitácora de resultados citables (runs, evaluaciones) |
| `docs/escenarios-referencias.md` | revisión de papers que justifica las plantillas |

**Nota sobre rangos de las plantillas:** los huecos de *puerta* y *paso estrecho* empiezan en **0.85 m**, no en 0.8 m. Con celdas de 5 cm, un hueco de exactamente 0.80 m puede quedar cerrado por el inflado de 0.40 m. El hueco mínimo garantizado sigue siendo 0.8 m.

---

### Tarea 1: Estructura del paquete y dependencias

**Files:**
- Create: `package.xml`, `setup.py`, `setup.cfg`, `resource/martha_nav` (vacío), `requirements.txt`, `tools/ct`, `.gitignore`
- Create: `martha_nav/__init__.py`, `martha_nav/sim2d/__init__.py`, `martha_nav/learning/__init__.py` (vacíos)
- Create: `worlds/` (copia de 7 mundos)
- Modify: `~/ros2_ws/Dockerfile` (fuera del repo, no se hace commit)
- Test: `test/test_package.py`

**Interfaces:**
- Produces: el paquete importable `martha_nav` y el comando `./tools/ct`, que usan todas las tareas.

- [ ] **Step 1: Crear los archivos del paquete**

`package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>martha_nav</name>
  <version>0.1.0</version>
  <description>PPO local planner for the Martha robot (thesis).</description>
  <maintainer email="nicolas58david@hotmail.com">Mocolax</maintainer>
  <license>MIT</license>

  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>python3-scipy</exec_depend>
  <exec_depend>python3-yaml</exec_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`setup.py`:
```python
from glob import glob

from setuptools import find_packages, setup

package_name = 'martha_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mocolax',
    maintainer_email='nicolas58david@hotmail.com',
    description='PPO local planner for the Martha robot (thesis).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
```

`setup.cfg`:
```ini
[develop]
script_dir=$base/lib/martha_nav
[install]
install_scripts=$base/lib/martha_nav
```

`requirements.txt`:
```text
# Installed on top of the ROS 2 Humble image (see ~/ros2_ws/Dockerfile).
# numpy stays on the system 1.21 so ROS packages keep working.
stable-baselines3==2.3.2
pandas<2
numpy<2
tensorboard<2.18
```

`tools/ct` (después: `chmod +x tools/ct`):
```bash
#!/usr/bin/env bash
# Run a command inside the ros2_humble container, in this repository.
exec docker exec -w /home/ros/ros2_ws/src/martha_nav ros2_humble "$@"
```

`.gitignore`:
```text
__pycache__/
*.pyc
.pytest_cache/
runs/
```

```bash
mkdir -p resource martha_nav/sim2d martha_nav/learning test worlds
touch resource/martha_nav martha_nav/__init__.py martha_nav/sim2d/__init__.py martha_nav/learning/__init__.py
chmod +x tools/ct
cp ../martha/worlds/{four_rooms,hall,lab,multi,roblab,room,tube}.world worlds/
```

- [ ] **Step 2: Escribir la prueba**

`test/test_package.py`:
```python
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_package_imports():
    import martha_nav.learning  # noqa: F401
    import martha_nav.sim2d  # noqa: F401


def test_worlds_copied():
    names = sorted(p.stem for p in (REPO / 'worlds').glob('*.world'))
    assert names == ['four_rooms', 'hall', 'lab', 'multi', 'roblab', 'room', 'tube']
```

- [ ] **Step 3: Instalar las dependencias en el contenedor**

Añadir al `~/ros2_ws/Dockerfile`, justo después de la línea `gymnasium==0.29.1 \` / `pyserial==3.5` del bloque pip existente, un bloque nuevo para que una reconstrucción de la imagen conserve la instalación:

```dockerfile
RUN python3 -m pip install --no-cache-dir \
    "stable-baselines3==2.3.2" "pandas<2" "numpy<2" "tensorboard<2.18"
```

E instalar ya en el contenedor en marcha:

Run: `./tools/ct pip install --user -r requirements.txt`
Run: `./tools/ct python3 -c "import stable_baselines3, numpy; print(stable_baselines3.__version__, numpy.__version__)"`
Expected: `2.3.2 1.21.5`. **numpy debe seguir en 1.21.5**; si cambió, algo arrastró numpy 2 y hay que revisar antes de seguir.

Run: `./tools/ct bash -c 'source /opt/ros/humble/setup.bash && python3 -c "import rclpy; print(\"ok\")"'`
Expected: `ok` (ROS sigue funcionando).

- [ ] **Step 4: Ejecutar la prueba**

Run: `./tools/ct python3 -m pytest -q test/test_package.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add package.xml setup.py setup.cfg resource requirements.txt tools .gitignore martha_nav worlds test/test_package.py
git commit -m "Scaffold the martha_nav package and its dependencies" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 2: Geometría (grid, raycasting, colisión)

**Files:**
- Create: `martha_nav/sim2d/geometry.py`
- Test: `test/test_geometry.py`

**Interfaces:**
- Produces:
  - Constantes `RESOLUTION = 0.05`, `ROBOT_LENGTH = 0.56`, `ROBOT_WIDTH = 0.41`, `LIDAR_OFFSET_X = 0.2325`.
  - `Grid(occ: bool[rows, cols], origin: (x, y), resolution=0.05)`, con:
    - `.shape`, `.copy()`;
    - `.crop(x0, y0, x1, y1) -> Grid`: copia recortada que conserva las coordenadas del mundo;
    - `.to_cell(x, y) -> (row, col)`;
    - `.cell_center(row, col) -> (x, y)`;
    - `.occupied(x_array, y_array) -> bool array`. Fuera del grid cuenta como ocupado.
  - `empty_grid(width, height, origin=(0, 0)) -> Grid`.
  - `draw_box(grid, cx, cy, sx, sy, yaw=0.0)` y `draw_circle(grid, cx, cy, radius)`, que modifican el grid in-place y solo recorren las celdas cercanas a la forma (importante para la velocidad de `reset()` en mundos de 20 × 20 m).
  - `raycast(grid, ox, oy, angles, max_range) -> distances`, con los ángulos en coordenadas del mundo.
  - `footprint_collides(grid, x, y, theta) -> bool`.

- [ ] **Step 1: Escribir las pruebas**

`test/test_geometry.py`:
```python
import numpy as np

from martha_nav.sim2d.geometry import draw_box, draw_circle, empty_grid, footprint_collides, raycast


def test_grid_cells_and_outside_is_occupied():
    g = empty_grid(2.0, 1.0)
    assert g.shape == (20, 40)
    assert not g.occupied(np.array([1.0]), np.array([0.5]))[0]
    assert g.occupied(np.array([-0.1]), np.array([0.5]))[0]
    assert g.occupied(np.array([1.0]), np.array([5.0]))[0]


def test_draw_box_and_circle():
    g = empty_grid(4.0, 4.0)
    draw_box(g, 1.0, 1.0, 0.5, 0.5)
    draw_circle(g, 3.0, 3.0, 0.3)
    assert g.occupied(np.array([1.0, 3.0, 2.0]), np.array([1.0, 3.0, 2.0])).tolist() == [True, True, False]


def test_rotated_box():
    g = empty_grid(4.0, 4.0)
    draw_box(g, 2.0, 2.0, 2.0, 0.2, yaw=np.pi / 2)   # long along y after rotation
    assert g.occupied(np.array([2.0]), np.array([2.9]))[0]
    assert not g.occupied(np.array([2.9]), np.array([2.0]))[0]


def test_raycast_hits_wall_at_expected_distance():
    g = empty_grid(10.0, 4.0)
    draw_box(g, 5.05, 2.0, 0.1, 4.0)                 # wall face at x = 5.0
    d = raycast(g, 1.0, 2.0, np.array([0.0, np.pi]), max_range=8.0)
    assert abs(d[0] - 4.0) <= 0.05
    assert abs(d[1] - 1.0) <= 0.05                   # grid border counts as occupied


def test_raycast_returns_max_range_when_nothing_is_hit():
    g = empty_grid(20.0, 20.0)
    d = raycast(g, 10.0, 10.0, np.array([0.0]), max_range=8.0)
    assert d[0] == 8.0


def test_footprint_collision_respects_orientation():
    g = empty_grid(4.0, 4.0)
    draw_box(g, 2.0, 2.0 + 0.30, 2.0, 0.05)          # thin wall 0.30 m above the centre
    assert not footprint_collides(g, 2.0, 2.0, 0.0)          # half-width 0.205 < 0.30
    assert footprint_collides(g, 2.0, 2.0, np.pi / 2)        # half-length 0.28 reaches the wall cells


def test_crop_keeps_world_coordinates():
    g = empty_grid(10.0, 10.0)
    draw_box(g, 7.0, 7.0, 0.5, 0.5)
    c = g.crop(5.0, 5.0, 9.0, 9.0)
    assert c.shape[0] < g.shape[0]
    assert c.occupied(np.array([7.0, 6.0]), np.array([7.0, 6.0])).tolist() == [True, False]
    assert c.occupied(np.array([2.0]), np.array([2.0]))[0]     # outside the crop is occupied


def test_drawing_only_touches_cells_near_the_shape():
    g = empty_grid(10.0, 10.0)
    draw_box(g, 5.0, 5.0, 1.0, 0.5, yaw=0.3)
    draw_circle(g, 1.0, 1.0, 0.3)
    rows, cols = np.nonzero(g.occ)
    xs, ys = g.cell_center(rows, cols)
    assert np.all((np.hypot(xs - 5.0, ys - 5.0) < 0.6) | (np.hypot(xs - 1.0, ys - 1.0) < 0.31))
    assert 0.5 * 0.9 < g.occ.sum() * g.resolution ** 2 < (0.5 + np.pi * 0.09) * 1.2
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_geometry.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.geometry'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/geometry.py`:
```python
"""Occupancy grid, LiDAR raycasting and footprint collision. Pure numpy."""
from dataclasses import dataclass, field

import numpy as np

RESOLUTION = 0.05          # m per cell
ROBOT_LENGTH = 0.56        # m, along x of base_link
ROBOT_WIDTH = 0.41         # m, along y of base_link
LIDAR_OFFSET_X = 0.2325    # m, LiDAR position in base_link


@dataclass
class Grid:
    """Boolean occupancy grid. occ[row, col]; row grows with y, col with x.

    origin is the world (x, y) of the lower-left corner of cell (0, 0).
    Everything outside the grid counts as occupied.
    """

    occ: np.ndarray
    origin: tuple
    resolution: float = RESOLUTION
    _padded: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def shape(self):
        return self.occ.shape

    def copy(self):
        return Grid(self.occ.copy(), self.origin, self.resolution)

    def crop(self, x0, y0, x1, y1):
        """Copy of the cells covering [x0, x1] x [y0, y1], clipped to the grid."""
        (r0, r1), (c0, c1) = self._window(x0, y0, x1, y1)
        origin = (self.origin[0] + c0 * self.resolution, self.origin[1] + r0 * self.resolution)
        return Grid(self.occ[r0:r1, c0:c1].copy(), origin, self.resolution)

    def _window(self, x0, y0, x1, y1):
        """Row and column slices (start, stop) covering a world rectangle."""
        (ra, ca), (rb, cb) = self.to_cell(x0, y0), self.to_cell(x1, y1)
        rows, cols = self.occ.shape
        return ((int(np.clip(ra, 0, rows)), int(np.clip(rb + 1, 0, rows))),
                (int(np.clip(ca, 0, cols)), int(np.clip(cb + 1, 0, cols))))

    def to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return row, col

    def cell_center(self, row, col):
        x = self.origin[0] + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(row) + 0.5) * self.resolution
        return x, y

    def occupied(self, x, y):
        # A one-cell occupied border lets out-of-grid points clip onto it.
        if self._padded is None:
            self._padded = np.pad(self.occ, 1, constant_values=True)
        row, col = self.to_cell(x, y)
        rows, cols = self.occ.shape
        return self._padded[np.clip(row + 1, 0, rows + 1), np.clip(col + 1, 0, cols + 1)]


def empty_grid(width, height, origin=(0.0, 0.0), resolution=RESOLUTION):
    rows = int(np.ceil(height / resolution))
    cols = int(np.ceil(width / resolution))
    return Grid(np.zeros((rows, cols), dtype=bool), tuple(origin), resolution)


def _local_centers(grid, cx, cy, reach):
    """Window slices around (cx, cy) and the cell-centre coordinates inside it."""
    (r0, r1), (c0, c1) = grid._window(cx - reach, cy - reach, cx + reach, cy + reach)
    xs, ys = grid.cell_center(np.arange(r0, r1)[:, None], np.arange(c0, c1)[None, :])
    return (slice(r0, r1), slice(c0, c1)), xs, ys


def draw_box(grid, cx, cy, sx, sy, yaw=0.0):
    """Mark cells whose centers fall inside a rotated rectangle."""
    win, xs, ys = _local_centers(grid, cx, cy, np.hypot(sx, sy) / 2)
    dx, dy = xs - cx, ys - cy
    c, s = np.cos(yaw), np.sin(yaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    grid.occ[win] |= (np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)
    grid._padded = None


def draw_circle(grid, cx, cy, radius):
    win, xs, ys = _local_centers(grid, cx, cy, radius)
    grid.occ[win] |= (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    grid._padded = None


def raycast(grid, ox, oy, angles, max_range):
    """Distance from (ox, oy) along each world angle to the first occupied cell."""
    step = grid.resolution / 2
    ts = np.arange(step, max_range + step, step)
    angles = np.asarray(angles, dtype=float)
    xs = ox + np.cos(angles)[:, None] * ts[None, :]
    ys = oy + np.sin(angles)[:, None] * ts[None, :]
    hits = grid.occupied(xs, ys)
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), np.minimum(ts[first], max_range), max_range)


def _footprint_points(length=ROBOT_LENGTH, width=ROBOT_WIDTH, spacing=RESOLUTION / 2):
    hx, hy = length / 2, width / 2
    xs = np.linspace(-hx, hx, int(np.ceil(length / spacing)) + 1)
    ys = np.linspace(-hy, hy, int(np.ceil(width / spacing)) + 1)
    top = np.stack([xs, np.full_like(xs, hy)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, -hy)], axis=1)
    left = np.stack([np.full_like(ys, -hx), ys], axis=1)
    right = np.stack([np.full_like(ys, hx), ys], axis=1)
    return np.concatenate([top, bottom, left, right])


FOOTPRINT = _footprint_points()


def footprint_collides(grid, x, y, theta):
    """True when the robot rectangle's perimeter touches an occupied cell."""
    c, s = np.cos(theta), np.sin(theta)
    px = x + c * FOOTPRINT[:, 0] - s * FOOTPRINT[:, 1]
    py = y + s * FOOTPRINT[:, 0] + c * FOOTPRINT[:, 1]
    return bool(grid.occupied(px, py).any())
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_geometry.py`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/geometry.py test/test_geometry.py
git commit -m "Add occupancy grid, LiDAR raycasting and footprint collision" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 3: Planificador, ruta y zanahoria

**Files:**
- Create: `martha_nav/sim2d/planner.py`
- Test: `test/test_planner.py`

**Interfaces:**
- Consumes: `Grid`, `empty_grid`, `draw_box` (tarea 2).
- Produces:
  - `INFLATION = 0.40`.
  - `Path(points Nx2)`, con:
    - `.points`, `.s` (longitud de arco acumulada), `.length`;
    - `.point_at(s) -> (2,)`;
    - `.project(x, y, s_hint=None, back=1.0, fwd=2.0) -> s`.
  - `carrot(path, s, lookahead, scan_points=None, clearance=0.4, step=0.1) -> (point (2,), s_carrot)`.
  - `PlanningGrid(grid, inflation=0.40)`, con:
    - `.node_at(x, y) -> int` (−1 si no es libre);
    - `.node_xy(node) -> (2,)`;
    - `.largest_component() -> node ids`;
    - `.distances_from(node, limit=inf) -> (dist, predecessors)`;
    - `.route(start_xy, goal_xy, pred=None, start_node=None) -> Path | None`.

  La implementación usa Dijkstra de `scipy.sparse.csgraph` (en C) en lugar de un A* en Python puro. Da el mismo camino más corto y es mucho más rápido.

- [ ] **Step 1: Escribir las pruebas**

`test/test_planner.py`:
```python
import numpy as np

from martha_nav.sim2d.geometry import draw_box, empty_grid
from martha_nav.sim2d.planner import Path, PlanningGrid, carrot


def corridor_grid(gap=None):
    """8 x 3 m room; optional wall across x = 4 with a gap of `gap` metres centred at y = 1.5."""
    g = empty_grid(8.0, 3.0)
    if gap is not None:
        lower = 1.5 - gap / 2
        draw_box(g, 4.0, lower / 2, 0.1, lower)
        draw_box(g, 4.0, 3.0 - lower / 2, 0.1, lower)
    return g


def test_path_arc_length_and_point_at():
    p = Path([[0, 0], [3, 0], [3, 4]])
    assert p.length == 7.0
    assert np.allclose(p.point_at(5.0), [3, 2])
    assert np.allclose(p.point_at(99), [3, 4])


def test_project_uses_window_around_hint():
    p = Path([[0, 0], [4, 0], [4, 1], [0, 1]])     # U-turn: legs 1 m apart
    assert abs(p.project(1.0, 0.1) - 1.0) < 1e-9
    # Near the start of the return leg, with a hint on the first leg, stay on the first leg.
    assert p.project(1.0, 0.6, s_hint=1.0) < 2.0


def test_route_found_in_open_room_and_blocked_by_wall():
    open_pg = PlanningGrid(corridor_grid())
    route = open_pg.route((1.0, 1.5), (7.0, 1.5))
    assert route is not None and 6.0 <= route.length <= 6.2
    blocked = PlanningGrid(corridor_grid(gap=0.0))
    assert blocked.route((1.0, 1.5), (7.0, 1.5)) is None


def test_inflation_requires_gap_of_about_0_8_m():
    assert PlanningGrid(corridor_grid(gap=0.9)).route((1.0, 1.5), (7.0, 1.5)) is not None
    assert PlanningGrid(corridor_grid(gap=0.7)).route((1.0, 1.5), (7.0, 1.5)) is None


def test_route_endpoints_are_exact():
    route = PlanningGrid(corridor_grid()).route((1.01, 1.49), (6.97, 1.52))
    assert np.allclose(route.points[0], [1.01, 1.49])
    assert np.allclose(route.points[-1], [6.97, 1.52])


def test_carrot_lookahead_and_end_of_path():
    p = Path([[0, 0], [10, 0]])
    pt, s = carrot(p, 2.0, 1.5)
    assert np.allclose(pt, [3.5, 0]) and s == 3.5
    pt, s = carrot(p, 9.5, 1.5)
    assert np.allclose(pt, [10, 0])


def test_carrot_skips_points_near_scan_hits():
    p = Path([[0, 0], [10, 0]])
    scan = np.array([[3.5, 0.1], [3.8, 0.0]])       # obstacle right on the carrot
    pt, s = carrot(p, 2.0, 1.5, scan_points=scan, clearance=0.4)
    assert s >= 4.2 - 1e-9
    assert np.min(np.hypot(*(scan - pt).T)) >= 0.4
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_planner.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.planner'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/planner.py`:
```python
"""Grid shortest paths (Dijkstra on an inflated grid), route geometry and carrot."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

INFLATION = 0.40   # m; a path in the inflated grid implies a free gap >= 0.8 m


class Path:
    """Polyline route with arc-length parametrisation."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        seg = np.diff(self.points, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])

    def point_at(self, s):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, len(self.seg_len) - 1))
        if self.seg_len[i] == 0.0:
            return self.points[i].copy()
        t = (s - self.s[i]) / self.seg_len[i]
        return self.points[i] + t * (self.points[i + 1] - self.points[i])

    def project(self, x, y, s_hint=None, back=1.0, fwd=2.0):
        """Arc length of the closest route point, searched near s_hint if given."""
        a, b = self.points[:-1], self.points[1:]
        ab = b - a
        denom = np.maximum((ab ** 2).sum(axis=1), 1e-12)
        t = np.clip(((np.array([x, y]) - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = ((proj - np.array([x, y])) ** 2).sum(axis=1)
        s_proj = self.s[:-1] + t * self.seg_len
        if s_hint is not None:
            window = (s_proj >= s_hint - back) & (s_proj <= s_hint + fwd)
            if window.any():
                d2 = np.where(window, d2, np.inf)
        return float(s_proj[int(np.argmin(d2))])


def carrot(path, s, lookahead, scan_points=None, clearance=0.4, step=0.1):
    """Route point `lookahead` ahead of s, pushed forward past scanned obstacles.

    Returns (point (2,), s_carrot).
    """
    s_c = min(s + lookahead, path.length)
    if scan_points is None or len(scan_points) == 0:
        return path.point_at(s_c), s_c
    while True:
        p = path.point_at(s_c)
        if np.min(np.hypot(*(scan_points - p).T)) >= clearance or s_c >= path.length:
            return p, s_c
        s_c = min(s_c + step, path.length)


class PlanningGrid:
    """8-connected graph over the free cells of an inflated occupancy grid."""

    def __init__(self, grid, inflation=INFLATION):
        self.grid = grid
        clearance = ndimage.distance_transform_edt(~grid.occ) * grid.resolution
        self.free = clearance > inflation
        rows, cols = self.free.shape
        self.node_of = -np.ones(self.free.shape, dtype=np.int64)
        free_rc = np.argwhere(self.free)
        self.node_of[free_rc[:, 0], free_rc[:, 1]] = np.arange(len(free_rc))
        self.rc = free_rc
        src, dst, w = [], [], []
        for dr, dc, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            a = self.node_of[r0:r1, c0:c1]
            b = self.node_of[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ok = (a >= 0) & (b >= 0)
            src.append(a[ok])
            dst.append(b[ok])
            w.append(np.full(int(ok.sum()), cost * grid.resolution))
        n = len(free_rc)
        self.graph = coo_matrix(
            (np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(n, n)
        ).tocsr()
        _, labels = connected_components(self.graph, directed=False)
        self.labels = labels

    def node_at(self, x, y):
        row, col = self.grid.to_cell(x, y)
        if 0 <= row < self.free.shape[0] and 0 <= col < self.free.shape[1]:
            return int(self.node_of[row, col])
        return -1

    def node_xy(self, node):
        x, y = self.grid.cell_center(self.rc[node, 0], self.rc[node, 1])
        return np.array([float(x), float(y)])

    def largest_component(self):
        """Node ids of the largest connected free region."""
        if len(self.labels) == 0:
            return np.array([], dtype=np.int64)
        biggest = np.bincount(self.labels).argmax()
        return np.flatnonzero(self.labels == biggest)

    def distances_from(self, node, limit=np.inf):
        return dijkstra(self.graph, directed=False, indices=node,
                        return_predecessors=True, limit=limit)

    def route(self, start_xy, goal_xy, pred=None, start_node=None):
        """Path from start to goal through free cells, or None."""
        s = self.node_at(*start_xy) if start_node is None else start_node
        g = self.node_at(*goal_xy)
        if s < 0 or g < 0:
            return None
        if pred is None:
            _, pred = self.distances_from(s)
        if g != s and pred[g] < 0:
            return None
        nodes = [g]
        while nodes[-1] != s:
            nodes.append(pred[nodes[-1]])
        pts = [self.node_xy(n) for n in reversed(nodes)]
        pts[0] = np.asarray(start_xy, dtype=float)
        pts[-1] = np.asarray(goal_xy, dtype=float)
        if len(pts) == 1:
            pts.append(pts[0].copy())
        return Path(np.array(pts))
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_planner.py`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/planner.py test/test_planner.py
git commit -m "Add grid shortest paths, route projection and carrot selection" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 4: Rasterizador de mundos `.world`

**Files:**
- Create: `martha_nav/sim2d/worlds.py`
- Test: `test/test_worlds.py`

**Interfaces:**
- Consumes: `RESOLUTION`, `draw_box`, `draw_circle`, `empty_grid` (tarea 2); `PlanningGrid` (tarea 3, solo en las pruebas).
- Produces:
  - `WORLDS_DIR` (`<repo>/worlds`).
  - `world_shapes(path) -> list`, con elementos `('box', x, y, sx, sy, yaw)` o `('cylinder', x, y, r)`.
  - `rasterize_world(name) -> Grid`: cacheado y de **solo lectura**; hay que usar `.copy()` para modificarlo.

  Compone las poses de modelo, link y collision, e ignora `ground_plane` y `sun`. El grid se recorta al rectángulo que envuelve las formas, así que lo que queda fuera cuenta como ocupado. Esto evita el error de la bitácora vieja (§7) de muestrear metas fuera de la arena.

- [ ] **Step 1: Escribir las pruebas**

`test/test_worlds.py`:
```python
import numpy as np
import pytest

from martha_nav.sim2d.planner import PlanningGrid
from martha_nav.sim2d.worlds import rasterize_world, world_shapes, WORLDS_DIR

ALL_WORLDS = ['four_rooms', 'hall', 'multi', 'roblab', 'room', 'tube', 'lab']


def test_lab_shapes_parsed():
    shapes = world_shapes(WORLDS_DIR / 'lab.world')
    assert len(shapes) == 13
    assert all(s[0] == 'box' for s in shapes)


def test_lab_raster_matches_its_7_5_by_9_5_m_footprint():
    g = rasterize_world('lab')
    h, w = g.shape
    assert abs(w * g.resolution - 7.65) < 0.1      # 7.5 m + wall thickness
    assert abs(h * g.resolution - 9.65) < 0.1


def test_nested_model_poses_are_composed():
    # tube.world places walls through <model><pose>, not through collision poses.
    g = rasterize_world('tube')
    assert g.occ.mean() > 0.02


@pytest.mark.parametrize('name', ALL_WORLDS)
def test_every_world_has_a_large_free_region(name):
    pg = PlanningGrid(rasterize_world(name))
    area = len(pg.largest_component()) * pg.grid.resolution ** 2
    assert area > 20.0


def test_raster_is_cached_and_read_only():
    a = rasterize_world('room')
    assert a is rasterize_world('room')
    with pytest.raises(ValueError):
        a.occ[0, 0] = True
    b = a.copy()
    b.occ[0, 0] = not b.occ[0, 0]
    assert np.any(a.occ != b.occ)
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_worlds.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.worlds'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/worlds.py`:
```python
"""Rasterise the static boxes and cylinders of a Gazebo .world into a Grid."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from martha_nav.sim2d.geometry import RESOLUTION, draw_box, draw_circle, empty_grid

WORLDS_DIR = Path(__file__).resolve().parents[2] / 'worlds'
SKIP_MODELS = {'ground_plane', 'sun'}


def _pose(elem):
    """(x, y, yaw) of an element's <pose>, identity if absent."""
    node = elem.find('pose')
    if node is None or not node.text:
        return np.zeros(3)
    v = [float(t) for t in node.text.split()]
    return np.array([v[0], v[1], v[5]])


def _compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def world_shapes(path):
    """List of ('box', x, y, sx, sy, yaw) and ('cylinder', x, y, r) in world coordinates."""
    root = ET.parse(path).getroot()
    shapes = []
    for model in root.iter('model'):
        if model.get('name') in SKIP_MODELS:
            continue
        m_pose = _pose(model)
        for link in model.findall('link'):
            l_pose = _compose(m_pose, _pose(link))
            for col in link.findall('collision'):
                pose = _compose(l_pose, _pose(col))
                box = col.find('geometry/box/size')
                cyl = col.find('geometry/cylinder/radius')
                if box is not None:
                    sx, sy, _ = (float(t) for t in box.text.split())
                    shapes.append(('box', pose[0], pose[1], sx, sy, pose[2]))
                elif cyl is not None:
                    shapes.append(('cylinder', pose[0], pose[1], float(cyl.text)))
    return shapes


def _extent(shape):
    if shape[0] == 'box':
        _, x, y, sx, sy, yaw = shape
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        hx, hy = (c * sx + s * sy) / 2, (s * sx + c * sy) / 2
        return x - hx, y - hy, x + hx, y + hy
    _, x, y, r = shape
    return x - r, y - r, x + r, y + r


@lru_cache(maxsize=None)
def rasterize_world(name, resolution=RESOLUTION):
    """Grid of worlds/<name>.world clipped to the bounding box of its shapes."""
    shapes = world_shapes(WORLDS_DIR / f'{name}.world')
    ext = np.array([_extent(s) for s in shapes])
    x0, y0 = ext[:, 0].min(), ext[:, 1].min()
    x1, y1 = ext[:, 2].max(), ext[:, 3].max()
    grid = empty_grid(x1 - x0, y1 - y0, origin=(x0, y0), resolution=resolution)
    for shape in shapes:
        if shape[0] == 'box':
            draw_box(grid, *shape[1:])
        else:
            draw_circle(grid, *shape[1:])
    grid.occ.setflags(write=False)
    return grid
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_worlds.py`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/worlds.py test/test_worlds.py
git commit -m "Rasterize the static shapes of Gazebo worlds" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 5: Escenarios (plantillas, episodios, obstáculos sorpresa)

**Files:**
- Create: `martha_nav/sim2d/scenarios.py`
- Test: `test/test_scenarios.py`

**Interfaces:**
- Consumes: `Grid`, `draw_box`, `draw_circle`, `empty_grid` (tarea 2); `INFLATION`, `PlanningGrid`, `Path` (tarea 3); `rasterize_world` (tarea 4).
- Produces:
  - `TEMPLATES`: diccionario nombre → función `(rng) -> Grid`. Incluye `open_room`, que se usa solo en la puerta de convergencia.
  - `WORLD_SOURCES`, `TRAIN_SOURCES` (6 plantillas + 6 mundos, sin `open_room` ni `lab`) y `EVAL_ONLY_SOURCES = ('lab',)`.
  - `Obstacle(kind, x, y, sx, sy, yaw, radius)`, con `.extent` y `.draw(grid)`.
  - `ScenarioConfig(sources, obstacle_mode='mixed'|'none'|'always', p_no_obstacles=0.2, max_obstacles=4, route_min=3.0, route_max=12.0, inflation=0.40, obstacle_attempts=20, start_clearance=1.0, goal_clearance=0.6, lateral_offset=0.5)`.
  - `Scenario(source, static, full, start (x, y, yaw), goal (x, y), path, shortest, obstacles, obstacles_dropped)`.
  - `generate(seed, cfg=ScenarioConfig()) -> Scenario`, determinista por semilla.

- [ ] **Step 1: Escribir las pruebas**

`test/test_scenarios.py`:
```python
import numpy as np
import pytest

from martha_nav.sim2d.planner import PlanningGrid
from martha_nav.sim2d.scenarios import (EVAL_ONLY_SOURCES, TEMPLATES, TRAIN_SOURCES,
                                        ScenarioConfig, generate)


def test_lab_is_never_a_training_source():
    assert 'lab' in EVAL_ONLY_SOURCES
    assert 'lab' not in TRAIN_SOURCES
    assert 'open_room' not in TRAIN_SOURCES


def test_same_seed_same_episode():
    a, b = generate(123), generate(123)
    assert a.source == b.source
    assert np.allclose(a.start, b.start) and np.allclose(a.goal, b.goal)
    assert np.array_equal(a.full.occ, b.full.occ)


@pytest.mark.parametrize('name', list(TEMPLATES))
def test_every_template_yields_episodes(name):
    for seed in range(5):
        sc = generate(seed, ScenarioConfig(sources=(name,), obstacle_mode='none'))
        assert 3.0 <= sc.path.length <= 12.5


def test_obstacle_modes():
    none = [generate(s, ScenarioConfig(obstacle_mode='none')) for s in range(20)]
    assert all(len(sc.obstacles) == 0 for sc in none)
    always = [generate(s, ScenarioConfig(obstacle_mode='always')) for s in range(20)]
    assert sum(len(sc.obstacles) > 0 for sc in always) >= 18


def test_500_mixed_episodes_respect_every_rule():
    cfg = ScenarioConfig()
    with_obstacles = 0
    for seed in range(500):
        sc = generate(seed, cfg)
        assert sc.source in TRAIN_SOURCES
        assert cfg.route_min <= sc.path.length <= cfg.route_max + 0.5
        for ob in sc.obstacles:
            assert np.hypot(ob.x - sc.start[0], ob.y - sc.start[1]) - ob.extent >= cfg.start_clearance
            assert np.hypot(ob.x - sc.goal[0], ob.y - sc.goal[1]) - ob.extent >= cfg.goal_clearance
        if sc.obstacles:
            with_obstacles += 1
            detour = PlanningGrid(sc.full, cfg.inflation).route(sc.start[:2], sc.goal)
            assert detour is not None                        # gap >= 0.8 m exists
            assert detour.length <= 1.5 * sc.path.length + 2.0
    assert 0.6 <= with_obstacles / 500 <= 0.9               # ~80% nominal minus drops
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_scenarios.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.scenarios'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/scenarios.py`:
```python
"""Episode generation: static map, start/goal, static route, surprise obstacles."""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from martha_nav.sim2d.geometry import Grid, draw_box, draw_circle, empty_grid
from martha_nav.sim2d.planner import INFLATION, PlanningGrid
from martha_nav.sim2d.worlds import rasterize_world

WALL = 0.15


def _room(w, h):
    """Empty room with interior w x h; returns (grid, outer_w, outer_h)."""
    W, H = w + 2 * WALL, h + 2 * WALL
    g = empty_grid(W, H)
    draw_box(g, W / 2, WALL / 2, W, WALL)
    draw_box(g, W / 2, H - WALL / 2, W, WALL)
    draw_box(g, WALL / 2, H / 2, WALL, H)
    draw_box(g, W - WALL / 2, H / 2, WALL, H)
    return g, W, H


def _random_flip(g, rng):
    occ = g.occ
    if rng.random() < 0.5:
        occ = occ[:, ::-1]
    if rng.random() < 0.5:
        occ = occ[::-1, :]
    return Grid(np.ascontiguousarray(occ), g.origin, g.resolution)


def open_room(rng):
    g, _, _ = _room(rng.uniform(4, 8), rng.uniform(4, 8))
    return g


def corridor(rng):
    length, width = rng.uniform(6, 12), rng.uniform(1.2, 2.5)
    g, _, _ = _room(length, width) if rng.random() < 0.5 else _room(width, length)
    return g


def doorway(rng):
    w1, w2, h = rng.uniform(3, 5), rng.uniform(3, 5), rng.uniform(3, 5)
    g, W, H = _room(w1 + WALL + w2, h)
    gap = rng.uniform(0.85, 1.2)
    y_gap = rng.uniform(WALL + 0.3, H - WALL - 0.3 - gap)
    x = WALL + w1 + WALL / 2
    draw_box(g, x, y_gap / 2, WALL, y_gap)
    top = y_gap + gap
    draw_box(g, x, (top + H) / 2, WALL, H - top)
    return _random_flip(g, rng)


def l_turn(rng):
    g, W, H = _room(rng.uniform(5, 8), rng.uniform(5, 8))
    c1, c2 = rng.uniform(1.2, 2.0), rng.uniform(1.2, 2.0)
    x0, y0 = WALL + c1, WALL + c2
    draw_box(g, (x0 + W) / 2, (y0 + H) / 2, W - x0, H - y0)
    return _random_flip(g, rng)


def furniture_walls(rng):
    g, W, H = _room(rng.uniform(4, 8), rng.uniform(4, 8))
    for _ in range(rng.integers(2, 7)):
        sx, sy = rng.uniform(0.4, 1.5), rng.uniform(0.4, 0.8)
        side = rng.integers(4)
        if side == 0:
            draw_box(g, rng.uniform(WALL, W - WALL), WALL + sy / 2, sx, sy)
        elif side == 1:
            draw_box(g, rng.uniform(WALL, W - WALL), H - WALL - sy / 2, sx, sy)
        elif side == 2:
            draw_box(g, WALL + sy / 2, rng.uniform(WALL, H - WALL), sy, sx)
        else:
            draw_box(g, W - WALL - sy / 2, rng.uniform(WALL, H - WALL), sy, sx)
    return g


def furniture_center(rng):
    g, W, H = _room(rng.uniform(5, 8), rng.uniform(5, 8))
    for _ in range(rng.integers(1, 5)):
        draw_box(g, rng.uniform(WALL + 1.0, W - WALL - 1.0), rng.uniform(WALL + 1.0, H - WALL - 1.0),
                 rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5), rng.uniform(0, np.pi))
    return g


def narrow_passage(rng):
    g, W, H = _room(rng.uniform(5, 8), rng.uniform(4, 7))
    gap, thick = rng.uniform(0.85, 1.0), rng.uniform(0.5, 1.5)
    y_gap = rng.uniform(WALL + 0.5, H - WALL - 0.5 - gap)
    x = W / 2
    draw_box(g, x, y_gap / 2, thick, y_gap)
    top = y_gap + gap
    draw_box(g, x, (top + H) / 2, thick, H - top)
    return _random_flip(g, rng)


TEMPLATES = {
    'open_room': open_room,
    'corridor': corridor,
    'doorway': doorway,
    'l_turn': l_turn,
    'furniture_walls': furniture_walls,
    'furniture_center': furniture_center,
    'narrow_passage': narrow_passage,
}
WORLD_SOURCES = ('four_rooms', 'hall', 'multi', 'roblab', 'room', 'tube')
TRAIN_SOURCES = ('corridor', 'doorway', 'l_turn', 'furniture_walls', 'furniture_center',
                 'narrow_passage') + WORLD_SOURCES
EVAL_ONLY_SOURCES = ('lab',)


@dataclass
class Obstacle:
    kind: str            # 'box' or 'cylinder'
    x: float
    y: float
    sx: float = 0.0      # box size x
    sy: float = 0.0      # box size y
    yaw: float = 0.0
    radius: float = 0.0  # cylinder

    @property
    def extent(self):
        return self.radius if self.kind == 'cylinder' else float(np.hypot(self.sx, self.sy) / 2)

    def draw(self, grid):
        if self.kind == 'box':
            draw_box(grid, self.x, self.y, self.sx, self.sy, self.yaw)
        else:
            draw_circle(grid, self.x, self.y, self.radius)


@dataclass
class ScenarioConfig:
    sources: tuple = TRAIN_SOURCES
    obstacle_mode: str = 'mixed'     # 'mixed' | 'none' | 'always'
    p_no_obstacles: float = 0.2
    max_obstacles: int = 4
    route_min: float = 3.0
    route_max: float = 12.0
    inflation: float = INFLATION
    obstacle_attempts: int = 20
    start_clearance: float = 1.0
    goal_clearance: float = 0.6
    lateral_offset: float = 0.5


@dataclass
class Scenario:
    source: str
    static: Grid
    full: Grid
    start: np.ndarray          # (x, y, yaw)
    goal: np.ndarray           # (x, y)
    path: object               # planner.Path over the static map
    shortest: float            # route length with obstacles, for SPL
    obstacles: list = field(default_factory=list)
    obstacles_dropped: bool = False


@lru_cache(maxsize=None)
def _world_planning(name, inflation):
    grid = rasterize_world(name)
    return grid, PlanningGrid(grid, inflation)


def static_map(source, rng, inflation=INFLATION):
    if source in TEMPLATES:
        grid = TEMPLATES[source](rng)
        return grid, PlanningGrid(grid, inflation)
    return _world_planning(source, inflation)


def _sample_obstacle(rng, path, start, goal, cfg):
    for _ in range(50):
        s = rng.uniform(0.0, path.length)
        p = path.point_at(s)
        t = path.point_at(min(s + 0.1, path.length)) - path.point_at(max(s - 0.1, 0.0))
        n = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-9)
        c = p + n * rng.uniform(-cfg.lateral_offset, cfg.lateral_offset)
        if rng.random() < 0.5:
            ob = Obstacle('box', c[0], c[1], sx=rng.uniform(0.2, 0.6), sy=rng.uniform(0.2, 0.6),
                          yaw=rng.uniform(0, np.pi))
        else:
            ob = Obstacle('cylinder', c[0], c[1], radius=rng.uniform(0.1, 0.3))
        if (np.hypot(*(c - start[:2])) - ob.extent >= cfg.start_clearance
                and np.hypot(*(c - goal)) - ob.extent >= cfg.goal_clearance):
            return ob
    return None


def _detour(full, path, start, goal, cfg, margin=2.0):
    """Route around the obstacles, planned on a crop around the static route (faster)."""
    lo = path.points.min(axis=0) - margin
    hi = path.points.max(axis=0) + margin
    return PlanningGrid(full.crop(lo[0], lo[1], hi[0], hi[1]), cfg.inflation).route(start[:2], goal)


def _n_obstacles(rng, cfg):
    if cfg.obstacle_mode == 'none':
        return 0
    if cfg.obstacle_mode == 'mixed' and rng.random() < cfg.p_no_obstacles:
        return 0
    return int(rng.integers(1, cfg.max_obstacles + 1))


def generate(seed, cfg=ScenarioConfig()):
    """Deterministic episode for an integer seed."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        source = str(cfg.sources[rng.integers(len(cfg.sources))])
        static, pg = static_map(source, rng, cfg.inflation)
        nodes = pg.largest_component()
        if len(nodes) == 0:
            continue
        start_node = int(nodes[rng.integers(len(nodes))])
        dist, pred = pg.distances_from(start_node, limit=cfg.route_max)
        cand = np.flatnonzero((dist >= cfg.route_min) & (dist <= cfg.route_max))
        if len(cand) == 0:
            continue
        goal = pg.node_xy(int(cand[rng.integers(len(cand))]))
        start_xy = pg.node_xy(start_node)
        path = pg.route(start_xy, goal, pred=pred, start_node=start_node)
        start = np.array([start_xy[0], start_xy[1], rng.uniform(-np.pi, np.pi)])
        break
    else:
        raise RuntimeError(f'no valid start/goal for seed {seed}')

    n = _n_obstacles(rng, cfg)
    if n == 0:
        return Scenario(source, static, static, start, goal, path, path.length)
    for _ in range(cfg.obstacle_attempts):
        obstacles = [o for o in (_sample_obstacle(rng, path, start, goal, cfg) for _ in range(n)) if o]
        full = static.copy()
        for ob in obstacles:
            ob.draw(full)
        detour = _detour(full, path, start, goal, cfg)
        if detour is not None and detour.length <= 1.5 * path.length + 2.0:
            return Scenario(source, static, full, start, goal, path, detour.length, obstacles)
    return Scenario(source, static, static, start, goal, path, path.length, obstacles_dropped=True)
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_scenarios.py`
Expected: `11 passed`. La prueba de 500 semillas tarda ~15 s. Un `reset()` mide ~12–15 ms de media (máximo ~50 ms), gracias al dibujo local y a que la comprobación de hueco se planifica sobre un recorte de ±2 m alrededor de la ruta. Esto importa porque `SubprocVecEnv` es síncrono: un reset lento detiene a los 16 entornos.

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/scenarios.py test/test_scenarios.py
git commit -m "Generate episodes from templates and worlds with surprise obstacles" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 6: Dinámica

**Files:**
- Create: `martha_nav/sim2d/dynamics.py`
- Test: `test/test_dynamics.py`

**Interfaces:**
- Produces:
  - `DT = 0.1`, `SUBSTEPS = 5`.
  - `DynamicsRanges(tau=(0.05, 0.4), acc_v=(0.3, 1.0), acc_w=(1.0, 3.0), gain=(0.9, 1.1))`.
  - `DynamicsParams(tau_v, tau_w, acc_v, acc_w, gain_v, gain_w)`.
  - `sample_params(rng, ranges) -> DynamicsParams`.
  - `Dynamics(params)`, con:
    - `.reset(pose)`;
    - `.step(cmd_v, cmd_w, collides) -> bool`, donde `collides(x, y, theta)`;
    - los atributos `.pose (x, y, θ)`, `.v` y `.w`.

  El comando enviado en el paso k actúa en el paso k+1 (1 paso de retardo).

- [ ] **Step 1: Escribir las pruebas**

`test/test_dynamics.py`:
```python
import numpy as np

from martha_nav.sim2d.dynamics import Dynamics, DynamicsParams, DynamicsRanges, sample_params

never = lambda x, y, t: False  # noqa: E731


def test_command_acts_one_step_later():
    d = Dynamics(DynamicsParams())
    d.step(0.3, 0.0, never)
    assert d.v == 0.0
    d.step(0.3, 0.0, never)
    assert d.v > 0.0


def test_converges_to_gain_times_command():
    d = Dynamics(DynamicsParams(tau_v=0.3, gain_v=1.1))
    for _ in range(50):
        d.step(0.3, 0.0, never)
    assert abs(d.v - 0.33) < 1e-3


def test_acceleration_is_limited():
    d = Dynamics(DynamicsParams(tau_v=0.01, acc_v=0.5))
    d.step(0.35, 0.0, never)
    d.step(0.35, 0.0, never)
    assert d.v <= 0.5 * 0.1 + 1e-9


def test_straight_line_and_turning():
    d = Dynamics(DynamicsParams(tau_v=0.01, tau_w=0.01, acc_v=100, acc_w=100))
    for _ in range(11):
        d.step(0.2, 0.0, never)
    assert abs(d.pose[0] - 0.2) < 0.01 and abs(d.pose[1]) < 1e-9
    d.reset(np.zeros(3))
    for _ in range(11):
        d.step(0.0, 0.5, never)
    assert abs(d.pose[2] - 0.5) < 0.01


def test_collision_keeps_last_free_pose_and_stops():
    d = Dynamics(DynamicsParams(tau_v=0.01, acc_v=100))
    wall = lambda x, y, t: x > 0.05  # noqa: E731
    hit = False
    for _ in range(5):
        hit = d.step(0.35, 0.0, wall) or hit
    assert hit and d.pose[0] <= 0.05 and d.v == 0.0


def test_sampled_params_stay_in_ranges():
    r = DynamicsRanges()
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = sample_params(rng, r)
        assert r.tau[0] <= p.tau_v <= r.tau[1] and r.acc_w[0] <= p.acc_w <= r.acc_w[1]
        assert r.gain[0] <= p.gain_v <= r.gain[1]
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_dynamics.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.dynamics'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/dynamics.py`:
```python
"""Unicycle (v, w) model with first-order lag, acceleration limits and 1-step delay."""
from dataclasses import dataclass

import numpy as np

DT = 0.1        # s, control period (10 Hz)
SUBSTEPS = 5


@dataclass
class DynamicsRanges:
    tau: tuple = (0.05, 0.4)      # s
    acc_v: tuple = (0.3, 1.0)     # m/s^2
    acc_w: tuple = (1.0, 3.0)     # rad/s^2
    gain: tuple = (0.9, 1.1)


@dataclass
class DynamicsParams:
    tau_v: float = 0.1
    tau_w: float = 0.1
    acc_v: float = 1.0
    acc_w: float = 3.0
    gain_v: float = 1.0
    gain_w: float = 1.0


def sample_params(rng, r=DynamicsRanges()):
    return DynamicsParams(
        tau_v=rng.uniform(*r.tau), tau_w=rng.uniform(*r.tau),
        acc_v=rng.uniform(*r.acc_v), acc_w=rng.uniform(*r.acc_w),
        gain_v=rng.uniform(*r.gain), gain_w=rng.uniform(*r.gain),
    )


class Dynamics:
    """Integrates the robot pose; the command sent at step k acts at step k+1."""

    def __init__(self, params, dt=DT, substeps=SUBSTEPS):
        self.p = params
        self.h = dt / substeps
        self.substeps = substeps
        self.reset(np.zeros(3))

    def reset(self, pose):
        self.pose = np.array(pose, dtype=float)
        self.v = 0.0
        self.w = 0.0
        self.pending = (0.0, 0.0)

    def _approach(self, current, target, tau, acc):
        alpha = 1.0 - np.exp(-self.h / tau)
        limit = acc * self.h
        dv = min(max((target - current) * alpha, -limit), limit)
        return current + dv

    def step(self, cmd_v, cmd_w, collides):
        """Advance one control period. collides(x, y, theta) -> bool.

        Returns True on collision; the pose then stays at the last free pose
        and the velocities are zeroed.
        """
        target_v = self.p.gain_v * self.pending[0]
        target_w = self.p.gain_w * self.pending[1]
        self.pending = (cmd_v, cmd_w)
        for _ in range(self.substeps):
            self.v = self._approach(self.v, target_v, self.p.tau_v, self.p.acc_v)
            self.w = self._approach(self.w, target_w, self.p.tau_w, self.p.acc_w)
            x, y, th = self.pose
            nxt = np.array([x + self.v * np.cos(th) * self.h,
                            y + self.v * np.sin(th) * self.h,
                            th + self.w * self.h])
            nxt[2] = (nxt[2] + np.pi) % (2 * np.pi) - np.pi
            if collides(*nxt):
                self.v = self.w = 0.0
                return True
            self.pose = nxt
        return False
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_dynamics.py`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/dynamics.py test/test_dynamics.py
git commit -m "Add the (v, w) dynamics model with lag, limits and randomization" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 7: Observación (contrato compartido con ROS)

**Files:**
- Create: `martha_nav/sim2d/observation.py`
- Test: `test/test_observation.py`

**Interfaces:**
- Consumes: `draw_box`, `empty_grid`, `raycast` (tarea 2, solo en las pruebas).
- Produces:
  - Constantes `N_SECTORS = 90`, `LIDAR_MAX = 8.0`, `WAYPOINT_MAX = 3.0`, `V_MAX = 0.35`, `V_REVERSE = 0.15`, `W_MAX = 0.8`, `OBS_DIM = 96`.
  - `reduce_scan(ranges, angles) -> (90,)` en metros.
  - `build_observation(ranges, angles, velocity, waypoint_rel, prev_action) -> float32 (96,)`.
  - `action_to_cmd(action) -> (v, w)`.

  **Orden de la observación:**
  - `[0:90]` LiDAR;
  - `[90]` distancia al waypoint;
  - `[91]` ángulo al waypoint;
  - `[92:94]` velocidad medida;
  - `[94:96]` acción anterior.

  El nodo ROS del plan 2 debe llamar a estas mismas funciones.

- [ ] **Step 1: Escribir las pruebas**

`test/test_observation.py`:
```python
import numpy as np

from martha_nav.sim2d.geometry import draw_box, empty_grid, raycast
from martha_nav.sim2d.observation import (LIDAR_MAX, OBS_DIM, action_to_cmd, build_observation,
                                          reduce_scan)


def test_sector_zero_is_front_and_sectors_grow_counter_clockwise():
    ranges = np.array([1.0, 2.0, 3.0])
    angles = np.array([0.0, np.deg2rad(4.0), np.deg2rad(-4.0)])
    s = reduce_scan(ranges, angles)
    assert s[0] == 1.0 and s[1] == 2.0 and s[89] == 3.0
    assert s[45] == LIDAR_MAX


def test_invalid_readings_count_as_max_range():
    s = reduce_scan(np.array([np.inf, np.nan, 0.0, 20.0]), np.zeros(4))
    assert s[0] == LIDAR_MAX


def test_minimum_per_sector():
    s = reduce_scan(np.array([2.0, 1.5]), np.deg2rad([0.5, -1.0]))
    assert s[0] == 1.5


def test_reduction_is_independent_of_beam_density():
    g = empty_grid(10.0, 10.0)
    draw_box(g, 6.5, 5.0, 0.4, 0.4)
    a360 = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    a720 = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    s360 = reduce_scan(raycast(g, 5.0, 5.0, a360, LIDAR_MAX), a360)
    s720 = reduce_scan(raycast(g, 5.0, 5.0, a720, LIDAR_MAX), a720)
    assert np.max(np.abs(s360 - s720)) < 0.1


def test_observation_layout_and_bounds():
    obs = build_observation(np.full(180, 4.0), np.linspace(-np.pi, np.pi, 180, endpoint=False),
                            velocity=(0.35, -0.8), waypoint_rel=(0.0, 1.5), prev_action=(0.5, -1.0))
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.allclose(obs[:90], 0.5)
    assert np.isclose(obs[90], 0.5) and np.isclose(obs[91], 0.5)   # 1.5/3 m, +90 deg
    assert np.allclose(obs[92:94], [1.0, -1.0])
    assert np.allclose(obs[94:], [0.5, -1.0])
    far = build_observation(np.full(4, 1.0), np.zeros(4), (0.5, 0), (-10.0, 0.0), (0, 0))
    assert far.min() >= -1.0 and far.max() <= 1.0 and np.isclose(far[90], 1.0)


def test_action_mapping_is_asymmetric():
    assert action_to_cmd([1.0, 1.0]) == (0.35, 0.8)
    assert action_to_cmd([-1.0, -1.0]) == (-0.15, -0.8)
    assert action_to_cmd([5.0, 0.0]) == (0.35, 0.0)
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_observation.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.observation'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/observation.py`:
```python
"""The single observation contract shared by the 2D env and the ROS nodes."""
import numpy as np

N_SECTORS = 90
LIDAR_MAX = 8.0     # m, RPLIDAR A2M8
WAYPOINT_MAX = 3.0  # m
V_MAX = 0.35        # m/s forward
V_REVERSE = 0.15    # m/s backward
W_MAX = 0.8         # rad/s
OBS_DIM = N_SECTORS + 6


def reduce_scan(ranges, angles):
    """Minimum range per 4-degree sector.

    angles are relative to the robot's forward axis. Sector 0 is centred on
    the front and sectors grow counter-clockwise. Invalid readings (inf, NaN,
    <= 0) and empty sectors count as LIDAR_MAX.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    width = 2 * np.pi / N_SECTORS
    valid = np.isfinite(ranges) & (ranges > 0)
    r = np.where(valid, np.minimum(ranges, LIDAR_MAX), LIDAR_MAX)
    idx = (np.floor(np.mod(angles + width / 2, 2 * np.pi) / width).astype(int)) % N_SECTORS
    out = np.full(N_SECTORS, LIDAR_MAX)
    np.minimum.at(out, idx, r)
    return out


def build_observation(ranges, angles, velocity, waypoint_rel, prev_action):
    """96-value observation in [-1, 1].

    ranges/angles: raw scan, angles relative to the robot's forward axis.
    velocity: measured (v, w). waypoint_rel: (dx, dy) in the robot frame.
    prev_action: last action sent, already in [-1, 1].
    """
    lidar = reduce_scan(ranges, angles) / LIDAR_MAX
    dx, dy = waypoint_rel
    wp = [min(np.hypot(dx, dy), WAYPOINT_MAX) / WAYPOINT_MAX, np.arctan2(dy, dx) / np.pi]
    vel = [velocity[0] / V_MAX, velocity[1] / W_MAX]
    obs = np.concatenate([lidar, wp, vel, np.asarray(prev_action, dtype=float)])
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def action_to_cmd(action):
    """[-1, 1]^2 -> (v, w); reverse is capped lower than forward on purpose."""
    a_v, a_w = np.clip(action, -1.0, 1.0)
    v = a_v * (V_MAX if a_v >= 0 else V_REVERSE)
    return float(v), float(a_w * W_MAX)
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_observation.py`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/observation.py test/test_observation.py
git commit -m "Add the shared observation contract and action mapping" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 8: Recompensa

**Files:**
- Create: `martha_nav/sim2d/reward.py`
- Test: `test/test_reward.py`

**Interfaces:**
- Produces:
  - `RewardConfig(progress=1.0, goal=20.0, collision=-10.0, step=-0.005, proximity=0.0, proximity_dist=0.5, turn=0.0)`.
  - `compute_reward(progress_gain, reached, collided, min_range, delta_turn, cfg) -> (total, terms)`, con `terms` de claves `progress, goal, collision, step, proximity, turn`.

- [ ] **Step 1: Escribir las pruebas**

`test/test_reward.py`:
```python
from martha_nav.sim2d.reward import RewardConfig, compute_reward


def test_default_terms():
    total, t = compute_reward(0.1, False, False, 2.0, 0.0)
    assert t['progress'] == 0.1 and t['step'] == -0.005
    assert t['proximity'] == 0.0 and t['turn'] == 0.0
    assert abs(total - 0.095) < 1e-12


def test_terminal_values():
    _, t = compute_reward(0.0, True, False, 2.0, 0.0)
    assert t['goal'] == 20.0
    _, t = compute_reward(0.0, False, True, 0.1, 0.0)
    assert t['collision'] == -10.0


def test_negative_progress_is_not_paid():
    _, t = compute_reward(-0.5, False, False, 2.0, 0.0)
    assert t['progress'] == 0.0


def test_optional_terms_when_enabled():
    cfg = RewardConfig(proximity=0.1, turn=0.02)
    _, t = compute_reward(0.0, False, False, 0.25, 1.0, cfg)
    assert abs(t['proximity'] + 0.05) < 1e-12      # half of proximity_dist
    assert abs(t['turn'] + 0.02) < 1e-12
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_reward.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.reward'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/reward.py`:
```python
"""Reward terms. Edit RewardConfig to experiment; every term is logged apart."""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    progress: float = 1.0        # per metre of new-record route progress
    goal: float = 20.0
    collision: float = -10.0
    step: float = -0.005
    proximity: float = 0.0       # optional, off by default
    proximity_dist: float = 0.5  # m
    turn: float = 0.0            # optional, off by default; per unit of |delta a_w|


def compute_reward(progress_gain, reached, collided, min_range, delta_turn, cfg=RewardConfig()):
    """Return (total, terms). progress_gain is metres beyond the episode's best."""
    terms = {
        'progress': cfg.progress * max(progress_gain, 0.0),
        'goal': cfg.goal if reached else 0.0,
        'collision': cfg.collision if collided else 0.0,
        'step': cfg.step,
        'proximity': -cfg.proximity * max(0.0, 1.0 - min_range / cfg.proximity_dist),
        'turn': -cfg.turn * abs(delta_turn),
    }
    return sum(terms.values()), terms
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_reward.py`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/sim2d/reward.py test/test_reward.py
git commit -m "Add the four-term reward with optional terms off" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 9: Entorno Gymnasium

**Files:**
- Create: `martha_nav/sim2d/env.py`
- Create: `docs/resultados.md`
- Test: `test/test_env.py`

**Interfaces:**
- Consumes: todo lo de las tareas 2–8.
- Produces:
  - `TRAIN_SEED_LIMIT = 1_000_000`.
  - `EnvConfig(scenario, dynamics, reward, carrot_range=(1.0, 2.5), carrot_clearance=0.4, goal_tolerance=0.3, no_progress_time=15.0, n_rays=180, lidar_noise=(0.01, 0.02), lidar_dropout=0.01, vel_noise=0.05, episode_seeds=())`.
  - `NavEnv(cfg)`: espacio de observación `Box(-1, 1, (96,))` y de acción `Box(-1, 1, (2,))`.
  - Al terminar un episodio, `info` contiene:
    - `outcome` (`success` / `collision` / `timeout` / `stalled`);
    - `episode_seed`, `source`, `n_obstacles`, `obstacles_dropped`;
    - `route_length`, `shortest`, `travelled`, `spl`, `steps`;
    - la suma de cada término de recompensa (`r_progress`, `r_goal`, `r_collision`, `r_step`, `r_proximity`, `r_turn`).
  - Con `episode_seeds` no vacío, cada `reset()` juega la siguiente semilla de la lista. Esto es lo que usa la evaluación.

- [ ] **Step 1: Escribir las pruebas**

`test/test_env.py`:
```python
from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig, NavEnv
from martha_nav.sim2d.scenarios import ScenarioConfig

OPEN = EnvConfig(scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none'))


def pursue(obs):
    """Scripted controller: turn toward the carrot, drive when roughly aligned."""
    ang = obs[91]
    return np.array([1.0 if abs(ang) < 0.25 else 0.0, np.clip(4 * ang, -1, 1)])


def test_gymnasium_api():
    check_env(NavEnv(), skip_render_check=True)


def test_reset_is_deterministic_per_seed():
    env = NavEnv()
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b)
    assert env.episode_seed < TRAIN_SEED_LIMIT


def test_episode_seeds_are_played_in_order():
    env = NavEnv(replace(OPEN, episode_seeds=(TRAIN_SEED_LIMIT + 5, TRAIN_SEED_LIMIT + 9)))
    env.reset()
    assert env.episode_seed == TRAIN_SEED_LIMIT + 5
    env.reset()
    assert env.episode_seed == TRAIN_SEED_LIMIT + 9


def test_scripted_controller_reaches_goals_in_open_rooms():
    env = NavEnv(OPEN)
    outcomes = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(pursue(obs))
            done = term or trunc
        outcomes.append(info['outcome'])
        if info['outcome'] == 'success':
            assert info['r_goal'] == 20.0 and 0.0 < info['spl'] <= 1.0
    assert outcomes.count('success') >= 18


def test_progress_is_paid_once_per_metre():
    env = NavEnv(OPEN)
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(pursue(obs))
        done = term or trunc
    assert info['outcome'] == 'success'
    assert info['r_progress'] <= info['route_length'] + 1e-6


def test_driving_into_a_wall_is_a_terminal_collision():
    env = NavEnv(OPEN)
    env.reset(seed=2)
    for _ in range(2000):
        _, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
        if term or trunc:
            break
    # Straight ahead from a random pose ends at a wall or, rarely, at the goal.
    assert info['outcome'] in ('collision', 'success')
    if info['outcome'] == 'collision':
        assert r < -9.0


def test_standing_still_is_truncated_as_stalled():
    env = NavEnv(OPEN)
    env.reset(seed=3)
    steps = 0
    while True:
        _, _, term, trunc, info = env.step(np.zeros(2))
        steps += 1
        if term or trunc:
            break
    assert trunc and info['outcome'] == 'stalled' and steps == 150
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_env.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.sim2d.env'`

- [ ] **Step 3: Implementar**

`martha_nav/sim2d/env.py`:
```python
"""Gymnasium environment: Martha following a carrot along an A* route."""
from collections import defaultdict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from martha_nav.sim2d.dynamics import DT, Dynamics, DynamicsRanges, sample_params
from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, footprint_collides, raycast
from martha_nav.sim2d.observation import (LIDAR_MAX, OBS_DIM, V_MAX, action_to_cmd,
                                          build_observation)
from martha_nav.sim2d.planner import carrot
from martha_nav.sim2d.reward import RewardConfig, compute_reward
from martha_nav.sim2d.scenarios import ScenarioConfig, generate

TRAIN_SEED_LIMIT = 1_000_000   # training episode seeds are < this; evaluation seeds are >=


@dataclass
class EnvConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    dynamics: DynamicsRanges = field(default_factory=DynamicsRanges)
    reward: RewardConfig = field(default_factory=RewardConfig)
    carrot_range: tuple = (1.0, 2.5)
    carrot_clearance: float = 0.4
    goal_tolerance: float = 0.3
    no_progress_time: float = 15.0
    n_rays: int = 180
    lidar_noise: tuple = (0.01, 0.02)
    lidar_dropout: float = 0.01
    vel_noise: float = 0.05
    episode_seeds: tuple = ()    # evaluation: play exactly these seeds, in order


class NavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=None):
        self.cfg = cfg or EnvConfig()
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.ray_angles = np.linspace(-np.pi, np.pi, self.cfg.n_rays, endpoint=False)
        self._seed_index = 0

    # ---- episode ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.cfg.episode_seeds:
            seeds = self.cfg.episode_seeds
            self.episode_seed = int(seeds[self._seed_index % len(seeds)])
            self._seed_index += 1
        else:
            self.episode_seed = int(self.np_random.integers(TRAIN_SEED_LIMIT))
        self.sc = generate(self.episode_seed, self.cfg.scenario)
        self.rng = np.random.default_rng([self.episode_seed, 1])
        self.dyn = Dynamics(sample_params(self.rng, self.cfg.dynamics))
        self.dyn.reset(self.sc.start)
        self.lookahead = self.rng.uniform(*self.cfg.carrot_range)
        self.lidar_sigma = self.rng.uniform(*self.cfg.lidar_noise)
        self.s = self.s_best = 0.0
        self.steps = self.since_progress = 0
        self.max_steps = int(np.ceil((3 * self.sc.path.length / V_MAX + 10) / DT))
        self.prev_action = np.zeros(2)
        self.travelled = 0.0
        self.terms = defaultdict(float)
        self._scan()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        before = self.dyn.pose[:2].copy()
        collided = self.dyn.step(*action_to_cmd(action),
                                 lambda x, y, t: footprint_collides(self.sc.full, x, y, t))
        x, y, _ = self.dyn.pose
        self.travelled += float(np.hypot(*(self.dyn.pose[:2] - before)))
        self.steps += 1
        self.s = self.sc.path.project(x, y, s_hint=self.s)
        gain = max(0.0, self.s - self.s_best)
        self.s_best = max(self.s_best, self.s)
        self.since_progress = 0 if gain > 1e-3 else self.since_progress + 1
        to_goal = np.hypot(x - self.sc.goal[0], y - self.sc.goal[1])
        reached = not collided and to_goal < self.cfg.goal_tolerance
        self._scan()
        reward, terms = compute_reward(gain, reached, collided, float(self.ranges.min()),
                                       action[1] - self.prev_action[1], self.cfg.reward)
        for k, v in terms.items():
            self.terms[k] += v
        self.prev_action = action
        terminated = collided or reached
        stalled = self.since_progress * DT >= self.cfg.no_progress_time
        truncated = not terminated and (self.steps >= self.max_steps or stalled)
        info = {}
        if terminated or truncated:
            outcome = ('collision' if collided else 'success' if reached
                       else 'stalled' if stalled else 'timeout')
            info = self._summary(outcome)
        return self._obs(), float(reward), terminated, truncated, info

    # ---- sensing ---------------------------------------------------------
    def _scan(self):
        x, y, th = self.dyn.pose
        ox, oy = x + LIDAR_OFFSET_X * np.cos(th), y + LIDAR_OFFSET_X * np.sin(th)
        r = raycast(self.sc.full, ox, oy, th + self.ray_angles, LIDAR_MAX)
        r = r + self.rng.normal(0.0, self.lidar_sigma, r.shape)
        r[self.rng.random(r.shape) < self.cfg.lidar_dropout] = LIDAR_MAX
        self.ranges = np.clip(r, 0.0, LIDAR_MAX)
        hit = self.ranges < LIDAR_MAX
        a = th + self.ray_angles[hit]
        self.scan_points = np.stack([ox + self.ranges[hit] * np.cos(a),
                                     oy + self.ranges[hit] * np.sin(a)], axis=1)

    def _obs(self):
        x, y, th = self.dyn.pose
        point, _ = carrot(self.sc.path, self.s, self.lookahead, self.scan_points,
                          self.cfg.carrot_clearance)
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy)
        noise = 1.0 + self.rng.normal(0.0, self.cfg.vel_noise, 2)
        vel = (self.dyn.v * noise[0], self.dyn.w * noise[1])
        return build_observation(self.ranges, self.ray_angles, vel, rel, self.prev_action)

    def _summary(self, outcome):
        success = outcome == 'success'
        spl = self.sc.shortest / max(self.sc.shortest, self.travelled) if success else 0.0
        info = {
            'outcome': outcome,
            'episode_seed': self.episode_seed,
            'source': self.sc.source,
            'n_obstacles': len(self.sc.obstacles),
            'obstacles_dropped': self.sc.obstacles_dropped,
            'route_length': self.sc.path.length,
            'shortest': self.sc.shortest,
            'travelled': self.travelled,
            'spl': spl,
            'steps': self.steps,
        }
        info.update({f'r_{k}': v for k, v in self.terms.items()})
        return info
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_env.py`
Expected: `7 passed`

- [ ] **Step 5: Medir el rendimiento (puerta de la fase 1)**

Run:
```bash
./tools/ct python3 -c "
import time
from martha_nav.sim2d.env import NavEnv
env = NavEnv(); env.reset(seed=0); t = time.time(); n = 0
while n < 4000:
    _, _, te, tr, _ = env.step(env.action_space.sample()); n += 1
    if te or tr: env.reset()
print(f'{n / (time.time() - t):.0f} steps/s per process')
"
```
Expected: entre 800 y 1100 pasos/s por proceso (el prototipo midió 850–1170 según el modo de energía del PC; hay que medir con el perfil **rendimiento**). Si baja de 400, revisar antes de seguir: el cuello de botella esperado es `raycast`.

- [ ] **Step 6: Crear la bitácora de resultados**

`docs/resultados.md`, completando la cifra medida en el paso 5:

```markdown
# Resultados — martha_nav

Bitácora de resultados citables. Toda cifra apunta a su evidencia (carpeta de run o CSV).
Las configuraciones se leen del `config.yaml` del run, nunca del código fuente.

## Rendimiento del simulador 2D

- Entorno de un proceso: <PASOS> pasos/s (medido con la tarea 9, paso 5; CPU i9-13900HX).
```

(`<PASOS>` se reemplaza por el número medido; es un dato, no un marcador pendiente del plan).

- [ ] **Step 7: Correr toda la suite y commit**

Run: `./tools/ct python3 -m pytest -q test`
Expected: todas pasan (62 en este punto).

```bash
git add martha_nav/sim2d/env.py test/test_env.py docs/resultados.md
git commit -m "Add the Gymnasium navigation environment" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 10: Arquitecturas de la política (CNN-1D y MLP)

**Files:**
- Create: `martha_nav/learning/policy.py`
- Test: `test/test_policy.py`

**Interfaces:**
- Consumes: `N_SECTORS`, `OBS_DIM` (tarea 7).
- Produces:
  - `LidarCnnExtractor(observation_space)`, con `features_dim = 160`.
  - `policy_kwargs(arch) -> dict` para `arch` en `'cnn'` o `'mlp'`: incluye `net_arch=dict(pi=[256, 256], vf=[256, 256])`, `share_features_extractor=False` y `log_std_init=-0.5`.

- [ ] **Step 1: Escribir las pruebas**

`test/test_policy.py`:
```python
import numpy as np
import pytest
import torch
from gymnasium import spaces

from martha_nav.learning.policy import LidarCnnExtractor, policy_kwargs
from martha_nav.sim2d.observation import OBS_DIM

OBS_SPACE = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)


def test_cnn_extractor_shape():
    ext = LidarCnnExtractor(OBS_SPACE)
    out = ext(torch.zeros(4, OBS_DIM))
    assert out.shape == (4, 160) and ext.features_dim == 160


def test_cnn_is_circular_over_the_lidar():
    """Rotating the scan by one full turn of sectors must not change the output;
    and a hit next to sector 0 must affect it like a hit next to sector 89."""
    ext = LidarCnnExtractor(OBS_SPACE)
    obs = torch.rand(1, OBS_DIM)
    rolled = obs.clone()
    rolled[0, :90] = torch.roll(obs[0, :90], 90)
    assert torch.allclose(ext(obs), ext(rolled))
    conv = ext.cnn[0]
    x = torch.zeros(1, 1, 90)
    x[0, 0, 89] = 1.0
    assert conv(x)[0, :, 0].abs().sum() > 0      # sector 89 leaks into sector 0


def test_policy_kwargs():
    assert policy_kwargs('cnn')['features_extractor_class'] is LidarCnnExtractor
    assert 'features_extractor_class' not in policy_kwargs('mlp')
    assert policy_kwargs('mlp')['share_features_extractor'] is False
    with pytest.raises(ValueError):
        policy_kwargs('lstm')
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_policy.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.learning.policy'`

- [ ] **Step 3: Implementar**

`martha_nav/learning/policy.py`:
```python
"""Feature extractors and SB3 policy kwargs for the two compared architectures."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from martha_nav.sim2d.observation import N_SECTORS


class LidarCnnExtractor(BaseFeaturesExtractor):
    """Circular 1D CNN over the LiDAR sectors plus a linear branch for the rest."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=160)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, padding_mode='circular'), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2, padding_mode='circular'), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, 1, N_SECTORS)).shape[1]
        self.lidar_head = nn.Sequential(nn.Linear(n_flat, 128), nn.ReLU())
        rest = observation_space.shape[0] - N_SECTORS
        self.rest_head = nn.Sequential(nn.Linear(rest, 32), nn.ReLU())

    def forward(self, obs):
        lidar = self.cnn(obs[:, None, :N_SECTORS])
        return torch.cat([self.lidar_head(lidar), self.rest_head(obs[:, N_SECTORS:])], dim=1)


def policy_kwargs(arch):
    """SB3 policy_kwargs for 'cnn' (main) or 'mlp' (baseline)."""
    kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                  share_features_extractor=False, log_std_init=-0.5)
    if arch == 'cnn':
        kwargs['features_extractor_class'] = LidarCnnExtractor
    elif arch != 'mlp':
        raise ValueError(f'unknown arch {arch!r}')
    return kwargs
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_policy.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/learning/policy.py test/test_policy.py
git commit -m "Add the circular 1D-CNN extractor and the MLP baseline" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 11: Evaluación determinista

**Files:**
- Create: `martha_nav/learning/evaluate.py`
- Test: `test/test_evaluate.py`

**Interfaces:**
- Consumes: `TRAIN_SEED_LIMIT`, `EnvConfig`, `NavEnv` (tarea 9); `TRAIN_SOURCES` (tarea 5).
- Produces:
  - `CONDITIONS = {'clean': 'none', 'obstacles': 'always', 'mixed': 'mixed'}`.
  - `eval_seeds(n, offset=0) -> list[int]`, con semillas `≥ 1_000_000`.
  - `run_episodes(model, env_cfg, seeds, n_envs=8, deterministic=True) -> list[dict]`, ordenada por semilla. `model` solo necesita `predict(obs, deterministic)`.
  - `wilson(k, n) -> (lo, hi)`.
  - `summarize(rows) -> dict`, con `episodes`, `spl`, `success`, `collision`, `timeout`, `stalled` y `<outcome>_ci`.
  - `write_csv(rows, path)`.
  - CLI: `python3 -m martha_nav.learning.evaluate --model M --episodes N --condition clean|obstacles|mixed [--sources ...] [--out CSV]`.

- [ ] **Step 1: Escribir las pruebas**

`test/test_evaluate.py`:
```python
import numpy as np

from martha_nav.learning.evaluate import eval_seeds, run_episodes, summarize, wilson
from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig
from martha_nav.sim2d.scenarios import ScenarioConfig

OPEN = EnvConfig(scenario=ScenarioConfig(sources=('open_room',), obstacle_mode='none'))


def test_eval_seeds_are_disjoint_from_training():
    assert min(eval_seeds(10)) >= TRAIN_SEED_LIMIT


def test_wilson_interval():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88


class Pursuit:
    """Stand-in for a trained model (same predict signature as SB3)."""

    def predict(self, obs, deterministic=True):
        ang = obs[:, 91]
        return np.stack([np.where(np.abs(ang) < 0.25, 1.0, 0.0), np.clip(4 * ang, -1, 1)], 1), None


def test_run_episodes_plays_each_seed_once():
    seeds = eval_seeds(6)
    rows = run_episodes(Pursuit(), OPEN, seeds, n_envs=2)
    assert [r['episode_seed'] for r in rows] == seeds
    s = summarize(rows)
    assert s['episodes'] == 6 and s['success'] >= 5 / 6
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_evaluate.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.learning.evaluate'`

- [ ] **Step 3: Implementar**

`martha_nav/learning/evaluate.py`:
```python
"""Deterministic evaluation on fixed seeds; CSV rows and a summary with 95% CIs."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from martha_nav.sim2d.env import TRAIN_SEED_LIMIT, EnvConfig, NavEnv
from martha_nav.sim2d.scenarios import TRAIN_SOURCES

CONDITIONS = {'clean': 'none', 'obstacles': 'always', 'mixed': 'mixed'}


def eval_seeds(n, offset=0):
    """Evaluation episode seeds; disjoint from training seeds by construction."""
    return [TRAIN_SEED_LIMIT + offset + i for i in range(n)]


def make_vec_env(cfgs):
    fns = [lambda c=c: NavEnv(c) for c in cfgs]
    return DummyVecEnv(fns) if len(fns) == 1 else SubprocVecEnv(fns)


def run_episodes(model, env_cfg, seeds, n_envs=8, deterministic=True):
    """Play every seed exactly once; returns one info dict per episode, in seed order."""
    n_envs = max(1, min(n_envs, len(seeds)))
    chunks = [tuple(c) for c in np.array_split(np.asarray(seeds), n_envs)]
    venv = make_vec_env([replace(env_cfg, episode_seeds=c) for c in chunks])
    done_count = [0] * n_envs
    rows = []
    obs = venv.reset()
    while any(done_count[i] < len(chunks[i]) for i in range(n_envs)):
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = venv.step(actions)
        for i, done in enumerate(dones):
            if done and done_count[i] < len(chunks[i]):
                done_count[i] += 1
                rows.append({k: v for k, v in infos[i].items()
                             if k not in ('terminal_observation', 'TimeLimit.truncated')})
    venv.close()
    return sorted(rows, key=lambda r: r['episode_seed'])


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(rows):
    n = len(rows)
    out = {'episodes': n, 'spl': float(np.mean([r['spl'] for r in rows])) if n else 0.0}
    for outcome in ('success', 'collision', 'timeout', 'stalled'):
        k = sum(r['outcome'] == outcome for r in rows)
        lo, hi = wilson(k, n)
        out[outcome] = k / n if n else 0.0
        out[f'{outcome}_ci'] = (lo, hi)
    return out


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser(description='Evaluate a trained policy in the 2D simulator.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--condition', choices=list(CONDITIONS), default='obstacles')
    ap.add_argument('--sources', nargs='+', default=list(TRAIN_SOURCES))
    ap.add_argument('--n-envs', type=int, default=8)
    ap.add_argument('--out', default=None, help='CSV path (default: next to the model)')
    args = ap.parse_args(argv)

    cfg = EnvConfig()
    cfg = replace(cfg, scenario=replace(cfg.scenario, sources=tuple(args.sources),
                                        obstacle_mode=CONDITIONS[args.condition]))
    model = PPO.load(args.model, device='cpu')
    rows = run_episodes(model, cfg, eval_seeds(args.episodes), args.n_envs)
    out = args.out or Path(args.model).with_name(
        f'eval_{args.condition}_{"-".join(args.sources) if len(args.sources) < 3 else "train"}.csv')
    write_csv(rows, out)
    s = summarize(rows)
    print(f'{args.condition}: episodes={s["episodes"]} success={s["success"]:.3f} '
          f'[{s["success_ci"][0]:.3f}, {s["success_ci"][1]:.3f}] collision={s["collision"]:.3f} '
          f'timeout={s["timeout"]:.3f} stalled={s["stalled"]:.3f} spl={s["spl"]:.3f} -> {out}')
    return s


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_evaluate.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/learning/evaluate.py test/test_evaluate.py
git commit -m "Add deterministic fixed-seed evaluation with Wilson intervals" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 12: Entrenamiento

**Files:**
- Create: `martha_nav/learning/train.py`
- Test: `test/test_train.py`

**Interfaces:**
- Consumes: `eval_seeds`, `run_episodes`, `summarize` (tarea 11); `policy_kwargs` (tarea 10); `EnvConfig`, `NavEnv` (tarea 9); `TRAIN_SOURCES` (tarea 5).
- Produces:
  - `PRESETS`:
    - `'gate'`: `open_room`, sin obstáculos, 500k pasos;
    - `'full'`: `TRAIN_SOURCES`, obstáculos `mixed`, 5M pasos.
  - `PPO_PARAMS` y `LEARNING_RATE = 3e-4`, con decaimiento lineal.
  - `main(argv) -> run_dir`. `--device auto` es el valor por defecto: usa CUDA si está disponible. En la RTX 4070 fue ~40% más rápido que la CPU (1 473 frente a 1 035 pasos/s, entrenamiento completo con 16 entornos).
  - Cada run escribe en `runs/<name>/`:
    - `config.yaml`, `episodes.csv`, `evals.csv`;
    - `best_model.zip`, `last_model.zip`, `vecnormalize.pkl`;
    - `tb/`.

- [ ] **Step 1: Escribir la prueba de humo**

`test/test_train.py`:
```python
import numpy as np
from stable_baselines3 import PPO

from martha_nav.learning.train import main as train_main
from martha_nav.sim2d.observation import OBS_DIM


def test_train_smoke(tmp_path):
    run_dir = train_main(['--preset', 'gate', '--arch', 'cnn', '--steps', '2048', '--n-envs', '2',
                          '--eval-every', '1024', '--eval-episodes', '2',
                          '--runs-dir', str(tmp_path), '--name', 'smoke'])
    for f in ('config.yaml', 'episodes.csv', 'evals.csv', 'best_model.zip', 'last_model.zip',
              'vecnormalize.pkl'):
        assert (run_dir / f).exists(), f
    model = PPO.load(run_dir / 'last_model.zip', device='cpu')
    action, _ = model.predict(np.zeros(OBS_DIM, np.float32), deterministic=True)
    assert action.shape == (2,)
```

- [ ] **Step 2: Verificar que falla**

Run: `./tools/ct python3 -m pytest -q test/test_train.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.learning.train'`

- [ ] **Step 3: Implementar**

`martha_nav/learning/train.py`:
```python
"""Train PPO (Stable-Baselines3) on the 2D simulator.

python3 -m martha_nav.learning.train --preset gate --arch cnn --seed 0
python3 -m martha_nav.learning.train --preset full --arch cnn --seed 0
"""
import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from martha_nav.learning.evaluate import eval_seeds, run_episodes, summarize
from martha_nav.learning.policy import policy_kwargs
from martha_nav.sim2d.env import EnvConfig, NavEnv
from martha_nav.sim2d.scenarios import TRAIN_SOURCES

RUNS_DIR = Path(__file__).resolve().parents[2] / 'runs'

PRESETS = {
    # Convergence gate: open room, no obstacles, must reach >= 80% success.
    'gate': dict(sources=('open_room',), obstacle_mode='none', steps=500_000),
    # Main training: every training source, mixed obstacles.
    'full': dict(sources=TRAIN_SOURCES, obstacle_mode='mixed', steps=5_000_000),
}

PPO_PARAMS = dict(n_steps=512, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)
LEARNING_RATE = 3e-4


class EpisodeLogger(BaseCallback):
    """Appends one CSV row per finished training episode."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.file = self.writer = None

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if not done or 'outcome' not in info:
                continue
            row = {'timesteps': self.num_timesteps}
            row.update({k: v for k, v in info.items()
                        if k not in ('terminal_observation', 'TimeLimit.truncated', 'episode')})
            if self.writer is None:
                self.file = open(self.path, 'w', newline='')
                self.writer = csv.DictWriter(self.file, fieldnames=list(row), extrasaction='ignore')
                self.writer.writeheader()
            self.writer.writerow(row)
        return True

    def _on_training_end(self):
        if self.file:
            self.file.close()


class PeriodicEval(BaseCallback):
    """Every `every` steps: deterministic evaluation on fixed seeds; keeps best_model.zip."""

    def __init__(self, env_cfg, run_dir, every, episodes, n_envs=8):
        super().__init__()
        self.env_cfg, self.run_dir = env_cfg, run_dir
        self.every, self.seeds, self.n_envs = every, eval_seeds(episodes), n_envs
        self.next_eval, self.best = every, -1.0

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.every
        s = summarize(run_episodes(self.model, self.env_cfg, self.seeds, self.n_envs))
        line = {'timesteps': self.num_timesteps, 'success': s['success'],
                'collision': s['collision'], 'timeout': s['timeout'],
                'stalled': s['stalled'], 'spl': s['spl']}
        path = self.run_dir / 'evals.csv'
        new = not path.exists()
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(line))
            if new:
                w.writeheader()
            w.writerow(line)
        for k, v in line.items():
            if k != 'timesteps':
                self.logger.record(f'eval/{k}', v)
        if s['success'] > self.best:
            self.best = s['success']
            self.model.save(self.run_dir / 'best_model.zip')
        print(f'[eval] {self.num_timesteps} steps: success={s["success"]:.3f} '
              f'collision={s["collision"]:.3f} spl={s["spl"]:.3f} (best {self.best:.3f})', flush=True)
        return True


def build_config(preset):
    p = PRESETS[preset]
    cfg = EnvConfig()
    return replace(cfg, scenario=replace(cfg.scenario, sources=tuple(p['sources']),
                                         obstacle_mode=p['obstacle_mode']))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Train the PPO local planner.')
    ap.add_argument('--preset', choices=list(PRESETS), default='gate')
    ap.add_argument('--arch', choices=['cnn', 'mlp'], default='cnn')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None, help='override the preset budget')
    ap.add_argument('--n-envs', type=int, default=16)
    ap.add_argument('--eval-every', type=int, default=250_000)
    ap.add_argument('--eval-episodes', type=int, default=200)
    ap.add_argument('--device', default='auto', help="'auto' uses CUDA when available (~40%% faster)")
    ap.add_argument('--runs-dir', default=str(RUNS_DIR))
    ap.add_argument('--name', default=None)
    args = ap.parse_args(argv)

    steps = args.steps or PRESETS[args.preset]['steps']
    name = args.name or f'{args.preset}_{args.arch}_s{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir = Path(args.runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    env_cfg = build_config(args.preset)
    config = {'preset': args.preset, 'arch': args.arch, 'seed': args.seed, 'steps': steps,
              'n_envs': args.n_envs, 'learning_rate': LEARNING_RATE, 'ppo': PPO_PARAMS,
              'eval_every': args.eval_every, 'eval_episodes': args.eval_episodes,
              'env': json.loads(json.dumps(asdict(env_cfg)))}
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    torch.set_num_threads(4)
    venv = SubprocVecEnv([lambda: NavEnv(env_cfg) for _ in range(args.n_envs)])
    venv.seed(args.seed)
    venv = VecNormalize(VecMonitor(venv), norm_obs=False, norm_reward=True,
                        gamma=PPO_PARAMS['gamma'])
    model = PPO('MlpPolicy', venv, learning_rate=lambda f: LEARNING_RATE * f,
                policy_kwargs=policy_kwargs(args.arch), seed=args.seed, device=args.device,
                tensorboard_log=str(run_dir / 'tb'), verbose=0, **PPO_PARAMS)
    callbacks = [EpisodeLogger(run_dir / 'episodes.csv'),
                 PeriodicEval(env_cfg, run_dir, args.eval_every, args.eval_episodes)]
    start = time.time()
    model.learn(total_timesteps=steps, callback=callbacks, tb_log_name='ppo')
    model.save(run_dir / 'last_model.zip')
    venv.save(str(run_dir / 'vecnormalize.pkl'))
    venv.close()
    minutes = (time.time() - start) / 60
    print(f'done: {steps} steps in {minutes:.1f} min ({steps / (minutes * 60):.0f} steps/s) -> {run_dir}')
    return run_dir


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verificar que pasa, y la suite completa**

Run: `./tools/ct python3 -m pytest -q test/test_train.py`
Expected: `1 passed` (~15 s)

Run: `./tools/ct python3 -m pytest -q test`
Expected: `69 passed`

Run: `./tools/ct python3 -m flake8 --max-line-length 110 martha_nav test`
Expected: sin salida

- [ ] **Step 5: Commit**

```bash
git add martha_nav/learning/train.py test/test_train.py
git commit -m "Add PPO training with per-episode logs and periodic evaluation" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 13: Puerta de convergencia

Sin código nuevo. Es la verificación de que el aprendizaje funciona en el caso más simple **antes** de añadir dificultad (spec §8).

**Files:**
- Modify: `docs/resultados.md`

- [ ] **Step 1: Entrenar la puerta con la CNN**

Run (unos minutos; en segundo plano si se quiere): `./tools/ct python3 -m martha_nav.learning.train --preset gate --arch cnn --seed 0 --name gate_cnn_s0`
Expected: líneas `[eval] ... success=...` a los 250k y 500k pasos, y al final `done: 500000 steps in ... min`.

- [ ] **Step 2: Evaluar con 200 episodios nuevos**

Run: `./tools/ct python3 -m martha_nav.learning.evaluate --model runs/gate_cnn_s0/best_model.zip --episodes 200 --condition clean --sources open_room`
Expected: `success ≥ 0.80`. El prototipo obtuvo **0.995 [IC95 0.972, 0.999]**, con 500k pasos en 11.4 min en CPU; ya a los 250k pasos daba 0.995.

**Si no llega a 0.80, no se sigue adelante.** Revisar, en este orden:
1. `episodes.csv`: ¿la tasa de `stalled` baja con el tiempo?
2. `r_progress` medio frente a `route_length`.
3. La recompensa en `tb/`.

- [ ] **Step 3: Registrar en la bitácora y commit**

Añadir a `docs/resultados.md` (con los valores obtenidos en el paso 2):

```markdown
## Puerta de convergencia (open_room, sin obstáculos)

Run `runs/gate_cnn_s0` (CNN, semilla 0, 500k pasos, <MIN> min).
Evaluación determinista, 200 episodios de semillas reservadas:
éxito <EXITO> [IC95 <LO>, <HI>], colisión <COL>, SPL <SPL>.
Evidencia: `runs/gate_cnn_s0/eval_clean_open_room.csv`.
```

```bash
git add docs/resultados.md
git commit -m "Record the convergence gate result" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

(Los runs están en `.gitignore`: son pesados. Se versiona la bitácora.)

---

### Tarea 14: Primer run completo con obstáculos

Sin código nuevo. Es la primera medición del problema real, y la línea base para decidir si hace falta ajustar la recompensa antes de E1.

**Files:**
- Modify: `docs/resultados.md`

- [ ] **Step 1: Entrenar**

Run (en segundo plano; estimado ~1 h con GPU, a ~1 500 pasos/s): `./tools/ct python3 -m martha_nav.learning.train --preset full --arch cnn --seed 0 --name full_cnn_s0`

- [ ] **Step 2: Evaluar en las tres condiciones de la spec**

Run: `./tools/ct python3 -m martha_nav.learning.evaluate --model runs/full_cnn_s0/best_model.zip --episodes 500 --condition clean`
Run: `./tools/ct python3 -m martha_nav.learning.evaluate --model runs/full_cnn_s0/best_model.zip --episodes 500 --condition obstacles`
Run: `./tools/ct python3 -m martha_nav.learning.evaluate --model runs/full_cnn_s0/best_model.zip --episodes 200 --condition obstacles --sources lab`

Expected: tres líneas de resumen y tres CSV junto al modelo. No hay umbral de aprobado: es una medición.

- [ ] **Step 3: Diagnóstico por términos**

Run:
```bash
./tools/ct python3 -c "
import pandas as pd
d = pd.read_csv('runs/full_cnn_s0/episodes.csv')
d['window'] = d.timesteps // 500_000
cols = ['r_progress', 'r_goal', 'r_collision', 'r_step']
print(d.groupby('window').outcome.value_counts(normalize=True).unstack().round(3))
print(d.groupby('window')[cols].mean().round(2))
print(d.groupby([d.n_obstacles > 0, 'window']).outcome.value_counts(normalize=True).unstack().round(3))
"
```
Expected: tablas por ventana de 500k pasos. Hay que registrar si el éxito con obstáculos sigue subiendo al final (y más pasos ayudarían) o si se aplanó (y toca revisar la recompensa antes de E1).

- [ ] **Step 4: Registrar y commit**

Añadir a `docs/resultados.md` una sección `## Primer run completo (full_cnn_s0)`, con las tres evaluaciones (éxito con IC95, colisión, SPL), la tabla por ventanas y una conclusión de una línea: ¿se estabilizó o sigue subiendo?

```bash
git add docs/resultados.md
git commit -m "Record the first full training run" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 15: Revisión bibliográfica de las plantillas de escenarios

Independiente del resto: puede hacerse en cualquier momento. Justifica en la tesis las familias de la spec §4.2.

**Files:**
- Create: `docs/escenarios-referencias.md`

- [ ] **Step 1: Buscar y leer**

Buscar los benchmarks y papers de navegación local con DRL que definen escenarios de entrenamiento o prueba. Como mínimo:
- Arena-Rosnav (Kästner et al.);
- DRL-VO (Xie & Dames);
- Long et al. 2018 (*Towards Optimally Decentralized Multi-Robot Collision Avoidance via DRL*);
- BARN (*Benchmark Autonomous Robot Navigation*, Perille et al. 2020).

Para cada uno anotar: qué tipos de escenario usan, sus rangos (anchos de paso, densidad de obstáculos) y si generan escenarios proceduralmente.

- [ ] **Step 2: Escribir el documento**

`docs/escenarios-referencias.md` con:
- una tabla `plantilla de martha_nav → papers que usan algo equivalente → rangos citados`, con una fila por cada uno de: `corridor`, `doorway`, `l_turn`, `furniture_walls`, `furniture_center`, `narrow_passage`;
- una sección de "diferencias", donde se anotan los rangos de la literatura que difieren de los de `scenarios.py`;
- la lista de referencias completas.

Si la revisión sugiere cambiar un rango, **no se cambia el código en esta tarea**: se anota como propuesta y se decide con el usuario.

- [ ] **Step 3: Commit**

```bash
git add docs/escenarios-referencias.md
git commit -m "Document the literature behind the scenario templates" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Después de este plan

Con las tareas 13 y 14 registradas se decide:

1. si la recompensa necesita ajustes antes de E1 (según el diagnóstico de la tarea 14);
2. el **plan 2**: E1 (CNN frente a MLP, 3 semillas por brazo), URDF, `sim.launch.py`, nodos ROS y `gazebo_eval.py` (spec §7 y §9).

El plan 2 incluye también la prueba *golden* de la spec §8: un `LaserScan` sintético convertido por el adaptador de `policy_node` debe dar la misma observación que los arrays del simulador. Esa prueba necesita el adaptador de ROS, así que no cabe en este plan.
