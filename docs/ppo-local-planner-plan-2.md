# Plan de implementación 2: ROS 2, Gazebo y la demo en `lab.world`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ejecutar la política ya entrenada (`runs/long_c_kl_s0/best_model.zip`) en Gazebo, sobre `lab.world`, con un planificador global A* y el mismo contrato de observación del simulador 2D; y medir la brecha 2D → Gazebo (experimento E2 de la spec).

**Architecture:** Dos nodos ROS 2 nuevos. `planner_node` hace A* sobre `/map` y publica `/plan`. `policy_node` calcula la zanahoria con `/plan` y `/scan`, arma la observación con la **misma** función del simulador 2D y publica `/cmd_vel`. Gazebo aporta el robot (`gazebo_ros_planar_move`), el LiDAR y el sensor de contacto. Un nodo auxiliar publica `map → odom` con la pose real del simulador y otro publica el mapa rasterizado del `.world`.

**Tech Stack:** ROS 2 Humble, Gazebo Classic 11, `gazebo_ros_pkgs`, rclpy, `tf2_ros`, Stable-Baselines3 (solo inferencia en CPU), pytest.

**Spec:** `docs/ppo-local-planner-design.md` (sección 7; E2 en la sección 9)

**Estado de verificación:** ⚠️ **a diferencia del plan 1, este plan NO se prototipó.** El plan 1 se escribió a partir de código ya ejecutado; aquí no, porque el experimento E1 ocupaba la máquina. El código de las tareas es correcto por construcción pero **no está probado**, así que cada tarea trae sus verificaciones y hay que tratarlas como reales: si algo no da lo esperado, hay que investigar, no forzar.

Datos que sí se verificaron en el contenedor antes de escribir el plan:
- existen `libgazebo_ros_planar_move.so`, `libgazebo_ros_ray_sensor.so`, `libgazebo_ros_bumper.so`, `libgazebo_ros_factory.so`, `libgazebo_ros_state.so` y `libgazebo_ros_init.so`;
- el paquete `martha` ya usa el sensor `ray` con `min_angle = -pi`, `max_angle = pi` y `update_rate = 10`, y su bumper publica `gazebo_msgs/ContactsState`.

## Restricciones globales

- Python y pytest **sin ROS** siguen ejecutándose con `./tools/ct <comando>`. Todo lo que importe `rclpy`, `nav_msgs`, `sensor_msgs`, `gazebo_msgs` o `tf2_ros` usa **`./tools/ct_ros <comando>`** (creado en la tarea 1), que primero hace `source /opt/ros/humble/setup.bash`.
- `martha_nav/sim2d/` **no importa nada de ROS**. La dependencia va en un solo sentido: `ros/` importa de `sim2d/`.
- La observación se construye **únicamente** con `martha_nav.sim2d.observation.build_observation`. Ningún nodo reimplementa esa lógica.
- Valores por defecto vigentes (spec actualizada): LiDAR `d/(d+1)` con `d ≤ 8 m`, 90 sectores, choque −20, `target_kl = 0.02`.
- Geometría: huella 0.56 × 0.41 m, LiDAR en `x = +0.2325 m`, inflado del planificador 0.40 m.
- Modelo de referencia: `runs/long_c_kl_s0/best_model.zip` (limpio 0.958, obstáculos 0.868, `lab` 2D 0.865).
- Gazebo se lanza **siempre** desde `sim.launch.py`; no se arranca `gzserver` a mano.
- Los mensajes de commit terminan en `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Estilo: `flake8 --max-line-length 110`.

## Mapa de archivos

| archivo | responsabilidad |
|---|---|
| `tools/ct_ros` | ejecutar un comando en el contenedor **con ROS cargado** |
| `urdf/martha.urdf.xacro` | robot mínimo: caja 0.56 × 0.41, LiDAR, bumper, `planar_move` |
| `martha_nav/ros/occupancy.py` | `Grid` ↔ `nav_msgs/OccupancyGrid` |
| `martha_nav/ros/scan_adapter.py` | `sensor_msgs/LaserScan` → `(ranges, angles)` en el marco del robot |
| `martha_nav/ros/planner_core.py` | lógica pura del planificador: cuándo replanificar, ruta, estado |
| `martha_nav/ros/policy_core.py` | lógica pura de la política: zanahoria, observación, acción, paradas de seguridad |
| `martha_nav/ros/planner_node.py` | nodo rclpy que envuelve `planner_core` |
| `martha_nav/ros/policy_node.py` | nodo rclpy que envuelve `policy_core` |
| `martha_nav/ros/map_publisher.py` | rasteriza un `.world` y publica `/map` (latched) |
| `martha_nav/ros/ground_truth_tf.py` | `map → odom` a partir de la pose real de Gazebo |
| `launch/sim.launch.py` | gzserver + robot + mapa + TF + los dos nodos (+ RViz opcional) |
| `martha_nav/ros/gazebo_eval.py` | evaluación por episodios en Gazebo: teletransporte, obstáculos, CSV |
| `test/test_occupancy.py`, `test_scan_adapter.py`, `test_planner_core.py`, `test_policy_core.py`, `test_urdf.py` | pruebas sin grafo ROS |

**Diseño clave:** toda la lógica vive en `*_core.py`, que son clases puras sin rclpy. Los nodos solo traducen mensajes. Así las pruebas no necesitan un grafo ROS ni Gazebo, igual que en el plan 1.

---

### Tarea 1: `tools/ct_ros` y el robot de Gazebo

**Files:**
- Create: `tools/ct_ros`, `urdf/martha.urdf.xacro`
- Test: `test/test_urdf.py`

**Interfaces:**
- Produces: un URDF con `base_link`, `lidar` (en `x = 0.2325`), el plugin `planar_move` (que consume `/cmd_vel` y publica `/odom` y el TF `odom → base_link`), el LiDAR (`/scan`, 360 rayos, 8 m, 10 Hz) y el bumper (`/bumper_states`).

- [ ] **Step 1: Crear `tools/ct_ros`**

```bash
cat > tools/ct_ros <<'EOF'
#!/usr/bin/env bash
# Run a command inside the container with ROS 2 Humble sourced.
exec docker exec -w /home/ros/ros2_ws/src/martha_nav ros2_humble \
  bash -lc 'source /opt/ros/humble/setup.bash && exec "$@"' _ "$@"
EOF
chmod +x tools/ct_ros
./tools/ct_ros python3 -c "import rclpy, nav_msgs, gazebo_msgs; print('ros ok')"
```
Expected: `ros ok`

- [ ] **Step 2: Escribir la prueba del URDF**

`test/test_urdf.py`:

```python
from pathlib import Path

import xacro

URDF = Path(__file__).resolve().parents[1] / 'urdf' / 'martha.urdf.xacro'


def robot():
    return xacro.process_file(str(URDF)).toprettyxml()


def test_xacro_expands():
    assert '<robot' in robot()


def test_footprint_and_lidar_pose():
    doc = robot()
    assert '0.56 0.41' in doc                 # chassis box, length x width
    assert '0.2325' in doc                    # LiDAR offset on x


def test_required_plugins_and_topics():
    doc = robot()
    for needle in ('libgazebo_ros_planar_move.so', 'libgazebo_ros_ray_sensor.so',
                   'libgazebo_ros_bumper.so', '~/out:=/scan', 'cmd_vel:=/cmd_vel',
                   '<frame_name>lidar</frame_name>'):
        assert needle in doc, needle
    assert doc.count('<sensor ') == 2          # ray + contact
```

- [ ] **Step 3: Verificar que falla**

Run: `./tools/ct_ros python3 -m pytest -q test/test_urdf.py`
Expected: FAIL (el archivo `urdf/martha.urdf.xacro` no existe)

- [ ] **Step 4: Escribir el URDF**

`urdf/martha.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="martha">
  <!-- Minimal Martha for Gazebo: the wheels are not simulated; planar_move
       consumes /cmd_vel directly. Geometry matches sim2d/geometry.py. -->
  <xacro:property name="length" value="0.56"/>
  <xacro:property name="width" value="0.41"/>
  <xacro:property name="height" value="0.25"/>
  <xacro:property name="lidar_x" value="0.2325"/>
  <xacro:arg name="lidar_samples" default="360"/>
  <xacro:arg name="lidar_visualize" default="false"/>

  <link name="base_footprint"/>

  <link name="base_link">
    <visual>
      <origin xyz="0 0 ${height/2}"/>
      <geometry><box size="0.56 0.41 ${height}"/></geometry>
      <material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 ${height/2}"/>
      <geometry><box size="0.56 0.41 ${height}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 ${height/2}"/>
      <mass value="12.0"/>
      <inertia ixx="0.25" ixy="0" ixz="0" iyy="0.40" iyz="0" izz="0.50"/>
    </inertial>
  </link>

  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="base_link"/>
    <origin xyz="0 0 0.05"/>
  </joint>

  <link name="lidar">
    <visual>
      <geometry><cylinder radius="0.0475" length="0.04"/></geometry>
    </visual>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar"/>
    <origin xyz="${lidar_x} 0 ${height + 0.02}"/>
  </joint>

  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
    <sensor name="contact_sensor" type="contact">
      <always_on>true</always_on>
      <update_rate>50</update_rate>
      <contact>
        <collision>base_footprint_fixed_joint_lump__base_link_collision</collision>
      </contact>
      <plugin name="gazebo_ros_bumper" filename="libgazebo_ros_bumper.so">
        <ros>
          <remapping>bumper_states:=/bumper_states</remapping>
        </ros>
        <frame_name>base_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="lidar">
    <sensor name="lidar_sensor" type="ray">
      <always_on>true</always_on>
      <visualize>$(arg lidar_visualize)</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>$(arg lidar_samples)</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.15</min>
          <max>8.0</max>
          <resolution>0.01</resolution>
        </range>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </ray>
      <plugin name="gazebo_ros_lidar" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <remapping>~/out:=/scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo>
    <plugin name="planar_move" filename="libgazebo_ros_planar_move.so">
      <ros>
        <remapping>cmd_vel:=/cmd_vel</remapping>
        <remapping>odom:=/odom</remapping>
      </ros>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_footprint</robot_base_frame>
      <odometry_rate>50.0</odometry_rate>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
    </plugin>
  </gazebo>
</robot>
```

- [ ] **Step 5: Verificar que pasa**

Run: `./tools/ct_ros python3 -m pytest -q test/test_urdf.py`
Expected: `3 passed`

**Si falla el nombre de la colisión del bumper:** Gazebo renombra las colisiones al convertir de URDF a SDF. Para obtener el nombre real:

```bash
./tools/ct_ros bash -c 'xacro urdf/martha.urdf.xacro > /tmp/m.urdf && gz sdf -p /tmp/m.urdf | grep -n "<collision name"'
```
Se usa el nombre que imprima y se actualiza el URDF y la prueba.

- [ ] **Step 6: Commit**

```bash
git add tools/ct_ros urdf/martha.urdf.xacro test/test_urdf.py
git commit -m "Add the minimal Gazebo robot and the ROS-aware container helper" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 2: Conversión de mapas y publicador de `/map`

**Files:**
- Create: `martha_nav/ros/__init__.py` (vacío), `martha_nav/ros/occupancy.py`, `martha_nav/ros/map_publisher.py`
- Test: `test/test_occupancy.py`

**Interfaces:**
- Consumes: `Grid`, `rasterize_world` (plan 1).
- Produces:
  - `grid_to_msg(grid, frame_id='map', stamp=None) -> nav_msgs/OccupancyGrid`;
  - `msg_to_grid(msg, occupied_threshold=50) -> Grid` (lo desconocido, −1, cuenta como ocupado);
  - nodo `map_publisher`, parámetro `world` (por defecto `lab`), que publica `/map` latched.

- [ ] **Step 1: Escribir las pruebas**

`test/test_occupancy.py`:

```python
import numpy as np

from martha_nav.ros.occupancy import grid_to_msg, msg_to_grid
from martha_nav.sim2d.geometry import draw_box, empty_grid


def sample_grid():
    g = empty_grid(2.0, 1.0, origin=(-1.0, -0.5))
    draw_box(g, 0.5, 0.0, 0.2, 0.2)
    return g


def test_grid_to_msg_layout():
    g = sample_grid()
    msg = grid_to_msg(g)
    assert msg.header.frame_id == 'map'
    assert msg.info.resolution == 0.05
    assert (msg.info.width, msg.info.height) == (g.shape[1], g.shape[0])
    assert (msg.info.origin.position.x, msg.info.origin.position.y) == (-1.0, -0.5)
    assert len(msg.data) == g.occ.size
    assert set(msg.data) == {0, 100}


def test_round_trip_preserves_cells_and_origin():
    g = sample_grid()
    back = msg_to_grid(grid_to_msg(g))
    assert np.array_equal(back.occ, g.occ)
    assert back.origin == g.origin and back.resolution == g.resolution


def test_unknown_cells_count_as_occupied():
    g = sample_grid()
    msg = grid_to_msg(g)
    msg.data = [-1] * len(msg.data)
    assert msg_to_grid(msg).occ.all()
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct_ros python3 -m pytest -q test/test_occupancy.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.ros'`

- [ ] **Step 3: Implementar**

`martha_nav/ros/occupancy.py`:

```python
"""nav_msgs/OccupancyGrid <-> sim2d.geometry.Grid."""
import numpy as np
from nav_msgs.msg import OccupancyGrid

from martha_nav.sim2d.geometry import Grid


def grid_to_msg(grid, frame_id='map', stamp=None):
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution)
    msg.info.height, msg.info.width = (int(n) for n in grid.occ.shape)
    msg.info.origin.position.x = float(grid.origin[0])
    msg.info.origin.position.y = float(grid.origin[1])
    msg.info.origin.orientation.w = 1.0
    msg.data = np.where(grid.occ, 100, 0).astype(np.int8).ravel().tolist()
    return msg


def msg_to_grid(msg, occupied_threshold=50):
    """Unknown (-1) counts as occupied: the planner must not route through it."""
    data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    occ = (data >= occupied_threshold) | (data < 0)
    origin = (float(msg.info.origin.position.x), float(msg.info.origin.position.y))
    return Grid(np.ascontiguousarray(occ), origin, float(msg.info.resolution))
```

`martha_nav/ros/map_publisher.py`:

```python
"""Publish a rasterised .world as a latched /map."""
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from martha_nav.ros.occupancy import grid_to_msg
from martha_nav.sim2d.worlds import rasterize_world

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        world = self.declare_parameter('world', 'lab').value
        frame = self.declare_parameter('frame_id', 'map').value
        msg = grid_to_msg(rasterize_world(world), frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.pub.publish(msg)
        self.get_logger().info(
            f'published {world}: {msg.info.width}x{msg.info.height} cells @ {msg.info.resolution} m')


def main():
    rclpy.init()
    node = MapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct_ros python3 -m pytest -q test/test_occupancy.py`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros/__init__.py martha_nav/ros/occupancy.py martha_nav/ros/map_publisher.py test/test_occupancy.py
git commit -m "Convert grids to OccupancyGrid and publish the rasterised world" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 3: Adaptador del LiDAR y prueba *golden*

Es la prueba que la spec (§8) pide y que el plan 1 no podía hacer: que un `LaserScan` real produzca **exactamente** la misma observación que los arrays del simulador.

**Files:**
- Create: `martha_nav/ros/scan_adapter.py`
- Test: `test/test_scan_adapter.py`

**Interfaces:**
- Produces: `scan_to_arrays(msg, lidar_yaw=0.0) -> (ranges, angles)`, con los ángulos **relativos al frente del robot**, listos para `build_observation`.

- [ ] **Step 1: Escribir las pruebas**

`test/test_scan_adapter.py`:

```python
import numpy as np
from sensor_msgs.msg import LaserScan

from martha_nav.ros.scan_adapter import scan_to_arrays
from martha_nav.sim2d.geometry import draw_box, empty_grid, raycast
from martha_nav.sim2d.observation import LIDAR_MAX, build_observation


def make_scan(ranges, angle_min=-np.pi, angle_max=np.pi):
    msg = LaserScan()
    msg.angle_min = float(angle_min)
    msg.angle_max = float(angle_max)
    msg.angle_increment = float((angle_max - angle_min) / len(ranges))
    msg.range_min, msg.range_max = 0.15, LIDAR_MAX
    msg.ranges = [float(r) for r in ranges]
    return msg


def test_angles_follow_the_message_fields():
    msg = make_scan([1.0, 2.0, 3.0, 4.0])
    ranges, angles = scan_to_arrays(msg)
    assert np.allclose(ranges, [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(angles, msg.angle_min + np.arange(4) * msg.angle_increment)


def test_lidar_yaw_rotates_the_angles():
    msg = make_scan([1.0, 2.0])
    _, angles = scan_to_arrays(msg, lidar_yaw=np.pi / 2)
    assert np.allclose(angles, [-np.pi / 2, np.pi / 2])


def test_golden_same_observation_from_a_laserscan_and_from_sim_arrays():
    """The ROS path and the 2D simulator must produce the identical vector."""
    grid = empty_grid(10.0, 10.0)
    draw_box(grid, 7.0, 5.0, 0.4, 0.4)
    draw_box(grid, 5.0, 8.0, 3.0, 0.2)
    sim_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    sim_ranges = raycast(grid, 5.0, 5.0, sim_angles, LIDAR_MAX)
    sim_obs = build_observation(sim_ranges, sim_angles, (0.2, -0.1), (1.4, 0.3), (0.5, 0.0))

    msg = make_scan(np.where(sim_ranges >= LIDAR_MAX, np.inf, sim_ranges))
    ros_ranges, ros_angles = scan_to_arrays(msg)
    ros_obs = build_observation(ros_ranges, ros_angles, (0.2, -0.1), (1.4, 0.3), (0.5, 0.0))

    assert np.array_equal(sim_obs, ros_obs)
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct_ros python3 -m pytest -q test/test_scan_adapter.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.ros.scan_adapter'`

- [ ] **Step 3: Implementar**

`martha_nav/ros/scan_adapter.py`:

```python
"""sensor_msgs/LaserScan -> the arrays build_observation expects."""
import numpy as np


def scan_to_arrays(msg, lidar_yaw=0.0):
    """Ranges and beam angles relative to the robot's forward axis.

    lidar_yaw is the LiDAR frame's yaw in base_link (0.0 when it points forward,
    pi when the sensor is mounted backwards). Invalid readings are left as they
    come: build_observation already treats inf/NaN/<=0 as max range.
    """
    ranges = np.asarray(msg.ranges, dtype=float)
    angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment + lidar_yaw
    return ranges, angles
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct_ros python3 -m pytest -q test/test_scan_adapter.py`
Expected: `3 passed`. **La tercera es la prueba golden de la spec §8.**

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros/scan_adapter.py test/test_scan_adapter.py
git commit -m "Adapt LaserScan to the observation contract, with the golden test" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 4: Lógica del planificador (`planner_core`)

**Files:**
- Create: `martha_nav/ros/planner_core.py`
- Test: `test/test_planner_core.py`

**Interfaces:**
- Consumes: `PlanningGrid`, `Path`, `INFLATION` (plan 1); `msg_to_grid` (tarea 2).
- Produces:
  - `PlannerCore(inflation=0.40, replan_distance=1.0, goal_tolerance=0.3)`, con:
    - `.set_map(grid)`;
    - `.set_goal(x, y)`;
    - `.update(x, y) -> status`, devuelve `'idle' | 'active' | 'succeeded' | 'failed'`;
    - `.path` (un `Path` o `None`).
  - Replanifica cuando la meta cambia o el robot se aleja más de `replan_distance` de la ruta.

- [ ] **Step 1: Escribir las pruebas**

`test/test_planner_core.py`:

```python
import numpy as np

from martha_nav.ros.planner_core import PlannerCore
from martha_nav.sim2d.geometry import draw_box, empty_grid


def room(blocked=False):
    g = empty_grid(8.0, 4.0)
    if blocked:
        draw_box(g, 4.0, 2.0, 0.2, 4.0)
    return g


def test_idle_without_map_or_goal():
    core = PlannerCore()
    assert core.update(1.0, 2.0) == 'idle'
    core.set_map(room())
    assert core.update(1.0, 2.0) == 'idle'


def test_plans_once_the_goal_is_set():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    assert core.update(1.0, 2.0) == 'active'
    assert core.path is not None and core.path.length > 5.5


def test_reports_failure_when_there_is_no_route():
    core = PlannerCore()
    core.set_map(room(blocked=True))
    core.set_goal(7.0, 2.0)
    assert core.update(1.0, 2.0) == 'failed'
    assert core.path is None


def test_succeeds_inside_the_tolerance():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    assert core.update(6.9, 2.05) == 'succeeded'


def test_replans_only_when_far_from_the_route():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    first = core.path
    core.update(2.0, 2.05)
    assert core.path is first                      # still on the route
    core.update(2.0, 3.4)
    assert core.path is not first                  # 1.4 m away: replanned


def test_a_new_goal_replans():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    first = core.path
    core.set_goal(7.0, 3.0)
    core.update(1.0, 2.0)
    assert core.path is not first
    assert np.allclose(core.path.points[-1], [7.0, 3.0])
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_planner_core.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.ros.planner_core'`

(Esta prueba no necesita ROS: `./tools/ct` basta.)

- [ ] **Step 3: Implementar**

`martha_nav/ros/planner_core.py`:

```python
"""Global planner logic, without ROS: same A* as the 2D simulator."""
import numpy as np

from martha_nav.sim2d.planner import INFLATION, PlanningGrid


class PlannerCore:
    def __init__(self, inflation=INFLATION, replan_distance=1.0, goal_tolerance=0.3):
        self.inflation = inflation
        self.replan_distance = replan_distance
        self.goal_tolerance = goal_tolerance
        self.planning_grid = None
        self.goal = None
        self.path = None
        self._needs_plan = False

    def set_map(self, grid):
        self.planning_grid = PlanningGrid(grid, self.inflation)
        self._needs_plan = self.goal is not None

    def set_goal(self, x, y):
        self.goal = np.array([float(x), float(y)])
        self.path = None
        self._needs_plan = True

    def update(self, x, y):
        """Advance the state machine for the robot pose; returns the status."""
        if self.planning_grid is None or self.goal is None:
            return 'idle'
        if np.hypot(x - self.goal[0], y - self.goal[1]) <= self.goal_tolerance:
            return 'succeeded'
        if self.path is not None:
            s = self.path.project(x, y)
            point = self.path.point_at(s)
            if np.hypot(x - point[0], y - point[1]) > self.replan_distance:
                self._needs_plan = True
        if self._needs_plan or self.path is None:
            self.path = self.planning_grid.route((x, y), self.goal)
            self._needs_plan = False
            if self.path is None:
                return 'failed'
        return 'active'
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_planner_core.py`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros/planner_core.py test/test_planner_core.py
git commit -m "Add the global planner state machine" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 5: Lógica de la política (`policy_core`)

**Files:**
- Create: `martha_nav/ros/policy_core.py`
- Test: `test/test_policy_core.py`

**Interfaces:**
- Consumes: `carrot`, `Path` (plan 1); `build_observation`, `action_to_cmd`, `LIDAR_MAX` (plan 1).
- Produces:
  - `PolicyCore(model, lookahead=1.5, carrot_clearance=0.4, lidar_encoding='inverse')`, con:
    - `.reset()`;
    - `.compute(path, pose, ranges, angles, velocity) -> (v, w, info)`.
  - `footprint_blocked(ranges, angles, margin=0.05) -> bool`: comprueba si algún punto del scan cae dentro de la huella inflada, **medido desde el LiDAR**.
  - `model` solo necesita `predict(obs, deterministic=True)`.

- [ ] **Step 1: Escribir las pruebas**

`test/test_policy_core.py`:

```python
import numpy as np

from martha_nav.ros.policy_core import PolicyCore, footprint_blocked
from martha_nav.sim2d.observation import LIDAR_MAX, V_MAX, W_MAX
from martha_nav.sim2d.planner import Path


class FakeModel:
    """Records the observation it was given and returns a fixed action."""

    def __init__(self, action=(1.0, 0.0)):
        self.action = np.array(action)
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = np.asarray(obs)
        return self.action, None


def straight_path():
    return Path([[0.0, 0.0], [10.0, 0.0]])


def clear_scan(n=360):
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return np.full(n, LIDAR_MAX), angles


def test_action_is_mapped_to_velocities():
    core = PolicyCore(FakeModel((1.0, -1.0)))
    ranges, angles = clear_scan()
    v, w, _ = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert (v, w) == (V_MAX, -W_MAX)


def test_observation_matches_the_contract():
    model = FakeModel()
    core = PolicyCore(model, lookahead=1.5)
    ranges, angles = clear_scan()
    core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    obs = model.last_obs
    assert obs.shape == (96,)
    assert np.isclose(obs[90], 1.5 / 3.0)        # carrot 1.5 m ahead
    assert np.isclose(obs[91], 0.0)              # straight ahead


def test_previous_action_is_fed_back_and_cleared_on_reset():
    model = FakeModel((0.5, 0.25))
    core = PolicyCore(model)
    ranges, angles = clear_scan()
    core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    core.compute(straight_path(), (0.1, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert np.allclose(model.last_obs[94:], [0.5, 0.25])
    core.reset()
    core.compute(straight_path(), (0.2, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert np.allclose(model.last_obs[94:], [0.0, 0.0])


def test_an_obstacle_inside_the_footprint_stops_the_robot():
    core = PolicyCore(FakeModel())
    ranges, angles = clear_scan()
    ranges[0] = 0.04                              # right in front of the LiDAR
    v, w, info = core.compute(straight_path(), (0.0, 0.0, 0.0), ranges, angles, (0.0, 0.0))
    assert (v, w) == (0.0, 0.0) and info['blocked']


def test_footprint_check_uses_the_lidar_offset():
    angles = np.array([0.0, np.pi])
    # Forward: the footprint ends 0.28 - 0.2325 = 0.0475 m ahead of the LiDAR.
    assert footprint_blocked(np.array([0.04, 8.0]), angles)
    assert not footprint_blocked(np.array([0.5, 8.0]), angles)
    # Backwards: the footprint reaches 0.28 + 0.2325 = 0.5125 m behind it.
    assert footprint_blocked(np.array([8.0, 0.40]), angles)
```

- [ ] **Step 2: Verificar que fallan**

Run: `./tools/ct python3 -m pytest -q test/test_policy_core.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.ros.policy_core'`

- [ ] **Step 3: Implementar**

`martha_nav/ros/policy_core.py`:

```python
"""Local-planner logic, without ROS: carrot, observation, action and safety stop."""
import numpy as np

from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH
from martha_nav.sim2d.observation import LIDAR_MAX, action_to_cmd, build_observation
from martha_nav.sim2d.planner import carrot


def footprint_blocked(ranges, angles, margin=0.05):
    """True when a scan point falls inside the robot rectangle plus a margin.

    Ranges are measured from the LiDAR, which sits LIDAR_OFFSET_X ahead of the
    footprint centre, so the rectangle is shifted by that amount.
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    valid = np.isfinite(ranges) & (ranges > 0)
    x = ranges[valid] * np.cos(angles[valid]) + LIDAR_OFFSET_X
    y = ranges[valid] * np.sin(angles[valid])
    inside = (np.abs(x) <= ROBOT_LENGTH / 2 + margin) & (np.abs(y) <= ROBOT_WIDTH / 2 + margin)
    return bool(inside.any())


class PolicyCore:
    def __init__(self, model, lookahead=1.5, carrot_clearance=0.4, lidar_encoding='inverse'):
        self.model = model
        self.lookahead = lookahead
        self.carrot_clearance = carrot_clearance
        self.lidar_encoding = lidar_encoding
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(2)
        self.s = 0.0

    def compute(self, path, pose, ranges, angles, velocity):
        """One control step. pose is (x, y, yaw) in the map frame.

        Returns (v, w, info); info carries 'carrot', 's' and 'blocked'.
        """
        x, y, yaw = pose
        if footprint_blocked(ranges, angles):
            self.prev_action = np.zeros(2)
            return 0.0, 0.0, {'blocked': True, 's': self.s, 'carrot': None}

        self.s = path.project(x, y, s_hint=self.s)
        points = self._scan_points(ranges, angles, pose)
        point, _ = carrot(path, self.s, self.lookahead, points, self.carrot_clearance)
        dx, dy = point[0] - x, point[1] - y
        rel = (np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy)
        obs = build_observation(ranges, angles, velocity, rel, self.prev_action,
                                self.lidar_encoding)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        self.prev_action = action
        v, w = action_to_cmd(action)
        return v, w, {'blocked': False, 's': self.s, 'carrot': point}

    def _scan_points(self, ranges, angles, pose):
        """Scan hits in map coordinates, for the carrot's obstacle skipping."""
        x, y, yaw = pose
        ranges = np.asarray(ranges, dtype=float)
        angles = np.asarray(angles, dtype=float)
        hit = np.isfinite(ranges) & (ranges > 0) & (ranges < LIDAR_MAX)
        if not hit.any():
            return np.empty((0, 2))
        ox = x + LIDAR_OFFSET_X * np.cos(yaw)
        oy = y + LIDAR_OFFSET_X * np.sin(yaw)
        world = yaw + angles[hit]
        return np.stack([ox + ranges[hit] * np.cos(world), oy + ranges[hit] * np.sin(world)], axis=1)
```

- [ ] **Step 4: Verificar que pasan**

Run: `./tools/ct python3 -m pytest -q test/test_policy_core.py`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros/policy_core.py test/test_policy_core.py
git commit -m "Add the local-planner logic with the footprint safety stop" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 6: Los nodos ROS y el launch

**Files:**
- Create: `martha_nav/ros/planner_node.py`, `martha_nav/ros/policy_node.py`, `martha_nav/ros/ground_truth_tf.py`, `launch/sim.launch.py`
- Modify: `setup.py` (entry points y `data_files` de `launch/` y `urdf/`)

**Interfaces:**
- Produces los ejecutables `map_publisher`, `planner_node`, `policy_node` y `ground_truth_tf`, y el launch `sim.launch.py` con los argumentos `world`, `gui`, `checkpoint` y `rviz`.
- Tópicos: `planner_node` publica `/plan` (`nav_msgs/Path`) y `/nav_status` (`std_msgs/String`); `policy_node` publica `/cmd_vel` y `/carrot` (`geometry_msgs/PointStamped`, para RViz).

- [ ] **Step 1: Escribir `planner_node.py`**

```python
"""A* over /map -> /plan, driven by /goal_pose and the map -> base_link transform."""
import rclpy
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from martha_nav.ros.occupancy import msg_to_grid
from martha_nav.ros.planner_core import PlannerCore

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.core = PlannerCore(
            inflation=self.declare_parameter('inflation', 0.40).value,
            replan_distance=self.declare_parameter('replan_distance', 1.0).value,
            goal_tolerance=self.declare_parameter('goal_tolerance', 0.3).value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, LATCHED)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.plan_pub = self.create_publisher(PathMsg, '/plan', LATCHED)
        self.status_pub = self.create_publisher(String, '/nav_status', LATCHED)
        self.status = None
        self.create_timer(0.2, self.tick)

    def on_map(self, msg):
        self.core.set_map(msg_to_grid(msg))
        self.get_logger().info('map received')

    def on_goal(self, msg):
        self.core.set_goal(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def pose(self):
        try:
            tf = self.buffer.lookup_transform(self.map_frame, self.base_frame,
                                              rclpy.time.Time(), Duration(seconds=0.2))
        except Exception:                                   # noqa: BLE001 - TF errors are expected
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def tick(self):
        pose = self.pose()
        if pose is None:
            return
        previous, path = self.status, self.core.path
        self.status = self.core.update(*pose)
        if self.status != previous:
            self.status_pub.publish(String(data=self.status))
            self.get_logger().info(f'status: {self.status}')
        if self.core.path is not None and self.core.path is not path:
            self.plan_pub.publish(self.to_msg(self.core.path))

    def to_msg(self, path):
        msg = PathMsg()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path.points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg


def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Escribir `policy_node.py`**

```python
"""/plan + /scan + /odom -> /cmd_vel at 10 Hz, using the trained policy."""
import math

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from stable_baselines3 import PPO
from tf2_ros import Buffer, TransformListener

from martha_nav.ros.policy_core import PolicyCore
from martha_nav.ros.scan_adapter import scan_to_arrays
from martha_nav.sim2d.planner import Path

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
STALE = 0.3      # s; older sensor data stops the robot


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        checkpoint = self.declare_parameter('checkpoint', '').value
        if not checkpoint:
            raise RuntimeError('parameter "checkpoint" is required')
        torch.set_num_threads(1)
        model = PPO.load(checkpoint, device='cpu')
        self.core = PolicyCore(model,
                               lookahead=self.declare_parameter('lookahead', 1.5).value,
                               lidar_encoding=self.declare_parameter('lidar_encoding',
                                                                     'inverse').value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.scan = self.scan_time = None
        self.velocity = (0.0, 0.0)
        self.odom_time = None
        self.path = self.path_time = None
        self.status = 'idle'
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PathMsg, '/plan', self.on_plan, LATCHED)
        self.create_subscription(String, '/nav_status', self.on_status, LATCHED)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.carrot_pub = self.create_publisher(PointStamped, '/carrot', 10)
        self.create_timer(0.1, self.tick)

    # ---- inputs ----
    def on_scan(self, msg):
        self.scan = scan_to_arrays(msg)
        self.scan_time = self.get_clock().now()

    def on_odom(self, msg):
        self.velocity = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        self.odom_time = self.get_clock().now()

    def on_plan(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path = Path(np.array(points)) if len(points) >= 2 else None
        self.path_time = self.get_clock().now()
        self.core.reset()

    def on_status(self, msg):
        self.status = msg.data

    def pose(self):
        try:
            tf = self.buffer.lookup_transform(self.map_frame, self.base_frame,
                                              rclpy.time.Time(), Duration(seconds=0.2))
        except Exception:                                   # noqa: BLE001
            return None
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        return tf.transform.translation.x, tf.transform.translation.y, yaw

    # ---- control ----
    def stop(self, reason):
        self.cmd_pub.publish(Twist())
        self.get_logger().warning(reason, throttle_duration_sec=2.0)

    def fresh(self, stamp, limit):
        return stamp is not None and (self.get_clock().now() - stamp) < Duration(seconds=limit)

    def tick(self):
        if self.status != 'active' or self.path is None:
            self.cmd_pub.publish(Twist())
            return
        if not self.fresh(self.scan_time, STALE) or not self.fresh(self.odom_time, STALE):
            return self.stop('stale sensor data')
        pose = self.pose()
        if pose is None:
            return self.stop('no map -> base transform')
        ranges, angles = self.scan
        v, w, info = self.core.compute(self.path, pose, ranges, angles, self.velocity)
        if info['blocked']:
            self.stop('obstacle inside the footprint')
            return
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
        if info['carrot'] is not None:
            point = PointStamped()
            point.header.frame_id = self.map_frame
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x, point.point.y = float(info['carrot'][0]), float(info['carrot'][1])
            self.carrot_pub.publish(point)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Escribir `ground_truth_tf.py`**

```python
"""Publish map -> odom from Gazebo's true pose, so map -> base_link is exact.

Evaluation in Gazebo measures the policy, not the localisation (spec 7.2).
"""
import math

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


class GroundTruthTf(Node):
    def __init__(self):
        super().__init__('ground_truth_tf')
        self.model = self.declare_parameter('model_name', 'martha').value
        self.truth = None      # (x, y, yaw) of base_footprint in map
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.on_states, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)

    def on_states(self, msg):
        if self.model not in msg.name:
            return
        pose = msg.pose[msg.name.index(self.model)]
        self.truth = (pose.position.x, pose.position.y, yaw_of(pose.orientation))

    def on_odom(self, msg):
        """map -> odom = (map -> base) * inverse(odom -> base), in the plane."""
        if self.truth is None:
            return
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        oyaw = yaw_of(msg.pose.pose.orientation)
        tx, ty, tyaw = self.truth
        dyaw = tyaw - oyaw
        c, s = math.cos(dyaw), math.sin(dyaw)
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'odom'
        tf.transform.translation.x = tx - (c * ox - s * oy)
        tf.transform.translation.y = ty - (s * ox + c * oy)
        tf.transform.rotation.z = math.sin(dyaw / 2)
        tf.transform.rotation.w = math.cos(dyaw / 2)
        self.broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = GroundTruthTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Escribir `launch/sim.launch.py`**

```python
"""ros2 launch martha_nav sim.launch.py checkpoint:=/abs/path/best_model.zip

Gazebo + Martha + the rasterised map + ground-truth TF + the two navigation nodes.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('martha_nav')
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    checkpoint = LaunchConfiguration('checkpoint')
    rviz = LaunchConfiguration('rviz')
    urdf = ParameterValue(
        Command(['xacro ', PathJoinSubstitution([share, 'urdf', 'martha.urdf.xacro'])]),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='lab'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),

        ExecuteProcess(
            cmd=['gzserver', PathJoinSubstitution([share, 'worlds', [world, '.world']]),
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so',
                 '-s', 'libgazebo_ros_state.so'],
            output='screen'),
        ExecuteProcess(cmd=['gzclient'], condition=IfCondition(gui), output='screen'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': urdf, 'use_sim_time': True}], output='screen'),
        Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
             arguments=['-topic', 'robot_description', '-entity', 'martha',
                        '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'), '-z', '0.05']),

        Node(package='martha_nav', executable='map_publisher', output='screen',
             parameters=[{'world': world, 'use_sim_time': True}]),
        Node(package='martha_nav', executable='ground_truth_tf', output='screen',
             parameters=[{'use_sim_time': True}]),
        Node(package='martha_nav', executable='planner_node', output='screen',
             parameters=[{'use_sim_time': True}]),
        Node(package='martha_nav', executable='policy_node', output='screen',
             parameters=[{'checkpoint': checkpoint, 'use_sim_time': True}]),
        Node(package='rviz2', executable='rviz2', condition=IfCondition(rviz), output='screen',
             parameters=[{'use_sim_time': True}]),
    ])
```

- [ ] **Step 5: Declarar los ejecutables y los datos del paquete**

En `setup.py`, añadir a `data_files`:

```python
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
```

y reemplazar `entry_points`:

```python
    entry_points={
        'console_scripts': [
            'map_publisher = martha_nav.ros.map_publisher:main',
            'ground_truth_tf = martha_nav.ros.ground_truth_tf:main',
            'planner_node = martha_nav.ros.planner_node:main',
            'policy_node = martha_nav.ros.policy_node:main',
        ],
    },
```

Añadir también a `package.xml`, antes de `<export>`:

```xml
  <exec_depend>rclpy</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>gazebo_msgs</exec_depend>
  <exec_depend>tf2_ros</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>xacro</exec_depend>
```

- [ ] **Step 6: Compilar y verificar que los nodos arrancan**

```bash
./tools/ct_ros bash -c 'cd /home/ros/ros2_ws && colcon build --packages-select martha_nav --symlink-install'
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 pkg executables martha_nav'
```
Expected: los cuatro ejecutables listados.

- [ ] **Step 7: Prueba de humo en Gazebo (verificación manual)**

Terminal 1:
```bash
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 launch martha_nav sim.launch.py world:=lab gui:=false checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/long_c_kl_s0/best_model.zip x:=-2.0 y:=-3.0'
```

Terminal 2, comprobaciones en orden:
```bash
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 topic hz /scan'          # ~10 Hz
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 topic echo --once /map --field info'
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 run tf2_ros tf2_echo map base_footprint'
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: 2.0, y: 2.0}, orientation: {w: 1.0}}}"'
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 topic echo /nav_status'   # active -> succeeded
```
Expected: `/scan` a ~10 Hz, el mapa con el tamaño del laboratorio, `map → base_footprint` coincidiendo con la posición de spawn, y Martha llegando a la meta con `/nav_status` en `succeeded`.

**Puntos donde esto puede fallar, y qué mirar:**
- `/scan` vacío o sin publicar → el nombre del sensor o el remapeo del plugin ray.
- TF `map → odom` ausente → el nombre del modelo en `/gazebo/model_states` (debe ser `martha`).
- El robot no se mueve → comprobar `ros2 topic echo /cmd_vel`; si hay comandos pero no movimiento, revisar `robot_base_frame` en `planar_move`.
- Da vueltas o choca enseguida → comparar la observación del nodo con la del simulador 2D antes de tocar nada más (`/carrot` en RViz ayuda).

- [ ] **Step 8: Commit**

```bash
git add martha_nav/ros/planner_node.py martha_nav/ros/policy_node.py martha_nav/ros/ground_truth_tf.py launch/sim.launch.py setup.py package.xml
git commit -m "Add the ROS nodes and the Gazebo launch" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Tarea 7: Evaluación por episodios en Gazebo (E2)

**Files:**
- Create: `martha_nav/ros/gazebo_eval.py`
- Modify: `setup.py` (un ejecutable más), `docs/resultados.md`

**Interfaces:**
- `gazebo_eval` es un nodo que corre **contra un `sim.launch.py` ya lanzado**. Por episodio:
  1. genera el escenario con `scenarios.generate(seed, cfg)` usando la fuente `lab`;
  2. teletransporta a Martha al inicio con `/gazebo/set_entity_state`;
  3. genera los obstáculos sorpresa con `/spawn_entity` y los borra al final con `/delete_entity`;
  4. publica la meta en `/goal_pose` y espera `succeeded`, contacto (`/bumper_states`) o timeout;
  5. escribe una fila en el CSV con las mismas columnas que `learning/evaluate.py`.

- [ ] **Step 1: Implementar**

`martha_nav/ros/gazebo_eval.py`:

```python
"""Episodic evaluation in Gazebo, against a running sim.launch.py.

ros2 run martha_nav gazebo_eval --ros-args -p episodes:=100 -p out:=/tmp/eval_gazebo.csv
"""
import csv
import math
import time

import numpy as np
import rclpy
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from martha_nav.learning.evaluate import eval_seeds
from martha_nav.sim2d.scenarios import ScenarioConfig, generate

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)

BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static><link name="link">
<collision name="c"><geometry><box><size>{sx} {sy} 0.6</size></box></geometry></collision>
<visual name="v"><geometry><box><size>{sx} {sy} 0.6</size></box></geometry></visual>
</link></model></sdf>"""

CYLINDER_SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static><link name="link">
<collision name="c"><geometry><cylinder><radius>{r}</radius><length>0.6</length></cylinder></geometry></collision>
<visual name="v"><geometry><cylinder><radius>{r}</radius><length>0.6</length></cylinder></geometry></visual>
</link></model></sdf>"""


class GazeboEval(Node):
    def __init__(self):
        super().__init__('gazebo_eval')
        self.episodes = self.declare_parameter('episodes', 100).value
        self.out = self.declare_parameter('out', '/tmp/eval_gazebo.csv').value
        self.timeout = self.declare_parameter('episode_timeout', 120.0).value
        self.cfg = ScenarioConfig(sources=('lab',), obstacle_mode='always')
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', LATCHED)
        self.create_subscription(String, '/nav_status', self.on_status, LATCHED)
        self.create_subscription(ContactsState, '/bumper_states', self.on_contact, 10)
        self.spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete = self.create_client(DeleteEntity, '/delete_entity')
        self.set_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.status = 'idle'
        self.contact = False

    # ---- callbacks ----
    def on_status(self, msg):
        self.status = msg.data

    def on_contact(self, msg):
        if msg.states:
            self.contact = True

    # ---- helpers ----
    def call(self, client, request):
        client.wait_for_service()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        return future.result()

    def teleport(self, x, y, yaw):
        request = SetEntityState.Request()
        request.state.name = 'martha'
        request.state.pose.position.x, request.state.pose.position.y = float(x), float(y)
        request.state.pose.position.z = 0.05
        request.state.pose.orientation.z = math.sin(yaw / 2)
        request.state.pose.orientation.w = math.cos(yaw / 2)
        request.state.reference_frame = 'world'
        self.call(self.set_state, request)

    def spawn_obstacles(self, obstacles):
        names = []
        for i, ob in enumerate(obstacles):
            name = f'obstacle_{i}'
            sdf = (BOX_SDF.format(name=name, sx=ob.sx, sy=ob.sy) if ob.kind == 'box'
                   else CYLINDER_SDF.format(name=name, r=ob.radius))
            request = SpawnEntity.Request()
            request.name, request.xml = name, sdf
            request.initial_pose.position.x = float(ob.x)
            request.initial_pose.position.y = float(ob.y)
            request.initial_pose.position.z = 0.3
            request.initial_pose.orientation.z = math.sin(ob.yaw / 2)
            request.initial_pose.orientation.w = math.cos(ob.yaw / 2)
            self.call(self.spawn, request)
            names.append(name)
        return names

    def clear_obstacles(self, names):
        for name in names:
            request = DeleteEntity.Request()
            request.name = name
            self.call(self.delete, request)

    def send_goal(self, goal):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(goal[0]), float(goal[1])
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    # ---- episodes ----
    def run_episode(self, seed):
        scenario = generate(seed, self.cfg)
        self.teleport(*scenario.start)
        names = self.spawn_obstacles(scenario.obstacles)
        self.status, self.contact = 'idle', False
        time.sleep(1.0)
        self.send_goal(scenario.goal)
        start = time.time()
        outcome = 'timeout'
        while time.time() - start < self.timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.contact:
                outcome = 'collision'
                break
            if self.status == 'succeeded':
                outcome = 'success'
                break
            if self.status == 'failed':
                outcome = 'failed'
                break
        self.clear_obstacles(names)
        return {'episode_seed': seed, 'outcome': outcome, 'source': 'lab',
                'n_obstacles': len(scenario.obstacles), 'route_length': scenario.path.length,
                'shortest': scenario.shortest, 'seconds': round(time.time() - start, 2)}

    def run(self):
        rows = []
        for seed in eval_seeds(self.episodes):
            row = self.run_episode(seed)
            rows.append(row)
            self.get_logger().info(f'{seed}: {row["outcome"]} ({len(rows)}/{self.episodes})')
        with open(self.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts = {o: sum(r['outcome'] == o for r in rows) / len(rows) for o in
                  {r['outcome'] for r in rows}}
        self.get_logger().info(f'done: {counts} -> {self.out}')


def main():
    rclpy.init()
    node = GazeboEval()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
```

Añadir a `entry_points`: `'gazebo_eval = martha_nav.ros.gazebo_eval:main',`

- [ ] **Step 2: Ejecutar E2 (100 episodios)**

Terminal 1: `sim.launch.py` con el checkpoint, `gui:=false`.
Terminal 2:
```bash
./tools/ct_ros bash -c 'source /home/ros/ros2_ws/install/setup.bash && ros2 run martha_nav gazebo_eval --ros-args -p episodes:=100 -p out:=/home/ros/ros2_ws/src/martha_nav/runs/long_c_kl_s0/eval_gazebo_lab.csv'
```
Expected: 100 filas y un resumen. **Las semillas son las mismas que las del `lab` en 2D**, así que la comparación es pareada.

- [ ] **Step 3: Comparar con el 2D y registrar (esto es E2)**

```bash
./tools/ct python3 -c "
import pandas as pd
a = pd.read_csv('runs/long_c_kl_s0/eval_obstacles_lab.csv')
b = pd.read_csv('runs/long_c_kl_s0/eval_gazebo_lab.csv')
common = set(a.episode_seed) & set(b.episode_seed)
a, b = a[a.episode_seed.isin(common)], b[b.episode_seed.isin(common)]
for name, d in (('2D', a), ('gazebo', b)):
    print(name, len(d), d.outcome.value_counts(normalize=True).round(3).to_dict())
"
```

Añadir a `docs/resultados.md` una sección `## E2: brecha 2D → Gazebo (lab.world)` con: la tabla de las dos columnas, la diferencia en puntos de éxito, y una conclusión de una línea sobre si hace falta un fine-tune en Gazebo (la spec lo deja fuera de alcance y se decidiría con el usuario).

- [ ] **Step 4: Commit**

```bash
git add martha_nav/ros/gazebo_eval.py setup.py docs/resultados.md
git commit -m "Evaluate the policy in Gazebo and record the 2D-to-Gazebo gap" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Después de este plan

1. Según la brecha medida en E2, decidir con el usuario si hace falta un fine-tune en Gazebo (spec §9).
2. Robot real: el contrato ya está fijado (spec §7.5). Falta el launch del hardware y sustituir `ground_truth_tf` por `slam_toolbox` en modo localización.
3. E1 (CNN frente a MLP) corre aparte con `tools/run_e1.sh`; su tabla entra en la tesis, no en este plan.
