# Comandos

Todo se ejecuta desde la raíz del repo (`~/ros2_ws/src/martha_nav`) en el anfitrión.
`./tools/ct` corre dentro del contenedor `ros2_humble`; `./tools/ct_ros` además
hace `source` de ROS 2 Humble y del workspace, así que acepta `ros2 run martha_nav ...`
directamente. `git` se ejecuta fuera del contenedor.

```bash
docker start ros2_humble          # si no está arriba
```

## 1. Pruebas del paquete

Todas las pruebas, incluidas las que importan mensajes de ROS y `xacro`:

```bash
./tools/ct_ros python3 -m pytest test -q
```

Solo lo del simulador 2D y el aprendizaje, que es más rápido y no necesita ROS:

```bash
./tools/ct python3 -m pytest test/test_env.py test/test_reward.py test/test_scenarios.py -q
```

## 2. Entrenamiento (simulador 2D)

Puerta de convergencia, ~15 min, sala abierta sin obstáculos (debe pasar de 80%):

```bash
./tools/ct_ros ros2 run martha_nav train_policy --preset gate --arch cnn --seed 0 --name gate_check
```

Entrenamiento completo, 5M pasos, todas las fuentes con obstáculos mezclados:

```bash
./tools/ct_ros ros2 run martha_nav train_policy --preset full --arch cnn --seed 0 --name mi_run
```

Opciones útiles de `train_policy`:

| bandera | para qué |
|---|---|
| `--steps N` | cambia el presupuesto del preset |
| `--arch cnn\|mlp` | arquitectura del extractor |
| `--n-envs 16` | entornos en paralelo |
| `--action-dim 3` | añade `vy` (mecanum holonómico) |
| `--recurrent` | política LSTM (usar con `--device cpu`) |
| `--lidar-encoding inverse\|linear` | `inverse` = `d/(d+1)`, el que funciona |
| `--reward-collision -20` | penalización de choque |
| `--reward-stalled -5` | hace terminal el atasco (empeoró, no usar) |
| `--eval-every 250000` | cada cuánto evalúa y guarda `best_model.zip` |

Lanzarlo en segundo plano y seguir el avance:

```bash
nohup ./tools/ct_ros ros2 run martha_nav train_policy --preset full --arch cnn --seed 0 --name mi_run > runs/mi_run.log 2>&1 &
```

```bash
tail -f runs/mi_run.log
```

Un run deja en `runs/<nombre>/`: `config.yaml`, `episodes.csv`, `evals.csv`,
`best_model.zip`, `last_model.zip`, `vecnormalize.pkl` y los registros de TensorBoard en `tb/`.

## 3. Evaluación determinista en 2D

La batería estándar de un run (limpio, obstáculos, laboratorio y sus puntos fijos) y su informe:

```bash
./tools/evaluate_run.sh runs/mi_run
```

O cada condición por separado:

```bash
./tools/ct_ros ros2 run martha_nav evaluate_2d --model runs/mi_run/best_model.zip --episodes 500 --condition obstacles
```

```bash
./tools/ct_ros ros2 run martha_nav evaluate_2d --model runs/mi_run/best_model.zip --episodes 500 --condition clean
```

Generalización al laboratorio, que nunca se entrenó:

```bash
./tools/ct_ros ros2 run martha_nav evaluate_2d --model runs/mi_run/best_model.zip --episodes 200 --condition obstacles --sources lab
```

Con los puntos fijos del paquete anterior (90 pares de `config/training_points.yaml`):

```bash
./tools/ct_ros ros2 run martha_nav evaluate_2d --model runs/mi_run/best_model.zip --condition obstacles --points lab
```

## 4. Gráficas

Informe por run (`learning_report.png` y `ppo_diagnostics.png` dentro de la carpeta del run):

```bash
./tools/ct python3 tools/plot_report.py runs/mi_run
```

```bash
./tools/ct python3 tools/plot_report.py --all
```

Mismo contenido con el formato exacto del `ppo_plot` anterior:

```bash
./tools/ct python3 tools/plot_report_legacy.py runs/mi_run
```

Comparación entre runs y figuras de los experimentos:

```bash
./tools/ct python3 tools/compare_runs.py --out docs/figures/comparacion.png base=runs/long_c_kl_s0 H=runs/armH_holonomic_s0 L=runs/armL_lstm_s0
```

```bash
./tools/ct python3 tools/experiments/plot_e1.py --out docs/figures/e1_cnn_vs_mlp.png
```

```bash
./tools/ct python3 tools/plot_e2.py --run runs/long_c_kl_s0 --out docs/figures/e2_2d_vs_gazebo.png
```

TensorBoard, si se quiere mirar en vivo:

```bash
./tools/ct tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

## 5. ROS 2 y Gazebo

Compilar el paquete (solo hace falta tras tocar `setup.py`, `launch/`, `urdf/`,
`config/`, `rviz/` o `worlds/`; el código Python va con `--symlink-install`):

```bash
./tools/ct_ros bash -c 'cd /home/ros/ros2_ws && colcon build --packages-select martha_nav --symlink-install'
```

Levantar la simulación con una política cargada, con ventana de Gazebo y RViz:

```bash
./tools/ct_ros ros2 launch martha_nav sim.launch.py world:=lab rviz:=true checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/long_c_kl_s0/best_model.zip
```

Argumentos del launch: `world` (`lab`, `room`, `hall`, `multi`, `tube`, `four_rooms`,
`roblab`), `drive` (`mecanum` por defecto, `planar` como alternativa rápida), `gui`,
`rviz`, `checkpoint`, `x`, `y`, `lidar_samples`, `sim_speed_factor`, `physics_step_size`.

Mandarle una meta a mano (o con la herramienta *2D Goal Pose* de RViz):

```bash
./tools/ct_ros ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: 2.0, y: 2.0}, orientation: {w: 1.0}}}"
```

```bash
./tools/ct_ros ros2 topic echo /nav_status
```

Cancelar la meta (el estado pasa a `idle` y el robot se detiene):

```bash
./tools/ct_ros ros2 topic pub --once /cancel_goal std_msgs/msg/Empty {}
```

RViz por separado, contra una simulación ya levantada:

```bash
./tools/ct_ros ros2 run rviz2 rviz2 -d /home/ros/ros2_ws/src/martha_nav/rviz/nav.rviz --ros-args -p use_sim_time:=true
```

## 6. Evaluación en Gazebo (E2)

Con `sim.launch.py` ya corriendo (conviene `gui:=false` para que vaya más rápido),
en **otra terminal**. Las mismas semillas que en 2D, así que la comparación es pareada:

```bash
./tools/ct_ros ros2 run martha_nav evaluate_gazebo --ros-args -p episodes:=100 -p mode:=seeds -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/runs/mi_run/eval_gazebo_lab.csv
```

Con los puntos fijos del paquete anterior:

```bash
./tools/ct_ros ros2 run martha_nav evaluate_gazebo --ros-args -p mode:=points -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/runs/mi_run/eval_gazebo_lab_points.csv
```

O el guion que encadena las dos:

```bash
./tools/run_e2.sh
```

## 6.1. Demostración en todos los mundos

`tools/run_demo_worlds.sh` recorre los mundos uno por uno: levanta `sim.launch.py`,
espera a que la política esté cargada, corre unos cuantos episodios con `evaluate_gazebo`,
cierra Gazebo y pasa al siguiente. Deja un CSV por mundo y una tabla en
`runs/demo_worlds/resumen.md`.

A diferencia del resto, **este guion se ejecuta dentro del contenedor**, desde la raíz
del repo, porque necesita la ventana de Gazebo:

```bash
docker exec -it ros2_humble bash -lc 'cd /home/ros/ros2_ws/src/martha_nav && ./tools/run_demo_worlds.sh'
```

Se controla por variables de entorno:

| variable | por defecto | para qué |
|---|---|---|
| `MODEL` | `runs/long_c_kl_s0/best_model.zip` | política a demostrar (ruta relativa al repo) |
| `EPISODES` | `10` | episodios por mundo |
| `WORLDS` | los siete mundos | lista separada por espacios |
| `GUI` | `false` | `true` abre Gazebo y RViz para verlo |
| `SPEED` | `1.0` | factor de tiempo real (súbelo si solo quieres los números) |
| `RVIZ` | igual que `GUI` | RViz con el mapa, el LiDAR, el plan y la zanahoria |
| `CONDITION` | `obstacles` | `obstacles`, `clean` o `mixed` |
| `OUT` | `runs/demo_worlds` | carpeta de salida |

Para verlo en vivo en pocos mundos:

```bash
EPISODES=5 GUI=true WORLDS="lab room tube" ./tools/run_demo_worlds.sh
```

Y para una demo larga en el laboratorio, que es el mundo de la sustentación:

```bash
EPISODES=30 GUI=true WORLDS=lab ./tools/run_demo_worlds.sh
```

## 7. Guiones de experimentos ya hechos

`tools/experiments/` guarda las colas que produjeron los resultados de `docs/resultados*.md`,
tal como se corrieron, para poder repetirlas: `run_experiments_abc.sh` (A/B/C),
`run_e1.sh` (CNN contra MLP), `run_night.sh` y `run_arms_stuck.sh` (brazos H, L, HL, S, SL),
`train_gl_geo_10m.sh` (brazo GL geodésico, 10M pasos), `plot_e1.py` y `plot_run.py`.

```bash
./tools/experiments/run_e1.sh
```

## 8. Limpieza cuando algo queda colgado

```bash
docker exec ros2_humble pkill -f '[g]zserver'; docker exec ros2_humble pkill -f '[g]zclient'
```

```bash
docker restart ros2_humble
```
