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
./tools/ct_ros ros2 run martha_nav train_policy --preset gate --seed 0 --name gate_check
```

Entrenamiento completo, 5M pasos, todas las fuentes con obstáculos mezclados:

```bash
./tools/ct_ros ros2 run martha_nav train_policy --preset full --seed 0 --name mi_run
```

Opciones útiles de `train_policy`:

| bandera | para qué |
|---|---|
| `--steps N` | cambia el presupuesto del preset |
| `--n-envs 16` | entornos en paralelo |
| `--action-dim 3` | añade `vy` (mecanum holonómico) |
| `--recurrent` | política LSTM (usar con `--device cpu`) |
| `--lidar-encoding inverse\|linear` | `inverse` = `d/(d+1)`, el que funciona |
| `--reward-collision -20` | penalización de choque |
| `--reward-stalled -5` | hace terminal el atasco (empeoró, no usar) |
| `--eval-every 250000` | cada cuánto evalúa y guarda `best_model.zip` |

Lanzarlo en segundo plano y seguir el avance:

```bash
nohup ./tools/ct_ros ros2 run martha_nav train_policy --preset full --seed 0 --name mi_run > runs/mi_run.log 2>&1 &
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

O las dos de una vez (de ~1.5 h a ~13 min), sin levantar la simulación antes:
`tools/evaluate_run_gazebo.sh` levanta 3 Gazebos sin GUI, cada uno con la física sin límite,
y les reparte las semillas (`shard:=i/3`). Deja los CSV en el run, o en `out_dir` si se da:

```bash
./tools/evaluate_run_gazebo.sh runs/mi_run _v5
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

## 6.2. Prueba de localización con slam_toolbox

¿Se pierde slam_toolbox con cajas que no están en su mapa? Por defecto la simulación usa la
pose verdadera de Gazebo; con `slam:=localization` la pose sale de slam_toolbox, y
`evaluate_gazebo` compara las dos en cada episodio (columnas `loc_err_mean`, `loc_err_max`
en metros y `yaw_err_max_deg`).

1. Mapear, con el robot en el origen para que el mapa de SLAM quede en el marco de Gazebo:

```bash
./tools/ct_ros ros2 launch martha_nav sim.launch.py world:=lab gui:=false x:=0.0 y:=0.0 slam:=mapping checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/long_c_kl_s0/best_model.zip
```

En otra terminal, el recorrido por los puntos fijos del mundo, que al final guarda el mapa:

```bash
./tools/ct_ros python3 tools/map_world.py --world lab --out /home/ros/ros2_ws/src/martha_nav/maps/lab
```

Deja `maps/lab.posegraph` y `maps/lab.data` (unos 9 MB). `maps/` no está en git: si no existe,
este paso lo regenera (unos 2 minutos).

2. Cerrar la simulación y relanzarla localizando con ese mapa:

```bash
./tools/ct_ros ros2 launch martha_nav sim.launch.py world:=lab gui:=false x:=0.95 y:=1.35 slam:=localization slam_map:=/home/ros/ros2_ws/src/martha_nav/maps/lab checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/wide_dyn_s0/best_model.zip
```

3. Evaluar sin obstáculos y con ellos (1 a 4 cajas o cilindros cerca de la ruta, que el mapa
   no tiene). Cada teleport manda la pose inicial a `/initialpose`, como el operador en RViz:

```bash
./tools/ct_ros ros2 run martha_nav evaluate_gazebo --ros-args -p episodes:=100 -p condition:=clean -p out:=/home/ros/ros2_ws/src/martha_nav/runs/long_c_kl_s0/eval_gazebo_lab_slam_clean.csv
```

```bash
./tools/ct_ros ros2 run martha_nav evaluate_gazebo --ros-args -p episodes:=100 -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/runs/long_c_kl_s0/eval_gazebo_lab_slam.csv
```

## 7. Guiones de experimentos ya hechos

`tools/experiments/` guarda las colas que produjeron los resultados de `docs/resultados*.md`,
tal como se corrieron, para poder repetirlas: `run_experiments_abc.sh` (A/B/C),
`run_e1.sh` (CNN contra MLP), `run_night.sh` y `run_arms_stuck.sh` (brazos H, L, HL, S, SL),
`train_gl_geo_10m.sh` (brazo GL geodésico, 10M pasos), `run_e2.sh` (el primer E2, a 1×),
`plot_e1.py` y `plot_run.py`.

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

## 9. Robot real

Diseño: [robot-real-design.md](robot-real-design.md). Todo lo de ROS corre en el PC
montado sobre el robot; la ESP32 solo hace motores, encoders, gyro y protecciones.

### 9.1 Puertos

La ESP32 y el RPLIDAR usan el mismo chip USB (CP2102) y `ttyUSB0/1` pueden
intercambiarse entre arranques: usa siempre las rutas estables.

```bash
ls -l /dev/serial/by-id/
```

Si los dos adaptadores USB-serie tienen el mismo número de serie, `by-id` muestra
un solo enlace: usa `/dev/serial/by-path/` en ese caso.

### 9.2 Firmware (en el anfitrión)

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 --warnings all firmware
arduino-cli upload -p /dev/serial/by-id/<esp32> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware
arduino-cli monitor -p /dev/serial/by-id/<esp32> -c baudrate=115200
```

En el monitor se escriben los comandos a mano: `cmd_vel,0.1,0,0` o `reset`.

### 9.3 Antes de flashear (hardware, una vez)

1. Divisor en los cables amarillo (A) y blanco (B) de cada encoder, del lado de la
   ESP32: señal → 4.7 kΩ → GPIO, y 10 kΩ del GPIO a GND. Con la rueda quieta en
   alto el pin debe leer entre 2.5 y 3.6 V.
2. Desoldar R4 (el firmware usa el pull-up interno).
3. RV1 con los encoders conectados y los drivers despiertos: VD ≈ 0.7 V (~3.5 A por
   motor). Anotar R2/R3.
4. Recomendado e irreversible: `espefuse.py --port /dev/serial/by-id/<esp32> set_flash_voltage 3.3V`
   (si no, GPIO12 —encoder A de M1— puede impedir que la ESP32 arranque al reabrir el puerto con los drivers despiertos).

### 9.4 Puesta en marcha (en orden)

1. Ruedas al aire, monitor serial: aparece `ready`; `battery,V` coincide con el
   multímetro (si no, ajustar `BATTERY_GAIN`). Con el robot armado, cierra y reabre
   el monitor dos veces: debe salir `ready` en las dos (si la ESP32 no arranca, es
   GPIO12 — ver el eFuse de 9.3.4).
2. `cmd_vel,0.1,0,0`: las 4 ruedas hacia adelante y `odom` con vx > 0. Si una rueda
   gira al revés, su entrada de `MOTOR_SIGN`; si cuenta al revés, la de `ENCODER_SIGN`.
   Repetir con `cmd_vel,0,0.1,0` (vy > 0, hacia la izquierda) y `cmd_vel,0,0,0.5` (wz > 0).
3. La velocidad medida sigue a la ordenada; si no, ajustar `KP` y `KI`.
4. Giro antihorario a mano **con las ruedas apoyadas en el piso**: el último campo
   de `odom` (gz) > 0; quieto, ≈ 0. Con las ruedas al aire el firmware cree que el
   robot está quieto y reaprende el bias del gyro (~2 s), así que el giro no se ve;
   gíralo rápido o hazlo en el piso.
5. Un solo `cmd_vel,0.1,0,0`: las ruedas giran ~0.5 s y aparece `cmd_vel_timeout`.
   Puente de D23 a GND: `motor_overcurrent`; quitarlo y `reset` → `ready`.
   Fuente de laboratorio < 11 V: `battery_too_low`. Sujeta una rueda con la mano
   (con guantes) y manda un `cmd_vel` pequeño: `motor_overcurrent` debe salir en
   ~2 s; si nunca sale, D23 conmuta con el PWM (anotarlo, pendiente de revisar).
6. ROS en modo mapeo (9.5) con RViz: activa el display TF y comprueba el árbol
   `map → odom → base_link → lidar`; el scan alineado con el frente del robot (si
   sale girado 180°, `flip_x_axis: true` en `config/rplidar.yaml`); sin puntos del
   scan sobre el propio robot (PC, mástil) — el PPO los tomaría como obstáculos
   dentro del footprint; empujarlo 1 m → `/odom` ~1 m; girarlo 360° → ~2π.

### 9.5 Mapear un lugar

Pon el robot sobre una marca de cinta en el piso, marcando también su orientación:
es donde arrancará la demo (el mapa guardado arranca en (0, 0, 0), yaw incluido).

```bash
./tools/ct_ros ros2 launch martha_nav real.launch.py rviz:=true \
    esp32_port:=/dev/serial/by-id/<esp32> lidar_port:=/dev/serial/by-id/<rplidar>
```

Para detener: Ctrl-C en esa terminal para todo. Si se cerró la terminal o algo
quedó corriendo: `docker exec ros2_humble pkill -INT -f '[r]eal.launch.py'`. El
paro físico es el interruptor de alimentación del robot.

En otra terminal (el teleop necesita una terminal interactiva, por eso `-it`; con
Shift se mueve lateral):

```bash
docker exec -it ros2_humble bash -lc 'source /opt/ros/humble/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.2 -p turn:=0.5'
```

Mantener una tecla apretada tartamudea (el autorepeat del terminal contra el
timeout de 500 ms); pulsa repetido o convive con eso.

En RViz agrega un display Map sobre `/slam_map` para ver el mapa crecer. Al terminar:

```bash
./tools/ct_ros tools/save_map.sh /home/ros/ros2_ws/src/martha_nav/maps/<lugar>_real
```

Deja `<lugar>_real.pgm/.yaml` (el mapa del A*; se puede limpiar en GIMP) y
`<lugar>_real.posegraph/.data` (para localizarse).

### 9.6 Demo

Robot sobre la marca de cinta con su orientación (o "2D Pose Estimate" en RViz),
sin las cajas en el mapa:

```bash
./tools/ct_ros ros2 launch martha_nav real.launch.py rviz:=true \
    map:=/home/ros/ros2_ws/src/martha_nav/maps/<lugar>_real \
    checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/armH_holonomic_s0/best_model.zip \
    esp32_port:=/dev/serial/by-id/<esp32> lidar_port:=/dev/serial/by-id/<rplidar>
```

La meta se da con "2D Goal Pose" en RViz.

### 9.7 Tras un latch (`motor_overcurrent` o `battery_too_low`)

Revisa la causa, cancela la meta para que el PPO no arranque al rearmar, y resetea;
la respuesta (`ready` o `reset_blocked`) sale en el log de `esp32_bridge`:

```bash
./tools/ct_ros ros2 topic pub --once /cancel_goal std_msgs/msg/Empty
./tools/ct_ros ros2 service call /esp32_bridge/reset std_srvs/srv/Trigger
```

Los eventos se mandan una sola vez: si el robot no se mueve y no viste ninguno,
puede haber quedado latcheado antes de que el bridge conectara; llama igual al
reset y mira el log del bridge (`ready` o `reset_blocked`).
