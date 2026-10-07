# Correr la tesis en el TurtleBot3 Burger — diseño

Fecha: 2026-10-06. Estado: aprobado por partes en conversación; pendiente de revisión escrita.

## Objetivo

Que el TurtleBot3 Burger pueda hacer la demo de la tesis (mapear, localizarse y navegar con 2–4
obstáculos fuera del mapa) mientras no lleguen los motores de Martha. El Burger **reemplaza** a
Martha en la demo; no se presenta como un resultado de generalización.

`martha_nav` sigue siendo un solo paquete: cada mejora del método vale para los dos robots. Lo que
hoy supone "el robot es Martha" pasa a leerse de un perfil de robot.

## Parte 0: limpieza (commits aparte, antes de todo)

**Solo la arquitectura CNN.** La MLP solo existía para E1 y no se va a usar: se quitan el flag
`--arch` de `train_policy` y la rama `mlp` de `policy_kwargs()`. Los runs `e1_mlp_*` siguen
cargándose (el zip guarda su arquitectura).

**Código sin historia.** Se borran `tools/experiments/` (guiones de experimentos ya corridos) y
`tools/plot_report_legacy.py`, y los comentarios que nombran runs viejos o el "paquete anterior".
`docs/comandos.md` explica cómo recuperar esos guiones del historial de git.

**Nombres.**

Los nombres E1/E2 ya no dicen nada. Lo que queda tras la limpieza del 2026-10-06 (`9a61050`):

- `tools/plot_e2.py` → `tools/plot_2d_vs_gazebo.py`.
- El comentario "(E2)" de `tools/evaluate_run_gazebo.sh` y el título "Evaluación en Gazebo (E2)"
  de `docs/comandos.md`: sin "E2".
- `docs/resultados.md`: los títulos con "E2" pasan a "Brecha 2D → Gazebo"; cifras e historia no
  cambian.

## Parte 1: el perfil del robot

`martha_nav/robots.py`: un `dataclass(frozen=True)` `Robot` y un diccionario `ROBOTS` con dos
perfiles en Python (no YAML: son dos y se versionan con el código).

| campo | Martha | Burger |
|---|---|---|
| `length`, `width` (contorno de colisión) | 0.56, 0.41 m | 0.14, 0.178 m |
| `footprint_offset_x` (centro del contorno en `base_link`) | 0.0 | −0.032 m |
| `lidar_offset_x` | 0.2325 m | −0.032 m |
| `lidar_range` (máx.), `lidar_min` | 8.0, 0.0 m | según `range_max`: 3.5 / 0.12 (LDS-01) o 8.0 / 0.16 (LDS-02) |
| `lidar_rate` | 10 Hz | 5 Hz |
| `v_max`, `v_reverse`, `v_lateral` | 0.35, 0.15, 0.25 m/s | 0.22, 0.22, — |
| `w_max` | 0.8 rad/s | 1.5 rad/s (el robot da 2.84; más rango haría la acción más gruesa) |
| `holonomic` (acepta `action_dim` 3) | sí | no |
| `inflation` del planificador | 0.40 m | 0.20 m |

Las cifras del Burger salen de su URDF oficial (`turtlebot3_burger.urdf`: caja de colisión
0.140 × 0.140 m centrada en x = −0.032, ruedas hasta 0.178 m de ancho, `base_scan` en x = −0.032)
y de su ficha (0.22 m/s, 2.84 rad/s). Antes de entrenar se verifican contra el robot: `range_max`
de `/scan`, y sentido y escala de `/cmd_vel`. Martha tiene `lidar_min` 0 porque el 2D de hoy no
modela un alcance mínimo; así sus episodios no cambian.

**Flujo.**

- `EnvConfig.robot: str = 'martha'`. `NavEnv` toma de `ROBOTS[cfg.robot]` el contorno (colisión),
  la posición y el alcance del LiDAR (raycast), las velocidades (escala de acciones y de la
  velocidad observada) y el tiempo límite. `ScenarioConfig` y el A* toman el inflado del perfil.
- `train_policy --robot burger` lo guarda en el `config.yaml`. `trained_env_config()` lo lee; los
  runs sin el campo son `martha`.
- `global_planner`, `ppo_local_planner` y `policy_core` (contorno de la parada de seguridad,
  posición del LiDAR, escala de `/cmd_vel`) usan el perfil del modelo cargado. Si el launch dice
  un robot y el modelo es de otro, el nodo no arranca y dice por qué.
- Desaparecen de los módulos `V_MAX`, `V_REVERSE`, `V_LATERAL`, `W_MAX`, `LIDAR_MAX`,
  `ROBOT_LENGTH`, `ROBOT_WIDTH`, `LIDAR_OFFSET_X`, `FOOTPRINT`, `INFLATION`. `N_SECTORS` (90) y
  `DT` (0.1 s) no son del robot y se quedan.

**Garantía.** La huella de episodios 2D (`runs/_fingerprint.py`; el hash esperado está en su
línea `# expected:`) debe salir idéntica con `robot: martha`, y las evaluaciones 2D de
`wide_dyn_s0` deben repetir sus cifras. Con eso, nada de lo ya entrenado ni reportado cambia.

Los mapas de entrenamiento no cambian: hechos para Martha, son holgados para el Burger.

## Parte 2: entrenar y evaluar el Burger

**Entrenamiento 2D**, misma receta que el modelo final:

```bash
train_policy --preset full --seed 0 --robot burger --wide-dynamics --name burger_s0
```

Dos rasgos del sensor real que el 2D debe imitar (la lección del retardo de actuación: lo que no
se simula cuesta en el robot real):

- **Frecuencia del LiDAR.** Con `lidar_rate` 5 Hz y control a 10 Hz, el entorno repite el scan
  anterior en uno de cada dos pasos (fase aleatoria por episodio).
- **Alcance mínimo.** Las lecturas por debajo de `lidar_min` se devuelven como sin dato (alcance
  máximo), como el LDS.

**Evaluación 2D:** los mismos cuatro conjuntos con `tools/evaluate_run.sh runs/burger_s0`, sin
cambios (el robot sale del `config.yaml`).

**Gazebo:**

- `urdf/burger.urdf.xacro`, propio y con geometría simple (sin depender de los paquetes
  `turtlebot3_*`): las medidas del URDF oficial, `libgazebo_ros_diff_drive` (`/cmd_vel`, `/odom`,
  TF `odom → base_footprint`, ruedas a 0.160 m, radio 0.033 m), un sensor de rayos con el alcance,
  el mínimo y la frecuencia del perfil, y un sensor de contacto que publica en `/bumper_states`
  como el de Martha.
- `sim.launch.py robot:=martha|burger`, por defecto el del `config.yaml` del checkpoint; si se pasan
  los dos y no coinciden, falla. Con `burger` no se lanzan `ros2_control` ni
  `mecanum_cmd_vel_bridge`.
- `evaluate_gazebo` y `gazebo_ground_truth_tf` buscan la entidad `robot` en vez de `martha`; los
  dos URDF se spawnean con ese nombre.
- `tools/evaluate_run_gazebo.sh runs/burger_s0` funciona sin cambios.

**Criterio de éxito:** en las mismas semillas, el éxito del Burger en Gazebo sin diferencia
significativa frente a su 2D (McNemar, p > 0.05), como con Martha.

## Parte 3: el Burger real

**Un solo launch que corre en cualquier máquina**, `launch/burger.launch.py`:
slam_toolbox (`slam:=mapping|localization`, mismo módulo `martha_nav/ros/slam.py` que Martha, con
`max_laser_range` del perfil),
`world_map_publisher`, `global_planner` y `ppo_local_planner` con la política exportada. Se corre
en la Pi 4 (4 GB) del Burger para la demo; en el PC es el mismo comando, para depurar con los logs
a mano. El bringup oficial del TurtleBot (`turtlebot3_bringup robot.launch.py`) sigue aparte, en la
Pi, como ya funciona. En el PC queda RViz para ver y mandar metas: si el WiFi cae, el robot termina
la meta que tiene.

`real.launch.py` queda solo para Martha.

**Política sin PyTorch.**

- `export_policy --model runs/burger_s0/best_model.zip` escribe `policy.npz` junto al modelo:
  pesos de la red de la acción determinista, el perfil del robot y la configuración de observación
  (`lidar_encoding`, `action_dim`, `stuck_signal`, `target`).
- `ppo_local_planner` acepta `checkpoint:=...zip` (PyTorch, como hoy) o `...npz` (numpy). Con
  `.npz` no importa SB3 ni torch: la configuración del entorno sale del propio `.npz`, no de
  `trained_env_config` (que importa SB3).
- Un test exige que, sobre observaciones reales del 2D, la acción en numpy sea igual a la de
  PyTorch (`atol` 1e-5), para el `wide_dyn_s0` y para una red CNN recién creada.
- Solo para políticas no recurrentes (el modelo final y el del Burger lo son).

**Despliegue.** `tools/deploy_burger.sh <usuario@ip>`: `rsync` de `martha_nav` (sin `runs/`,
salvo el `policy.npz` indicado) al workspace de la Pi y `colcon build --symlink-install` allí.
Dependencias en la Pi, una vez: `ros-humble-slam-toolbox` y `python3-scipy` por apt. Los pasos
quedan en `docs/comandos.md`.

**Red.** PC y robot comparten `ROS_DOMAIN_ID` (el TurtleBot suele traer 30); se documenta.

**Latencia.** `ppo_local_planner` registra cada 10 s la latencia entre el sello del `/scan` y la
publicación de `/cmd_vel` (media y máxima): para mostrar el desempeño a bordo.

**Seguridad.** La parada de seguridad del planificador local usa el contorno del perfil. Parámetro
`speed_scale` (por defecto 1.0) que escala `/cmd_vel`; la primera prueba real se hace con 0.5.

## Pruebas

- `robots.py`: los perfiles existen y Martha reproduce las constantes de hoy.
- Huella de episodios 2D idéntica con `robot: martha`; cifras 2D de `wide_dyn_s0` idénticas.
- `NavEnv` con `robot: burger`: colisión con su contorno, escala de acciones a sus velocidades,
  scan repetido a 5 Hz, lecturas bajo `lidar_min` como sin dato.
- `trained_env_config` sin campo `robot` → `martha`; con campo → ese robot.
- Nodos: rechazan un modelo de otro robot; `policy_core` usa el contorno y el LiDAR del perfil.
- Exportación: acciones numpy = PyTorch.
- URDF del Burger: se procesa con xacro y tiene el sensor de contacto y el de rayos.

## Fuera de alcance

- Firmware o bringup del TurtleBot.
- Martha real: sigue como está, esperando los motores.
- Políticas recurrentes en numpy.
- Mapas de entrenamiento nuevos para robots pequeños (solo si el Burger lo necesita).
- Renombrar el paquete `martha_nav`.
