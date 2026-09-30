# Contexto del proyecto (resumen para retomar rápido)

Planificador local con PPO para el robot mecanum **Martha**. El global sigue siendo
clásico (Dijkstra sobre rejilla) y la política solo resuelve el tramo local hacia una
zanahoria. Entrenamiento en un simulador 2D propio y rápido; validación en Gazebo.
`lab.world` es el gemelo del laboratorio y **nunca se entrena con él**.

## Decisiones fijas

- Acción continua. **Para la presentación se usa el holonómico `(vx, vy, w)`**
  (`armH_holonomic_s0`, decisión del 2026-09-24): es difícil justificar un modelo diferencial
  sobre una plataforma mecanum. `(v, w)` (`long_c_kl_s0`) queda como línea base: choca
  3 a 5 veces menos, y el holonómico cambia estancamientos por choques.
- LiDAR: 360 rayos a 8 m (A2M8), codificados **`d/(d+1)`** (`inverse`). Esta es la
  decisión que hizo converger el modelo; con `d/8` (`linear`) se estanca en ~37%.
- Ángulo a la zanahoria **directo, no `sin/cos`** (provoca trayectorias circulares).
- Sin apilado de fotogramas.
- Recompensa: **+20 meta, −20 choque**, en `martha_nav/sim2d/reward.py` (archivo aparte
  a propósito, para poder tocarla sin abrir el entorno).
- `target_kl = 0.02`. Sin esto el `std` de la política colapsa (0.6 → 0.007) y el
  rendimiento se degrada con más entrenamiento.
- Rutas de 3 a 12 m, zanahoria con lookahead variable.
- Escenarios: plantillas paramétricas + los `.world` existentes rasterizados, **todos
  mezclados**, sin niveles de dificultad.
- Las semillas de evaluación son disjuntas de las de entrenamiento por construcción
  (`TRAIN_SEED_LIMIT`), así que la evaluación determinista es honesta.

## Mapa del repositorio

| ruta | qué hay |
|---|---|
| `martha_nav/sim2d/` | `geometry.py` (rejilla, raycast), `planner.py` (Dijkstra, zanahoria, inflado 0.40 m), `worlds.py` (rasteriza SDF), `scenarios.py` (plantillas y `generate`), `dynamics.py`, `observation.py`, `reward.py`, `env.py` |
| `martha_nav/learning/` | `policy.py` (CNN 1D sobre el LiDAR), `train.py`, `evaluate.py` |
| `martha_nav/ros/` | `global_planner.py`, `ppo_local_planner.py`, `world_map_publisher.py`, `gazebo_ground_truth_tf.py`, `mecanum_cmd_vel_bridge.py`, `world_speed.py`, `evaluate_gazebo.py` |
| `urdf/martha.urdf.xacro` | reusado del paquete anterior; `drive:=mecanum\|planar` |
| `launch/sim.launch.py` | Gazebo + controladores + nodos + RViz opcional |
| `tools/` | `ct`, `ct_ros`, `evaluate_run.sh`, `run_demo_worlds.sh`, `run_e2.sh`, gráficas; `experiments/` guarda las colas ya corridas |
| `docs/resultados.md` | bitácora citable (puerta, A/B/C, `target_kl`, E1, E2) |
| `docs/resultados-noche.md` | brazos H (holonómico) y L (LSTM) |

Todo lo de Python corre **dentro del contenedor** `ros2_humble` con `./tools/ct`;
`git` corre en el anfitrión. Los comandos están en [comandos.md](comandos.md).

## Resultados clave

| run | limpio | obstáculos | `lab` (no visto) |
|---|---|---|---|
| `full_cnn_s0` (base, LiDAR lineal) | 0.682 | 0.366 | 0.390 |
| `expC_col20_inverse` (LiDAR `d/(d+1)`) | 0.910 | 0.712 | 0.590 |
| **`long_c_kl_s0`** (C + `target_kl`, referencia) | **0.958** | **0.868** | **0.865** |
| `armH_holonomic_s0` (`vx, vy, w`) | 0.982 | 0.912 | 0.815 |
| `armL_lstm_s0` (LSTM) | 0.974 | 0.840 | 0.670 |

E2 (mismas semillas en 2D y en Gazebo, `lab.world`): 0.880 contra 0.830, dentro de la
repetibilidad del propio Gazebo (±5 puntos). **La transferencia 2D → Gazebo no pierde
nada medible**; el fallo que queda es bloqueo/atasco, no choque.

## Tropiezos ya resueltos (no repetirlos)

- **URDF**: no puede contener `: ` ni `:=` **ni siquiera en los comentarios**; se pasa
  como regla de parámetro y se parsea como YAML. Lo vigila `test_urdf.py`.
- Los comentarios XML no admiten `--`.
- Con `mecanum_drive_controller` la odometría y el TF salen en topics propios; el launch
  remapea `/odom` y `gazebo_ground_truth_tf` reenvía `/mecanum_drive_controller/tf_odometry`.
- `libgazebo_ros_state.so` cargado con `-s` no publica solo: `create_scaled_world`
  le inyecta el plugin con `update_rate 50`.
- `evaluate_gazebo` toma la posición de `/gazebo/model_states`, no de `/odom` (con mecanum
  ese topic no existe y la regla de atasco quedaba muerta en silencio).
- La parada de seguridad del `ppo_local_planner` congelaba al robot contra las paredes; ahora
  el guardia es **direccional** (clasifica la intrusión por el eje dominante). Eso subió
  Gazebo de 76% a 88%.
- `evaluate` fija `torch.set_num_threads(4)`: sin eso torch usaba los 24 hilos y dejaba
  sin CPU a los entornos.
- Los runs recurrentes van con `--device cpu` y **no** se solapan con Gazebo (hubo un
  OOM de GPU que tumbó el equipo).
- Al matar procesos, usar patrones tipo `[g]zserver` para que `pkill` no se mate a sí mismo.
- Si algo queda raro, reiniciar el contenedor entero: los `robot_state_publisher`
  huérfanos confunden la búsqueda de parámetros del plugin.

## Pendiente

- Brazo **S**: añadir señal de atasco a la observación (para el ciclo de
  "intento, no quepo, retrocedo, vuelvo por el mismo lado" que se vio en evaluación).
- Brazo **H2**: `(vx, vy, w)` con choque −40, para bajar las colisiones del brazo H.
- Vídeo de la demo en `lab.world` con `gui:=true`.
- Integración con el robot real cuando el hardware esté arreglado.
