# Diseño: planificador local con PPO para Martha

Fecha: 2026-09-21. Estado: aprobado en brainstorming, pendiente de plan de implementación.

## 1. Objetivo y alcance

Tesis de navegación autónoma con aprendizaje por refuerzo (PPO) en ROS 2 Humble y
Gazebo Classic. El paquete `martha_nav` reemplaza por completo al código PPO de
`martha/`, que no convergía con obstáculos (techo de ~45% de éxito, ver
`martha/HALLAZGOS_TESIS.md`). El código viejo queda solo como referencia.

**Qué debe demostrar la tesis:**

- Martha navega hasta una meta esquivando obstáculos colocados a propósito cuya
  posición no se conoce de antemano (cajas, sillas, cilindros).
- La política **generaliza**: se entrena en habitaciones generadas y se prueba en
  `lab.world` (gemelo digital del laboratorio), que nunca ve durante el
  entrenamiento.
- Funciona en Gazebo y, cuando el hardware esté listo, en el robot real sin
  reentrenar.

**Dentro del alcance:** simulador 2D de entrenamiento, entrenamiento con
Stable-Baselines3, evaluación en 2D y en Gazebo, nodos ROS de planificación y
política, robot simplificado para Gazebo.

**Fuera del alcance:** lanzamiento del robot físico, identificación del sistema
(sysid), fine-tune en Gazebo, planificadores clásicos de comparación (opcional si
sobra tiempo). Del robot real solo se fija el contrato de interfaz (sección 7.5).

**Restricción:** plazo corto (entrega hacia finales de octubre de 2026). Ante la
duda, lo más simple que converja.

## 2. Decisiones principales

| decisión | elección | motivo |
|---|---|---|
| papel de la red | **planificador local**: sigue waypoints de un A* global | extremo a extremo cae en mínimos locales tras paredes |
| dónde se entrena | **simulador 2D propio (Gymnasium)** | Gazebo daba ~22 pasos/s, ~1 día por experimento |
| algoritmo | **PPO de Stable-Baselines3** | implementación probada; elimina una fuente de bugs |
| acciones | **continuas `(v, ω)`**, `vy = 0` | espacio pequeño; el mecanum lo ejecuta sin problema |
| memoria | **ninguna** (sin LSTM, sin apilar frames) | obstáculos estáticos, la velocidad va en la observación |
| red principal | **CNN-1D circular** sobre el LiDAR; MLP como línea base | continuidad angular del LiDAR |

## 3. Arquitectura y estructura

Paquete ROS 2 nuevo `martha_nav` (ament_python), repo git propio. De `martha/`
solo se reutilizan los archivos `.world`.

```
martha_nav/
  martha_nav/
    sim2d/            puro Python + numpy, SIN dependencias de ROS
      geometry.py       grid de ocupación, raycasting LiDAR, colisión de la huella
      scenarios.py      plantillas, rasterizador de .world, obstáculos sorpresa
      planner.py        A* sobre grid inflado, ruta, progreso y zanahoria
      dynamics.py       modelo (v, ω) con retardo, aceleración limitada, ruido
      observation.py    build_observation(): contrato compartido sim / ROS
      reward.py         RewardConfig + compute_reward()
      env.py            entorno Gymnasium que une todo
    learning/
      policy.py         extractores CNN-1D y MLP para SB3
      train.py          PPO de SB3 con entornos en paralelo
      evaluate.py       evaluación determinista, métricas, CSV
    ros/
      planner_node.py   A* sobre /map -> /plan
      policy_node.py    zanahoria + red -> /cmd_vel
      map_publisher.py  .world rasterizado -> /map
      ground_truth_tf.py  pose de Gazebo -> TF map -> odom
      gazebo_eval.py    evaluación por episodios en Gazebo
  urdf/martha.urdf.xacro
  launch/sim.launch.py
  worlds/             copia de los .world de martha/
  test/
  docs/
```

**Contrato central:** `build_observation(ranges, angles, velocity, waypoint_rel,
prev_action)` en `observation.py` es la única forma de construir la observación.
La usan el entorno 2D, el `policy_node` en Gazebo y el `policy_node` en el robot
real. `sim2d/` no importa nada de ROS; `ros/` importa de `sim2d/`.

Tamaño objetivo: 1500–2000 líneas en total.

## 4. Escenarios (`scenarios.py`, `planner.py`)

### 4.1 Mapa

Grid de ocupación de **5 cm/celda** con dos capas:

- **Estática** (paredes, muebles): la conoce el A*.
- **Sorpresa** (cajas, cilindros): solo la ve el LiDAR.

La pose del robot `(x, y, θ)` es **continua**. El grid solo se usa para el
raycasting, la colisión y el A*.

### 4.2 Fuentes de la capa estática

Mezcladas al azar con probabilidad uniforme, **sin niveles de dificultad**:

1. **Plantillas paramétricas** (lista final ajustada tras revisar papers como
   Arena-Rosnav y DRL-VO):
   - pasillo, ancho 1.2–2.5 m;
   - puerta entre dos habitaciones, hueco 0.8–1.2 m;
   - giro en L;
   - habitación con muebles contra las paredes;
   - habitación con muebles en el centro;
   - paso estrecho, 0.8–1.0 m.
2. **Mundos existentes rasterizados:** `four_rooms`, `hall`, `multi`, `roblab`,
   `room`, `tube`. Se leen las cajas y cilindros estáticos del SDF.
3. **`lab.world` queda excluido del entrenamiento**, y una prueba lo verifica.

Plan de respaldo si no converge: introducir escenarios difíciles de forma
gradual. No se implementa salvo que haga falta.

### 4.3 Generación de un episodio

1. Se elige la fuente y se genera la capa estática.
2. Se toman un inicio y una meta en espacio libre, con una **ruta A* estática de
   3 a 12 m**.
3. Se colocan **0–4 obstáculos sorpresa** (~20% de episodios sin obstáculos):
   - cajas de lado 0.2–0.6 m o cilindros de radio 0.1–0.3 m;
   - centrados en puntos de la ruta A* con desplazamiento lateral de ±0.5 m;
   - nunca a menos de 1.0 m del inicio ni a menos de 0.6 m de la meta.
4. **Garantía de paso:** A* sobre estática + sorpresa infladas **0.40 m** (hueco
   libre ≥ 0.8 m; Martha mide 0.41 m de ancho). La ruta con obstáculos debe medir
   ≤ 1.5 × ruta estática + 2 m. Si falla, se re-sortean los obstáculos hasta 20
   veces y, si sigue fallando, el episodio va sin obstáculos (y se cuenta en el
   log).
5. Todo depende de una semilla. Las semillas de evaluación son fijas y disjuntas
   de las de entrenamiento.

### 4.4 Zanahoria (waypoint local)

- Se calcula sobre la ruta A* **estática**.
- `s` = proyección de la pose del robot sobre la ruta (longitud de arco).
- La zanahoria es el punto de la ruta a distancia `L` por delante de `s`.
  - `L` es aleatoria por episodio en el entrenamiento, uniforme en [1.0, 2.5] m.
  - En ROS es un parámetro fijo, 1.5 m por defecto.
- **Salto de obstáculos:** si la zanahoria está a menos de 0.4 m de algún punto
  del scan actual, avanza por la ruta hasta el primer punto libre, o hasta el
  final de la ruta. Solo usa el scan, así que funciona igual en el robot real.
- La meta final es la zanahoria cuando queda a menos de `L`.

## 5. Entorno (`env.py`, `dynamics.py`, `observation.py`, `reward.py`)

### 5.1 Observación: 96 valores

| parte | tamaño | codificación |
|---|---|---|
| LiDAR | 90 | mínimo por sector de 4°, `d / (d + 1 m)` con `d ≤ 8 m`; rayos desde la posición del LiDAR (+0.2325 m en x). *Cambiado el 2026-09-22: la versión inicial `d / 8 m` rendía la mitad con obstáculos (ver `docs/resultados.md`, experimentos A/B/C).* |
| waypoint | 2 | distancia `min(d, 3 m) / 3 m`, ángulo `atan2 / π` en [−1, 1] (**sin seno/coseno**) |
| velocidad medida | 2 | `v / 0.35`, `ω / 0.8` |
| acción anterior | 2 | en [−1, 1] |

El sector 0 apunta al frente del robot y los sectores crecen en sentido
antihorario. Los rayos inválidos (inf/NaN) cuentan como alcance máximo.

### 5.2 Acción

`[a_v, a_ω] ∈ [−1, 1]²`, mapeada a:

- `v`: los negativos a [−0.15, 0) m/s y los positivos a [0, 0.35] m/s (asimétrica
  a propósito);
- `ω ∈ [−0.8, 0.8] rad/s`.

### 5.3 Dinámica

- Control a 10 Hz, 5 subpasos de integración y colisión comprobada en cada
  subpaso.
- Retardo de primer orden con τ ∈ [0.05, 0.4] s.
- Aceleración máxima: lineal en [0.3, 1.0] m/s², angular en [1.0, 3.0] rad/s².
- Retardo de actuación de 1 paso y ganancia ×[0.9, 1.1].
- Los parámetros se aleatorizan por episodio.
- Ruido del LiDAR: gaussiano con σ ∈ [0.01, 0.02] m y 1% de rayos perdidos
  (alcance máximo).
- Ruido de la velocidad medida: 5%.
- La recompensa y la zanahoria usan la pose verdadera.

### 5.4 Recompensa (`reward.py`)

`RewardConfig` reúne todos los pesos. `compute_reward()` es una función pura que
devuelve el total y cada término por separado.

| término | valor |
|---|---|
| progreso | +1.0 × (metros de **nuevo récord** de `s` sobre la ruta estática) |
| llegada | +20 |
| choque | −20 (−10 en la versión inicial; ver `docs/resultados.md`) |
| tiempo | −0.005 por paso |
| proximidad (opcional) | 0 por defecto |
| giro brusco (opcional) | 0 por defecto |

El progreso solo se paga al superar el mejor `s` del episodio: cada metro se paga
una vez y oscilar no rinde.

### 5.5 Fin de episodio

- **Éxito:** distancia a la meta final < 0.3 m (terminal).
- **Choque:** el rectángulo 0.56 × 0.41 m del robot solapa una celda ocupada
  (terminal).
- **Truncado** (bootstrap del valor): tiempo límite de `3 × ruta / 0.35 + 10` s,
  o 15 s sin nuevo récord de progreso.

## 6. Red y entrenamiento (`learning/`)

### 6.1 Arquitecturas

**CNN-1D (principal):**

```
LiDAR (90) -> Conv1d 1->16 k5 circ -> ReLU -> Conv1d 16->32 k5 s2 circ -> ReLU
           -> Conv1d 32->32 k3 s2 circ -> ReLU -> Flatten -> Linear 128 -> ReLU
resto (6)  -> Linear 32 -> ReLU
concat (160) -> cabezas [256, 256] (actor -> (v, ω); crítico -> V)
```

**MLP (línea base):** los 96 valores entran a `[256, 256]`.

En ambos casos el actor y el crítico tienen extractores separados
(`share_features_extractor=False`).

### 6.2 PPO

| parámetro | valor |
|---|---|
| entornos | 16, `SubprocVecEnv` |
| `n_steps` | 512 (8192 transiciones por actualización) |
| `batch_size`, `n_epochs` | 256, 10 |
| `learning_rate` | 3e-4, decaimiento lineal |
| `gamma`, `gae_lambda` | 0.99, 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `log_std_init` | −0.5 |
| normalización | `VecNormalize(norm_obs=False, norm_reward=True)` |
| presupuesto | ~5 M de pasos por run |

La observación no se normaliza con estadísticas, así que el modelo se despliega
sin archivos extra.

### 6.3 Registro

Cada run crea una carpeta con:

- `config.yaml` completo (escenarios, dinámica, recompensa, PPO, semilla);
- `episodes.csv` con: resultado, cada término de recompensa, longitud de la ruta,
  SPL, número de obstáculos y fuente del escenario;
- logs de TensorBoard;
- `best_model.zip`, `last_model.zip` y las estadísticas de `VecNormalize`.

Cada 250k pasos se hace una evaluación determinista de 200 episodios con
semillas fijas; `best_model` se elige por su tasa de éxito.

## 7. ROS 2 y Gazebo

### 7.1 Robot de Gazebo (`urdf/martha.urdf.xacro`)

- Chasis: caja de 0.56 × 0.41 m.
- LiDAR a +0.2325 m en x: 360 rayos, 0.15–8 m, 10 Hz, ruido gaussiano de 0.01 m.
- Bumper en el chasis.
- Movimiento: `gazebo_ros_planar_move`, que recibe `/cmd_vel` y publica `/odom`
  y el TF `odom -> base_link`.

Valida el pipeline ROS, el LiDAR de Gazebo y los tiempos reales. **No valida la
dinámica de ruedas.**

### 7.2 `sim.launch.py`

Lanza:

- `gzserver` con `world` (por defecto `lab.world`) y `gzclient` opcional;
- `robot_state_publisher` y el spawn;
- `map_publisher` y `ground_truth_tf`;
- `planner_node` y `policy_node` (con `checkpoint:=`);
- RViz opcional.

### 7.3 Nodos

- **`map_publisher`:** rasteriza el `.world` con `scenarios.py` y publica `/map`
  (latched).
- **`ground_truth_tf`:** publica `map -> odom` para que `map -> base_link` sea la
  pose verdadera de Gazebo.
- **`planner_node`:**
  - recibe `/map`, `/goal_pose` y el TF `map -> base_link`;
  - hace el A* inflado 0.40 m con `planner.py`;
  - publica `/plan` (`nav_msgs/Path`) y `/nav_status` (`active` / `succeeded` /
    `failed`);
  - vuelve a planificar si cambia la meta o si el robot se aleja más de 1 m de la
    ruta;
  - marca `succeeded` a menos de 0.3 m de la meta y `failed` si no hay ruta.
- **`policy_node`:**
  - recibe `/plan`, `/scan`, `/odom` y el TF;
  - calcula la zanahoria (sección 4.4, `L` = 1.5 m);
  - construye la observación con `build_observation`;
  - ejecuta el modelo en CPU y publica `/cmd_vel` a 10 Hz.
- **Seguridad del `policy_node`:** publica velocidad cero si:
  - `/scan` u `/odom` tienen más de 0.3 s;
  - no hay ruta activa o el estado es `succeeded`;
  - algún punto del LiDAR entra en la huella + 5 cm.

### 7.4 `gazebo_eval.py`

Por episodio:

1. Teletransporta a Martha al inicio.
2. Genera los obstáculos sorpresa sobre `lab.world` con las reglas de la sección
   4.3 y `spawn_entity`.
3. Publica la meta y espera `succeeded`, choque (bumper) o timeout.
4. Borra los obstáculos.

Escribe un CSV con las mismas columnas que `evaluate.py`.

### 7.5 Robot real (contrato únicamente)

- **Entradas:** `/scan` (`LaserScan`, frame `lidar`) y `/odom` (`Odometry`).
- **Salida:** `/cmd_vel` (`Twist`, solo `linear.x` y `angular.z`).
- **TF:** `map -> odom` lo dará un localizador (p. ej. `slam_toolbox` en modo
  localización), además de `odom -> base_link` y `base_link -> lidar`.

Los nodos no cambian. El launch del robot real se diseña cuando el hardware
funcione.

## 8. Pruebas

Pytest, sin ROS:

- **geometry:** distancia de un rayo contra una caja conocida (±5 cm); colisión
  de la huella rotada; origen de los rayos en el offset del LiDAR.
- **planner:** hay ruta en un pasillo; no la hay si está bloqueado; se respeta la
  inflación; la proyección `s` y la zanahoria; el salto de obstáculos.
- **scenarios:** en 500 semillas todos los episodios con obstáculos cumplen el
  hueco ≥ 0.8 m y las distancias al inicio y a la meta; `lab.world` nunca es una
  fuente de entrenamiento.
- **dynamics:** convergencia a la velocidad comandada; límites de aceleración.
- **reward:** cada término; el progreso se paga una sola vez.
- **observation:** tamaño 96, valores en [−1, 1]; prueba golden: un scan dado como
  arrays y como `LaserScan` sintético (convertido por el adaptador de
  `policy_node`) produce la misma observación.
- **env:** `gymnasium.utils.env_checker.check_env`; reproducibilidad por semilla.

**Puerta de convergencia:** en una habitación abierta sin obstáculos, PPO debe
llegar a ≥ 80% de éxito en ≤ 500k pasos. No se añade dificultad hasta pasarla.

## 9. Experimentos

- **E1: CNN-1D frente a MLP.**
  - 3 semillas por brazo, ~5 M de pasos cada una.
  - Evaluación en 500 episodios de semillas reservadas, en condición limpia y con
    obstáculos.
  - Métricas: éxito, colisión, timeout y SPL, con IC95%.
- **E2: generalización y brecha de simulación.** El mejor modelo se evalúa en:
  - (a) semillas reservadas del 2D;
  - (b) `lab.world` rasterizado en 2D;
  - (c) `lab.world` en Gazebo, con 100–200 episodios y obstáculos sorpresa.

  La diferencia entre (b) y (c) decide si hace falta un fine-tune en Gazebo (spec
  aparte).
- **Opcionales:** término de proximidad encendido; comparación con DWA.

## 10. Orden de trabajo

Orientativo: define el orden y las puertas, no fechas. El usuario tiene otras
entregas (documento de tesis) que se planifican aparte.

| fase | entregable | puerta para pasar a la siguiente |
|---|---|---|
| 1 | `sim2d` + pruebas, rasterizador de `.world`, revisión de papers para las plantillas | `check_env` pasa; pasos/s medidos |
| 2 | `learning/`, primeros runs | puerta de convergencia |
| 3 | E1 en segundo plano; URDF, launch, nodos, `gazebo_eval.py` | Martha navega en Gazebo |
| 4 | E2, análisis de la brecha, decisión sobre el fine-tune | tablas de E1 y E2 |
| 5 | gráficas; robot real si el hardware está listo | — |

**Riesgo principal:** la demo real depende de reparar el hardware, lo cual está
fuera de este diseño. Plan B: defender con E1 y E2 más la demo de `lab.world` en
Gazebo.

## 11. Entorno de ejecución

Todo corre dentro del contenedor Docker existente (ROS 2 Humble, Python 3.10,
torch 2.2.2 + CUDA, gymnasium 0.29.1). Se añade `stable-baselines3` con una
versión compatible con gymnasium 0.29 (2.3.x). El entrenamiento corre en CPU
(16 procesos) y la red, que es pequeña, en CPU o GPU según lo que mida más rápido
el primer benchmark.
