# Resultados — martha_nav

Bitácora de resultados citables. Toda cifra apunta a su evidencia (carpeta de run o CSV) y las
configuraciones se leen del `config.yaml` del run. Éxito con IC95 de Wilson; evaluación
determinista sobre semillas reservadas; las comparaciones pareadas usan McNemar exacto.

## Resumen

Éxito con obstáculos del mejor modelo de cada etapa, sobre las mismas semillas:

| etapa | run | cambio | fuentes de entrenamiento (500) | `lab` 2D (200) | `lab` Gazebo (100) |
|---|---|---|---|---|---|
| primer run | `full_cnn_s0` | — | 0.366 | 0.390 | — |
| C | `expC_col20_inverse` | LiDAR `d/(d+1)`, choque −20 | 0.712 | 0.590 | — |
| C + `target_kl` | `long_c_kl_s0` | PPO con actualizaciones limitadas | 0.868 | 0.865 | 0.79 |
| **modelo final** | **`wide_dyn_s0`** | dinámica amplia | **0.896** | **0.855** | **0.87** |

`lab` es el gemelo del laboratorio y nunca se usó para entrenar ni para elegir el modelo: es la
medida de generalización. `wide_dyn_s0` transfiere a Gazebo sin pérdida significativa (sección 6).
Con la torre montada, `policies/martha.npz` es `martha_tower_scratch_s0`, la misma receta
entrenada desde cero con el perfil `martha_tower` (sección 9).

## 1. Puerta de convergencia (`open_room`, sin obstáculos)

`runs/gate_cnn_s0` (CNN, semilla 0, 500k pasos, 10 min): éxito **0.995** [0.972, 0.999], colisión
0.005, SPL 0.995 en 200 episodios (`eval_clean_open_room.csv`). La red aprende a navegar en ~250k
pasos; umbral de la puerta, 0.80. El simulador 2D da 935 pasos/s en un proceso (i9-13900HX).

## 2. Primer run completo (`full_cnn_s0`)

CNN, semilla 0, 5M pasos, todas las fuentes de entrenamiento, obstáculos `mixed` (99 min, 840
pasos/s). Figura: `docs/figures/full_cnn_s0.png`.

| condición | episodios | éxito [IC95] | colisión | estancado | SPL |
|---|---|---|---|---|---|
| limpio (fuentes de entrenamiento) | 500 | 0.682 [0.640, 0.721] | 0.046 | 0.244 | 0.664 |
| con obstáculos (fuentes de entrenamiento) | 500 | 0.366 [0.325, 0.409] | 0.266 | 0.336 | 0.358 |
| con obstáculos, `lab` | 200 | 0.390 [0.325, 0.459] | 0.360 | 0.215 | 0.383 |

La evaluación periódica llega a su máximo (0.495) a 1.5M pasos y luego cae: la colisión sube de
0.155 a 0.49. Navegar en limpio sigue mejorando (0.38 → 0.79 en los episodios de entrenamiento),
pero esquivar se estanca y después empeora. `lab` rinde igual que las fuentes de entrenamiento
(0.39 frente a 0.37): **la generalización funciona; lo que falla es la evitación.**

## 3. Experimentos A / B / C (2M pasos, semilla 0)

Figura: `docs/figures/exp_abc.png` (`tools/compare_runs.py`). Mismas semillas reservadas que el base.

| run | cambio frente al base | limpio | obstáculos [IC95] | `lab` | colisión (obst.) | estancado (obst.) |
|---|---|---|---|---|---|---|
| base `full_cnn_s0` | — | 0.682 | 0.366 [0.325, 0.409] | 0.390 | 0.266 | 0.336 |
| A `expA_col20` | choque −20 | 0.566 | 0.340 [0.300, 0.383] | 0.350 | 0.282 | 0.324 |
| B `expB_col20_stall5` | A + atasco terminal −5 | 0.274 | 0.148 [0.120, 0.182] | 0.230 | 0.266 | 0.408 |
| **C `expC_col20_inverse`** | A + LiDAR `d/(d+1)` | **0.910** | **0.712 [0.671, 0.750]** | **0.590** | **0.046** | 0.232 |

- **La codificación del LiDAR era el cuello de botella.** Con `d/(d+1)` el éxito con obstáculos pasa
  de 0.37 a 0.71 (los IC95 no se solapan) y la colisión baja de 0.27 a 0.05. A muestra que el choque
  −20 por sí solo no mejora, así que la ganancia de C viene de la codificación.
- **Un atasco terminal es contraproducente** (B): una penalización grande frente al progreso inicial
  enseña a no moverse.
- En C el mejor modelo sigue apareciendo pronto (250k pasos) y después se degrada (sección 4).

## 4. Estabilidad de PPO: `target_kl`

Leído de TensorBoard (`runs/*/tb`):

| indicador | referencia en PPO | `full_cnn_s0` (1–3M) | `expC_col20_inverse` (0.8–1.2M) |
|---|---|---|---|
| `approx_kl` | 0.01–0.02 | 1.2–2.0 | 0.13–0.17 |
| `clip_fraction` | 0.1–0.2 | 0.63–0.71 | 0.39–0.41 |
| `std` | baja poco a poco | 0.60 → 0.087 (1M) → 0.007 (5M) | 0.60 → 0.077 (2M) |

Con `ent_coef = 0` el ruido de la política colapsa; con un `std` diminuto cualquier cambio de la
media es un cambio enorme de probabilidad, y 10 épocas por lote lo aplican entero. Cada
actualización reescribe la política. **Arreglo:** `target_kl = 0.02` en `PPO_PARAMS` (SB3 corta
las épocas del lote cuando el KL lo supera).

### Verificación: `long_c_kl_s0` (configuración C + `target_kl = 0.02`, 5M pasos)

78 min, ~1 070 pasos/s. Figuras: `docs/figures/target_kl.png` y `docs/figures/long_c_kl_s0.png`.
Indicadores sanos: `approx_kl` 0.015–0.022, `clip_fraction` 0.13–0.18, `std` 0.60 → 0.15 de forma
gradual, `explained_variance` 0.85–0.90. La evaluación periódica queda estable entre 0.86 y 0.91
desde 2.5M pasos, **sin degradación**.

| modelo | limpio | obstáculos | `lab` | colisión (obst. / `lab`) | estancado (obst.) |
|---|---|---|---|---|---|
| mejor (3.2M) | **0.958** [0.937, 0.972] | **0.868** [0.836, 0.895] | **0.865** [0.811, 0.906] | 0.012 / 0.025 | 0.116 |
| final (5M) | 0.976 [0.959, 0.986] | 0.878 [0.846, 0.904] | 0.780 [0.718, 0.832] | 0.042 / 0.125 | 0.078 |

Con `target_kl` entrenar más sí mejora, y la política generaliza a `lab` igual que a las fuentes de
entrenamiento (0.865 frente a 0.868). El fallo dominante restante es el estancamiento, no el choque.

## 5. Acción holonómica y memoria recurrente

Brazos comparados contra `long_c_kl_s0`, 5M pasos, todo lo demás igual; comparación pareada.

### Brazo H: acción holonómica `(vx, vy, w)`

`runs/armH_holonomic_s0`, 64 min.

| condición | referencia `(v, w)` | brazo H | Δ | McNemar |
|---|---|---|---|---|
| limpio (500) | 0.958 | **0.982** | +0.024 | — |
| obstáculos, fuentes de entrenamiento (500) | 0.868 | **0.912** | **+0.044** | **p = 0.005** |
| `lab`, semillas (200) | **0.865** | 0.815 | −0.050 | p = 0.087 |
| `lab`, puntos fijos (90) | 0.722 | **0.811** | +0.089 | p = 0.077 |

**La holonomía resuelve el estancamiento y lo paga en colisiones.** Los bloqueos caen en todas las
condiciones (puntos fijos: 0.222 → 0.044) y las colisiones suben (`lab`: 0.025 → 0.130). Moverse de
lado saca al robot de un hueco estrecho, pero presenta su lado largo (0.56 m) al obstáculo. Solo la
mejora en las fuentes de entrenamiento es significativa; en `lab` los signos se oponen según el
conjunto.

### Brazo L: política recurrente (LSTM)

`runs/armL_lstm_s0`, `RecurrentPPO` de sb3-contrib, acciones `(v, w)`.

| condición | referencia | brazo H | brazo L |
|---|---|---|---|
| limpio (500) | 0.958 | **0.982** | 0.974 |
| obstáculos, entrenamiento (500) | 0.868 | **0.912** | 0.840 |
| `lab`, semillas (200) | **0.865** | 0.815 | 0.670 |
| `lab`, puntos fijos (90) | 0.722 | **0.811** | 0.700 |

**El LSTM empeora chocando:** colisión 0.14 frente a 0.01 en las fuentes de entrenamiento y 0.27
frente a 0.03 en `lab`, con buen rendimiento en limpio. La tarea es casi markoviana (obstáculos
estáticos; velocidad y acción anterior ya en la observación), así que la memoria aporta poco y
añade parámetros que ajustar con el mismo presupuesto. Se descarta.

## 6. Transferencia 2D → Gazebo en `lab.world`

Tracción **mecanum** con ruedas y rodillos simulados. La semilla fija mapa, inicio, meta y
obstáculos, así que los episodios son los mismos en los dos simuladores y la comparación es
pareada: 100 semillas generadas y 90 puntos fijos del paquete anterior. Figura:
`docs/figures/e2_2d_vs_gazebo.png`.

| medición | modelo | semillas: 2D → Gazebo | puntos: 2D → Gazebo |
|---|---|---|---|
| primera | `long_c_kl_s0` | 88 → 76 % | 72 → 68 % |
| guarda direccional | `long_c_kl_s0` | 88 → 88 % | — |
| mismas reglas que el 2D (v4) | `long_c_kl_s0` | 88 → 79 % (13 vs 4, p = 0.049) | 72 → 68 % (11 vs 7, p = 0.48) |
| `action_delay:=1` | `long_c_kl_s0` | 88 → 87 % | 72 → 72 % |
| dinámica amplia | `wide_dyn_s0` | 84 → 83 % (7 vs 6, p = 1.0) | 86 → 88 % (6 vs 8, p = 0.79) |
| **re-evaluación v5** | **`wide_dyn_s0`** | **84 → 87 % (4 vs 7, p = 0.55)** | **85.6 → 91.1 % (4 vs 9, p = 0.27)** |

Correcciones del evaluador y del nodo que explican la secuencia:

1. **Guarda direccional.** La parada de seguridad publicaba velocidad cero en todas las direcciones
   y dejaba al robot congelado a 5 cm del obstáculo hasta el timeout. Ahora bloquea solo el
   movimiento que chocaría: de los 22 episodios que agotaban el tiempo, 14 pasan a éxitos.
2. **Mismas reglas que `NavEnv`** (commit `5bcf412`): misma clase `RouteProgress` para el
   estancamiento, timeout según el largo de la ruta, tiempo simulado, cancelación de la meta
   anterior entre episodios y posición desde la pose real de Gazebo. La pérdida que queda en v4 es
   casi toda estancamiento; las colisiones son iguales o menores que en 2D.

### Causa de la brecha: la actuación, no la percepción

El 2D aplicaba el comando **un periodo tarde** y con una dinámica lenta; el
`mecanum_drive_controller` obedece en el mismo periodo y acelera a 2.5 m/s² y 6.5 rad/s². Jugar el
2D sin retardo y con esas aceleraciones reproduce los números de Gazebo (77 % y 64 %), y retener
cada comando un periodo en Gazebo (`action_delay:=1`) recupera los del 2D. El guardia, el *carrot*
y la odometría quedaron descartados.

### Corrección: entrenar con dinámica amplia (`wide_dyn_s0`)

`train --wide-dynamics` aleatoriza el retardo entre 0 y 2 periodos, `tau` entre 0.02 y 0.4 s y
aceleraciones hasta 3 m/s² y 7 rad/s² (`WIDE_DYNAMICS`); el resto es igual que `long_c_kl_s0`, 5M
pasos. Evaluación 2D: limpio 97.2 %, con obstáculos 89.6 %, `lab` 85.5 %, puntos de `lab` 85.6 %.

En Gazebo, sin retrasar los comandos, la brecha desaparece. Frente a `long_c_kl_s0` gana en los
puntos fijos 18 episodios sin perder ninguno (68 → 88 %, p < 0.001). `wide_dyn_s0` es el modelo de
referencia, también para el robot real.

### Evaluación rápida y repetibilidad de Gazebo

`tools/evaluate_run_gazebo.sh` corre 3 Gazebos en paralelo, cada uno con su `ROS_DOMAIN_ID`, con la
física sin límite (~3× tiempo real) y reparte las semillas con `shard:=i/n`: los 190 episodios
pasan de ~1.5 h a ~13 min. Con `wide_dyn_s0`, 100 semillas:

| corrida | éxito | mismo resultado que la original |
|---|---|---|
| original, 1× | 83 % | — |
| repetición, 1× | 87 % | 94 / 100 |
| rápida, ~3× | 84 % | 95 / 100 |

Acelerar no cambia más que repetir: Gazebo varía ~±4 puntos entre corridas idénticas. Con este
script se re-evaluaron todas las configuraciones del paquete de datos (`tools/build_entrega.py`,
`runs/<run>/v5/`); son las cifras v5 de la tabla anterior.

## 7. El robot quieto contra una pared

Demo de 10 episodios en `lab` con el brazo H (`runs/demo_worlds/demo_lab.csv`): 8 éxitos y 2
`stalled` (semillas 1000007 y 1000008) con la guarda disparándose sin interrupción 12–17 s. La guarda
no es la causa: en el 2D, que no tiene guarda, esas dos semillas terminan en colisión. La política
manda avanzar contra la pared, la guarda anula la componente lineal y deja la angular, y el robot
gira sobre su eje: **la guarda convierte un choque en un bloqueo**, no inventa un fallo nuevo. La
semilla 1000007 arranca con 0.44 m de holgura para un radio circunscrito de 0.347 m; solo el 1 % de
las semillas de `lab` arranca con menos de 0.45 m.

## 8. Localización con slam_toolbox

Mapa de `lab.world` hecho con `tools/map_world.py`; `sim.launch.py slam:=localization` y
`evaluate_gazebo` comparando la pose de SLAM con la verdadera. El éxito exige que la pose
**verdadera** quede a menos de 0.5 m de la meta. 100 semillas, pareadas
(`runs/wide_dyn_s0/eval_gazebo_lab_slam{_clean,}_v2.csv`):

| condición | `long_c_kl_s0` | `wide_dyn_s0` | ganados / perdidos | error medio | error máximo |
|---|---|---|---|---|---|
| sin obstáculos | 92 % | **97 %** | 6 / 1 (p = 0.13) | 2.5 cm | 9.5 cm |
| 2–4 cajas no mapeadas | 74 % | **87 %** | 17 / 4 (p = 0.007) | 6.2 cm | 4.5 m (1 episodio) |

Las cajas fuera del mapa no hacen que slam_toolbox se pierda: el error medio queda en centímetros.
Solo dos episodios con 4 cajas se desvían (1.0 m, que llega igual, y 4.5 m, que se atasca), y
ninguno termina `lost`. Con localización por SLAM el modelo final rinde como con la pose verdadera.

## 9. Martha con la torre (`martha_tower_scratch_s0`)

La torre de aluminio puso cuatro postes de 20 × 40 mm en el plano del LiDAR y lo movió a 0.19 m
delante de `base_link` (perfil `martha_tower`). La observación pierde los sectores que tapan los
postes, así que `wide_dyn_s0` no sirve tal cual. `martha_tower_scratch_s0` repite su receta
desde cero (`--preset full --robot martha_tower --wide-dynamics`, 5M pasos, semilla 0): 71 min,
~1 180 pasos/s. Pasa del 85 % en 600k pasos y del 90 % en 2M; el mejor punto, 91 % en 4.1M, es el
exportado. En los últimos 1.5M pasos se mueve entre 87.5 y 91 %, dentro del ruido de 200
episodios. Figuras: `docs/figures/martha_tower_scratch_s0.png` y
`docs/figures/martha_tower_scratch_s0_ppo.png`.

Evaluación 2D sobre las mismas semillas que `wide_dyn_s0`, que se evalúa sin postes:

| evaluación | `wide_dyn_s0` | `martha_tower_scratch_s0` | ganados / perdidos |
|---|---|---|---|
| limpio (500) | 97.2 % | **98.2 %** | 12 / 7 (p = 0.36) |
| obstáculos (500) | 89.6 % | **88.8 %** | 30 / 34 (p = 0.71) |
| `lab` (200) | 85.5 % | **82.5 %** | 12 / 18 (p = 0.36) |
| puntos de `lab` (90) | 85.6 % | **81.1 %** | 5 / 9 (p = 0.42) |

Ninguna diferencia es significativa: con los postes en la observación, el modelo rinde como el de
referencia sin ellos. Las colisiones en `lab` suben de 7.5 a 11.5 %, y en sus puntos fijos de 6.7
a 14.4 %.

En Gazebo (`tools/evaluate_run_gazebo.sh`, `runs/martha_tower_scratch_s0/v5/`), con la torre en el
URDF, frente a la re-evaluación v5 de `wide_dyn_s0` sin ella:

| condición | `wide_dyn_s0` | `martha_tower_scratch_s0` | ganados / perdidos |
|---|---|---|---|
| semillas de `lab` (100) | 87 % | **84 %** | 4 / 7 (p = 0.55) |
| puntos fijos de `lab` (90) | 91.1 % | **82.2 %** | 2 / 10 (p = 0.04) |

En los puntos fijos la pérdida es significativa: de los 10 episodios perdidos, 5 son choques y 5
estancamientos, y las colisiones pasan de 3 a 7–8 % en los dos conjuntos. Los dos episodios
`failed` de los puntos fijos fallan igual con los dos modelos. La comparación no es entre modelos
sobre el mismo robot: `wide_dyn_s0` se evalúa en un robot que ve hacia atrás y no puede conducir
a Martha con la torre, así que no separa cuánto cuesta la torre y cuánto el entrenamiento.
