# Resultados — martha_nav

Bitácora de resultados citables. Toda cifra apunta a su evidencia (carpeta de run o CSV).
Las configuraciones se leen del `config.yaml` del run, nunca del código fuente.

## Rendimiento del simulador 2D

- Entorno de un proceso: 935 pasos/s (medido con la tarea 9, paso 5; CPU i9-13900HX,
  perfil de energía `balanced`).

## Puerta de convergencia (open_room, sin obstáculos)

Run `runs/gate_cnn_s0` (CNN, semilla 0, 500k pasos, 10.1 min, perfil `balanced`).
Evaluación interna: 0.995 a los 250k pasos y 0.990 a los 500k.
Evaluación determinista, 200 episodios de semillas reservadas (`best_model.zip`, 250k pasos):
éxito 0.995 [IC95 0.972, 0.999], colisión 0.005, SPL 0.995.
Evidencia: `runs/gate_cnn_s0/eval_clean_open_room.csv`.

**Puerta superada** (umbral 0.80). La red aprende la navegación básica en ~250k pasos.

## Primer run completo (full_cnn_s0)

Run `runs/full_cnn_s0`: CNN, semilla 0, 5M pasos, todas las fuentes de entrenamiento, obstáculos
`mixed`. Duró 99.2 min (840 pasos/s, perfil `balanced`).

**Evaluación interna (200 episodios, mixed):** el máximo es **0.495 a 1.5M pasos**; después el éxito
cae hasta 0.39 y la colisión sube de 0.155 a 0.49. `best_model.zip` es el de 1.5M.

**Evaluación determinista de `best_model.zip`** (semillas reservadas):

| condición | episodios | éxito [IC95] | colisión | estancado | SPL |
|---|---|---|---|---|---|
| limpio (fuentes de entrenamiento) | 500 | 0.682 [0.640, 0.721] | 0.046 | 0.244 | 0.664 |
| con obstáculos (fuentes de entrenamiento) | 500 | 0.366 [0.325, 0.409] | 0.266 | 0.336 | 0.358 |
| con obstáculos, `lab` (nunca visto) | 200 | 0.390 [0.325, 0.459] | 0.360 | 0.215 | 0.383 |
| `last_model.zip`, con obstáculos | 500 | 0.270 [0.233, 0.311] | 0.566 | 0.150 | 0.267 |

**Episodios de entrenamiento por ventanas de 500k pasos** (`episodes.csv`):

| ventana | limpios: éxito | limpios: colisión | con obstáculos: éxito | con obstáculos: colisión | con obstáculos: estancado |
|---|---|---|---|---|---|
| 0–0.5M | 0.38 | 0.08 | 0.25 | 0.19 | 0.42 |
| 1.0–1.5M | 0.60 | 0.07 | 0.43 | 0.24 | 0.31 |
| 2.0–2.5M | 0.79 | 0.09 | 0.33 | 0.45 | 0.20 |
| 4.0–4.5M | 0.75 | 0.14 | 0.29 | 0.56 | 0.13 |

Las colisiones ocurren de media al 40% de la ruta, tras unos 203 pasos.

**Conclusión:** navegar en limpio sigue mejorando (0.38 → 0.79), pero esquivar se estanca hacia
1.5M pasos y luego **empeora**. La política cambia estancamientos (que no cuestan nada, porque se
truncan con bootstrap) por choques. `lab`, que nunca vio, rinde igual que las fuentes de entrenamiento
(0.39 frente a 0.37): **la generalización funciona; lo que falla es la evitación.** Hay que revisar la
recompensa antes de E1.

## Experimentos A / B / C (2M pasos cada uno, semilla 0)

Script: `tools/run_experiments_abc.sh`. Figura: `docs/figures/exp_abc.png`
(`tools/compare_runs.py`). Perfil `performance`, ~1 150 pasos/s.

| run | cambio frente al base | limpio | obstáculos | obstáculos, `lab` | colisión (obst.) | estancado (obst.) |
|---|---|---|---|---|---|---|
| base `full_cnn_s0` (mejor, 1.5M) | — | 0.682 | 0.366 [0.325, 0.409] | 0.390 | 0.266 | 0.336 |
| A `expA_col20` | choque −20 | 0.566 | 0.340 [0.300, 0.383] | 0.350 | 0.282 | 0.324 |
| B `expB_col20_stall5` | A + atasco terminal −5 | 0.274 | 0.148 [0.120, 0.182] | 0.230 | 0.266 | 0.408 |
| **C `expC_col20_inverse`** | A + LiDAR `d/(d+1)` | **0.910** | **0.712 [0.671, 0.750]** | **0.590** | **0.046** | 0.232 |

Evaluación determinista del `best_model.zip` de cada run: 500 episodios (200 en `lab`) con las mismas
semillas reservadas que el base.

**Conclusiones:**
- **La codificación del LiDAR era el cuello de botella.** Con `d/(d+1)`, el éxito con obstáculos pasa de
  0.37 a **0.71** (los IC95 no se solapan) y la colisión baja de 0.27 a 0.05. En `lab` (no visto) sube de
  0.39 a 0.59. A aísla el efecto de la recompensa (choque −20 solo no mejora), así que la ganancia de C
  viene de la codificación.
- **Choque −20 solo no ayuda** (A, dentro del error respecto al base, peor en limpio).
- **Hacer terminal el atasco es contraproducente** (B): la política no despega. Coincide con la §13 de
  la bitácora anterior: una penalización grande respecto al progreso inicial enseña a no moverse.
- **Queda un problema:** en C el mejor modelo es el de **250k pasos**; después la evaluación periódica
  baja de 0.765 a ~0.69 y la colisión sube de 0.025 a ~0.21. La degradación con el entrenamiento
  persiste, aunque en un nivel mucho más alto. En `lab`, el fallo dominante de C es el estancamiento (0.275).

## Por qué empeoraba con el entrenamiento: colapso del `std` y actualizaciones sin límite

Leído de TensorBoard (`runs/*/tb`):

| indicador | referencia en PPO | `full_cnn_s0` (1–3M pasos) | `expC_col20_inverse` (0.8–1.2M) |
|---|---|---|---|
| `approx_kl` | 0.01–0.02 | 1.2–2.0 | 0.13–0.17 |
| `clip_fraction` | 0.1–0.2 | 0.63–0.71 | 0.39–0.41 |
| `std` | baja poco a poco | 0.60 → 0.087 (1M) → 0.007 (5M) | 0.60 → 0.077 (2M) |

Con `ent_coef = 0` el ruido de la política colapsa. Con un `std` diminuto, cualquier cambio de la
media es un cambio enorme de probabilidad, y 10 épocas por lote aplican el cambio aunque la mayoría de
las muestras estén recortadas. Cada actualización reescribe la política: el mejor modelo aparece pronto
y después se degrada.

**Arreglo:** `target_kl = 0.02` en `PPO_PARAMS` (SB3 corta las épocas del lote cuando el KL lo supera).
Run de verificación: `runs/long_c_kl_s0` (configuración C + `target_kl`, 5M pasos). El run
`runs/long_c_s0_stopped` (sin `target_kl`) se detuvo a los 100k pasos, a favor de este.

### Verificación: `long_c_kl_s0` (configuración C + `target_kl = 0.02`, 5M pasos)

78 min, ~1 070 pasos/s, perfil `performance`. Figuras: `docs/figures/target_kl.png` (comparación) y
`docs/figures/long_c_kl_s0.png`.

**Indicadores de PPO, ya sanos:** `approx_kl` 0.015–0.022, `clip_fraction` 0.13–0.18, `std` baja de forma
gradual (0.60 → 0.38 → 0.30 → 0.25 → 0.19 → 0.15 cada 1M pasos) y `explained_variance` 0.85–0.90.

**Evaluación periódica:** estable entre 0.86 y 0.91 de éxito desde 2.5M pasos, con colisión 0.02–0.06.
**No hay degradación.** Mejor punto: 0.91 a 3.2M.

| modelo | limpio | obstáculos | obstáculos, `lab` (no visto) | colisión (obst. / `lab`) | estancado (obst.) |
|---|---|---|---|---|---|
| mejor (3.2M) | **0.958** [0.937, 0.972] | **0.868** [0.836, 0.895] | **0.865** [0.811, 0.906] | 0.012 / 0.025 | 0.116 |
| final (5M) | 0.976 [0.959, 0.986] | 0.878 [0.846, 0.904] | 0.780 [0.718, 0.832] | 0.042 / 0.125 | 0.078 |

Evolución del éxito con obstáculos (mejor modelo, mismas semillas):
base 0.37 → C 0.71 → **C + `target_kl` 0.87**. En `lab`: 0.39 → 0.59 → **0.87**.

**Conclusiones:**
- Con `target_kl` el entrenamiento largo **sí mejora** y ya no se degrada: era la condición para que
  "entrenar más" sirviera.
- La política generaliza a `lab` casi igual que a las fuentes de entrenamiento (0.865 frente a 0.868).
- El modelo final es algo más agresivo en `lab` (colisión 0.125 frente a 0.025): la selección del modelo
  por evaluación importa. **Modelo candidato para Gazebo: `runs/long_c_kl_s0/best_model.zip`.**
- El fallo restante dominante es el **estancamiento** (~0.1), no el choque.

## E2: brecha 2D → Gazebo en `lab.world`

Modelo `runs/long_c_kl_s0/best_model.zip`, tracción **mecanum** con ruedas y rodillos
simulados, física a 1 ms y tiempo real 1×. Los episodios son **los mismos en los dos
simuladores**: la semilla fija el mapa, el inicio, la meta y los obstáculos, así que la
comparación es pareada. Figura: `docs/figures/e2_2d_vs_gazebo.png`.

| condición | simulador | éxito [IC95] | colisión | estancado o timeout |
|---|---|---|---|---|
| semillas generadas (100) | 2D | 0.880 [0.802, 0.930] | 0.020 | 0.100 |
| semillas generadas (100) | **Gazebo** | **0.760 [0.668, 0.833]** | 0.000 | 0.220 |
| puntos fijos del paquete anterior (90) | 2D | 0.722 [0.622, 0.804] | 0.056 | 0.222 |
| puntos fijos del paquete anterior (90) | **Gazebo** | **0.678 [0.576, 0.765]** | 0.011 | 0.300 |

**Brecha pareada:** −12.0 puntos con las semillas generadas y −4.4 con los puntos fijos.
En las semillas, 71 episodios salen bien en ambos, 17 solo en 2D y 5 solo en Gazebo.

**La política no choca más en Gazebo; se queda quieta.** Las colisiones bajan (2.0% → 0.0% y
5.6% → 1.1%) y los bloqueos suben en la misma proporción (10% → 22% y 22% → 30%). La dinámica
real de las ruedas hace que empujar contra un obstáculo no termine en contacto reportado, sino
en un robot parado. Los episodios perdidos son además los de **ruta más larga** (6.9 m de media
frente a 6.2 m del total; 8.0 m frente a 6.7 m en los puntos fijos).

**Tiempos:** 25 s por episodio con éxito y 121 s los que agotan el tiempo límite.

**Salvedad metodológica:** el entorno 2D trunca un episodio tras 15 s sin progreso, y
`gazebo_eval` no tiene ese detector, así que espera hasta el timeout de 120 s. Por eso las dos
columnas de "estancado o timeout" no son exactamente la misma medida, aunque describen el mismo
fallo.

> **Conclusión citable:** la política entrenada en el simulador 2D **transfiere a Gazebo con una
> pérdida de 4 a 12 puntos** de éxito, sin aumentar las colisiones. El fallo dominante en ambos
> simuladores es quedarse bloqueado, no chocar.

### Corrección de E2: la brecha era nuestra propia parada de seguridad

Los 1 276 avisos de "obstáculo dentro de la huella" registrados durante E2 revelaron un fallo de
diseño: la protección del `policy_node` publicaba velocidad cero **en todas las direcciones**, así
que el robot quedaba congelado a 5 cm del obstáculo, sin poder girar ni retroceder, hasta agotar el
tiempo límite. Nunca llegaba a tocar, así que tampoco contaba como colisión.

Dos arreglos: la protección pasa a ser **direccional** (bloquea solo el movimiento que chocaría,
clasificando la intrusión por su eje dominante) y `gazebo_eval` corta el episodio tras **15 s sin
avanzar**, la misma regla que el entorno 2D.

Repetición con las 100 semillas, pareada episodio a episodio:

| | éxito [IC95] | colisión | bloqueo | sin ruta |
|---|---|---|---|---|
| 2D | 0.880 [0.802, 0.930] | 0.020 | 0.100 | 0.000 |
| Gazebo, antes | 0.760 [0.668, 0.833] | 0.000 | 0.220 | 0.020 |
| **Gazebo, corregido** | **0.880 [0.802, 0.930]** | 0.020 | 0.080 | 0.020 |

De los 22 episodios que antes agotaban el tiempo, **14 pasan a ser éxitos**, 7 siguen bloqueados y 1
resulta ser una colisión real. El tiempo medio por episodio baja de 46 a 36 s.

> **Conclusión revisada, que sustituye a la anterior:** la política entrenada en el simulador 2D
> **transfiere a Gazebo sin pérdida medible** (0.880 frente a 0.880, con la misma tasa de colisión).
> La brecha de 12 puntos que medimos primero no era del simulador: era un guardia de seguridad mal
> diseñado por nuestra parte. Queda pendiente repetir también la condición de puntos fijos.

### E2 final, con la regla de estancamiento activa

Segundo fallo encontrado al revisar los CSV: `gazebo_eval` tomaba la posición de `/odom`, que la
tracción mecanum **no publica** (su odometría va en `/mecanum_drive_controller/odometry`). La
posición quedaba en `None` y la regla de los 15 s sin avanzar no se activaba nunca: todos los
bloqueos agotaban los 120 s y se etiquetaban `timeout`. Ahora la posición viene de la pose real de
Gazebo, que existe con cualquier tracción.

Repetición final de las dos condiciones, pareada contra el 2D:

| condición | simulador | éxito [IC95] | colisión | estancado | timeout |
|---|---|---|---|---|---|
| semillas generadas (100) | 2D | 0.880 [0.802, 0.930] | 0.020 | 0.090 | 0.010 |
| semillas generadas (100) | Gazebo | 0.830 [0.745, 0.891] | 0.010 | 0.080 | 0.070 |
| puntos fijos (90) | 2D | 0.722 [0.622, 0.804] | 0.056 | 0.222 | 0.000 |
| puntos fijos (90) | Gazebo | 0.733 [0.634, 0.814] | 0.011 | 0.067 | 0.178 |

**Repetibilidad de Gazebo:** dos pasadas de los **mismos** 100 episodios dan 0.880 y 0.830, y
coinciden en 85 de 100. Es decir, el propio simulador tiene un ruido de ±5 puntos entre
repeticiones, del orden de la brecha que queremos medir.

> **Conclusión final de E2:** la política transfiere del simulador 2D a Gazebo **sin pérdida
> distinguible del ruido del propio simulador**: −5.0 puntos con las semillas generadas y +1.1 con
> los puntos fijos, con intervalos que se solapan ampliamente. Las colisiones son iguales o menores
> en Gazebo. Entrenar en 2D, unas 50 veces más rápido, no se paga con un peor resultado.

Queda un 7–18% de episodios que agotan el tiempo sin ser estancamiento: el robot se mueve, pero
avanza demasiado poco para llegar. Es el mismo fallo que en 2D, visto con otro reloj.
