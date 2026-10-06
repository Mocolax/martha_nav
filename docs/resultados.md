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

### E2 v4 (2026-09-29): las mismas reglas que el 2D y sin arrastre de la meta anterior

La revisión de código del 29/09 encontró tres diferencias entre las dos columnas de E2, ya
corregidas (commit `5bcf412`):

1. `gazebo_eval` no cancelaba la meta al terminar un episodio: tras el teleport, el planificador
   volvía a planificar hacia la meta **anterior** y el robot salía del punto de inicio antes de
   recibir la nueva.
2. El estancamiento se medía distinto: en Gazebo bastaba moverse 0.1 m para reiniciar el contador,
   en 2D hace falta batir el récord de avance sobre la ruta. Un robot que oscila o da vueltas
   estaba estancado en 2D pero no en Gazebo, y agotaba un timeout fijo de 120 s.
3. Los tiempos eran de reloj de pared, no simulados.

Ahora Gazebo usa las reglas de `NavEnv` (misma clase `RouteProgress`, mismo timeout según el largo
de la ruta) en tiempo simulado. Mismos episodios, pareados contra el 2D (`*_v4.csv`):

| condición | simulador | éxito [IC95] | colisión | estancado | timeout | sin ruta |
|---|---|---|---|---|---|---|
| semillas generadas (100) | 2D | 0.88 [0.80, 0.93] | 0.02 | 0.09 | 0.01 | 0.00 |
| semillas generadas (100) | Gazebo v4 | 0.79 [0.70, 0.86] | 0.01 | 0.17 | 0.03 | 0.00 |
| puntos fijos (90) | 2D | 0.72 [0.62, 0.80] | 0.06 | 0.22 | 0.00 | 0.00 |
| puntos fijos (90) | Gazebo v4 | 0.68 [0.58, 0.77] | 0.00 | 0.28 | 0.02 | 0.02 |

Comparación episodio a episodio (McNemar): con las semillas, 13 episodios salen bien solo en 2D y
4 solo en Gazebo (p = 0.049); con los puntos fijos, 11 contra 7 (p = 0.48). De los 13, 9 son
estancamientos, 3 timeouts y 1 colisión.

> **Conclusión revisada de E2:** medida con las mismas reglas, la transferencia 2D → Gazebo pierde
> **unos 9 puntos con las semillas generadas** (en el límite de la significancia) y 4 con los puntos
> fijos (no significativo). La pérdida es casi toda **estancamiento**, no choques: en Gazebo el robot
> se bloquea más a menudo, y las colisiones siguen siendo iguales o menores que en 2D. La conclusión
> anterior, "sin pérdida distinguible del ruido", venía en parte de la regla de estancamiento más
> laxa de Gazebo.

## Qué pasa cuando el robot se queda quieto contra una pared

Demostración de 10 episodios en `lab` con el brazo H (`runs/demo_worlds/demo_lab.csv`): 8 éxitos y
los episodios 8 y 9 (semillas 1000007 y 1000008) etiquetados `stalled`, con el robot inmóvil. El
registro del `policy_node` muestra la parada de seguridad disparándose **de forma continua** durante
los dos bloqueos, y solo una vez en los ocho episodios que sí terminaron:

| episodio | semilla | resultado | avisos de la parada de seguridad |
|---|---|---|---|
| 1–7, 10 | 1000000–1000006, 1000009 | éxito | 1 aviso aislado en total |
| 8 | 1000007 | estancado a los 15.7 s | `['front','left']`, luego `['rear','right']` sin interrupción 12 s |
| 9 | 1000008 | estancado a los 37.8 s | `['front','left']` sin interrupción 17 s |

**La parada no es la causa.** Las mismas dos semillas, jugadas en el simulador 2D, donde no existe
ninguna parada de seguridad, terminan en **colisión**. Las otras ocho terminan igual que en Gazebo.

| semilla | 2D, referencia | 2D, brazo H | 2D, brazo L | Gazebo, brazo H |
|---|---|---|---|---|
| 1000007 | colisión | colisión | colisión | estancado |
| 1000008 | estancado | colisión | éxito | estancado |

El mecanismo es este: la política manda avanzar contra la pared, el guardia anula la componente
lineal que chocaría y deja libre la angular, así que **el robot gira sobre su eje sin desplazarse**.
Como la regla de estancamiento mide desplazamiento (menos de 10 cm en 15 s), eso se registra como
`stalled`. En 2D ese mismo comando se ejecuta entero y el episodio se registra como colisión. La
parada convierte un choque en un bloqueo; no inventa un fallo nuevo.

La semilla 1000007 además es difícil por construcción: el arranque tiene **0.44 m de holgura** y el
radio circunscrito del robot es **0.347 m**, es decir, nueve centímetros de margen para girar. Los
tres modelos fallan en ella. Es un caso raro: sobre 200 semillas de evaluación en `lab`, solo el 1%
arranca con menos de 0.45 m de holgura y ninguna por debajo del inflado de 0.40 m del planificador.

> **Lectura:** el fallo que queda no es de percepción ni del guardia, sino de memoria. La
> observación sí contiene la evidencia del bloqueo (ordenó avanzar y la velocidad medida es cero),
> pero una política sin estado no puede encadenar esa evidencia entre pasos. Es el argumento para el
> brazo S, que añade la señal de atasco explícita a la observación.

## Causa de la brecha de E2: la actuación, no la percepción (2026-09-30)

Los fallos extra de Gazebo en E2 v4 eran atascos. El 2D siempre aplicaba el comando **un periodo
tarde** y con una dinámica lenta; el `mecanum_drive_controller` de Gazebo obedece en el mismo periodo
y acelera a 2.5 m/s² y 6.5 rad/s². Jugar el 2D sin retardo y con esas aceleraciones reproduce los
números de Gazebo (77 % en semillas, 64 % en puntos, con los atascos de Gazebo). El guardia, el
carrot y la odometría quedaron descartados.

**Confirmación en Gazebo:** el mismo modelo (`long_c_kl_s0`) con `action_delay:=1` en
`sim.launch.py`, que retiene cada comando un periodo como en el 2D
(`runs/long_c_kl_s0/eval_gazebo_lab{,_points}_delay1.csv`):

| conjunto | 2D | Gazebo (E2 v4) | Gazebo, `action_delay:=1` |
|---|---|---|---|
| semillas (100) | 88 % | 79 % | **87 %** |
| puntos (90) | 72 % | 68 % | **72 %** |

## Corrección: entrenar con dinámica amplia (`wide_dyn_s0`)

En vez de retrasar el robot real, se entrena para cubrir los dos casos: `train --wide-dynamics`
aleatoriza el retardo entre 0 y 2 periodos, `tau` 0.02–0.4 s y aceleraciones hasta 3 m/s² y
7 rad/s² (`WIDE_DYNAMICS`). Misma configuración que `long_c_kl_s0` en lo demás, 5M pasos.

Evaluación 2D estándar: limpio 97.2 %, con obstáculos 89.6 %, `lab` 85.5 %, puntos de `lab` 85.6 %.

E2 en Gazebo **sin** `action_delay` (`runs/wide_dyn_s0/eval_gazebo_lab{,_points}.csv`), pareado
(McNemar exacto):

| comparación | semillas (100) | puntos (90) |
|---|---|---|
| 2D → Gazebo, `wide_dyn_s0` | 84 → 83 % (7 vs 6, p = 1.0) | 86 → 88 % (6 vs 8, p = 0.79) |
| Gazebo, `long_c_kl_s0` → `wide_dyn_s0` | 79 → 83 % (4 vs 8, p = 0.39) | 68 → **88 %** (0 vs 18, p < 0.001) |

> **Lectura:** con dinámica amplia la brecha 2D → Gazebo desaparece sin retrasar los comandos, y en
> los puntos fijos el modelo nuevo gana 18 episodios sin perder ninguno. Los atascos en Gazebo bajan
> a 13 % (semillas) y 7 % (puntos); las colisiones quedan en 3–4 %. `wide_dyn_s0` pasa a ser el
> modelo de referencia, también para el robot real, cuya actuación no conocemos todavía.

## Localización con slam_toolbox frente a obstáculos fuera del mapa

Mapa de `lab.world` hecho con `tools/map_world.py` (`maps/lab.posegraph`), luego
`sim.launch.py slam:=localization` y `evaluate_gazebo` comparando la pose de SLAM con la verdadera
(`runs/long_c_kl_s0/eval_gazebo_lab_slam{_clean,}.csv`, modelo `long_c_kl_s0`, 100 semillas):

| condición | éxito | fallos | error de posición medio / máximo |
|---|---|---|---|
| sin obstáculos | 92 % | 8 atascos | 8.5 cm / 2.65 m (un episodio) |
| 2–4 cajas no mapeadas | 74 % | 23 atascos, 2 timeouts, 1 colisión | 6.9 cm / 0.85 m |

> **Lectura:** las cajas que no están en el mapa no hacen que slam_toolbox se pierda: el error medio
> se queda en centímetros y los fallos son atascos, como con la pose verdadera (79 % en E2 v4). El
> pico de 2.65 m sin obstáculos es un episodio aislado aún sin analizar. Queda repetir con
> `wide_dyn_s0`.

### Repetición con `wide_dyn_s0` y el evaluador corregido

Dos fallos del evaluador aparecieron con SLAM: `/initialpose` salía antes de que Gazebo mostrara el
robot en el inicio (el localizador leía el teletransporte como odometría y arrancaba desviado
metros), y el éxito se tomaba del planificador, que se guía por la pose estimada. Ahora la pose
inicial espera al teletransporte y el éxito exige que la pose **verdadera** quede a menos de 0.5 m
de la meta (si no, `lost`). Resultados en `runs/wide_dyn_s0/eval_gazebo_lab_slam{_clean,}_v2.csv`,
pareados con `long_c_kl_s0` en las mismas semillas:

| condición | `long_c_kl_s0` | `wide_dyn_s0` | ganados / perdidos | error medio | error máximo |
|---|---|---|---|---|---|
| sin obstáculos | 92 % | **97 %** | 6 / 1 (p = 0.13) | 2.5 cm | 9.5 cm |
| 2–4 cajas no mapeadas | 74 % | **87 %** | 17 / 4 (p = 0.007) | 6.2 cm | 4.5 m (1 episodio) |

> **Lectura:** sin obstáculos ningún episodio pasa de 10 cm: los errores de metros de antes eran
> del evaluador. Con cajas fuera del mapa quedan dos episodios desviados, las semillas 1000062
> (1.0 m, llega igual) y 1000078 (4.5 m, se atasca). Ambos tienen 4 cajas y se repiten igual antes
> y después del arreglo, así que son pérdidas reales de slam_toolbox cuando el mapa difiere mucho de
> lo que ve el LiDAR: 2 de 100 episodios. Ningún episodio terminó `lost`.

## Evaluación rápida en Gazebo (2026-10-04)

`tools/evaluate_run_gazebo.sh` corre 3 Gazebos en paralelo (cada uno con su `ROS_DOMAIN_ID` y
puerto de Gazebo) con la física sin límite (`gz physics -u 0`, ~3× tiempo real cada uno) y reparte
las semillas con `shard:=i/n`. Validación con `wide_dyn_s0`, 100 semillas en `lab`:

| corrida | éxito | mismo resultado que la original a 1× |
|---|---|---|
| original, 1× | 83 % | — |
| repetición, 1× | 87 % | 94 / 100 |
| rápida, ~3× | 84 % | 95 / 100 |

Acelerar no cambia los resultados más que repetir a 1×: Gazebo varía ~±4 puntos entre corridas
idénticas. Los 190 episodios de E2 pasan de ~1.5 h a ~13 min.

Con esto se re-evaluaron todas las configuraciones del paquete de datos de la tesis
(`tools/build_entrega.py`, resultados en `runs/<run>/v5/`): ver `entrega_tesis/resumen_modelos.csv`.
