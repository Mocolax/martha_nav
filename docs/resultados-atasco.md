# Ablación de la señal de atasco

Cuatro combinaciones de señal de atasco y memoria recurrente, con todo lo demás igual a
`long_c_kl_s0` (CNN, semilla 0, `(v, w)`, LiDAR `d/(d+1)`, choque -20, `target_kl` 0.02,
5M pasos). Éxito con IC95, evaluación determinista sobre las semillas reservadas.

| run | limpio | obstáculos | lab | lab, puntos fijos | estancado (obst.) | colisión (obst.) |
|---|---|---|---|---|---|---|
| sin señal, sin LSTM (referencia) | 0.958 [0.937, 0.972] | 0.868 [0.836, 0.895] | 0.865 [0.811, 0.906] | 0.722 [0.622, 0.804] | 0.116 | 0.012 |
| sin señal, con LSTM (brazo L) | 0.974 [0.956, 0.985] | 0.840 [0.805, 0.870] | 0.670 [0.602, 0.731] | 0.700 [0.599, 0.785] | 0.022 | 0.138 |
| con señal, sin LSTM (brazo S) | 0.956 [0.934, 0.971] | 0.832 [0.797, 0.862] | 0.765 [0.702, 0.818] | 0.811 [0.718, 0.879] | 0.112 | 0.052 |
| con señal, con LSTM (brazo SL) | 0.938 [0.913, 0.956] | 0.802 [0.765, 0.835] | 0.635 [0.566, 0.699] | 0.644 [0.541, 0.736] | 0.054 | 0.144 |

**Lectura:** los dos factores empeoran el resultado en `lab` y se suman: 0.865 sin ninguno, 0.765
con la señal, 0.670 con la LSTM y 0.635 con las dos. La señal no ayuda a la LSTM a desatascarse;
la LSTM sigue cambiando estancamientos por choques (0.290 de colisión en `lab` con las dos).
El estancamiento no es un fallo independiente del choque: es su versión segura. Todo lo que lo
reduce en estos brazos lo convierte en colisiones.

## Brazo G: solo la meta, sin zanahoria (`armG_goal_s0`)

Igual que el brazo H (holonómico `(vx, vy, w)`, CNN, semilla 0, 5M pasos) salvo el punto que ve
la política: la **meta** (distancia normalizada a 12 m) en vez de la zanahoria sobre la ruta A*.
La ruta se sigue usando en entrenamiento para medir el progreso de la recompensa, pero la política
no la ve, así que desplegada no necesita planificador global.

| condición | H éxito | G éxito | McNemar | H estancado | G estancado | H colisión | G colisión |
|---|---|---|---|---|---|---|---|
| limpio (500) | **0.982** | 0.880 | p < 0.0001 | 0.010 | 0.080 | 0.008 | 0.040 |
| obstáculos (500) | **0.912** | 0.820 | p < 0.0001 | 0.036 | 0.118 | 0.048 | 0.062 |
| `lab` semillas (200) | **0.815** | 0.560 | p < 0.0001 | 0.050 | **0.330** | 0.130 | 0.110 |
| `lab` puntos fijos (90) | **0.811** | 0.644 | p = 0.003 | 0.044 | **0.289** | 0.122 | 0.067 |

**Lectura:** sin la zanahoria la política aprende (0.82 en las fuentes de entrenamiento) pero
generaliza mucho peor: pierde 26 puntos en `lab` frente a los 10 del brazo H. El fallo nuevo es el
estancamiento (un tercio de los episodios en `lab`): sin la ruta no sabe rodear una pared que la
separa de la meta y se queda frente a ella. La colisión no sube. Es el argumento experimental
para mantener la arquitectura jerárquica: el A* aporta la topología del lugar, que el LiDAR solo
no ve más allá del primer obstáculo.

## Brazo GL: solo la meta, holonómico, con LSTM (`armGL_s0`)

Igual que el brazo G más una política recurrente (RecurrentPPO), sin señal de atasco. La
hipótesis: sin la ruta, la memoria podría recordar por dónde ya intentó pasar y rodear la pared.

| condición | G éxito | GL éxito | McNemar | G estancado | GL estancado | G colisión | GL colisión |
|---|---|---|---|---|---|---|---|
| limpio (500) | **0.880** | 0.812 | p < 0.001 | 0.080 | 0.086 | 0.040 | 0.102 |
| obstáculos (500) | **0.820** | 0.630 | p < 0.0001 | 0.118 | 0.120 | 0.062 | 0.250 |
| `lab` semillas (200) | **0.560** | 0.410 | p < 0.001 | 0.330 | 0.310 | 0.110 | 0.280 |
| `lab` puntos fijos (90) | **0.644** | 0.344 | p < 0.0001 | 0.289 | 0.344 | 0.067 | 0.311 |

**Lectura:** la hipótesis no se cumple. La memoria **no reduce el estancamiento** (0.31 contra
0.33 en `lab`) y más que duplica las colisiones. Esta vez ni siquiera cambia bloqueos por choques:
solo añade choques. La evaluación periódica seguía subiendo al final (máximo 0.69 a 4.75M), así que
el modelo podría estar algo corto de entrenamiento, pero la distancia a G (0.82 en esas mismas
evaluaciones desde los 2M pasos) es demasiado grande para cerrarla así.

## Resumen de todos los brazos en `lab` (nunca visto, 200 episodios con obstáculos)

| brazo | acción | objetivo | LSTM | señal | éxito | colisión | estancado |
|---|---|---|---|---|---|---|---|
| `long_c_kl_s0` (línea base) | `(v, w)` | zanahoria | — | — | **0.865** | **0.025** | 0.100 |
| `armH_holonomic_s0` (**presentación**) | `(vx, vy, w)` | zanahoria | — | — | 0.815 | 0.130 | **0.050** |
| `armS_stuck_s0` | `(v, w)` | zanahoria | — | sí | 0.765 | 0.080 | 0.155 |
| `armL_lstm_s0` | `(v, w)` | zanahoria | sí | — | 0.670 | 0.270 | 0.060 |
| `armSL_s0` | `(v, w)` | zanahoria | sí | sí | 0.635 | 0.290 | 0.075 |
| `armG_goal_s0` | `(vx, vy, w)` | meta | — | — | 0.560 | 0.110 | 0.330 |
| `armGL_s0` | `(vx, vy, w)` | meta | sí | — | 0.410 | 0.280 | 0.310 |

Tres conclusiones para la tesis: la zanahoria del A* es la pieza que más aporta (quitarla cuesta
25 puntos); la memoria recurrente empeora en todas las combinaciones probadas; y lo que reduce
el estancamiento casi siempre lo paga en colisiones.
