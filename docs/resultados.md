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
