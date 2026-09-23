
## Cola nocturna del 2026-09-23

Brazos comparados contra la referencia `long_c_kl_s0`. Éxito con IC95, evaluación determinista.

### Brazo H: acción holonómica `(vx, vy, w)`

Run `runs/armH_holonomic_s0`, 5M pasos, 64 min. Todo lo demás igual que la referencia
`long_c_kl_s0`. Evaluación determinista, comparación **pareada** episodio a episodio:

| condición | referencia `(v, w)` | brazo H `(vx, vy, w)` | Δ | McNemar |
|---|---|---|---|---|
| obstáculos, fuentes de entrenamiento (500) | 0.868 | **0.912** | **+0.044** | **p = 0.005** |
| `lab`, semillas generadas (200) | **0.865** | 0.815 | −0.050 | p = 0.087 |
| `lab`, puntos fijos del paquete anterior (90) | 0.722 | **0.811** | +0.089 | p = 0.077 |
| limpio (500) | 0.958 | **0.982** | +0.024 | — |

**La holonomía resuelve el estancamiento, y lo paga en colisiones.** Los bloqueos caen en todas
las condiciones, de forma muy marcada en los puntos fijos (0.222 → **0.044**), mientras que las
colisiones suben (en `lab`, 0.025 → 0.130). En el entrenamiento se ve el mismo canje: último
millón de pasos, estancamiento 6.1% → 3.5% y colisión 3.9% → 8.2%.

**Interpretación:** moverse de lado es justo la maniobra que saca al robot de un hueco estrecho,
y por eso desaparecen los atascos. Pero al desplazarse lateralmente el robot presenta su lado
largo (0.56 m) al obstáculo y la política, entrenada con la misma penalización de choque, aún no
ha aprendido a respetar ese margen.

**Pendiente:** el único resultado significativo es el de las fuentes de entrenamiento (+4.4,
p = 0.005). En `lab` los dos signos son opuestos según el muestreo de episodios, así que conviene
una segunda semilla antes de decidir. Un candidato claro de ajuste es subir la penalización de
choque para el espacio holonómico.

### Brazo L: política recurrente (LSTM)

Run `runs/armL_lstm_s0`, 5M pasos con `RecurrentPPO` de sb3-contrib, acciones `(v, w)`.

| condición | referencia | brazo H | **brazo L** |
|---|---|---|---|
| limpio (500) | 0.958 | **0.982** | 0.974 |
| obstáculos, entrenamiento (500) | 0.868 | **0.912** | 0.840 |
| `lab`, semillas (200) | **0.865** | 0.815 | 0.670 |
| `lab`, puntos fijos (90) | 0.722 | **0.811** | 0.700 |

**El LSTM empeora, y empeora chocando.** Su tasa de colisión es la más alta de los tres en todas
las condiciones con obstáculos: 0.14 frente a 0.01 de la referencia en las fuentes de
entrenamiento, y 0.27 frente a 0.03 en `lab`. En limpio rinde bien (0.974), así que no es que
navegue peor: es que evita peor.

**Interpretación:** la tarea es casi markoviana (obstáculos estáticos, velocidad y acción anterior
ya en la observación), así que la memoria aporta poco, y a cambio la política recurrente tiene más
parámetros que ajustar con el mismo presupuesto de pasos. Coincide con lo que la bitácora anterior
sugería: el LSTM era parte de la complejidad que no pagaba.

> **Recomendación:** descartar el LSTM. El brazo H es el que merece seguimiento.

### Caída del equipo durante el brazo HL

El brazo HL (LSTM + holonómico) arrancó a las 04:23 y el equipo se apagó a las 05:39. El registro
del kernel muestra errores `NV_ERR_NO_MEMORY` de NVIDIA a las 04:22–04:23, justo al arrancar:
entrenaba en GPU mientras Gazebo también la usaba. No se perdió ningún resultado, porque todo
estaba commiteado, pero el run quedó incompleto.

**Medida:** los entrenamientos recurrentes se lanzan con `--device cpu`, y no se solapan con una
sesión de Gazebo abierta.
