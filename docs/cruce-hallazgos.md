# Cruce con la bitácora anterior (`martha/HALLAZGOS_TESIS.md`)

Qué queda en pie, qué se explica y qué se cae de los hallazgos del montaje anterior
(PPO propio, LSTM, brújula BFS, Gazebo) a la luz de `martha_nav` (`docs/resultados.md`).

**Advertencia de método:** los dos montajes difieren en muchas cosas a la vez (simulador 2D frente a
Gazebo, PPO de SB3 frente a implementación propia, sin LSTM, 90 sectores frente a 36, recompensa de 4
términos, waypoint de A* que esquiva lo que ve el LiDAR). Por eso, **una diferencia de resultado entre
los dos montajes no aísla ninguna causa**. La única comparación limpia es dentro de `martha_nav`, donde
los experimentos A/B/C cambian una cosa cada vez.

## 1. El techo del ~45% se explica: era la codificación del LiDAR

La bitácora dejaba abierto (§8, §11, §15) un techo reproducible de ~45% de éxito con obstáculos, sin
saber si era límite "de la observación o del objetivo".

En `martha_nav`, el experimento C aísla exactamente eso: misma recompensa, mismo entorno, misma red,
**solo cambia `d/8 m` por `d/(d+1 m)`**. El éxito con obstáculos pasa de 0.37 a 0.71, y la colisión de
0.27 a 0.05.

El código anterior usaba **la misma normalización lineal**:
`sector_values / valid_max` con `valid_max = 8.0` (`martha/PPO/observations.py:73`), sobre **36
sectores de 10°**. Con esa escala, un obstáculo a 0.3 m y otro a 0.5 m se diferencian en 0.025 de un
rango de 0 a 1: la red apenas puede distinguir "cerca" de "muy cerca" justo donde decide si esquiva.

> **Respuesta a la pregunta abierta de §15:** era la observación, y en concreto su escala. No hacía
> falta cambiar el objetivo.

Nota fina: la distancia a la meta **sí** usaba `d / (d + 3)` en el código viejo
(`observations.py:109`). La codificación saturante estaba ahí para la meta, pero no para el LiDAR.

## 2. Se matiza §10 (ablación del LiDAR)

§10 concluía: *"el fallo de esquiva no es de percepción... lo que falta está en el objetivo, no en la
observación"*, porque cegar el LiDAR derrumbaba la política.

Ese experimento demuestra que la política **usaba** el LiDAR, y eso sigue en pie. Pero no podía
detectar lo contrario: que la **resolución efectiva** de esa entrada fuera insuficiente de cerca. Una
entrada puede ser imprescindible y a la vez estar mal escalada. La conclusión correcta es más precisa:

> El fallo no era de *información ausente*, sino de *resolución de la información presente*.

## 3. Se cae §2: aprender a navegar y esquivar desde cero **sí** funciona

§2 concluía, tras replicarlo en dos máquinas, que entrenar desde cero con obstáculos era un fallo de
arranque (éxito 3–4%, estancamiento 60–71%), y §3 proponía el enfoque por etapas.

`runs/long_c_kl_s0` entrena **desde pesos aleatorios**, con obstáculos en ~80% de los episodios, y
llega a 0.87 de éxito con obstáculos y 0.96 sin ellos. No hay etapas ni currículum.

Lo que no se puede afirmar es *cuál* de las diferencias lo permite. Candidatos, en orden de sospecha:
la codificación del LiDAR (§1 de este documento), que el waypoint **salte los obstáculos que ve el
LiDAR** en vez de apuntar a través de ellos (§11c de la bitácora describía justo ese daño), y que no
haya ninguna penalización de holgura que castigue moverse al principio (§13).

## 4. Se confirma §13 (y §2): una penalización fuerte al principio enseña a no moverse

§13: la holgura direccional a escala 0.20 desde cero subió el estancamiento al 89.5%.
Lección citada: *"la magnitud correcta... es una fracción del progreso que la política es capaz de
conseguir"*.

`martha_nav` lo reproduce con otro término distinto: el experimento **B** (atasco terminal de −5) hunde
el éxito a 0.15 y sube el estancamiento a 0.41. Mismo mecanismo, penalización distinta.

> Vale como resultado replicado en dos montajes independientes: **es una propiedad del problema, no de
> una implementación.**

## 5. Se confirma §6: los premios intermedios son explotables

§6 mostró que pagar la meta "al tocar" se volvía explotable en ~500 episodios.
`martha_nav` adoptó desde el diseño el progreso **por récord** (cada metro se paga una vez) y no
presenta ese comportamiento. Es una confirmación por construcción, no un experimento nuevo.

## 6. Se confirma §12: las evaluaciones pequeñas son ruido

§12: con n = 10 el intervalo binomial es de ±15 puntos.
`martha_nav` usa 200 episodios durante el entrenamiento y 500 al final, todos con semillas reservadas.
En la práctica se notó: en el primer run completo, una ventana de entrenamiento sugería una tendencia
que la evaluación determinista no confirmó.

## 7. Queda en duda §11d: el horizonte de crédito no era el cuello de botella

§11d argumentaba que con γ = 0.997 y λ = 0.95 el horizonte efectivo de GAE (~19 pasos ≈ 0.8 m) no
alcanzaba para iniciar un desvío, y §14 probó λ = 0.99 sin efecto significativo.

`martha_nav` usa γ = 0.99 y λ = 0.95, es decir, un horizonte **todavía más corto** (~17 pasos), y llega
a 0.87 con obstáculos. El horizonte corto no impide aprender a esquivar en este montaje.

## 8. Fallo nuevo, ausente en la bitácora anterior: colapso del `std`

`martha_nav` sufrió un fallo que el montaje viejo **no tenía**: con `ent_coef = 0` y sin `target_kl`,
el ruido de la política colapsó (0.60 → 0.007) y el KL por actualización llegó a 1–2.

El código anterior ya se protegía de esto: `target_kl = 0.03`, `entropy_coef = 0.002` y un suelo de
`policy_std` en 0.25 (`martha/PPO/train.py`). De hecho, §11 reportaba `approx_kl` 0.0060–0.0072 y
`policy_std` fijo en 0.293: **su optimizador estaba sano y aun así no avanzaba**, lo que refuerza que su
límite estaba en la observación y no en la optimización.

> Lección cruzada: al reescribir, se perdieron protecciones que el código viejo ya tenía. Conviene
> revisar qué más de esa lista conviene recuperar (por ejemplo, un suelo de `std`).

## 9. Lo que sigue siendo válido de §7 (errores metodológicos)

Las seis correcciones de §7 se adoptaron desde el diseño de `martha_nav`:

| error de §7 | cómo se evita ahora |
|---|---|
| guía en marco equivocado | una sola función construye la observación, con prueba propia |
| arenas sin acotar | el mundo se rasteriza al rectángulo de sus paredes; fuera del grid es ocupado |
| sin normalización de recompensa | `VecNormalize(norm_reward=True)`; `explained_variance` 0.85–0.90 |
| obstáculos encima del robot | distancia mínima al inicio (1.0 m) y a la meta (0.6 m), con prueba de 500 semillas |
| `install/` desactualizado | se ejecuta desde el código fuente con `./tools/ct`, sin `colcon` de por medio |
| barra del "mejor modelo" heredada | cada run tiene su propia barra, y la evaluación es de 200 episodios |

## 10. Tabla resumen

| hallazgo anterior | estado |
|---|---|
| §1 navegador 72% sin obstáculos | superado: 0.958 |
| §2 desde cero con obstáculos no funciona | **se cae**: 0.87 desde cero |
| §3 el enfoque por etapas es necesario | ya no hace falta |
| §4 la forma del castigo láser no fue determinante | compatible: ahora no hay término de proximidad |
| §5 detenerse en la meta fue emergente | no aplica: el nodo detiene al llegar |
| §6 los premios al tocar son explotables | confirmado por diseño |
| §7 seis errores metodológicos | adoptados como reglas |
| §8 reentrenar con obstáculos, +12.8 puntos | superado por el entrenamiento directo |
| §9 A/B nulos por falta de potencia | resuelto: 2M pasos en 30 min y 500 episodios de evaluación |
| §10 el fallo no es de percepción | **matizado**: no faltaba información, faltaba resolución |
| §11 diagnóstico del estancamiento en ~45% | **explicado** por la codificación del LiDAR |
| §12 la evaluación de 10 episodios es ruido | confirmado y corregido |
| §13 una penalización fuerte paraliza | **replicado** con otro término (experimento B) |
| §14 holgura direccional y λ=0.99: nulo | consistente: no eran las palancas |
| §15 "¿observación u objetivo?" | **respondido: la observación** |
