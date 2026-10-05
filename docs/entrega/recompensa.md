# Justificación de los pesos de la recompensa

Recompensa por paso de control (0.1 s), `martha_nav/sim2d/reward.py`:

| término | peso | cuándo |
|---|---|---|
| progreso | +1 por metro | metros de **nuevo récord** de avance sobre la ruta A* |
| llegada | +20 | una vez, al quedar a menos de 0.3 m de la meta (termina el episodio) |
| choque | −20 | una vez, al tocar un obstáculo (termina el episodio) |
| tiempo | −0.005 | cada paso |
| proximidad, giro brusco, atasco | 0 | desactivados (ver abajo) |

`VecNormalize` divide la recompensa por la desviación estándar del retorno acumulado, así que lo
que importa son las **proporciones** entre términos, no su escala absoluta. Las cuentas usan las
cifras del entorno: velocidad máxima 0.35 m/s, rutas de 3 a 12 m y tiempo límite
3 × ruta / 0.35 + 10 s.

## Progreso: +1 por metro de nuevo récord

Es la señal densa que guía cada paso. Se paga solo al superar el mejor avance del episodio, así
que cada metro se cobra una vez: retroceder y volver a avanzar no rinde, y oscilar tampoco. Sobre
una ruta completa el progreso suma la longitud de la ruta, entre 3 y 12.

## Llegada: +20

Debe valer más que todo el progreso que se puede cobrar, para que terminar sea siempre mejor que
quedarse cerca de la meta. El progreso máximo es 12 (la ruta más larga), así que 20 lo supera en
cualquier episodio.

## Choque: −20

Debe hacer que un episodio que termina en choque valga menos que uno en que el robot no se mueve.

- Quedarse quieto hasta el timeout en la ruta más larga cuesta como mucho
  1 129 pasos × 0.005 = −5.6.
- Chocar después de cobrar todo el progreso de esa ruta vale 12 − 20 − 1.7 = **−9.7**: peor que
  no moverse, como se busca.
- Con el −10 del primer run, ese mismo choque valía 12 − 10 − 1.7 = **+0.3**: chocar cerca del
  final podía ser rentable. Por eso se pasó a −20.

Evidencia experimental (ajuste previo de la recompensa, 500 episodios con obstáculos): cambiar
solo el choque de −10 a −20 no movió el éxito de forma distinguible del ruido (0.366 → 0.342, los
IC95 se solapan). La mejora grande vino de codificar el LiDAR como d/(d+1): éxito 0.716 y colisión
0.260 → 0.048. El −20 se mantiene por el argumento de arriba.

## Tiempo: −0.005 por paso

A velocidad máxima el robot avanza 0.035 m por paso, así que el castigo equivale a 0.14 por metro:
el avance neto sigue siendo +0.86 por metro. Empuja a no demorarse sin competir con el progreso.
Un robot detenido pierde 0.05 por segundo, poco frente a lo que gana avanzando.

## Términos desactivados

- **Atasco terminal (−5):** se probó y el éxito con obstáculos cayó a 0.130 (frente a 0.342
  sin él). Un castigo grande comparado con el progreso de los primeros pasos enseña
  a no moverse. El estancamiento se mide igual como resultado, pero no se castiga.
- **Proximidad y giro brusco:** quedaron implementados y en 0. La colisión ya bajó a 0.048 con la
  codificación d/(d+1) del LiDAR, así que no hizo falta un castigo por acercarse.

## Hiperparámetros de PPO

`n_steps` 512 × 16 entornos, `batch_size` 256, 10 épocas, `gamma` 0.99, `gae_lambda` 0.95,
`clip_range` 0.2, `ent_coef` 0, tasa de aprendizaje 3e-4 con decaimiento lineal y
**`target_kl` 0.02**. El `target_kl` corta las épocas de un lote cuando la política cambia
demasiado. Sin él, la desviación estándar de la política colapsaba (0.60 → 0.007), cada
actualización reescribía la política y el mejor modelo aparecía pronto para luego degradarse.
