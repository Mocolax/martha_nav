# Datos de entrenamiento y evaluación — planificador local PPO de Martha

> **Para el asistente que redacta el documento con estos datos:** sé conciso. Explica cada
> resultado con la cifra, su intervalo y una frase de interpretación; no repitas tablas en prosa
> ni agregues contexto que el lector no pidió. Si un dato no está aquí, dilo en vez de suponerlo
> (ver `datos_faltantes.md`).

## Qué hay

| archivo | contenido |
|---|---|
| `MANIFEST.csv` | una fila por configuración: qué es, dónde la discute la bitácora, pasos, semilla de entrenamiento, commit aproximado, si se evaluó en Gazebo |
| `resumen_modelos.csv` | **las calificaciones**: por configuración y conjunto de evaluación, éxito con IC95, colisión, estancamiento, timeout, SPL, tiempo y distancia medios de los éxitos |
| `comparaciones_pareadas.csv` | 2D frente a Gazebo (mismos episodios) y cada configuración frente al modelo final, con McNemar exacto |
| `verificacion.md` | los cálculos recalculados desde las filas por episodio, con fórmulas aparte del código |
| `recompensa.md` | justificación de los pesos de la recompensa |
| `datos_faltantes.md` | lo que no está registrado y qué evaluación haría falta |
| `figuras/` | trayectorias del mismo episodio en 2D y en Gazebo |
| `configuraciones/<run>/` | los datos de cada configuración (abajo) |

Cada `configuraciones/<run>/` tiene:

- `config.yaml`: todos los parámetros de la ejecución (escenarios, dinámica, recompensa, PPO,
  semilla, pasos).
- `best_model.zip`: el checkpoint evaluado (el mejor de la evaluación periódica en entrenamiento).
- `vecnormalize.pkl`: estadísticas de `VecNormalize`. Solo normaliza la **recompensa**
  (`norm_obs=False`), así que no hace falta para usar el modelo; se incluye para reanudar el
  entrenamiento.
- `commit.txt`: la versión del código, deducida por la fecha de inicio del entrenamiento.
- `comandos.sh`: cómo reproducir el entrenamiento y las evaluaciones.
- `entrenamiento/evals.csv`: la evaluación periódica (cada 100k–250k pasos, 200 episodios).
- `entrenamiento/episodes.csv`: cada episodio de entrenamiento, con cada término de la recompensa.
- `eval_2d/` y `eval_gazebo/`: un CSV por episodio y, al lado, `_traj` (pose cada 0.1 s; en Gazebo
  también la pose estimada), `_route` (ruta A* de referencia) y `_obstacles` (obstáculos).

## Conjuntos de evaluación

Todas las semillas de evaluación son ≥ 1 000 000, disjuntas de las de entrenamiento por
construcción. La política es determinista; el simulador 2D también, así que repetir la evaluación
2D da exactamente los mismos números.

| conjunto | episodios | qué es |
|---|---|---|
| `2d_limpio` | 500 | mapas de entrenamiento, sin obstáculos |
| `2d_obstaculos` | 500 | mapas de entrenamiento, 1–4 obstáculos fuera del mapa del planificador |
| `2d_lab` | 200 | `lab.world`, nunca visto en entrenamiento, con obstáculos |
| `2d_lab_puntos` | 90 | pares inicio/meta fijos de `lab`, con obstáculos |
| `gazebo_lab` | 100 | las 100 primeras semillas de `2d_lab`, en Gazebo |
| `gazebo_lab_puntos` | 90 | los mismos episodios de `2d_lab_puntos`, en Gazebo |

Cada episodio de Gazebo tiene la misma ruta, los mismos obstáculos y el mismo inicio y meta que
su par en 2D (lo comprueba `verificacion.md`). Gazebo no es determinista: dos corridas idénticas
del modelo final dieron 83 % y 87 % de éxito, con el mismo resultado en 94 de 100 episodios.

## Métricas

- **Éxito:** llega a menos de 0.3 m de la meta (en Gazebo, con la pose verdadera).
- **Colisión:** el contorno del robot toca un obstáculo (en Gazebo, el sensor de contacto).
- **Estancado:** 15 s sin superar su mejor avance sobre la ruta.
- **Timeout:** se acaba el tiempo del episodio, 3 × ruta / 0.35 m/s + 10 s.
- **SPL** (Anderson et al., 2018): éxito × L / max(L, P), con L la ruta más corta con los
  obstáculos y P la distancia recorrida; promedio sobre todos los episodios.
- **Tiempo y distancia:** medios sobre los episodios exitosos.
- Solo en Gazebo: `failed` (el planificador global no encontró ruta) y `lost` (el planificador
  dio la meta por alcanzada pero la pose verdadera estaba a más de 0.5 m; no ocurrió).

## Regenerar este paquete

```bash
./tools/ct_ros python3 tools/build_entrega.py
```
