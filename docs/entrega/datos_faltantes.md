# Datos no registrados y evaluaciones pendientes

## Limitaciones que el documento debe declarar

1. **Una sola semilla de entrenamiento.** Salvo E1 (CNN con semillas 0, 1 y 2), cada configuración
   se entrenó una vez. E1 muestra cuánto varía el resultado solo por la semilla: en `2d_lab`,
   0.800, 0.710 y 0.670. Una diferencia de menos de ~10 puntos entre dos configuraciones en `lab`
   puede deberse a la semilla y no a la configuración. Distinguirlas exige entrenar 2–3 semillas
   más de cada una (~1 h por run en GPU).
2. **Sesgo leve en `2d_limpio` y `2d_obstaculos`.** El `best_model.zip` se elige durante el
   entrenamiento con 200 episodios de los mapas de entrenamiento y las semillas 1 000 000–1 000 199.
   Esas semillas son parte de los 500 episodios de esos dos conjuntos (con otros obstáculos), así que
   sus cifras pueden ser algo optimistas. `lab` (2D y Gazebo) nunca se usó para elegir el modelo:
   es la medida de generalización limpia.
3. **Gazebo no es determinista.** Dos corridas idénticas del modelo final dieron 83 % y 87 % de
   éxito en las mismas 100 semillas, con el mismo resultado en 94 de 100 episodios. Las cifras de
   Gazebo tienen esa variación además del intervalo de confianza.
4. **Versión del código aproximada.** Los runs no guardaron el commit: `commit.txt` es el último
   commit anterior al inicio del entrenamiento. Si en ese momento había cambios sin commitear, no
   quedaron registrados. `config.yaml` sí tiene todos los parámetros. Los comandos de entrenamiento
   exactos están en `tools/experiments/` del repositorio, para E1, A/B/C y los brazos.
5. **Métricas de recuperación del seguimiento.** No hay una métrica calculada de "cuánto se aleja
   de la ruta y cuánto tarda en volver". Se puede calcular con los archivos `_traj` y `_route`:
   distancia de cada pose a la ruta y tiempo hasta volver a menos de un umbral.

## Cifras de la bitácora que cambian con la re-evaluación

Todas las configuraciones se re-evaluaron el 2026-10-04 con el código actual. Las cifras de este
paquete reemplazan a las de la bitácora. `wide_dyn_s0` y los brazos G, H y L dan exactamente los
mismos resultados. Las configuraciones más antiguas se habían evaluado con reglas de fin de episodio
anteriores (el estancamiento medía el desplazamiento, no el avance sobre la ruta) y cambian:

| configuración | conjunto | bitácora | re-evaluado |
|---|---|---|---|
| `long_c_kl_s0` | `2d_lab` | 0.865 | 0.835 |
| `long_c_kl_s0` | `2d_lab_puntos` | 0.722 | 0.689 |
| `e1_cnn_s1` | `2d_lab` | 0.755 | 0.710 |
| `expB_col20_stall5` | `2d_obstaculos` | 0.148 | 0.130 |
| `expB_col20_stall5` | `2d_lab` | 0.230 | 0.180 |
| `expA_col20` | `2d_limpio` | 0.566 | 0.588 |

Las demás diferencias son de 2 puntos o menos. Las conclusiones de la bitácora no cambian: C sigue
muy por encima de A y B, y B sigue siendo la peor.

## Evaluaciones que no existen

| dato | estado | qué haría falta |
|---|---|---|
| Gazebo para E1, el primer run y A/B/C | no evaluado | `tools/evaluate_gazebo_fast.sh` por configuración, ~15 min cada una |
| E1 con MLP, brazos combinados (GL, SL, S, geodésicos) | fuera del paquete por decisión | están en `runs/` si se necesitan |
| Localización con slam_toolbox | solo `long_c_kl_s0` y `wide_dyn_s0`, 100 semillas, **sin** trayectorias ni distancia (evaluador anterior) | repetir con el evaluador actual si el documento analiza esas trayectorias |
| Robot real | sin datos (los motores no han llegado) | la demo con 2–4 obstáculos |
| Varias semillas de entrenamiento del modelo final | no existe | ver limitación 1 |
