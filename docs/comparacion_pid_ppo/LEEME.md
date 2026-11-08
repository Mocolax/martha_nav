# Línea base PID vs. PPO (simulador 2D)

Pregunta: ¿PPO aporta algo frente a una solución más sencilla? Se comparan dos controladores locales
sobre la misma ruta global (Dijkstra + carrot); solo cambia quién decide (v, w).

- **PPO**: `martha_tower_scratch_s0` (Martha con la torre, acción diferencial v, w).
- **PID**: control de rumbo hacia el carrot (en la práctica PD, la I es cero) más reglas if/else:
  frena según el espacio libre delante, se aparta de la pared lateral a menos de 25 cm y, si el
  frente está bloqueado, gira en el sitio hacia el lado más libre o retrocede si no cabe girando.

Mismos mundos, mismas semillas de prueba, misma dinámica aleatorizada y mismos sensores.
Los parámetros del PID se ajustaron a mano (unas 15 combinaciones) con semillas de **entrenamiento**;
las de prueba solo se usaron para la evaluación final. Solo simulador 2D: no se evaluó en Gazebo.

## Contenido
- `figura_pid_vs_ppo.png`: proporción de éxito, colisión y atasco/tiempo agotado por condición.
- `resumen.csv`: los mismos números, con el intervalo de confianza del 95 % (Wilson) del éxito y el SPL.
- `resultados/{pid,ppo}/eval_<condición>.csv`: una fila por episodio (`outcome`, `spl`, `source`, ...).
  Los `_obstacles` sirven solo para dibujar trayectorias; los `_traj` y `_route` no se incluyen en el repositorio por su tamaño (unos 55 MB).
  Condiciones: `clean_train` (limpio, 500), `obstacles_train` (obstáculos, 500),
  `obstacles_lab` (lab no visto, 200), `obstacles_lab-points` (puntos fijos del lab, 90).
- `codigo/pid.py`: el controlador; `codigo/config_pid.yaml`: el entorno (idéntico al de PPO);
  `codigo/_pid_tune.py`: el script de ajuste.

## Cómo leerlo
PPO llega a la meta mucho más seguido (89 % vs. 61 % con obstáculos; 82 % vs. 52 % en el lab no visto).
Pero el PID **choca menos** (2 % vs. 7 % con obstáculos): falla porque se atasca frente al obstáculo,
el problema típico de un controlador reactivo. Conclusión justa: "PPO alcanza la meta con más
frecuencia", no "PPO es más seguro". Es una línea base sencilla, no un método del estado del arte.
