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
