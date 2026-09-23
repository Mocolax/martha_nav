
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
