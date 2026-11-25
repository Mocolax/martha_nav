# Dinámica real de Martha en el piso (2026-11-25)

Martha con la torre de la garra, en el laboratorio. El simulador 2D mueve el robot con cinco
parámetros por eje (`martha_nav/sim2d/dynamics.py`): retardo, `tau` de primer orden,
aceleración máxima y ganancia (lo que el robot hace / lo que se le pide). Estas pruebas miden
esos mismos parámetros en el robot, para entrenar con lo que hace de verdad.

## Cómo se midió

Escalones de 1.2 s desde reposo, cada uno ida y vuelta (+ y -) para volver al sitio, a las
velocidades que usa la política. Por escalón: comando a 20 Hz, odometría de ruedas
(`/wheel/odometry`, 50 Hz), giróscopo (`/imu`) y el desplazamiento real medido con el LiDAR
(ICP entre el escaneo del inicio y el del final, sin los postes; residuo 0.3-1.1 cm). El LiDAR
ve el patinaje que los encoders no ven. Retardo: hasta el 20 % del comando; `tau`: del 20 % al
63 %; ganancia: media del último medio segundo.

## Lo que había mal antes de medir

1. **La rueda delantera izquierda (FL) no giraba**: conector del motor flojo en el driver.
   Moviendo solo la diagonal FL-RR, RR seguía el comando y FL quedaba quieta y en silencio aun
   con PWM 255. Martha anduvo con tres ruedas buena parte del día: el lateral mezclado con giro
   que se veía con la política venía en parte de ahí. Ajustado, en el aire las cuatro siguen el
   comando y un lateral puro ya no da giro. Conviene revisar los cuatro conectores.
2. **A baja velocidad las ruedas no arrancaban y pitaban.** El feedforward se calibró con las
   ruedas en el aire (arrancaban desde PWM ~10); en el piso hace falta mucho más. Un giro en el
   sitio a 0.5 rad/s movía a Martha 1-2° en 2 s, y a 1.5 rad/s (PWM saturado de entrada) 38° en
   1 s: no faltaba fuerza (batería a 15 V), faltaba PWM a baja velocidad. Con la política, en el
   30 % del tiempo alguna rueda recibía un feedforward de PWM 0-15, porque la mezcla mecanum de
   `vx`, `vy` y `w` deja ruedas casi en cero aunque el cuerpo vaya a 0.35 m/s.

## Firmware que queda (`firmware/firmware.ino`)

- `PWM_START` 40 (antes 4): el feedforward incluye la fricción del piso.
- Empuje anti-atasco: si una rueda va a menos de la mitad de su referencia (velocidad filtrada
  a 50 ms), su PWM sube 400 por segundo, hasta 200 más; se retira a 800 por segundo cuando la
  alcanza. Con retiro lento (150/s) el avance quedaba 25-60 % por encima del comando más de un
  segundo; detectar el atasco por falta total de pulsos no servía, porque una rueda que se
  arrastra cuenta algunos.
- `KI` 4 (antes 1.6): el integrador cubre antes la fuerza sostenida.
- `PWM_MIN` 20.
- Límite del empuje: una rueda que lleva 1.5 s trabada con PWM 255 descansa 1 s sin
  corriente, avisa `wheel_stall_fl` (`_fr`, `_rl`, `_rr`) y vuelve a intentar. Antes, con las
  cuatro ruedas trabadas a tope (lateral lento), los motores pitaban sin parar.
- Anti-windup sobre la salida total (feedforward + empuje + PI), no solo sobre el PI: con el
  feedforward ya en 255 el integrador seguía creciendo y la rueda se pasaba al despegar.
- Con referencia cero la rueda frena (motor en cortocircuito) en vez de quedar suelta: Martha
  seguía rodando por inercia cuando la política mandaba cero.

Con este firmware el giro lento arranca en ~0.4 s y gira 44-49° en 2 s (57° ideal).

## Resultado, firmware final

| Eje | Comando | Retardo | `tau` | Ganancia real (LiDAR) | Se cuela |
|---|---|---|---|---|---|
| Avance adelante | 0.15-0.35 m/s | 0.03-0.06 s | 0.22-0.30 s | 1.02-1.27 | casi nada |
| Avance atrás | 0.15-0.35 m/s | 0.05-0.15 s | 0.13-0.32 s | 0.60-1.03 | casi nada |
| Giro | 0.3-0.8 rad/s | 0.04-0.07 s | 0.22-0.56 s (una 1.1) | 0.44-0.57 | avance ≤ 0.11 m/s |
| Lateral, 0.10-0.25 m/s | desde reposo | - | - | **0**: no arranca | - |
| Lateral, 0.35-0.45 m/s | desde reposo | 0.04-0.39 s | - | **0.14-0.24** | **giro -0.34 a +0.29 rad/s** |

Con firmwares intermedios el giro dio 0.71-0.85 y el avance hasta 1.6: el rango de ganancias
depende del ajuste fino del control, así que el entrenamiento debe cubrirlo entero.

Combinados (firmware intermedio): avance + giro deja el giro en 0.47-0.59 de lo pedido;
avance + lateral da un lateral inconsistente (-0.04 a +0.30).

**El lateral es débil de verdad**, también con las cuatro ruedas bien y el PWM a tope: entrega
13-42 % del comando y mete un giro de ~0.3-0.9 rad/s por cada m/s de lateral, con signo que
cambió entre sesiones. La explicación más simple es mecánica: el peso repartido desigual entre
ejes (la garra y sus motores van adelante) hace que un eje frene más al deslizar de lado.
Moverle peso hacia atrás (la batería) sería el arreglo más directo.

## Lo que el simulador ahora modela

`DynamicsRanges` tiene tres campos nuevos; con sus valores por defecto las corridas anteriores
y sus episodios no cambian:
- `gain_vy`: ganancia lateral propia (antes el lateral usaba la del avance).
- `gain_w`: rango de la ganancia de giro aparte del avance.
- `yaw_leak`: rad/s de giro por m/s de lateral pedido, sorteado por episodio con su signo.

`--real-dynamics` (`REAL_DYNAMICS`) usa los rangos medidos, con margen:

| Parámetro | Rango | Por qué |
|---|---|---|
| `gain` (avance) | 0.6-1.3 | avance atrás 0.60, adelante hasta 1.27 |
| `gain_w` | 0.4-0.9 | 0.44-0.57 ahora, 0.71-0.85 con otro ajuste |
| `gain_vy` | 0.1-0.35 | 0.14-0.24 medido, 0 a baja velocidad |
| `yaw_leak` | -0.9 a 0.9 rad/s por m/s | ±0.3-0.9, signo variable |
| `tau` | 0.15-0.6 s | 0.13-0.56 medido |
| `delay` | 0-2 ciclos | 0.03-0.15 s del robot más ~0.13 s del escaneo al comando |
| `acc_v`, `acc_w` | 0.5-2.5 m/s², 1-5 rad/s² | alcanza 0.35 m/s en ~0.3 s |

## Para entrenar esta noche

Las dos variantes, con la receta de `martha_tower_holo_thin_rev` (patas delgadas, 360 rayos,
castigo por retroceder) y `--real-dynamics`, para compararlas mañana en el robot:

```bash
python3 -m martha_nav.learning.train --preset full --robot martha_tower --action-dim 3 \
    --real-dynamics --thin-obstacles 0.3 --n-rays 360 --reward-reverse 1.0 --seed 0 \
    --name martha_real_holo_s0
python3 -m martha_nav.learning.train --preset full --robot martha_tower --action-dim 2 \
    --real-dynamics --thin-obstacles 0.3 --n-rays 360 --reward-reverse 1.0 --seed 0 \
    --name martha_real_vw_s0
```

- **Holonómica** (`--action-dim 3`): aprende que el lateral rinde poco y gira el robot, y lo
  usa solo cuando le sirve.
- **(v, w)** (`--action-dim 2`): sin lateral. Con un lateral tan débil puede que no pierda nada
  en el robot y sea más predecible.

Límites de estas medidas: un solo piso, una batería a 15 V, ventanas de 1.2 s (la escala de
decisión de la política; un integrador lento puede acercar las ganancias a 1 en tramos largos).
Datos crudos: `sysid_1.npz` a `sysid_3.npz` en el scratchpad de la sesión del 2026-11-25.
