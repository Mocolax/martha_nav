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
- Cuando el comando entero es cero, las cuatro ruedas frenan (motor en cortocircuito) en vez
  de quedar sueltas: Martha seguía rodando por inercia cuando la política mandaba cero. Mientras
  se mueve, una rueda con referencia cero (la diagonal quieta de un movimiento en diagonal) queda
  suelta: frenada, FL+RR a 0.25 m/s no alcanzaba a arrancar (segunda ronda).

Con este firmware el giro lento arranca en ~0.4 s y gira 44-49° en 2 s (57° ideal).

## Resultado, firmware final

| Eje | Comando | Retardo | `tau` | Ganancia real (LiDAR) | Se cuela |
|---|---|---|---|---|---|
| Avance adelante | 0.15-0.35 m/s | 0.03-0.06 s | 0.22-0.30 s | 1.02-1.27 | casi nada |
| Avance atrás | 0.15-0.35 m/s | 0.05-0.15 s | 0.13-0.32 s | 0.60-1.03 | casi nada |
| Giro | 0.3-0.8 rad/s | 0.04-0.07 s | 0.22-0.56 s (una 1.1) | 0.44-0.57 | avance ≤ 0.11 m/s |
| Lateral, 0.10-0.25 m/s | desde reposo | - | - | **0**: no arranca | - |
| Lateral, 0.35-0.45 m/s | desde reposo | 0.04-0.39 s | - | **0.14-0.24** | **giro -0.34 a +0.29 rad/s** |

Ojo con la ganancia del avance de esta tabla: se midió con el escaneo tomado después de
parar, así que incluye lo que rodaba por inercia (hasta 18 cm). Mientras dura el comando, el
avance real va a 0.94-1.16 del pedido (segunda ronda).

Con firmwares intermedios el giro dio 0.71-0.85 y el avance hasta 1.6: el rango de ganancias
depende del ajuste fino del control, así que el entrenamiento debe cubrirlo entero.

Combinados (firmware intermedio): avance + giro deja el giro en 0.47-0.59 de lo pedido;
avance + lateral da un lateral inconsistente (-0.04 a +0.30).

**El lateral es débil de verdad**: entrega 13-42 % del comando y mete un giro de ~0.3-0.9 rad/s
por cada m/s de lateral. La segunda ronda descartó el reparto de peso y encontró la causa
principal en la rueda FL.

## Segunda ronda: firmware f51e403 y freno solo en parada total

Mismo método, pero con todos los escaneos de cada escalón seguidos por ICP contra el primero:
la trayectoria real da la velocidad estable (última mitad del comando) y el frenado después.
Datos: `check`, `brake`, `diag`, `diagfast`, `flrr`, `lat3`, `mix`, `latbase` y `latback`
(`.pkl`) en el scratchpad de la sesión, con `sysid2.py` y `analyze2.py`.

**Freno.** Desplazamiento después de mandar cero:

| Avance | Velocidad real | Recorre (LiDAR / ruedas) | Antes (ruedas, firmware sin freno) | Para en |
|---|---|---|---|---|
| +0.15 m/s | 0.15 | 1.4 / 2.0 cm | 4-5 cm | 0.14 s |
| +0.25 m/s | 0.25 | 5.3 / 4.9 cm | 11-12 cm | 0.27 s |
| +0.35 m/s | 0.40 | 6.9 / 6.2 cm | 17-18 cm | 0.32 s |
| -0.15 m/s | 0.15 | 2.6 / 2.9 cm | 1.6-2.2 cm | 0.30 s (ruedas) |
| -0.25 m/s | 0.34 | 7.1 / 6.0 cm | 5-6 cm | 0.51 s (ruedas) |
| -0.35 m/s | 0.46 | 9.8 / 8.0 cm | 10-13 cm | 0.46 s (ruedas) |

Adelante recorre de un tercio a la mitad que antes. Atrás frena con la misma desaceleración
(~1.1 m/s²), pero va 1.3-1.4 veces más rápido que lo pedido. Unos 2 cm de cada frenada son el
retardo del comando. La torre no cabecea.

**Diagonales** (solo un par de ruedas con referencia; las otras dos frenadas, con f51e403):

| Diagonal | Pedido por eje | Ganancia | Dirección real (pedido ±45°, ±135°) | Giro | Par motriz / par quieto |
|---|---|---|---|---|---|
| FL+RR | ±0.10-0.20 | 0.51-0.73 | -23 a -33° / +148 a +150° | ≤ 0.03 | 49-60 % de su referencia / arrastrado a 0.04-0.11 m/s |
| FL+RR, adelante | 0.25 | **0** (3 de 3) | - | - | FL y RR hacen fuerza y no giran |
| FL+RR, atrás | 0.25 | 0.24-0.40 | +149° | 0.12-0.22 | - |
| FR+RL, adelante | 0.10 | **0** | - | - | - |
| FR+RL, adelante | 0.15-0.25 | 0.52-0.64 | +25 a +31° | ≤ 0.04 | 58-63 % / 0.07-0.10 m/s |
| FR+RL, atrás | 0.10-0.25 | 0.67-0.83 | -162 a -167° | 0.02-0.14 | 68-80 % / hasta 0.29 m/s |

Las dos diagonales van al cuadrante correcto, así que las cuatro ruedas están bien montadas, y
en diagonal casi no se cuela giro. Pero todas se tuercen 12-32° hacia el avance, y las ruedas
que deberían quedar quietas giran arrastradas: sus rodillos no ruedan libres con el peso encima.
Girarlos a mano sin carga no lo muestra. Por eso el freno pasó a ser solo de parada total
(pendiente: repetir las diagonales con ese firmware).

**Lateral con ventanas de 3 s** (firmware final):

| Lateral | Ganancia | Dirección real | Giro | Giro / lateral pedido | Primera rueda en rendirse |
|---|---|---|---|---|---|
| ±0.15 m/s | 0.28-0.32 | +73° / -131° | ∓0.08-0.09 rad/s | -0.53 a -0.60 | FL, a 1.9-2.0 s |
| ±0.25 m/s | 0.22-0.37 | +51° / -133 a -135° | ∓0.16-0.20 | -0.64 a -0.80 | FL, a 1.8 s |
| ±0.40 m/s (1.2 s y 3 s) | 0.21-0.29 | +47 a +64° / -140° | ∓0.20-0.40 | -0.5 a -1.0 | FL; luego FR o RR |

**La rueda que se rinde es siempre FL**, en todas las corridas, en los dos sentidos. Las fugas
tienen su huella: quitándole a FL su aporte en la cinemática, un lateral mete -1/(4·0.385) =
-0.65 rad/s de giro y +0.25 m/s de avance por cada m/s, o sea unos 18° hacia adelante.
Medido: -0.44 a -1.0 rad/s y 17-41°. Que FL+RR no arranque hacia adelante a 0.25 m/s es la
misma rueda. Queda por saber si es la rueda (motor, caja, rodillos, cable) o el canal de la
placa (driver, conector): se sabe intercambiando los cables de FL y RL (mismos signos) y los
pines 0 ↔ 2 del firmware.

Hay ganancia lateral también entre 0.15 y 0.25 m/s (0.22-0.37). La primera ronda daba 0 porque
FL tenía el conector suelto.

**Batería movida atrás**: lateral ±0.25 y ±0.40 m/s, 3 s. No cambia nada medible: ganancia
0.15-0.36, giro -0.44 a -0.80 rad/s por m/s, y FL se rinde primero en las cuatro corridas. El
reparto de peso no es la causa.

**Combinados** (firmware final, ganancia por eje):

| Comando | Avance | Lateral | Giro | Primera ronda (firmware intermedio) |
|---|---|---|---|---|
| vx 0.25 + w 0.5 | 0.9-1.2 | - | 0.54-0.72 | giro 0.47-0.59 |
| vx 0.25 + vy 0.15 | 0.8-1.0 | 0.15-0.35 | 0.05-0.08 rad/s colado | lateral -0.04 a +0.30 |
| 0.15 + 0.15 + w 0.4 | ~0.9 | casi 0 | 0.35-0.53 | giro 0.39-0.76 |

**El signo del giro que se cuela.** El coeficiente giro / lateral pedido es negativo en todas
las corridas con ICP fiable: -0.53, -0.64, -0.76 y -0.64 en la primera ronda (sysid_2), y de
-0.44 a -1.0 en la segunda. El "-0.34 a +0.29 rad/s" de la tabla de arriba es el giro con el
lateral hacia un lado y hacia el otro, no un cambio de signo del coeficiente.

### Rangos de `REAL_DYNAMICS` tras la segunda ronda

Los rangos que salen de estas medidas describen a Martha con FL fallando. Si FL se arregla,
el giro que se cuela cae a ~0 y el lateral sube, así que los rangos laterales cubren las dos
Martha: la política no aprende a compensar un giro que puede desaparecer.

| Parámetro | Medido | Rango | Por qué |
|---|---|---|---|
| `gain_vy` | 0.15-0.45 | **0.1-0.7** | con FL bien debería subir; las diagonales muestran rodillos que arrastran con carga, así que no llegará a 1 |
| `yaw_leak` | -0.44 a -1.0 | **-1.0 a 0** | siempre negativo; el 0 es FL arreglada |
| `vx_leak` (nuevo) | +0.25 a +0.8 | **0 a 0.8** | m/s de avance por m/s de lateral pedido, mismo signo; el 0 es FL arreglada |
| `gain_w` | 0.35-0.53 combinado | **0.35-0.9** | |
| `gain` | atrás 1.3-1.4 | **0.6-1.4** | |

## Lo que el simulador ahora modela

`DynamicsRanges` tiene cuatro campos nuevos; con sus valores por defecto las corridas anteriores
y sus episodios no cambian:
- `gain_vy`: ganancia lateral propia (antes el lateral usaba la del avance).
- `gain_w`: rango de la ganancia de giro aparte del avance.
- `yaw_leak`: rad/s de giro por m/s de lateral pedido, sorteado por episodio con su signo.
- `vx_leak`: m/s de avance por m/s de lateral pedido.

`--real-dynamics` (`REAL_DYNAMICS`) usa los rangos medidos, con margen (los laterales y las
ganancias, de la tabla de la segunda ronda):

| Parámetro | Rango | Por qué |
|---|---|---|
| `gain` (avance) | 0.6-1.4 | avance atrás 0.60-1.4, adelante 0.94-1.16 |
| `gain_w` | 0.35-0.9 | 0.35-0.72 ahora, 0.71-0.85 con otro ajuste |
| `gain_vy` | 0.1-0.7 | 0.15-0.45 con FL fallando, más con FL arreglada |
| `yaw_leak` | -1.0 a 0 rad/s por m/s | -0.44 a -1.0 con FL fallando, 0 arreglada |
| `vx_leak` | 0 a 0.8 m/s por m/s | +0.25 a +0.8 con FL fallando, 0 arreglada |
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
