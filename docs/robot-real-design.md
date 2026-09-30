# Diseño: Martha en el robot real (firmware, bridge, fusión, mapeo y demo)

Fecha: 2026-09-29. Estado: aprobado en brainstorming, pendiente de plan de implementación.

## 1. Objetivo y alcance

Llevar el firmware de la ESP32 (`martha/arduino`) a `martha_nav`, limpio y con las
correcciones del code review, y cerrar el camino completo en el robot físico:
**mapear el lugar con slam_toolbox, guardar el mapa y hacer la demo** con
`global_planner` + `ppo_local_planner` esquivando obstáculos que el mapa no tiene.

Todo lo de ROS corre en el PC montado sobre el robot. La ESP32 solo hace motores,
encoders, IMU y protecciones.

**Criterio de éxito:** con un mapa guardado del lugar, un `/goal_pose` en RViz lleva
al robot a la meta, y las cajas puestas después del mapeo las esquiva el PPO, no el A*.

**Fuera del alcance:** reentrenar o ajustar la política para el robot real, reset
automático de protecciones, cambios al PCB (ya impreso), borrar `martha/arduino`
(queda como referencia).

**Regla de código:** mínimo y comprensible (skill `minimalist`): sin abstracciones
no pedidas, reutilizar lo que ya existe, el menor número de archivos.

## 2. Decisiones principales

| decisión | elección | motivo |
|---|---|---|
| dónde se fusiona | **EKF de `robot_localization` en el PC** | estándar y citable; la misma fusión que `martha` |
| qué fusiona | encoders: vx, vy, vyaw; IMU: **solo vyaw** | igual que `martha/config/ekf_hardware.yaml`; en 2D lo demás no aporta |
| qué calcula la ESP32 | velocidades medidas + gyro z, **sin pose** | la pose la integra el EKF; menos código y menos tráfico |
| autoridad de protecciones | **solo el firmware** | el bridge es un traductor sin estados |
| techo de velocidad | **en el firmware**, por encima de la política | protege de cualquier emisor sin acoplarse a `action_to_cmd` |
| escalado [-1, 1] → m/s | **se queda en ROS** (`action_to_cmd`) | es parte del contrato de la política; una sola copia |
| mapa del A* en la demo | **congelado** (guardado antes de poner las cajas) | es el supuesto del entrenamiento: el A* solo ve lo estático |
| `map → odom` | slam_toolbox en modo localización | una sola herramienta para mapear y localizar |
| reset tras un latch | **manual** (servicio `~/reset`) | una sobrecorriente pide que alguien revise antes |

## 3. Estructura

| ruta | qué es |
|---|---|
| `firmware/firmware.ino` | pines, `setup`/`loop`, protocolo serial, lazo de control, protecciones |
| `firmware/control.h` | PI con anti-windup + cinemática mecanum inversa y directa (matemática pura) |
| `firmware/sensors.h` | encoders con `pulse_cnt` + gyro z del MPU con su bias |
| `martha_nav/ros/esp32_bridge.py` | traductor serial ↔ ROS |
| `martha_nav/ros/world_map_publisher.py` | **se amplía**: parámetro `map_yaml` para publicar un mapa guardado |
| `config/ekf.yaml` | EKF recortado de `martha/config/ekf_hardware.yaml` |
| `config/rplidar.yaml` | copiado de `martha/config/rplidar_a2m8.yaml` |
| `launch/real.launch.py` | un launch, dos modos: sin `map:=` mapea, con `map:=` navega |
| `tools/save_map.sh` | guarda el mapa congelado y el posegraph |
| `maps/` | mapas guardados (`<lugar>_real.*`) |

Los módulos del firmware son headers sin `.cpp` (3 archivos en vez de 11). Arduino
exige que la carpeta se llame igual que el `.ino`.

Se reutilizan sin cambios: `urdf/martha.urdf.xacro` con `drive:=planar` (ruedas
fijas, no hace falta `/joint_states`), `global_planner`, `ppo_local_planner` y
`observation.py` (`action_to_cmd`). Si el LiDAR quedó montado al revés, se corrige con
`flip_x_axis` de `rplidar_ros` (`ppo_local_planner` no expone `lidar_yaw`).

### 3.1 Flujo en navegación

```
ESP32 ──serial──► esp32_bridge ──/wheel/odometry, /imu──► EKF ──/odom + TF odom→base_link──┐
RPLIDAR ──/scan──► slam_toolbox (localización) ──TF map→odom                              │
world_map_publisher(map_yaml) ──/map congelado──► global_planner ──/plan──► ppo_local_planner
                                                                               │  (/scan, /odom, TF)
ESP32 ◄──serial── esp32_bridge ◄──────────────── /cmd_vel ─────────────────────┘
```

En modo mapeo slam_toolbox mapea (y publica `map → odom`), no corren los planners y
`/cmd_vel` viene de `teleop_twist_keyboard`.

## 4. Firmware (`firmware/`)

### 4.1 Pines (verificados contra la placa impresa del 2026-09-21)

| función | GPIO | va a |
|---|---|---|
| encoders M1..M4 (A, B) | 12,13 · 36,39 · 4,16 · 21,22 | J1..J4 pines 5, 6 |
| PWM M1..M4 (IN1, IN2) | 27,14 · 32,33 · 15,2 · 18,19 | U1..U4 IN1 / IN2 |
| nSLEEP de los 4 drivers | 17 | U1..U4 SLEEP |
| sobrecorriente (LM339, activo bajo) | 23 | salidas del LM339 |
| batería (divisor 39k/10k, ganancia 4.9) | 34 | R5/R6/C2 |
| I2C del IMU | SDA 25, SCL 26 | J6.4, J6.3 |

Orden lógico de ruedas: FL, FR, RL, RR. `MOTOR_OUTPUT_SIGN` y `ENCODER_COUNT_SIGN`
se mantienen para corregir la polaridad en la puesta en marcha.

### 4.2 Constantes del robot real

r = 0.075 m, lx + ly = 0.385 m (el simulador usa 0.375), 3200 cuentas/vuelta
(Pololu 37D 50:1, 64 CPR, x4), 200 RPM máx., PWM 10 kHz / 8 bits, Kp = 2, Ki = 1.6
(se afinan en la puesta en marcha). Techo de seguridad: |vx|, |vy| ≤ 0.5 m/s,
|wz| ≤ 1.2 rad/s (la política llega a 0.35 / 0.25 / 0.8).

### 4.3 Ritmos del `loop()`

- **100 Hz, control.** Por rueda: `delta = cuenta − cuenta_anterior` (el contador
  de `pulse_cnt` con `accum_count` nunca se borra), RPM, PI, PWM. Al cambiar de
  sentido el motor queda un ciclo en *coast* (IN1 = IN2 = 0 es coast en el
  MPQ6612A, no freno; la función se llama así). Se acumulan los pulsos para el reporte.
- **50 Hz, reporte.** `odom,vx,vy,wz,gz`: velocidades por cinemática directa de los
  pulsos acumulados desde el reporte anterior (divididos por el tiempo transcurrido);
  `gz` del IMU en ese momento, o `nan` sin IMU.
- **Batería.** Una muestra por ciclo con filtro exponencial (τ ≈ 0.5 s), sin bucles
  bloqueantes. Latch si el valor filtrado baja de 11.0 V. `battery,V` a 1 Hz.
- **Sobrecorriente.** Latch solo si D23 está bajo **de forma continua 20 ms** y los
  drivers llevan más de 5 ms despiertos (el LM339 se alimenta del LDO de un driver,
  que se apaga en sleep).
- **Timeout de `cmd_vel`:** 500 ms.

### 4.4 `stopControl()`

Referencias a 0, reset del PI, motores en coast. La llaman el timeout, los dos
latches y el arranque/rearme: ninguna parada deja una referencia vieja.

### 4.5 PI con anti-windup

Integración condicional: no se integra cuando la salida está saturada y el error
empuja hacia el mismo lado. Con la rueda bloqueada a 80 RPM el término I queda en
~95 en vez de crecer ~128 PWM/s; al ordenar reversa la salida cambia de signo en
~0.7 s en vez de ~12 s.

### 4.6 IMU (solo gyro z)

Rango ±250 °/s (el robot gira como máximo ~46 °/s), filtro digital de 44 Hz.
Bias: 1 s de calibración al arrancar y después se reaprende **solo** con el robot
quieto según sus propios datos (todas las referencias en 0 y sin pulsos en los
últimos 0.5 s). El gyro nunca se pone a cero por su propia lectura y no hay banda
muerta. Si el IMU no responde: `imu_missing` y el firmware sigue sin él.

### 4.7 Protocolo serial (115200 baud, ~20 % de uso)

| dirección | línea | cuándo |
|---|---|---|
| PC → ESP32 | `cmd_vel,vx,vy,wz` | cada comando; se descarta si no es finito o si hay latch; se recorta al techo |
| PC → ESP32 | `reset` | pedido del operador |
| ESP32 → PC | `odom,vx,vy,wz,gz` | 50 Hz |
| ESP32 → PC | `battery,V` | 1 Hz |
| ESP32 → PC | `ready` | al arrancar y tras un reset aceptado |
| ESP32 → PC | `motor_overcurrent`, `battery_too_low` | al entrar en latch (una vez) |
| ESP32 → PC | `cmd_vel_timeout` | al vencer el timeout |
| ESP32 → PC | `reset_blocked` | reset rechazado |
| ESP32 → PC | `imu_missing` | al arrancar sin IMU |

Mientras hay latch los `cmd_vel` se ignoran en silencio.

### 4.8 Reset

Con `reset` el firmware exige batería ≥ 11.5 V; despierta los drivers, espera 5 ms
y exige D23 en alto. Si pasa: `stopControl()` y `ready` (el robot queda quieto hasta
el próximo `cmd_vel`). Si no: drivers dormidos y `reset_blocked`.

### 4.9 Lo que se elimina respecto a `martha/arduino`

Pose integrada en la ESP32, líneas `joint` y `wheel_ref_rpm`, escaneo I2C, roll /
pitch / yaw / aceleración / temperatura del IMU, `odom_reset`, ramas para el core
v2 de arduino-esp32, el driver PCNT legacy y el `while (true)` si falla el IMU.

## 5. `esp32_bridge` y EKF

### 5.1 `martha_nav/ros/esp32_bridge.py`

Traductor sin máquina de estados y sin límites propios.

- `/cmd_vel` → `cmd_vel,vx,vy,wz`.
- Timer que lee lo disponible sin bloquear y parte por líneas.
  - `odom,…` → `/wheel/odometry` (solo twist, con covarianza, frames `odom` /
    `base_link`) y, si `gz` es finito, `/imu` (solo `angular_velocity.z`, frame
    `base_link`; orientación y aceleración con `covariance[0] = -1`). El IMU va
    montado plano (`rpy 0 0 0` en el URDF viejo), así que su gyro z es el giro del robot.
  - Eventos → log como warning. Batería → log cada 30 s.
  - Sin líneas `odom` durante 1 s → un warning (el `ppo_local_planner` ya se detiene
    con `/odom` viejo).
- Servicio `~/reset` (`std_srvs/Trigger`): escribe `reset`. Responde solo que se
  envió; el resultado (`ready` / `reset_blocked`) sale en el log.
- Al cerrar manda un cero (si muere sin mandarlo, el timeout del firmware lo cubre).
- Puerto por parámetro, con rutas `/dev/serial/by-id/…` (el RPLIDAR A2 también usa
  un CP2102 y `ttyUSB0/1` pueden intercambiarse).
- Parseo de línea y formato de comando son funciones puras (tests sin ROS).

Uso: `ros2 service call /esp32_bridge/reset std_srvs/srv/Trigger`. Recomendado
cancelar antes la meta (`/cancel_goal`) para que el PPO no arranque al rearmar.

### 5.2 `config/ekf.yaml`

`two_d_mode: true`, 50 Hz, `world_frame: odom`, `publish_tf: true`.
`odom0: /wheel/odometry` con vx, vy, vyaw; `imu0: /imu` con vyaw. La salida
`odometry/filtered` se remapea a `/odom`.

## 6. Mapeo y demo (`launch/real.launch.py`)

Argumentos: `map` (ruta sin extensión), `checkpoint`, `esp32_port`, `lidar_port`,
`rviz`. Siempre arranca `robot_state_publisher` (`drive:=planar`), `esp32_bridge`,
`rplidar_node` y el EKF.

### 6.1 Fase 1: mapear (sin `map`)

- slam_toolbox en modo mapeo publica `map → odom` y su mapa en `/slam_map`.
- Teleop en otra terminal: `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
  (con Shift se mueve lateral).
- Empezar desde un punto marcado con cinta en el piso (sección 6.2).
- `tools/save_map.sh maps/<lugar>_real`:
  1. `ros2 run nav2_map_server map_saver_cli -f <out> -t /slam_map` → `.pgm/.yaml`,
     el mapa congelado del A* (editable en GIMP para borrar gente o sillas).
  2. `/slam_toolbox/serialize_map` → `.posegraph/.data`, para localizarse.

   Se llama a `map_saver_cli` directamente porque el servicio `save_map` de
   slam_toolbox buscaría `/map`, que está remapeado a `/slam_map`.

### 6.2 Fase 2: demo (`map:=maps/<lugar>_real checkpoint:=…`)

- slam_toolbox en modo localización carga el posegraph y publica `map → odom`.
  Arranca en la pose donde empezó el mapeo (el robot sobre la marca de cinta), o se
  da la pose con "2D Pose Estimate" en RViz (`/initialpose`).
- `world_map_publisher` con `map_yaml` publica el `/map` congelado, que no cambia.
  Lectura del PGM con numpy, filas invertidas (la imagen empieza arriba, la rejilla
  en el origen). Solo los píxeles blancos (≥ 250) son libres; desconocido (205) y
  ocupado bloquean, igual que `msg_to_grid` trata lo desconocido.
- `global_planner` y `ppo_local_planner` sin cambios; el `/goal_pose` sale de RViz.

### 6.3 slam_toolbox compartido con la simulación

Se usa el mismo esquema que `sim.launch.py` (en curso en otra sesión): la
configuración que trae slam_toolbox más overrides, y `/map` remapeado a `/slam_map`.
Cuando ese cambio esté en git, la función `slam_toolbox()` pasa a un módulo que
importan los dos launches. El robot real solo cambia `use_sim_time: false`,
`scan_topic: /scan` y que en mapeo sí publica TF.

**Riesgo principal:** que slam_toolbox pierda la localización con cajas que no están
en el mapa. La prueba en simulación de la otra sesión mide justo eso.

## 7. Hardware (fuera del PCB, antes de flashear)

1. **Divisor en el arnés de los encoders.** Los encoders se alimentan del LDO de 5 V
   de cada driver y sus salidas A/B llegan a 5 V a GPIO de 3.3 V. En los cables
   amarillo (A) y blanco (B), del lado de la ESP32: señal → 4.7 kΩ → GPIO, y 10 kΩ
   del GPIO a GND (≈ 3.4 V). 16 resistencias con termoencogible. **Verificar** con
   la rueda quieta en alto que el pin lee entre 2.5 y 3.6 V.
2. **Desoldar R4.** Sube GPIO23 a +5V4; el pull-up interno (`INPUT_PULLUP`, ~45 kΩ a
   3.3 V) hace lo mismo al voltaje correcto y además deja D23 en alto (sin falla)
   con los drivers dormidos.
3. **Calibrar RV1** con encoders conectados y drivers despiertos: VD ≈ 0.7 V
   (≈ 3.5 A por motor con VISEN de 200 mV/A; marcha normal < 2 A, bloqueo 5.5 A).
   Anotar los valores soldados de R2/R3 (en el esquemático figuran como "R").
4. **Opcional, eFuse:** `espefuse.py set_flash_voltage 3.3V` elimina el riesgo del
   pin de strapping GPIO12 (encoder A de M1). Irreversible; seguro en la DOIT
   (WROOM-32, flash de 3.3 V).

La placa impresa es `martha/pcb/martha_circuits/gerbers/` (2026-09-21); el
`Gerbers.zip` del 31 de julio ya no corresponde a lo fabricado.

## 8. Puesta en marcha (en orden; va a `docs/comandos.md`, sección "Robot real")

1. `arduino-cli compile/upload` con `esp32:esp32:esp32doit-devkit-v1`.
2. Ruedas al aire, monitor serial: `ready`; `battery,V` contra el multímetro.
3. Polaridad: `cmd_vel,0.1,0,0` → las 4 ruedas adelante y vx > 0; si no, ajustar
   `MOTOR_OUTPUT_SIGN` y `ENCODER_COUNT_SIGN`. Repetir con vy y wz.
4. PI: la velocidad medida sigue a la ordenada; si no, ajustar Kp / Ki.
5. IMU: giro antihorario → gz > 0; quieto → gz ≈ 0.
6. Protecciones: puente D23–GND → latch → `~/reset`; matar el bridge → ruedas
   paradas en ≤ 0.5 s; fuente de laboratorio < 11 V → latch de batería.
7. ROS en modo mapeo: árbol `map → odom → base_link → lidar` en RViz; scan alineado
   con el frente (si no, `flip_x_axis: true`); empujar 1 m → `/odom` ~1 m; girar 360° → ~2π.
8. Mapear, guardar y hacer la demo con el checkpoint.

## 9. Pruebas de código

- pytest sin ROS: parseo y formato de `esp32_bridge`; lector de PGM de
  `world_map_publisher` con un PGM sintético (filas invertidas, origen, libre vs
  ocupado).
- `arduino-cli compile` sin warnings de API deprecada.
- La prueba real es la puesta en marcha (sección 8); el hardware no se simula.

## 10. Dependencias y coordinación

`package.xml`: `robot_localization`, `rplidar_ros`, `nav2_map_server`,
`teleop_twist_keyboard`, `std_srvs`, `python3-serial`. Entrada `esp32_bridge` en
`setup.py`. Todo en un worktree aparte (`feat/real-robot`); `package.xml`,
`setup.py` y el módulo de slam_toolbox se integran después del commit de la sesión
que trabaja en `sim.launch.py`, sin tocar sus archivos antes.

## 11. Correcciones del code review cubiertas

| hallazgo | dónde se resuelve |
|---|---|
| referencias viejas tras reset por sobrecorriente | `stopControl()` (4.4) |
| encoders a 5 V en GPIO de 3.3 V | divisor en el arnés (7, punto 1) |
| PI sin anti-windup | integración condicional (4.5) |
| latch con una sola lectura de D23 | 20 ms continuos (4.3) |
| IMU anula giros lentos y contamina el bias | bias solo con el robot quieto (4.6) |
| leer y borrar el PCNT pierde pulsos | `pulse_cnt` con `accum_count` (4.3) |
| firmware colgado sin IMU | `imu_missing` y sigue (4.6) |
| bridge bloqueado si la ESP32 no reinicia | bridge sin estados (5.1) |
| enlace serial casi saturado | una línea de 50 Hz, ~20 % (4.7) |
| R4 sube GPIO23 a 5 V | desoldar R4 (7, punto 2) |
| GPIO12 pin de strapping | eFuse opcional (7, punto 4) |
| LDO de U4 cargado mueve la referencia VD | calibrar RV1 con encoders conectados (7, punto 3) |
| `brakeMotor()` en realidad es coast | nombre correcto (4.3) |
| PCNT legacy y ramas del core v2 | eliminados (4.9) |
| salidas de depuración sin consumidor | eliminadas (4.9) |
