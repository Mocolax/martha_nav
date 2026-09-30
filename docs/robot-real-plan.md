# Robot real: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Martha mapea un lugar con slam_toolbox y navega en él con `global_planner` + `ppo_local_planner` sobre el robot físico, con el firmware de la ESP32 portado y corregido.

**Architecture:** La ESP32 mide (velocidades de los encoders + gyro z) y es la única autoridad de las protecciones; `esp32_bridge` solo traduce serial ↔ ROS; el EKF de `robot_localization` fusiona y publica `/odom` + `odom → base_link`; slam_toolbox da `map → odom` (mapeo o localización) y `world_map_publisher` publica el mapa congelado para el A*.

**Tech Stack:** arduino-esp32 3.3.10 (ESP-IDF 5, driver `pulse_cnt`), ROS 2 Humble, rclpy, pyserial, robot_localization, slam_toolbox, nav2_map_server (`map_saver_cli`), rplidar_ros, numpy, pytest.

**Spec:** `docs/robot-real-design.md` (aprobado el 2026-09-29). Léelo junto con este plan.

**Estado de verificación:** todo el código de este plan **se prototipó y verificó** antes de escribirlo: el firmware compila con `--warnings all` sin avisos (344 líneas en 3 archivos; antes 1630 en 11), la comprobación de `control.h` en el PC pasa, los 177 tests del paquete pasan con los archivos nuevos, un `map_saver_cli` real se leyó de vuelta idéntico al mapa original, y `real.launch.py` corrió de punta a punta con una ESP32 falsa en un pseudo-terminal (`/odom`, `/imu`, TF `odom → base_link`, `cmd_vel` y `reset` llegando al puerto). Lo que falta es solo lo que exige el hardware (sección final).

## Global Constraints

- **Código mínimo** (skill `minimalist`): sin abstracciones, archivos ni logs que el plan no pida. Si algo parece faltar, pregunta antes de agregarlo.
- **Worktree:** todo se hace en `~/ros2_ws/build/wt/martha_nav_real` (rama `feat/real-robot`), **no** en `~/ros2_ws/src/martha_nav`, donde otra sesión tiene cambios sin commitear. Va dentro de `build/` porque colcon ignora esa carpeta (tiene `COLCON_IGNORE`), así el workspace no ve un segundo paquete `martha_nav`, y porque el contenedor la monta en `/home/ros/ros2_ws/build/`.
- **No tocar hasta la tarea 6** los archivos de la otra sesión: `launch/sim.launch.py`, `martha_nav/ros/gazebo_ground_truth_tf.py`, `martha_nav/ros/evaluate_gazebo.py`, `test/test_evaluate_gazebo.py`, `test/test_gazebo_ground_truth_tf.py`, `tools/map_world.py`, `tools/plot_e2.py`.
- **Pruebas en el contenedor**, desde el worktree. En cada terminal del anfitrión define primero:
  ```bash
  wt() { docker exec -i -w /home/ros/ros2_ws/build/wt/martha_nav_real ros2_humble bash -lc "source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && $*"; }
  ```
  `python3 -m pytest` pone el worktree primero en `sys.path`, así que importa su `martha_nav` y no el instalado.
- **Firmware:** FQBN `esp32:esp32:esp32doit-devkit-v1`, core `esp32:esp32` 3.3.10 (ya instalado en el anfitrión). Compila con `--warnings all` y **cero** avisos.
- **Protocolo serial** (115200 baud), textual y exacto: PC → ESP32 `cmd_vel,vx,vy,wz` y `reset`; ESP32 → PC `odom,vx,vy,wz,gz` (50 Hz), `battery,V` (1 Hz) y los eventos `ready`, `motor_overcurrent`, `battery_too_low`, `cmd_vel_timeout`, `reset_blocked`, `imu_missing`.
- **Constantes del robot real:** r = 0.075 m, lx + ly = 0.385 m, 3200 cuentas/vuelta, Kp = 2, Ki = 1.6, techo |vx|, |vy| ≤ 0.5 m/s y |wz| ≤ 1.2 rad/s, batería latch < 11.0 V y reset ≥ 11.5 V, timeout de `cmd_vel` 500 ms, sobrecorriente 20 ms continuos.
- **Commits:** en español o inglés como el resto del repo (inglés, en imperativo), terminando con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git` corre en el anfitrión, dentro del worktree.

---

### Task 1: Firmware de la ESP32

**Files:**
- Create: `firmware/control.h`, `firmware/sensors.h`, `firmware/firmware.ino`
- Check (fuera del repo): `/tmp/control_check/Arduino.h`, `/tmp/control_check/check.cpp`

**Interfaces:**
- Consumes: nada.
- Produces: el protocolo serial de las Global Constraints (lo consume `esp32_bridge`, tarea 2).

- [ ] **Step 1: Crear el worktree**

```bash
cd ~/ros2_ws/src/martha_nav
git worktree add ~/ros2_ws/build/wt/martha_nav_real -b feat/real-robot
cd ~/ros2_ws/build/wt/martha_nav_real
```

- [ ] **Step 2: Escribir la comprobación de `control.h` en el PC**

`control.h` es matemática pura: se comprueba en el PC con un `Arduino.h` mínimo. Los dos archivos quedan fuera del repo.

```bash
mkdir -p /tmp/control_check
```

`/tmp/control_check/Arduino.h`:
```cpp
// Just enough of Arduino.h to compile control.h on the PC.
#pragma once
#include <math.h>
#define PI 3.14159265358979f
#define constrain(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
```

`/tmp/control_check/check.cpp`:
```cpp
#include <assert.h>
#include <stdio.h>
#include "control.h"

int main() {
  Mecanum m{0.075f, 0.385f};
  float rpm[4], vx, vy, wz;
  m.toWheelsRpm(0.3f, -0.2f, 0.7f, rpm);
  m.toBody(rpm, vx, vy, wz);
  assert(fabsf(vx - 0.3f) < 1e-4f && fabsf(vy + 0.2f) < 1e-4f && fabsf(wz - 0.7f) < 1e-4f);

  m.toWheelsRpm(0.5f, 0.5f, 1.2f, rpm);  // the firmware's speed ceiling
  assert(rpm[1] < 200.0f);               // under the motors' 200 RPM: no wheel limiter needed

  WheelPI pi{2.0f, 1.6f, 255.0f};
  float out = 0;
  for (int i = 0; i < 500; ++i) out = pi.update(80.0f, 0.01f);  // wheel blocked for 5 s
  assert(out == 255.0f && 1.6f * pi.integral < 100.0f);         // integral frozen at saturation
  int ticks = 0;
  do {
    out = pi.update(-30.0f, 0.01f);
    ++ticks;
  } while (out > 0 && ticks < 5000);
  assert(ticks * 0.01f < 1.0f);  // reverses in under 1 s (about 12 s without anti-windup)
  puts("control.h ok");
}
```

- [ ] **Step 3: Verificar que falla**

Run: `cd /tmp/control_check && g++ -std=c++17 -Wall -I. -I ~/ros2_ws/build/wt/martha_nav_real/firmware check.cpp -o check`
Expected: FAIL con `fatal error: control.h: No such file or directory`.

- [ ] **Step 4: Escribir `firmware/control.h`**

```cpp
// Pure math of the drive: the wheel PI and the mecanum kinematics.
#pragma once
#include <Arduino.h>

// PI on one wheel, in PWM counts per RPM of error. Conditional integration: while the
// output is saturated and the error pushes it further, the integral stops growing, so a
// blocked wheel does not wind it up and the wheel reverses as soon as it is told to.
struct WheelPI {
  float kp, ki, limit, integral = 0;

  float update(float error, float dt) {
    float out = kp * error + ki * integral;
    if (fabsf(out) < limit || error * out < 0) {
      integral += error * dt;
      out = kp * error + ki * integral;
    }
    return constrain(out, -limit, limit);
  }

  void reset() { integral = 0; }
};

// Wheel order FL, FR, RL, RR and the sign convention of ros2_controllers'
// mecanum_drive_controller. k = lx + ly.
struct Mecanum {
  float radius, k;

  void toWheelsRpm(float vx, float vy, float wz, float rpm[4]) const {
    const float scale = 60.0f / (2.0f * PI * radius);
    rpm[0] = (vx - vy - k * wz) * scale;
    rpm[1] = (vx + vy + k * wz) * scale;
    rpm[2] = (vx + vy - k * wz) * scale;
    rpm[3] = (vx - vy + k * wz) * scale;
  }

  void toBody(const float rpm[4], float &vx, float &vy, float &wz) const {
    const float scale = 2.0f * PI * radius / 60.0f / 4.0f;
    vx = (rpm[0] + rpm[1] + rpm[2] + rpm[3]) * scale;
    vy = (-rpm[0] + rpm[1] + rpm[2] - rpm[3]) * scale;
    wz = (-rpm[0] + rpm[1] - rpm[2] + rpm[3]) * scale / k;
  }
};
```

- [ ] **Step 5: Verificar que pasa**

Run: `cd /tmp/control_check && g++ -std=c++17 -Wall -I. -I ~/ros2_ws/build/wt/martha_nav_real/firmware check.cpp -o check && ./check`
Expected: `control.h ok` (ida y vuelta exacta de la cinemática, el techo deja las ruedas en 186 RPM < 200, el integral se congela en ~96 y la rueda revierte en 0.76 s).

- [ ] **Step 6: Escribir `firmware/sensors.h`**

Encoders con el driver nuevo `pulse_cnt` (el contador nunca se borra) y solo el gyro z del MPU-6050. La configuración de flancos y niveles replica la del firmware viejo (`martha/arduino/EncoderPCNT.cpp`), así el signo de las cuentas no cambia.

```cpp
// Drivers of the two sensors: wheel encoders on the PCNT and the MPU-6050 gyro.
#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <driver/pulse_cnt.h>

// x4 quadrature on one PCNT unit. accum_count keeps the count across the unit's
// limits, so it is read as a running total and never cleared: no pulse is lost
// between reading and clearing.
struct Encoder {
  pcnt_unit_handle_t unit = nullptr;
  int last = 0;

  void begin(int a, int b) {
    pcnt_unit_config_t config = {};
    config.low_limit = -32768;
    config.high_limit = 32767;
    config.flags.accum_count = 1;
    ESP_ERROR_CHECK(pcnt_new_unit(&config, &unit));
    pcnt_glitch_filter_config_t filter = {.max_glitch_ns = 1000};
    ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(unit, &filter));
    addChannel(a, b, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE);
    addChannel(b, a, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE);
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(unit, config.low_limit));
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(unit, config.high_limit));
    ESP_ERROR_CHECK(pcnt_unit_enable(unit));
    ESP_ERROR_CHECK(pcnt_unit_clear_count(unit));
    ESP_ERROR_CHECK(pcnt_unit_start(unit));
  }

  void addChannel(int edge, int level, pcnt_channel_edge_action_t rising,
                  pcnt_channel_edge_action_t falling) {
    pcnt_chan_config_t config = {};
    config.edge_gpio_num = edge;
    config.level_gpio_num = level;
    pcnt_channel_handle_t channel;
    ESP_ERROR_CHECK(pcnt_new_channel(unit, &config, &channel));
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(channel, rising, falling));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(channel, PCNT_CHANNEL_LEVEL_ACTION_INVERSE,
                                                  PCNT_CHANNEL_LEVEL_ACTION_KEEP));
  }

  int delta() {
    int now;
    pcnt_unit_get_count(unit, &now);
    const int d = now - last;
    last = now;
    return d;
  }
};

// MPU-6050 gyro z only (the EKF fuses nothing else from it). The bias is averaged at
// boot and afterwards relearned only when the caller knows the robot is still: the
// gyro's own reading cannot tell a slow turn from no turn.
struct Gyro {
  static constexpr uint8_t ADDRESS = 0x68;
  static constexpr float LSB_PER_RAD_S = 131.0f * 180.0f / PI;  // +-250 deg/s
  bool ok = false;
  float bias = 0;

  bool begin(int sda, int scl) {
    Wire.begin(sda, scl, 400000);
    ok = write(0x6B, 0x00)    // wake up
         && write(0x1A, 0x03)  // 44 Hz low-pass
         && write(0x1B, 0x00); // +-250 deg/s
    if (!ok) return false;
    delay(100);
    long sum = 0;
    int16_t z;
    for (int i = 0; i < 200; ++i) {
      if (!raw(z)) return ok = false;
      sum += z;
      delay(5);
    }
    bias = sum / 200.0f;
    return true;
  }

  float read(bool still) {  // rad/s, NAN without an IMU
    int16_t z;
    if (!ok || !raw(z)) return NAN;
    if (still) bias += 0.01f * (z - bias);
    return (z - bias) / LSB_PER_RAD_S;
  }

  bool write(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(ADDRESS);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
  }

  bool raw(int16_t &z) {
    Wire.beginTransmission(ADDRESS);
    Wire.write(0x47);  // GYRO_ZOUT_H
    if (Wire.endTransmission(false) != 0 || Wire.requestFrom(ADDRESS, (uint8_t)2) != 2) return false;
    const uint8_t high = Wire.read();
    const uint8_t low = Wire.read();
    z = (int16_t)((high << 8) | low);
    return true;
  }
};
```

- [ ] **Step 7: Escribir `firmware/firmware.ino`**

`arm()` es el único camino para despertar los drivers: lo usan el arranque (con el umbral de batería baja) y el comando `reset` (con el de rearme). `stopControl()` es la única parada.

```cpp
// Martha's ESP32: PI control of the four mecanum wheels, and wheel odometry plus gyro
// to the PC. Protocol and design: docs/robot-real-design.md.
#include "control.h"
#include "sensors.h"

// Pins, checked against the printed board (2026-09-21). Wheel order FL, FR, RL, RR.
const int ENC_A[4] = {12, 36, 4, 21};
const int ENC_B[4] = {13, 39, 16, 22};
const int IN1[4] = {27, 32, 15, 18};
const int IN2[4] = {14, 33, 2, 19};
const int SLEEP_PIN = 17;        // nSLEEP of the four drivers
const int OVERCURRENT_PIN = 23;  // LM339, low = overcurrent; powered only while awake
const int BATTERY_PIN = 34;      // 39k / 10k divider
const int SDA_PIN = 25, SCL_PIN = 26;

// Flip one entry if that wheel turns or counts backwards.
const int MOTOR_SIGN[4] = {1, 1, 1, 1};
const int ENCODER_SIGN[4] = {1, 1, 1, 1};

const float WHEEL_RADIUS = 0.075f;       // m
const float LX_PLUS_LY = 0.385f;         // m
const float COUNTS_PER_REV = 3200.0f;    // 64 CPR x 50:1
const float MAX_V = 0.5f, MAX_W = 1.2f;  // safety ceiling, above the policy's limits
const float KP = 2.0f, KI = 1.6f;        // PWM counts per RPM
const float BATTERY_GAIN = 4.9f;
const float BATTERY_LOW = 11.0f, BATTERY_RESET = 11.5f;  // V

const unsigned long CONTROL_US = 10000;  // 100 Hz; odom every 2 ticks, battery every 100
const unsigned long CMD_TIMEOUT_MS = 500, OVERCURRENT_MS = 20, SETTLE_MS = 5, STILL_MS = 500;

Mecanum mecanum{WHEEL_RADIUS, LX_PLUS_LY};
WheelPI wheelPI[4] = {{KP, KI, 255}, {KP, KI, 255}, {KP, KI, 255}, {KP, KI, 255}};
Encoder encoders[4];
Gyro gyro;

float refRpm[4];
int direction[4];         // sign of the last PWM, to coast one cycle on a reversal
long reportCounts[4];     // pulses since the last odom line
float reportDt = 0, battery = 0;
unsigned tick = 0;
bool latched = true;      // motors asleep until arm() accepts; overcurrent or low battery
bool awake = false, cmdActive = false;
unsigned long awakeSinceMs = 0, overcurrentSinceMs = 0, lastCmdMs = 0, lastMotionMs = 0;
unsigned long lastControlUs = 0;

float readBattery() { return analogReadMilliVolts(BATTERY_PIN) * 1e-3f * BATTERY_GAIN; }

void setAwake(bool on) {
  if (on && !awake) awakeSinceMs = millis();
  awake = on;
  digitalWrite(SLEEP_PIN, on ? HIGH : LOW);
}

// IN1 = IN2 = 0 is coast on the MPQ6612A (brake would be 1/1).
void drive(int i, float out) {
  int dir = (out > 0) - (out < 0);
  if (dir != 0 && direction[i] != 0 && dir != direction[i]) dir = 0;
  const int duty = dir ? (int)fabsf(out) : 0;
  ledcWrite(IN1[i], dir > 0 ? duty : 0);
  ledcWrite(IN2[i], dir < 0 ? duty : 0);
  direction[i] = dir;
}

// Every stop goes through here, so no stop leaves an old reference behind.
void stopControl() {
  for (int i = 0; i < 4; ++i) {
    refRpm[i] = 0;
    wheelPI[i].reset();
    drive(i, 0);
  }
}

void latch(const char *event) {
  if (latched) return;
  latched = true;
  cmdActive = false;
  stopControl();
  setAwake(false);
  Serial.println(event);
}

// Wakes the drivers only if the battery and the overcurrent comparator allow it.
void arm(float minBattery) {
  if (battery < minBattery) {
    Serial.println("reset_blocked");
    return;
  }
  setAwake(true);
  delay(SETTLE_MS + 1);  // the LM339 runs on a driver's LDO, off while asleep
  if (digitalRead(OVERCURRENT_PIN) == LOW) {
    setAwake(false);
    Serial.println("reset_blocked");
    return;
  }
  latched = false;
  stopControl();
  Serial.println("ready");
}

void handle(const char *line) {
  float vx, vy, wz;
  if (sscanf(line, "cmd_vel,%f,%f,%f", &vx, &vy, &wz) == 3) {
    if (latched || !isfinite(vx) || !isfinite(vy) || !isfinite(wz)) return;
    mecanum.toWheelsRpm(constrain(vx, -MAX_V, MAX_V), constrain(vy, -MAX_V, MAX_V),
                        constrain(wz, -MAX_W, MAX_W), refRpm);
    lastCmdMs = millis();
    cmdActive = true;
  } else if (strcmp(line, "reset") == 0) {
    arm(BATTERY_RESET);
  }
}

void readSerial() {
  static char line[48];
  static size_t length = 0;
  while (Serial.available()) {
    const char c = Serial.read();
    if (c == '\n' || c == '\r') {
      line[length] = '\0';
      if (length) handle(line);
      length = 0;
    } else if (length < sizeof(line) - 1) {
      line[length++] = c;
    }
  }
}

void report() {
  float rpm[4], vx, vy, wz;
  for (int i = 0; i < 4; ++i) {
    rpm[i] = reportCounts[i] / COUNTS_PER_REV * 60.0f / reportDt;
    reportCounts[i] = 0;
  }
  reportDt = 0;
  mecanum.toBody(rpm, vx, vy, wz);
  const bool still = millis() - lastMotionMs > STILL_MS;
  Serial.printf("odom,%.4f,%.4f,%.4f,%.4f\n", vx, vy, wz, gyro.read(still));
}

void control() {
  const unsigned long now = micros();
  if (now - lastControlUs < CONTROL_US) return;
  const float dt = (now - lastControlUs) * 1e-6f;
  lastControlUs = now;

  for (int i = 0; i < 4; ++i) {
    const int counts = ENCODER_SIGN[i] * encoders[i].delta();
    reportCounts[i] += counts;
    if (counts != 0 || refRpm[i] != 0) lastMotionMs = millis();
    const float rpm = counts / COUNTS_PER_REV * 60.0f / dt;
    drive(i, MOTOR_SIGN[i] * wheelPI[i].update(refRpm[i] - rpm, dt));
  }

  battery += dt / 0.5f * (readBattery() - battery);  // 0.5 s low-pass
  if (battery < BATTERY_LOW) latch("battery_too_low");

  reportDt += dt;
  ++tick;
  if (tick % 2 == 0) report();
  if (tick % 100 == 0) Serial.printf("battery,%.2f\n", battery);
}

void protections() {
  const unsigned long now = millis();
  const bool low = awake && now - awakeSinceMs > SETTLE_MS &&
                   digitalRead(OVERCURRENT_PIN) == LOW;
  if (!low) {
    overcurrentSinceMs = now;
  } else if (now - overcurrentSinceMs >= OVERCURRENT_MS) {
    latch("motor_overcurrent");
  }

  if (cmdActive && now - lastCmdMs > CMD_TIMEOUT_MS) {
    cmdActive = false;
    stopControl();
    Serial.println("cmd_vel_timeout");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(SLEEP_PIN, OUTPUT);
  setAwake(false);
  pinMode(OVERCURRENT_PIN, INPUT_PULLUP);
  for (int i = 0; i < 4; ++i) {
    ledcAttach(IN1[i], 10000, 8);
    ledcAttach(IN2[i], 10000, 8);
    encoders[i].begin(ENC_A[i], ENC_B[i]);
  }
  if (!gyro.begin(SDA_PIN, SCL_PIN)) Serial.println("imu_missing");
  battery = readBattery();
  lastControlUs = micros();
  arm(BATTERY_LOW);
}

void loop() {
  readSerial();
  control();
  protections();
}
```

- [ ] **Step 8: Compilar sin avisos**

Run: `cd ~/ros2_ws/build/wt/martha_nav_real && arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 --warnings all firmware 2>&1 | grep -iE "warning|error|Sketch uses"`
Expected: una sola línea, `Sketch uses 352723 bytes (26%) ...` (el número exacto puede variar unos bytes); ninguna línea con `warning` ni `error`.

- [ ] **Step 9: Commit**

```bash
git add firmware/
git commit -m "Port the ESP32 firmware from martha/arduino with the review fixes

PI with conditional integration, pulse_cnt encoders read as running totals,
one stopControl() for every stop, a 20 ms overcurrent debounce, gyro z with
the bias learnt only while the robot is still, and a single 50 Hz odom line.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `esp32_bridge`

**Files:**
- Create: `martha_nav/ros/esp32_bridge.py`, `test/test_esp32_bridge.py`
- Modify: `setup.py` (entrada `esp32_bridge`), `package.xml` (`python3-serial`, `std_srvs`), `test/test_package.py` (`EXECUTABLES`)

**Interfaces:**
- Consumes: el protocolo serial de la tarea 1.
- Produces: el ejecutable `esp32_bridge` (parámetro `port`); publica `/wheel/odometry` (`nav_msgs/Odometry`, solo twist, frames `odom`/`base_link`) y `/imu` (`sensor_msgs/Imu`, solo `angular_velocity.z`, frame `base_link`); escucha `/cmd_vel`; servicio `/esp32_bridge/reset` (`std_srvs/Trigger`). Funciones puras `format_cmd(vx, vy, wz) -> bytes`, `parse_line(line) -> tuple | None`, `odometry_msg(vx, vy, wz, stamp)`, `imu_msg(gz, stamp)`.

- [ ] **Step 1: Escribir los tests**

`test/test_esp32_bridge.py`:
```python
import math

import pytest
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist

from martha_nav.ros import esp32_bridge
from martha_nav.ros.esp32_bridge import format_cmd, imu_msg, odometry_msg, parse_line


def test_command_line_the_firmware_parses():
    assert format_cmd(0.25, -0.1, 0.8) == b'cmd_vel,0.250,-0.100,0.800\n'


def test_parse_odom_keeps_a_missing_gyro_as_nan():
    kind, values = parse_line('odom,0.1,-0.2,0.3,nan')
    assert kind == 'odom' and values[:3] == [0.1, -0.2, 0.3] and math.isnan(values[3])


@pytest.mark.parametrize('line, parsed', [
    ('battery,12.34', ('battery', [12.34])),
    ('motor_overcurrent', ('event', 'motor_overcurrent')),
    ('dom,0.1,-0.2,0.3,0.4', None),     # the partial first line after opening the port
    ('odom,0.1,-0.2', None),            # missing fields
    ('odom,0.1,x,0.3,0.4', None),
])
def test_parse_line(line, parsed):
    assert parse_line(line) == parsed


def test_odometry_carries_velocities_for_the_ekf():
    msg = odometry_msg(0.2, -0.1, 0.5, Time(sec=3))
    assert (msg.header.frame_id, msg.child_frame_id) == ('odom', 'base_link')
    twist = msg.twist.twist
    assert (twist.linear.x, twist.linear.y, twist.angular.z) == (0.2, -0.1, 0.5)
    assert msg.twist.covariance[0] == 0.02 and msg.twist.covariance[35] == 0.05


def test_imu_is_yaw_rate_only():
    msg = imu_msg(0.3, Time(sec=3))
    assert msg.header.frame_id == 'base_link' and msg.angular_velocity.z == 0.3
    assert msg.angular_velocity_covariance[8] == 0.02
    assert msg.orientation_covariance[0] == -1.0 and msg.linear_acceleration_covariance[0] == -1.0


class FakeSerial:
    def __init__(self, *_args, **_kwargs):
        self.rx, self.tx = b'', []

    def read(self, size):
        data, self.rx = self.rx[:size], self.rx[size:]
        return data

    def write(self, data):
        self.tx.append(data)

    def close(self):
        pass


class Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(esp32_bridge.serial, 'Serial', FakeSerial)
    rclpy.init()
    node = esp32_bridge.Esp32Bridge()
    node.odom_pub, node.imu_pub = Recorder(), Recorder()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def test_lines_split_across_reads_are_joined(node):
    node.serial.rx = b'ery,12.1\nodom,0.1,0.0,0.2,0.05\nodom,0.2,0.0,0.'
    node.read()
    assert len(node.odom_pub.msgs) == 1 and len(node.imu_pub.msgs) == 1
    node.serial.rx = b'3,nan\n'
    node.read()
    assert node.odom_pub.msgs[-1].twist.twist.angular.z == 0.3
    assert len(node.imu_pub.msgs) == 1        # no gyro in that line: no /imu


def test_cmd_vel_and_reset_reach_the_port(node):
    twist = Twist()
    twist.linear.x, twist.linear.y, twist.angular.z = 0.3, 0.1, -0.5
    node.on_cmd_vel(twist)
    node.on_reset(None, esp32_bridge.Trigger.Response())
    assert node.serial.tx == [b'cmd_vel,0.300,0.100,-0.500\n', b'reset\n']
```

- [ ] **Step 2: Verificar que fallan**

Run: `wt python3 -m pytest test/test_esp32_bridge.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'martha_nav.ros.esp32_bridge'`.

- [ ] **Step 3: Escribir `martha_nav/ros/esp32_bridge.py`**

```python
"""/cmd_vel -> the ESP32 over serial; wheel odometry and gyro back to ROS.

The firmware owns the protections and the speed ceiling: this node only translates.
Protocol: docs/robot-real-design.md, section 4.7.
"""
import math
from contextlib import suppress

import serial
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_srvs.srv import Trigger

from martha_nav.ros.common import run_node

UNKNOWN = 1e6  # variance of what is not measured
# Same variances as martha/cmd_vel_serial_bridge.py: the EKF was tuned with them.
TWIST_VARIANCE = [0.02, 0.05, UNKNOWN, UNKNOWN, UNKNOWN, 0.05]
GYRO_Z_VARIANCE = 0.02
DATA_FIELDS = {'odom': 4, 'battery': 1}


def format_cmd(vx, vy, wz):
    return f'cmd_vel,{vx:.3f},{vy:.3f},{wz:.3f}\n'.encode()


def parse_line(line):
    """('odom', [vx, vy, wz, gz]), ('battery', [volts]), ('event', line), or None.

    None is a garbled line, typically the partial first line after opening the port.
    """
    kind, _, rest = line.partition(',')
    if kind not in DATA_FIELDS:
        return ('event', line) if line.isidentifier() else None
    try:
        values = [float(v) for v in rest.split(',')]
    except ValueError:
        return None
    return (kind, values) if len(values) == DATA_FIELDS[kind] else None


def odometry_msg(vx, vy, wz, stamp):
    """Velocities only: the EKF integrates the pose."""
    msg = Odometry()
    msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, 'odom', 'base_link'
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z = vx, vy, wz
    msg.twist.covariance = [0.0] * 36
    for i, variance in enumerate(TWIST_VARIANCE):
        msg.twist.covariance[i * 7] = variance
    return msg


def imu_msg(gz, stamp):
    """Yaw rate only; covariance[0] = -1 marks orientation and acceleration as absent."""
    msg = Imu()
    msg.header.stamp, msg.header.frame_id = stamp, 'base_link'
    msg.angular_velocity.z = gz
    msg.angular_velocity_covariance = [UNKNOWN, 0.0, 0.0, 0.0, UNKNOWN, 0.0, 0.0, 0.0, GYRO_Z_VARIANCE]
    msg.orientation_covariance[0] = -1.0
    msg.linear_acceleration_covariance[0] = -1.0
    return msg


class Esp32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')
        port = self.declare_parameter('port', '/dev/ttyUSB0').value
        self.serial = serial.Serial(port, 115200, timeout=0)
        self.buffer = b''
        self.last_odom = self.get_clock().now()
        self.odom_pub = self.create_publisher(Odometry, '/wheel/odometry', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_service(Trigger, '~/reset', self.on_reset)
        self.create_timer(0.01, self.read)

    def on_cmd_vel(self, msg):
        self.serial.write(format_cmd(msg.linear.x, msg.linear.y, msg.angular.z))

    def on_reset(self, _request, response):
        self.serial.write(b'reset\n')
        response.success = True
        response.message = 'reset sent; the answer (ready / reset_blocked) is in this log'
        return response

    def read(self):
        *lines, self.buffer = (self.buffer + self.serial.read(4096)).split(b'\n')
        now = self.get_clock().now()
        for raw in lines:
            parsed = parse_line(raw.decode(errors='replace').strip())
            if parsed is None:
                continue
            kind, values = parsed
            if kind == 'odom':
                self.last_odom = now
                stamp = now.to_msg()
                self.odom_pub.publish(odometry_msg(*values[:3], stamp))
                if math.isfinite(values[3]):
                    self.imu_pub.publish(imu_msg(values[3], stamp))
            elif kind == 'battery':
                self.get_logger().info(f'battery {values[0]:.2f} V', throttle_duration_sec=30.0)
            else:
                self.get_logger().warning(f'ESP32: {values}')
        if now - self.last_odom > Duration(seconds=1.0):
            self.get_logger().warning('no odometry from the ESP32 for 1 s',
                                      throttle_duration_sec=5.0)

    def destroy_node(self):
        with suppress(serial.SerialException):   # the port may be what failed (unplugged)
            self.serial.write(format_cmd(0.0, 0.0, 0.0))
        self.serial.close()
        super().destroy_node()


def main():
    run_node(Esp32Bridge)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verificar que pasan**

Run: `wt python3 -m pytest test/test_esp32_bridge.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Registrar el ejecutable y actualizar su test**

En `test/test_package.py`, reemplaza:

```python
EXECUTABLES = ['evaluate_2d', 'evaluate_gazebo', 'gazebo_ground_truth_tf', 'global_planner',
               'mecanum_cmd_vel_bridge', 'ppo_local_planner', 'train_policy',
               'world_map_publisher']
```

por:

```python
EXECUTABLES = ['esp32_bridge', 'evaluate_2d', 'evaluate_gazebo', 'gazebo_ground_truth_tf',
               'global_planner', 'mecanum_cmd_vel_bridge', 'ppo_local_planner', 'train_policy',
               'world_map_publisher']
```

Run: `wt python3 -m pytest test/test_package.py -q`
Expected: FAIL en `test_executables_say_what_they_do` (falta la entrada en `setup.py`).

En `setup.py`, después de la línea `'evaluate_gazebo = martha_nav.ros.evaluate_gazebo:main',` agrega:

```python
            'esp32_bridge = martha_nav.ros.esp32_bridge:main',
```

En `package.xml`, después de `<exec_depend>joint_state_broadcaster</exec_depend>` agrega:

```xml
  <exec_depend>python3-serial</exec_depend>
  <exec_depend>std_srvs</exec_depend>
```

Run: `wt python3 -m pytest test/test_package.py test/test_esp32_bridge.py -q`
Expected: todos pasan.

- [ ] **Step 6: Commit**

```bash
git add martha_nav/ros/esp32_bridge.py test/test_esp32_bridge.py test/test_package.py setup.py package.xml
git commit -m "Add esp32_bridge: /cmd_vel to the ESP32, wheel odometry and gyro back

A translator without state: the firmware owns the protections and the speed
ceiling. A reset service re-arms the motors after a latch.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `world_map_publisher` publica un mapa guardado

**Files:**
- Modify: `martha_nav/ros/world_map_publisher.py` (archivo completo abajo)
- Create: `test/test_world_map_publisher.py`

**Interfaces:**
- Consumes: el `.yaml` + `.pgm` que escribe `map_saver_cli` (tarea 4, `tools/save_map.sh`).
- Produces: parámetro nuevo `map_yaml` (ruta absoluta; vacío = rasterizar `world` como hasta ahora); funciones `read_pgm(path) -> np.ndarray` y `load_map_yaml(path) -> Grid`.

- [ ] **Step 1: Escribir el test**

`test/test_world_map_publisher.py`:
```python
import numpy as np

from martha_nav.ros.world_map_publisher import load_map_yaml

# 3 wide, 2 tall, as map_saver writes it: top row first; 254 free, 0 occupied, 205 unknown.
PGM = b'P5\n# CREATOR: map_saver.cpp 0.050 m/pix\n3 2\n255\n' + bytes([0, 254, 205, 254, 254, 254])
YAML = 'image: lab_real.pgm\nmode: trinary\nresolution: 0.05\norigin: [-1.5, -0.5, 0]\n' \
       'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n'


def test_saved_map_becomes_a_grid(tmp_path):
    (tmp_path / 'lab_real.pgm').write_bytes(PGM)
    (tmp_path / 'lab_real.yaml').write_text(YAML)
    grid = load_map_yaml(tmp_path / 'lab_real.yaml')
    assert grid.origin == (-1.5, -0.5) and grid.resolution == 0.05
    # Row 0 is the bottom of the image; the occupied and the unknown pixel block.
    assert np.array_equal(grid.occ, [[False, False, False], [True, False, True]])
```

- [ ] **Step 2: Verificar que falla**

Run: `wt python3 -m pytest test/test_world_map_publisher.py -q`
Expected: FAIL con `ImportError: cannot import name 'load_map_yaml'`.

- [ ] **Step 3: Reemplazar `martha_nav/ros/world_map_publisher.py`**

```python
"""Publish a latched /map: a rasterised .world, or a map saved on the real robot."""
import re
from pathlib import Path

import numpy as np
import yaml
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from martha_nav.ros.common import LATCHED, run_node
from martha_nav.ros.occupancy import grid_to_msg
from martha_nav.sim2d.geometry import Grid
from martha_nav.sim2d.worlds import rasterize_world


def read_pgm(path):
    """Binary 8-bit PGM (P5), as map_saver and GIMP write it."""
    data = Path(path).read_bytes()
    tokens = []
    for match in re.finditer(rb'#[^\n]*|(\S+)', data):
        if match.group(1):
            tokens.append(match.group(1))
        if len(tokens) == 4:
            break
    if tokens[0] != b'P5':
        raise ValueError(f'{path}: only binary PGM (P5) is supported')
    width, height = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data, np.uint8, width * height, match.end() + 1).reshape(height, width)


def load_map_yaml(path):
    """map_saver's .yaml + .pgm -> Grid. Only white pixels are free: unknown (205) and
    occupied block the planner, as msg_to_grid treats unknown cells."""
    meta = yaml.safe_load(Path(path).read_text())
    image = read_pgm(Path(path).parent / meta['image'])
    occ = np.flipud(image < 250)           # the image starts at the top, the grid at the origin
    origin = (float(meta['origin'][0]), float(meta['origin'][1]))
    return Grid(np.ascontiguousarray(occ), origin, float(meta['resolution']))


class WorldMapPublisher(Node):
    def __init__(self):
        super().__init__('world_map_publisher')
        world = self.declare_parameter('world', 'lab').value
        map_yaml = self.declare_parameter('map_yaml', '').value
        frame = self.declare_parameter('frame_id', 'map').value
        grid = load_map_yaml(map_yaml) if map_yaml else rasterize_world(world)
        msg = grid_to_msg(grid, frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.pub.publish(msg)
        self.get_logger().info(f'published {map_yaml or world}: '
                               f'{msg.info.width}x{msg.info.height} cells @ {msg.info.resolution} m')


def main():
    run_node(WorldMapPublisher)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verificar que pasa, y que lo de antes sigue igual**

Run: `wt python3 -m pytest test/test_world_map_publisher.py test/test_worlds.py test/test_occupancy.py -q`
Expected: todos pasan.

- [ ] **Step 5: Commit**

```bash
git add martha_nav/ros/world_map_publisher.py test/test_world_map_publisher.py
git commit -m "Let world_map_publisher publish a map saved on the real robot

map_yaml loads map_saver's .yaml + .pgm; unknown and occupied pixels block the
planner, like msg_to_grid treats unknown cells.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `real.launch.py`, slam_toolbox, EKF y guardado del mapa

**Files:**
- Create: `martha_nav/ros/slam.py`, `launch/real.launch.py`, `config/ekf.yaml`, `config/rplidar.yaml`, `tools/save_map.sh`
- Modify: `package.xml` (dependencias), `test/test_package.py` (el launch real también debe usar ejecutables instalados)
- Check (fuera del repo): `/tmp/fake_esp32.py`

**Interfaces:**
- Consumes: `esp32_bridge` (tarea 2), `world_map_publisher` con `map_yaml` (tarea 3), `global_planner` y `ppo_local_planner` sin cambios, `urdf/martha.urdf.xacro` con `drive:=planar`.
- Produces: `ros2 launch martha_nav real.launch.py` con los argumentos `map` (ruta sin extensión; vacío = mapear), `checkpoint`, `esp32_port`, `lidar_port`, `rviz`; `slam_toolbox(mode, scan_topic, use_sim_time, publish_tf=True, map_file='', start_pose=(0.0, 0.0, 0.0)) -> launch_ros Node` en `martha_nav/ros/slam.py` (la usa también `sim.launch.py` en la tarea 6); `tools/save_map.sh <ruta sin extensión>`.

- [ ] **Step 1: Extender el test de los launch**

En `test/test_package.py`, reemplaza:

```python
def test_the_launch_file_runs_installed_executables():
    import re
    launched = re.findall(r"package='martha_nav', executable='(\w+)'",
                          (REPO / 'launch' / 'sim.launch.py').read_text())
    assert launched and set(launched) <= set(entry_points())
```

por:

```python
def test_the_launch_files_run_installed_executables():
    import re
    for launch in ('sim.launch.py', 'real.launch.py'):
        launched = re.findall(r"package='martha_nav', executable='(\w+)'",
                              (REPO / 'launch' / launch).read_text())
        assert launched and set(launched) <= set(entry_points()), launch
```

- [ ] **Step 2: Verificar que falla**

Run: `wt python3 -m pytest test/test_package.py -q`
Expected: FAIL con `FileNotFoundError` de `launch/real.launch.py`.

- [ ] **Step 3: Escribir `martha_nav/ros/slam.py`**

Es la misma configuración que la otra sesión puso dentro de `sim.launch.py` (parámetros que trae slam_toolbox + overrides de Martha, `/map` → `/slam_map`), sacada a un módulo para que la usen los dos launch.

```python
"""slam_toolbox as the launch files run it: its packaged parameters plus Martha's.

Its map goes to /slam_map, so it never replaces the /map the global planner uses.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def slam_toolbox(mode, scan_topic, use_sim_time, publish_tf=True, map_file='',
                 start_pose=(0.0, 0.0, 0.0)):
    """mode 'mapping' or 'localization'; map_file is the saved posegraph without extension."""
    config = Path(get_package_share_directory('slam_toolbox')) / 'config'
    params = {'use_sim_time': use_sim_time, 'mode': mode, 'base_frame': 'base_link',
              'scan_topic': scan_topic, 'max_laser_range': 8.0}
    if mode == 'mapping':
        executable, defaults = 'sync_slam_toolbox_node', 'mapper_params_online_sync.yaml'
        if not publish_tf:
            params['transform_publish_period'] = 0.0
    else:
        executable, defaults = 'localization_slam_toolbox_node', 'mapper_params_localization.yaml'
        params['map_file_name'] = map_file
        params['map_start_pose'] = [float(v) for v in start_pose]
    return Node(package='slam_toolbox', executable=executable, name='slam_toolbox',
                output='screen', parameters=[str(config / defaults), params],
                remappings=[('/map', '/slam_map'), ('/map_metadata', '/slam_map_metadata')])
```

- [ ] **Step 4: Escribir las configuraciones**

`config/ekf.yaml`:
```yaml
# Encoders + gyro, the fusion of martha/config/ekf_hardware.yaml. The ESP32 sends
# velocities only, so the pose is integrated here, once.
ekf_filter_node:
  ros__parameters:
    frequency: 50.0
    sensor_timeout: 0.2
    two_d_mode: true
    publish_tf: true
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom

    odom0: /wheel/odometry        # vx, vy, vyaw
    odom0_config: [false, false, false,
                   false, false, false,
                   true,  true,  false,
                   false, false, true,
                   false, false, false]

    imu0: /imu                    # vyaw
    imu0_config: [false, false, false,
                  false, false, false,
                  false, false, false,
                  false, false, true,
                  false, false, false]
```

`config/rplidar.yaml`:
```yaml
# RPLIDAR A2M8, from martha/config/rplidar_a2m8.yaml. serial_port comes from the launch.
# flip_x_axis: true if the scan shows up rotated 180 degrees in RViz.
rplidar_node:
  ros__parameters:
    channel_type: serial
    serial_baudrate: 115200
    frame_id: lidar
    inverted: false
    angle_compensate: true
    flip_x_axis: false
    auto_standby: false
    topic_name: /scan
    scan_mode: Sensitivity
    scan_frequency: 10.0
```

- [ ] **Step 5: Escribir `launch/real.launch.py`**

```python
"""Martha on the real robot: ESP32, RPLIDAR, EKF and slam_toolbox.

Map a place, driving with teleop_twist_keyboard in another terminal, then save it:
    ros2 launch martha_nav real.launch.py esp32_port:=/dev/serial/by-id/... lidar_port:=/dev/serial/by-id/...
    tools/save_map.sh /abs/path/martha_nav/maps/lab_real
Navigate in it, starting on the spot where the mapping began (or use 2D Pose Estimate):
    ros2 launch martha_nav real.launch.py map:=/abs/path/martha_nav/maps/lab_real checkpoint:=/abs/best_model.zip
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from martha_nav.ros.slam import slam_toolbox

ARGUMENTS = [
    DeclareLaunchArgument('map', default_value='',
                          description='saved map without extension; empty maps the place'),
    DeclareLaunchArgument('checkpoint', default_value=''),
    DeclareLaunchArgument('esp32_port', default_value='/dev/ttyUSB0'),
    DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB1'),
    DeclareLaunchArgument('rviz', default_value='false'),
]


def launch_setup(context, *args, **kwargs):
    share = Path(FindPackageShare('martha_nav').perform(context))
    saved_map = LaunchConfiguration('map').perform(context)
    urdf = ParameterValue(Command(['xacro ', str(share / 'urdf' / 'martha.urdf.xacro'),
                                   ' drive:=planar']), value_type=str)
    actions = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': urdf}]),
        Node(package='martha_nav', executable='esp32_bridge', output='screen',
             parameters=[{'port': LaunchConfiguration('esp32_port')}]),
        Node(package='rplidar_ros', executable='rplidar_node', output='screen',
             parameters=[str(share / 'config' / 'rplidar.yaml'),
                         {'serial_port': LaunchConfiguration('lidar_port')}]),
        Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
             output='screen', parameters=[str(share / 'config' / 'ekf.yaml')],
             remappings=[('odometry/filtered', '/odom')]),
        Node(package='rviz2', executable='rviz2', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', str(share / 'rviz' / 'nav.rviz')]),
    ]
    if not saved_map:
        return actions + [slam_toolbox('mapping', '/scan', use_sim_time=False)]
    return actions + [
        slam_toolbox('localization', '/scan', use_sim_time=False, map_file=saved_map),
        Node(package='martha_nav', executable='world_map_publisher', output='screen',
             parameters=[{'map_yaml': saved_map + '.yaml'}]),
        Node(package='martha_nav', executable='global_planner', output='screen'),
        Node(package='martha_nav', executable='ppo_local_planner', output='screen',
             parameters=[{'checkpoint': LaunchConfiguration('checkpoint')}]),
    ]


def generate_launch_description():
    return LaunchDescription(ARGUMENTS + [OpaqueFunction(function=launch_setup)])
```

- [ ] **Step 6: Escribir `tools/save_map.sh`**

Llama a `map_saver_cli` directamente sobre `/slam_map`: el servicio `save_map` de slam_toolbox buscaría `/map`, que está remapeado.

```bash
#!/usr/bin/env bash
# Save the map slam_toolbox is building, for real.launch.py map:=<out>:
#   <out>.pgm/.yaml       the frozen /map of the global planner (editable in GIMP)
#   <out>.posegraph/.data what slam_toolbox localizes against
# Usage (with real.launch.py mapping): tools/save_map.sh /abs/path/martha_nav/maps/lab_real
set -euo pipefail
out=${1:?usage: $0 /abs/path/without/extension}
mkdir -p "$(dirname "$out")"
ros2 run nav2_map_server map_saver_cli -f "$out" -t /slam_map
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$out'}"
```

Run: `chmod +x tools/save_map.sh`

- [ ] **Step 7: Dependencias**

En `package.xml`, después de `<exec_depend>std_srvs</exec_depend>` (tarea 2) agrega:

```xml
  <exec_depend>robot_localization</exec_depend>
  <exec_depend>rplidar_ros</exec_depend>
  <exec_depend>slam_toolbox</exec_depend>
  <exec_depend>nav2_map_server</exec_depend>
  <exec_depend>teleop_twist_keyboard</exec_depend>
```

- [ ] **Step 8: Verificar que los tests pasan**

Run: `wt python3 -m pytest test -q`
Expected: todos pasan (177 al escribir este plan).

- [ ] **Step 9: Prueba de punta a punta con una ESP32 falsa**

Compila el worktree en un install aparte, para no tocar el del workspace:

```bash
wt 'cd /tmp && colcon build --base-paths /home/ros/ros2_ws/build/wt/martha_nav_real --build-base /tmp/pb --install-base /tmp/pi --packages-select martha_nav'
```

`/tmp/fake_esp32.py` (en el anfitrión; cópialo al contenedor con `docker cp /tmp/fake_esp32.py ros2_humble:/tmp/`):
```python
"""A fake ESP32 on a pseudo-terminal, to check esp32_bridge + EKF without hardware.

python3 fake_esp32.py SECONDS writes the pty path to /tmp/fake_esp32_path, sends
`odom` lines at 50 Hz (vx 0.1 m/s, wz 0.2 rad/s) and saves what it received in
/tmp/fake_esp32_rx.
"""
import os
import pty
import select
import sys
import time
import tty

master, slave = pty.openpty()
tty.setraw(slave)                     # no echo: only what the bridge writes comes back
open('/tmp/fake_esp32_path', 'w').write(os.ttyname(slave))
received = b''
end = time.time() + float(sys.argv[1])
os.write(master, b'ready\n')
while time.time() < end:
    os.write(master, b'odom,0.1000,0.0000,0.2000,0.2000\n')
    if select.select([master], [], [], 0.02)[0]:
        received += os.read(master, 4096)
open('/tmp/fake_esp32_rx', 'wb').write(received)
```

```bash
docker exec -i ros2_humble bash -lc 'source /opt/ros/humble/setup.bash; source /home/ros/ros2_ws/install/setup.bash; source /tmp/pi/setup.bash; export ROS_DOMAIN_ID=96
python3 /tmp/fake_esp32.py 20 & sleep 1
ros2 launch martha_nav real.launch.py esp32_port:=$(cat /tmp/fake_esp32_path) lidar_port:=/dev/null > /tmp/real_launch.log 2>&1 &
sleep 8
timeout 5 ros2 topic echo /odom --once --field twist.twist.linear.x
timeout 5 ros2 topic echo /imu --once --field angular_velocity.z
timeout 5 ros2 run tf2_ros tf2_echo odom base_link 2>&1 | grep -m1 Translation
ros2 topic pub -t 2 -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" > /dev/null
ros2 service call /esp32_bridge/reset std_srvs/srv/Trigger > /dev/null
sleep 12; sort /tmp/fake_esp32_rx | uniq -c; kill %2'
```

Expected: `0.1` (el EKF fusiona la odometría), `0.2` (el gyro), una línea `Translation` que avanza, y lo recibido por la ESP32 falsa:

```
      2 cmd_vel,0.200,0.000,0.000
      1 reset
```

`rplidar_node` muere porque `/dev/null` no es un LiDAR, y al terminar la ESP32 falsa el bridge sale con `device disconnected`: las dos cosas son esperadas aquí.

- [ ] **Step 10: Commit**

```bash
git add martha_nav/ros/slam.py launch/real.launch.py config/ekf.yaml config/rplidar.yaml tools/save_map.sh package.xml test/test_package.py
git commit -m "Add real.launch.py: map a place with slam_toolbox, then navigate in it

ESP32 bridge, RPLIDAR and the encoder + gyro EKF always; without map:= slam_toolbox
maps, with map:= it localizes and world_map_publisher serves the frozen map to the
planners. tools/save_map.sh saves both halves of a map.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Documentación

**Files:**
- Modify: `docs/comandos.md` (sección 9 nueva, al final), `docs/contexto.md` (mapa del repositorio y pendientes), `docs/robot-real-design.md` (estado)

**Interfaces:**
- Consumes: los comandos de las tareas 1 a 4.
- Produces: la guía que usará el usuario en el robot.

- [ ] **Step 1: Agregar al final de `docs/comandos.md`**

````markdown
## 9. Robot real

Diseño: [robot-real-design.md](robot-real-design.md). Todo lo de ROS corre en el PC
montado sobre el robot; la ESP32 solo hace motores, encoders, gyro y protecciones.

### 9.1 Puertos

La ESP32 y el RPLIDAR usan el mismo chip USB (CP2102) y `ttyUSB0/1` pueden
intercambiarse entre arranques: usa siempre las rutas estables.

```bash
ls -l /dev/serial/by-id/
```

### 9.2 Firmware (en el anfitrión)

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 --warnings all firmware
arduino-cli upload -p /dev/serial/by-id/<esp32> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware
arduino-cli monitor -p /dev/serial/by-id/<esp32> -c baudrate=115200
```

En el monitor se escriben los comandos a mano: `cmd_vel,0.1,0,0` o `reset`.

### 9.3 Antes de flashear (hardware, una vez)

1. Divisor en los cables amarillo (A) y blanco (B) de cada encoder, del lado de la
   ESP32: señal → 4.7 kΩ → GPIO, y 10 kΩ del GPIO a GND. Con la rueda quieta en
   alto el pin debe leer entre 2.5 y 3.6 V.
2. Desoldar R4 (el firmware usa el pull-up interno).
3. RV1 con los encoders conectados y los drivers despiertos: VD ≈ 0.7 V (~3.5 A por
   motor). Anotar R2/R3.
4. Opcional e irreversible: `espefuse.py --port /dev/serial/by-id/<esp32> set_flash_voltage 3.3V`.

### 9.4 Puesta en marcha (en orden)

1. Ruedas al aire, monitor serial: aparece `ready`; `battery,V` coincide con el
   multímetro (si no, ajustar `BATTERY_GAIN`).
2. `cmd_vel,0.1,0,0`: las 4 ruedas hacia adelante y `odom` con vx > 0. Si una rueda
   gira al revés, su entrada de `MOTOR_SIGN`; si cuenta al revés, la de `ENCODER_SIGN`.
   Repetir con `cmd_vel,0,0.1,0` (vy > 0, hacia la izquierda) y `cmd_vel,0,0,0.5` (wz > 0).
3. La velocidad medida sigue a la ordenada; si no, ajustar `KP` y `KI`.
4. Giro antihorario a mano: el último campo de `odom` (gz) > 0; quieto, ≈ 0.
5. Un solo `cmd_vel,0.1,0,0`: las ruedas giran ~0.5 s y aparece `cmd_vel_timeout`.
   Puente de D23 a GND: `motor_overcurrent`; quitarlo y `reset` → `ready`.
   Fuente de laboratorio < 11 V: `battery_too_low`.
6. ROS en modo mapeo (9.5) con RViz: árbol `map → odom → base_link → lidar`; el scan
   alineado con el frente del robot (si sale girado 180°, `flip_x_axis: true` en
   `config/rplidar.yaml`); empujarlo 1 m → `/odom` ~1 m; girarlo 360° → ~2π.

### 9.5 Mapear un lugar

Pon el robot sobre una marca de cinta en el piso: es donde arrancará la demo.

```bash
./tools/ct_ros ros2 launch martha_nav real.launch.py rviz:=true \
    esp32_port:=/dev/serial/by-id/<esp32> lidar_port:=/dev/serial/by-id/<rplidar>
```

En otra terminal (el teleop necesita una terminal interactiva, por eso `-it`; con
Shift se mueve lateral):

```bash
docker exec -it ros2_humble bash -lc 'source /opt/ros/humble/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard'
```

En RViz agrega un display Map sobre `/slam_map` para ver el mapa crecer. Al terminar:

```bash
./tools/ct_ros tools/save_map.sh /home/ros/ros2_ws/src/martha_nav/maps/<lugar>_real
```

Deja `<lugar>_real.pgm/.yaml` (el mapa del A*; se puede limpiar en GIMP) y
`<lugar>_real.posegraph/.data` (para localizarse).

### 9.6 Demo

Robot sobre la marca de cinta (o "2D Pose Estimate" en RViz), sin las cajas en el mapa:

```bash
./tools/ct_ros ros2 launch martha_nav real.launch.py rviz:=true \
    map:=/home/ros/ros2_ws/src/martha_nav/maps/<lugar>_real \
    checkpoint:=/home/ros/ros2_ws/src/martha_nav/runs/armH_holonomic_s0/best_model.zip \
    esp32_port:=/dev/serial/by-id/<esp32> lidar_port:=/dev/serial/by-id/<rplidar>
```

La meta se da con "2D Goal Pose" en RViz.

### 9.7 Tras un latch (`motor_overcurrent` o `battery_too_low`)

Revisa la causa, cancela la meta para que el PPO no arranque al rearmar, y resetea;
la respuesta (`ready` o `reset_blocked`) sale en el log de `esp32_bridge`:

```bash
./tools/ct_ros ros2 topic pub --once /cancel_goal std_msgs/msg/Empty
./tools/ct_ros ros2 service call /esp32_bridge/reset std_srvs/srv/Trigger
```
````

- [ ] **Step 2: Actualizar `docs/contexto.md`**

En la tabla "Mapa del repositorio", reemplaza la fila de `martha_nav/ros/`:

```markdown
| `martha_nav/ros/` | `global_planner.py`, `ppo_local_planner.py`, `world_map_publisher.py`, `gazebo_ground_truth_tf.py`, `mecanum_cmd_vel_bridge.py`, `world_speed.py`, `evaluate_gazebo.py` |
```

por:

```markdown
| `martha_nav/ros/` | `global_planner.py`, `ppo_local_planner.py`, `world_map_publisher.py` (`.world` o mapa guardado), `gazebo_ground_truth_tf.py`, `mecanum_cmd_vel_bridge.py`, `world_speed.py`, `evaluate_gazebo.py`, `esp32_bridge.py` y `slam.py` (robot real) |
| `firmware/` | ESP32 del robot real: `firmware.ino`, `control.h` (PI, cinemática), `sensors.h` (encoders, gyro) |
```

y la fila de `launch/sim.launch.py` por:

```markdown
| `launch/sim.launch.py` | Gazebo + controladores + nodos + RViz opcional |
| `launch/real.launch.py` | robot real: sin `map:=` mapea con slam_toolbox, con `map:=` navega |
```

En la lista de pendientes, reemplaza:

```markdown
- Integración con el robot real cuando el hardware esté arreglado.
```

por:

```markdown
- Robot real: código listo (`docs/robot-real-design.md`, `docs/robot-real-plan.md`); falta el hardware y la puesta en marcha de `comandos.md` §9.
```

- [ ] **Step 3: Marcar el spec como implementado**

En `docs/robot-real-design.md`, reemplaza:

```markdown
Fecha: 2026-09-29. Estado: aprobado en brainstorming, pendiente de plan de implementación.
```

por:

```markdown
Fecha: 2026-09-29. Estado: implementado (`docs/robot-real-plan.md`); falta la puesta en marcha con el hardware.
```

- [ ] **Step 4: Commit**

```bash
git add docs/comandos.md docs/contexto.md docs/robot-real-design.md
git commit -m "Document how to flash, bring up, map and demo the real robot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Integrar con la sesión de simulación

**Precondición:** la otra sesión hizo commit de su trabajo de slam_toolbox en `feat/sim2d-training` (`sim.launch.py` con `slam:=mapping|localization`). Compruébalo:

```bash
cd ~/ros2_ws/src/martha_nav && git status --short && git log --oneline -5
```

Si `launch/sim.launch.py` sigue apareciendo como modificado sin commit, **para aquí** y avisa al usuario.

**Files:**
- Modify: `launch/sim.launch.py` (usar `martha_nav.ros.slam.slam_toolbox`), `package.xml` (un solo `slam_toolbox`)

**Interfaces:**
- Consumes: `slam_toolbox(...)` de `martha_nav/ros/slam.py` (tarea 4).
- Produces: `feat/real-robot` al día con `feat/sim2d-training`, sin la función duplicada.

- [ ] **Step 1: Traer su trabajo al worktree**

```bash
cd ~/ros2_ws/build/wt/martha_nav_real
git rebase feat/sim2d-training
```

En el conflicto de `package.xml` deja **una sola** línea `<exec_depend>slam_toolbox</exec_depend>` (la traen las dos ramas).

- [ ] **Step 2: Comparar su función con `martha_nav/ros/slam.py`**

```bash
sed -n '/^def slam_toolbox/,/^def generate_launch_description/p' launch/sim.launch.py
```

Al escribir este plan, su versión era equivalente a `slam.py` con `use_sim_time=True`, `scan_topic=SCAN_TOPIC`, `publish_tf=False` al mapear y `start_pose=(x, y, 0.0)` al localizar. Si su versión commiteada agrega algún parámetro más, cópialo a `slam.py` antes de seguir.

- [ ] **Step 3: Usar el módulo compartido en `sim.launch.py`**

Agrega el import junto al de `create_scaled_world`:

```python
from martha_nav.ros.slam import slam_toolbox
```

Reemplaza la llamada:

```python
    if slam != 'off':
        actions.append(slam_toolbox(context, slam))
```

por:

```python
    if slam != 'off':
        actions.append(slam_toolbox(
            slam, SCAN_TOPIC, use_sim_time=True, publish_tf=False,
            map_file=LaunchConfiguration('slam_map').perform(context),
            start_pose=(float(LaunchConfiguration('x').perform(context)),
                        float(LaunchConfiguration('y').perform(context)), 0.0)))
```

y borra la función `slam_toolbox(context, mode)` completa de `sim.launch.py`. (`publish_tf=False` solo actúa al mapear, y `map_file` / `start_pose` solo al localizar, igual que en su versión.)

- [ ] **Step 4: Verificar**

Run: `wt python3 -m pytest test -q`
Expected: todos pasan.

Y que la simulación con slam siga arrancando (Gazebo sin ventana, 40 s):

```bash
wt 'cd /tmp && colcon build --base-paths /home/ros/ros2_ws/build/wt/martha_nav_real --build-base /tmp/pb --install-base /tmp/pi --packages-select martha_nav'
docker exec -i ros2_humble bash -lc 'source /opt/ros/humble/setup.bash; source /home/ros/ros2_ws/install/setup.bash; source /tmp/pi/setup.bash; export ROS_DOMAIN_ID=95
timeout 40 ros2 launch martha_nav sim.launch.py gui:=false slam:=mapping x:=0 y:=0 > /tmp/sim_slam.log 2>&1 &
sleep 30; ros2 node list | grep slam_toolbox; ros2 topic list | grep slam_map'
```

Expected: `/slam_toolbox` y `/slam_map`.

- [ ] **Step 5: Commit y pedir permiso para integrar**

```bash
git add launch/sim.launch.py package.xml martha_nav/ros/slam.py
git commit -m "Run slam_toolbox from one module in the simulated and the real launch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Pregunta al usuario antes de hacer merge de `feat/real-robot` en `feat/sim2d-training` o de hacer push.

---

## Después del plan: lo que exige el hardware

No es código y lo hace el usuario con el robot, en el orden de `docs/comandos.md` §9.3 y §9.4: el divisor de los encoders, desoldar R4, calibrar RV1, flashear, la polaridad, el PI, el gyro, las protecciones y el árbol de TF. Después, §9.5 (mapear) y §9.6 (demo). El riesgo principal es que slam_toolbox pierda la localización con cajas que no están en el mapa; la prueba de la otra sesión en simulación lo mide antes de llegar al robot.
