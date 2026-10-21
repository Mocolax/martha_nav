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
// The IMU board stands on its side with +X up (imu_test_uno, 2026-10-20): yaw is gyro X.
const uint8_t IMU_UP_REG = 0x43;  // GYRO_XOUT_H
const float IMU_UP_SIGN = 1.0f;

// The right motors and the left encoders are mounted mirrored (checked 2026-10-20, wheels
// in the air: forward PWM, forward wheel, positive counts). Flip an entry if that wheel
// turns or counts backwards.
const int MOTOR_SIGN[4] = {1, -1, 1, -1};
const int ENCODER_SIGN[4] = {-1, 1, -1, 1};

const float WHEEL_RADIUS = 0.075f;       // m
const float LX_PLUS_LY = 0.385f;         // m
const float COUNTS_PER_REV = 3200.0f;    // 64 CPR x 50:1
const float MAX_V = 0.5f, MAX_W = 1.2f;  // safety ceiling, above the policy's limits
const float KP = 1.0f, KI = 1.6f;        // PWM counts per RPM, on top of the feedforward
// Slow decay, wheels in the air (2026-10-21): RPM ~ 1.15 x (PWM - 4) on all four wheels,
// moving from PWM ~10. Fast decay needed PWM ~130 just to start, with hysteresis.
const float RPM_PER_PWM = 1.15f, PWM_START = 4.0f;
// Below these a command is a stop, and below PWM_MIN a wheel is coasted: a PWM too small to
// turn the motor only heats it and whistles at the PWM frequency.
const float V_DEADZONE = 0.01f, W_DEADZONE = 0.02f;  // m/s, rad/s
const float PWM_MIN = 8.0f;
const float BATTERY_GAIN = 4.9f;
const float BATTERY_LOW = 11.0f, BATTERY_RESET = 11.5f;  // V
// Circuits not fitted on the board yet (2026-10-21): the battery divider (R5/R6, GPIO34
// floats) and the overcurrent comparator (LM339, R4, RV1). Off, their checks are skipped:
// the robot arms without them and nothing stops it on a low battery or a stalled motor.
const bool BATTERY_SENSE = false;
const bool OVERCURRENT_SENSE = false;

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

// MPQ6612A, slow decay: one input held high, PWM inverted on the other, so the motor
// alternates drive and brake (1/1) instead of drive and coast. IN1 = IN2 = 0 is coast.
void drive(int i, float out) {
  int dir = (out > 0) - (out < 0);
  if (dir != 0 && direction[i] != 0 && dir != direction[i]) dir = 0;
  const int duty = dir ? (int)fminf(fabsf(out), 255.0f) : 0;
  ledcWrite(IN1[i], dir > 0 ? 255 : dir < 0 ? 255 - duty : 0);
  ledcWrite(IN2[i], dir < 0 ? 255 : dir > 0 ? 255 - duty : 0);
  direction[i] = dir;
}

// The PWM that gives `rpm` with the wheels in the air; the PI adds what load and wear take.
float feedforward(float rpm) {
  return rpm == 0 ? 0 : copysignf(PWM_START + fabsf(rpm) / RPM_PER_PWM, rpm);
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
  if (BATTERY_SENSE && battery < minBattery) {
    Serial.println("reset_blocked");
    return;
  }
  setAwake(true);
  delay(SETTLE_MS + 1);  // the LM339 runs on a driver's LDO, off while asleep
  if (OVERCURRENT_SENSE && digitalRead(OVERCURRENT_PIN) == LOW) {
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
    if (fabsf(vx) < V_DEADZONE && fabsf(vy) < V_DEADZONE && fabsf(wz) < W_DEADZONE) {
      vx = vy = wz = 0;
    }
    mecanum.toWheelsRpm(constrain(vx, -MAX_V, MAX_V), constrain(vy, -MAX_V, MAX_V),
                        constrain(wz, -MAX_W, MAX_W), refRpm);
    lastCmdMs = millis();
    cmdActive = true;
  } else if (strcmp(line, "reset") == 0) {
    if (latched) arm(BATTERY_RESET); else Serial.println("ready");
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
    // A wheel told to stand still coasts, with its PI cleared: no integral left over from
    // the last motion keeps pushing it.
    float out = 0;
    if (cmdActive && refRpm[i] != 0) {
      out = feedforward(refRpm[i]) + wheelPI[i].update(refRpm[i] - rpm, dt);
      if (fabsf(out) < PWM_MIN) out = 0;
    } else {
      wheelPI[i].reset();
    }
    drive(i, MOTOR_SIGN[i] * out);
  }

  if (BATTERY_SENSE) {
    battery += dt / 0.5f * (readBattery() - battery);  // 0.5 s low-pass
    if (battery < BATTERY_LOW) latch("battery_too_low");
  }

  reportDt += dt;
  ++tick;
  if (tick % 2 == 0) report();
  if (BATTERY_SENSE && tick % 100 == 0) Serial.printf("battery,%.2f\n", battery);
}

void protections() {
  const unsigned long now = millis();
  const bool low = OVERCURRENT_SENSE && awake && now - awakeSinceMs > SETTLE_MS &&
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
  if (!gyro.begin(SDA_PIN, SCL_PIN, IMU_UP_REG, IMU_UP_SIGN)) Serial.println("imu_missing");
  battery = readBattery();
  lastControlUs = micros();
  arm(BATTERY_LOW);
}

void loop() {
  readSerial();
  control();
  protections();
}
