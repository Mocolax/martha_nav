// Teleop without encoders: the same serial protocol as ../firmware.ino ("cmd_vel,vx,vy,wz"
// and "reset"), so esp32_bridge and teleop_twist_keyboard work unchanged, but open loop:
// each wheel's RPM from the mecanum kinematics goes straight to a PWM, no PI, no odometry.
// Run it with the wheels in the air first.
#include "../control.h"

const int IN1[4] = {27, 32, 15, 18};
const int IN2[4] = {14, 33, 2, 19};
const int SLEEP_PIN = 17;  // nSLEEP of the four drivers

// The right motors are mounted mirrored: the same PWM turns them the other way.
// Flip one entry if that wheel turns backwards.
const int MOTOR_SIGN[4] = {1, -1, 1, -1};  // FL, FR, RL, RR

const float WHEEL_RADIUS = 0.075f;       // m
const float LX_PLUS_LY = 0.385f;         // m
const float MAX_V = 0.5f, MAX_W = 1.2f;  // same ceiling as the firmware
// Full PWM (255) at the wheel RPM of MAX_V, so the whole PWM range is reachable from teleop.
const float MAX_RPM = MAX_V * 60.0f / (2.0f * PI * WHEEL_RADIUS);  // ~64 RPM
const int PWM_MIN = 40;                  // below this the motor does not start; tune it
const unsigned long CMD_TIMEOUT_MS = 500;

Mecanum mecanum{WHEEL_RADIUS, LX_PLUS_LY};
unsigned long lastCmdMs = 0;
bool moving = false;

// IN1 = IN2 = 0 is coast on the MPQ6612A.
void drive(int i, int pwm) {
  pwm = constrain(MOTOR_SIGN[i] * pwm, -255, 255);
  ledcWrite(IN1[i], pwm > 0 ? pwm : 0);
  ledcWrite(IN2[i], pwm < 0 ? -pwm : 0);
}

// Linear in RPM above the dead band; 0 stays 0.
int rpmToPwm(float rpm) {
  if (fabsf(rpm) < 0.5f) return 0;
  const float pwm = PWM_MIN + fabsf(rpm) / MAX_RPM * (255 - PWM_MIN);
  return (rpm > 0 ? 1 : -1) * (int)pwm;
}

void stopAll() {
  for (int i = 0; i < 4; ++i) drive(i, 0);
  moving = false;
}

void handle(const char *line) {
  float vx, vy, wz, rpm[4];
  if (sscanf(line, "cmd_vel,%f,%f,%f", &vx, &vy, &wz) == 3) {
    if (!isfinite(vx) || !isfinite(vy) || !isfinite(wz)) return;
    mecanum.toWheelsRpm(constrain(vx, -MAX_V, MAX_V), constrain(vy, -MAX_V, MAX_V),
                        constrain(wz, -MAX_W, MAX_W), rpm);
    for (int i = 0; i < 4; ++i) drive(i, rpmToPwm(rpm[i]));
    lastCmdMs = millis();
    moving = true;
  } else if (strcmp(line, "reset") == 0) {
    Serial.println("ready");
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

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; ++i) {
    ledcAttach(IN1[i], 10000, 8);
    ledcAttach(IN2[i], 10000, 8);
  }
  stopAll();
  pinMode(SLEEP_PIN, OUTPUT);
  digitalWrite(SLEEP_PIN, HIGH);
  Serial.println("ready");
}

void loop() {
  readSerial();
  if (moving && millis() - lastCmdMs > CMD_TIMEOUT_MS) {
    stopAll();  // teleop stopped sending
  }
}
