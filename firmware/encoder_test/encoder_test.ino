// Encoder test: the firmware's x4 PCNT encoders (../sensors.h), with the motors driven by
// hand. Prints, 5 times a second, each wheel's total count and its RPM.
//
// Serial 115200, one command per line:
//   <motor> <pwm>   motor 1-4 (M1=FL, M2=FR, M3=RL, M4=RR) or 0 for all four,
//                   pwm -255..255, positive = that wheel forward, 0 stops it.
//   z               zero the totals.
//
// What to check, wheels in the air:
//   1. "1 150": only enc M1 moves (each encoder on its own wheel).
//   2. Forward PWM gives positive counts and RPM; if not, -1 in ENCODER_SIGN.
//   3. Zero, turn a wheel exactly one revolution by hand: about COUNTS_PER_REV.
//   4. "0 255": about 200 RPM per wheel at 12 V without load.
#include "../sensors.h"

const int ENC_A[4] = {12, 36, 4, 21};
const int ENC_B[4] = {13, 39, 16, 22};
const int IN1[4] = {27, 32, 15, 18};
const int IN2[4] = {14, 33, 2, 19};
const int SLEEP_PIN = 17;  // nSLEEP of the four drivers

const int MOTOR_SIGN[4] = {1, -1, 1, -1};    // as in motor_test: the right motors are mirrored
const int ENCODER_SIGN[4] = {-1, 1, -1, 1};  // the left encoders count backwards
const float COUNTS_PER_REV = 3200.0f;        // 64 CPR x 50:1
const unsigned long PRINT_MS = 200;

Encoder encoders[4];
long totals[4];
unsigned long lastPrintMs = 0;

// IN1 = IN2 = 0 is coast on the MPQ6612A.
void drive(int i, int pwm) {
  pwm = constrain(MOTOR_SIGN[i] * pwm, -255, 255);
  ledcWrite(IN1[i], pwm > 0 ? pwm : 0);
  ledcWrite(IN2[i], pwm < 0 ? -pwm : 0);
}

void handle(String line) {
  line.trim();
  int motor, pwm;
  if (line == "z") {
    for (int i = 0; i < 4; ++i) totals[i] = 0;
  } else if (line.length() && isDigit(line[0]) &&
             sscanf(line.c_str(), "%d %d", &motor, &pwm) == 2 && motor >= 0 && motor <= 4) {
    for (int i = 0; i < 4; ++i)
      if (motor == 0 || motor == i + 1) drive(i, pwm);
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; ++i) {
    ledcAttach(IN1[i], 10000, 8);
    ledcAttach(IN2[i], 10000, 8);
    drive(i, 0);
    encoders[i].begin(ENC_A[i], ENC_B[i]);
  }
  pinMode(SLEEP_PIN, OUTPUT);
  digitalWrite(SLEEP_PIN, HIGH);
  Serial.println("encoder test: <motor 1-4, 0 = all> <pwm -255..255> | z");
  lastPrintMs = millis();
}

void loop() {
  if (Serial.available()) handle(Serial.readStringUntil('\n'));
  const unsigned long now = millis();
  if (now - lastPrintMs < PRINT_MS) return;
  const float dt = (now - lastPrintMs) * 1e-3f;
  lastPrintMs = now;
  Serial.print("enc");
  for (int i = 0; i < 4; ++i) {
    const int counts = ENCODER_SIGN[i] * encoders[i].delta();
    totals[i] += counts;
    Serial.printf(" | M%d %7ld %6.1f rpm", i + 1, totals[i], counts / COUNTS_PER_REV * 60.0f / dt);
  }
  Serial.println();
}
