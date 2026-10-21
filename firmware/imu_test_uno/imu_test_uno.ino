// MPU-6050 test on an Arduino Uno, with the same setup as the firmware's Gyro
// (../sensors.h, which only builds for the ESP32): wake, 44 Hz low-pass, +-250 deg/s,
// bias averaged over 1000 samples at boot, as in the firmware. Keep the IMU still
// until "ready".
//
// Wiring (GY-521): VCC -> 5V, GND -> GND, SDA -> A4, SCL -> A5, AD0 -> GND (address 0x68).
// Serial 115200, 10 lines a second: the three accelerations and rotation rates, which axis
// points up (the one reading about +-1 g), and the yaw integrated around that axis.
// Send "z" to zero the yaw, "c" to recalibrate the bias (IMU still, on the table).
//
// Mount the IMU exactly as it will sit on the robot, robot level, and check:
//   1. who_am_i 0x68.
//   2. "up" names one axis, e.g. "up -Y": that is the axis and sign the firmware must use.
//   3. Still: the yaw drifts less than a degree or two per minute.
//   4. Turned counterclockwise seen from above: yaw grows. Exactly 90 deg by hand: ~ 90.
#include <Wire.h>

const uint8_t ADDRESS = 0x68;
const float LSB_PER_DEG_S = 131.0f;  // +-250 deg/s
const float LSB_PER_G = 16384.0f;    // +-2 g, the default
const unsigned long PRINT_MS = 100;
const char AXIS_NAME[3] = {'X', 'Y', 'Z'};

float bias[3];
float yaw = 0;
unsigned long lastUs = 0, lastPrintMs = 0;

bool write(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADDRESS);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

// Three big-endian int16 starting at reg: 0x3B accel x, y, z; 0x43 gyro x, y, z.
bool read3(uint8_t reg, int16_t v[3]) {
  Wire.beginTransmission(ADDRESS);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(ADDRESS, (uint8_t)6) != 6) return false;
  for (int i = 0; i < 3; ++i) {
    const uint8_t high = Wire.read();
    v[i] = (int16_t)((high << 8) | Wire.read());
  }
  return true;
}

void scan() {
  Serial.print("i2c devices:");
  for (uint8_t a = 1; a < 127; ++a) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.print(" 0x");
      Serial.print(a, HEX);
    }
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  scan();
  Wire.beginTransmission(ADDRESS);
  Wire.write(0x75);
  Wire.endTransmission(false);
  Wire.requestFrom(ADDRESS, (uint8_t)1);
  Serial.print("who_am_i 0x");
  Serial.println(Wire.read(), HEX);
  if (!(write(0x6B, 0x00) && write(0x1A, 0x03) && write(0x1B, 0x00))) {
    Serial.println("imu_missing: check wiring and AD0");
    while (true) {}
  }
  delay(100);
  calibrate();
}

// 1000 samples over ~5 s; any movement meanwhile shows up as drift afterwards.
void calibrate() {
  Serial.println("calibrating, keep still...");
  long sum[3] = {0, 0, 0};
  int16_t g[3];
  for (int n = 0; n < 1000; ++n) {
    read3(0x43, g);
    for (int i = 0; i < 3; ++i) sum[i] += g[i];
    delay(5);
  }
  for (int i = 0; i < 3; ++i) bias[i] = sum[i] / 1000.0f;
  yaw = 0;
  Serial.println("ready");
  lastUs = micros();
}

void loop() {
  if (Serial.available()) {
    const char c = Serial.read();
    if (c == 'z') yaw = 0;
    if (c == 'c') calibrate();
  }
  int16_t a[3], g[3];
  if (!read3(0x3B, a) || !read3(0x43, g)) {
    Serial.println("read failed");
    delay(500);
    return;
  }
  float acc[3], rate[3];
  int up = 0;
  for (int i = 0; i < 3; ++i) {
    acc[i] = a[i] / LSB_PER_G;
    rate[i] = (g[i] - bias[i]) / LSB_PER_DEG_S;
    if (fabs(acc[i]) > fabs(acc[up])) up = i;
  }
  // The axis that reads +1 g points up; counterclockwise from above is positive about it.
  const float upSign = acc[up] > 0 ? 1 : -1;
  const unsigned long now = micros();
  yaw += upSign * rate[up] * (now - lastUs) * 1e-6f;
  lastUs = now;

  if (millis() - lastPrintMs < PRINT_MS) return;
  lastPrintMs = millis();
  Serial.print("acc g");
  for (int i = 0; i < 3; ++i) {
    Serial.print(' ');
    Serial.print(acc[i], 2);
  }
  Serial.print(" | rate deg/s");
  for (int i = 0; i < 3; ++i) {
    Serial.print(' ');
    Serial.print(rate[i], 1);
  }
  Serial.print(" | up ");
  Serial.print(upSign > 0 ? '+' : '-');
  Serial.print(AXIS_NAME[up]);
  Serial.print(" | yaw ");
  Serial.print(yaw, 1);
  Serial.println(" deg");
}
