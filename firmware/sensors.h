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

// MPU-6050 yaw rate only (the EKF fuses nothing else from it): the rate of the board axis
// that points up, given by its register (0x43 x, 0x45 y, 0x47 z) and sign (-1 when that
// axis points down), so a counterclockwise turn seen from above stays positive whatever
// way the board is mounted. The bias is averaged at boot and afterwards relearned only
// when the caller knows the robot is still: the gyro's own reading cannot tell a slow
// turn from no turn.
struct Gyro {
  static constexpr uint8_t ADDRESS = 0x68;
  static constexpr float LSB_PER_RAD_S = 131.0f * 180.0f / PI;  // +-250 deg/s
  bool ok = false;
  float bias = 0, sign = 1;
  uint8_t reg = 0x47;

  bool begin(int sda, int scl, uint8_t upReg, float upSign) {
    reg = upReg;
    sign = upSign;
    Wire.begin(sda, scl, 400000);
    ok = write(0x6B, 0x00)    // wake up
         && write(0x1A, 0x03)  // 44 Hz low-pass
         && write(0x1B, 0x00); // +-250 deg/s
    if (!ok) return false;
    delay(100);
    // 1000 samples, ~5 s: a finer zero than 200 (checked with imu_test_uno). The robot
    // must stay still meanwhile; any movement shows up later as drift.
    long sum = 0;
    int16_t z;
    for (int i = 0; i < 1000; ++i) {
      if (!raw(z)) return ok = false;
      sum += z;
      delay(5);
    }
    bias = sum / 1000.0f;
    return true;
  }

  float read(bool still) {  // rad/s, NAN without an IMU
    int16_t z;
    if (!ok || !raw(z)) return NAN;
    if (still) bias += 0.01f * (z - bias);
    return sign * (z - bias) / LSB_PER_RAD_S;
  }

  bool write(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(ADDRESS);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
  }

  bool raw(int16_t &z) {
    Wire.beginTransmission(ADDRESS);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0 || Wire.requestFrom(ADDRESS, (uint8_t)2) != 2) return false;
    const uint8_t high = Wire.read();
    const uint8_t low = Wire.read();
    z = (int16_t)((high << 8) | low);
    return true;
  }
};
