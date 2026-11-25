// Pure math of the drive: the wheel PI and the mecanum kinematics.
#pragma once
#include <Arduino.h>

// PI on one wheel, in PWM counts per RPM of error, on top of `base` (the feedforward and
// anything else already in the output); returns the whole output, clipped. Conditional
// integration on that whole output: while it is saturated and the error pushes it further,
// the integral stops growing, so a blocked wheel does not wind it up and the wheel neither
// overshoots when it breaks away nor lags when told to reverse. Checking the PI alone
// against the limit let the integral climb while the feedforward already held 255.
struct WheelPI {
  float kp, ki, limit, integral = 0;

  float update(float error, float dt, float base = 0) {
    float out = base + kp * error + ki * integral;
    if (fabsf(out) < limit || error * out < 0) {
      integral += error * dt;
      out = base + kp * error + ki * integral;
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
