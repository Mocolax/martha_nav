// Minimal motor test for Martha's PCB: a PWM value straight to the drivers, nothing else.
//
// Serial 115200, one command per line:
//   <motor> <pwm>   motor 1-4 (M1=FL, M2=FR, M3=RL, M4=RR) or 0 for all four,
//                   pwm -255..255, 0 stops it. Example: "2 150", "0 -100", "0 0".
// Lines that do not start with a digit are ignored, so an echo of our own replies
// cannot become a command.

// const int IN1[4] = {27, 32, 15, 18};
// const int IN2[4] = {14, 33, 2, 19};
const int IN1[4] = {27, 32, 15, 18};
const int IN2[4] = {14, 33, 2, 19};

const int SLEEP_PIN = 17;  // nSLEEP of the four drivers

// IN1 = IN2 = 0 is coast on the MPQ6612A.
void drive(int i, int pwm) {
  ledcWrite(IN1[i], pwm > 0 ? pwm : 0);
  ledcWrite(IN2[i], pwm < 0 ? -pwm : 0);
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; ++i) {
    ledcAttach(IN1[i], 10000, 8);
    ledcAttach(IN2[i], 10000, 8);
    drive(i, 0);
  }
  pinMode(SLEEP_PIN, OUTPUT);
  digitalWrite(SLEEP_PIN, HIGH);
  Serial.println("ready: <motor 1-4, 0 = all> <pwm -255..255>");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  int motor, pwm;
  if (line.length() == 0 || !isDigit(line[0])) return;
  if (sscanf(line.c_str(), "%d %d", &motor, &pwm) != 2 || motor < 0 || motor > 4) return;
  pwm = constrain(pwm, -255, 255);
  for (int i = 0; i < 4; ++i)
    if (motor == 0 || motor == i + 1) drive(i, pwm);
  Serial.printf("ok M%d pwm %d\n", motor, pwm);
}
