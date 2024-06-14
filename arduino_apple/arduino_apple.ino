#include <Servo.h>

Servo servo;
int pos = 180; // 舵机初始位置，设定为180度
unsigned long previousMillis = 0;
const long interval = 1000; // 5秒的间隔
bool rotating = false;

void setup() {
  servo.attach(3); // 将舵机连接到数字引脚 3
  servo.write(pos); // 设置舵机初始位置
  Serial.begin(9600); // 开启串口通信，波特率为9600
  Serial.println("Servo control started. Enter 1 or 0 to control the servo.");
}

void loop() {
  if (!rotating && Serial.available() > 0) {
    char input = Serial.read(); // 读取串口输入的一个字符

    if (input == '1') {
      Serial.println("Rotating counter-clockwise by 180 degrees.");
      Serial.end(); // 关闭串口通信
      servo.write(pos - 180); // 转到 0 度位置
      previousMillis = millis(); // 记录当前时间
      rotating = true; // 设置旋转标志
    } 
    else if (input == '0') {
      Serial.println("Rotating clockwise by 180 degrees.");
      Serial.end(); // 关闭串口通信
      servo.write(pos + 180); // 转到 360 度位置
      previousMillis = millis(); // 记录当前时间
      rotating = true; // 设置旋转标志
    } 
    else {
      Serial.println("Invalid input. Please enter 1 or 0.");
    }
  }

  // 检查是否需要复位舵机
  if (rotating) {
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= interval) {
      // 已经过了5秒，复位舵机
      servo.write(pos); // 回到初始位置（180度）
      delay(1000); // 等待1秒，确保舵机稳定
      rotating = false; // 重置旋转标志
      Serial.begin(9600); // 重新开启串口通信
      Serial.println("Servo reset to initial position.");
    }
  }
}
