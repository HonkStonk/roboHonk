import time
import hwDef
from adafruit_servokit import ServoKit
from gpiozero import LED


ledRed = LED(pin=hwDef.ledPinRed, active_high=False)
ledGreen = LED(pin=hwDef.ledPinGreen, active_high=False)
ledBlue = LED(pin=hwDef.ledPinBlue, active_high=False)

ledRed.off()
ledGreen.off()
ledBlue.off()

kit = ServoKit(channels=16)

kit.servo[hwDef.camServo].set_pulse_width_range(hwDef.camServoMin, hwDef.camServoMax)

kit.servo[hwDef.camServo].fraction = 0.90
for i in range(8):
    val = 0.90 + (0.01 * i)
    print("to servo: ", val)
    kit.servo[hwDef.camServo].fraction = val
    time.sleep(2)

kit.servo[hwDef.leftMotor].set_pulse_width_range(hwDef.motorMin, hwDef.motorMax)
kit.servo[hwDef.leftMotor].fraction = 0.5

kit.servo[hwDef.rightMotor].set_pulse_width_range(hwDef.motorMin, hwDef.motorMax)
kit.servo[hwDef.rightMotor].fraction = 0.5

ledRed.on()
time.sleep(1)
ledGreen.on()
time.sleep(1)
ledBlue.on()
time.sleep(1)

print("it worked")
