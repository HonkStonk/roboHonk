import time
import hwDef
from adafruit_servokit import ServoKit
from gpiozero import LED
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


def getPackVoltage():
    return batInput.voltage * hwDef.batDivRatio


ledRed = LED(pin=hwDef.ledPinRed, active_high=False)
ledGreen = LED(pin=hwDef.ledPinGreen, active_high=False)
ledBlue = LED(pin=hwDef.ledPinBlue, active_high=False)

ledRed.off()
ledGreen.off()
ledBlue.off()

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
batInput = AnalogIn(ads, hwDef.batInputPin)
print("divrat: ", hwDef.batDivRatio)
print("pack voltage: ", getPackVoltage())

kit = ServoKit(channels=16)
kit.servo[hwDef.camServo].set_pulse_width_range(hwDef.camServoMin, hwDef.camServoMax)

kit.servo[hwDef.camServo].fraction = 0.0
for i in range(8):
    val = 0.0 + (0.04 * i)
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

for i in range(50):
    print("pack voltage: ", getPackVoltage())
    time.sleep(0.3)
print("it worked")
