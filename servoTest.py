import time
from adafruit_servokit import ServoKit

camServo = 4
leftMotor = 5
rightMotor = 6

kit = ServoKit(channels=16)

kit.servo[camServo].set_pulse_width_range(1000, 2500)

kit.servo[camServo].fraction = 0.90
for i in range(8):
    val = 0.90 + (0.01 * i)
    print("to servo: ", val)
    kit.servo[camServo].fraction = val
    time.sleep(2)

kit.servo[leftMotor].set_pulse_width_range(1000, 2000)
kit.servo[leftMotor].fraction = 0.5

kit.servo[rightMotor].set_pulse_width_range(1000, 2000)
kit.servo[rightMotor].fraction = 0.5

print("it worked")
