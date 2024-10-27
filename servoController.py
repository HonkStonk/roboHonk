import hwDef
from adafruit_servokit import ServoKit


class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.kit.servo[hwDef.camServo].set_pulse_width_range(
            hwDef.camServoMin, hwDef.camServoMax
        )
        self.kit.servo[hwDef.leftMotor].set_pulse_width_range(
            hwDef.motorMin, hwDef.motorMax
        )
        self.kit.servo[hwDef.rightMotor].set_pulse_width_range(
            hwDef.motorMin, hwDef.motorMax
        )

        self.off()

    def off(self):
        self.kit.servo[hwDef.camServo].fraction = None  # releasing motor
        self.kit.servo[hwDef.leftMotor].fraction = None  # releasing motor
        self.kit.servo[hwDef.rightMotor].fraction = None  # releasing motor

    def camServo(self, val):
        self.kit.servo[hwDef.camServo].fraction = val

    def leftMotor(self, val):
        self.kit.servo[hwDef.leftMotor].fraction = val

    def rightMotor(self, val):
        self.kit.servo[hwDef.rightMotor].fraction = val
