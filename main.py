import time
import hwDef
from packSensor import packSensor
from LEDController import LEDController
from servoController import ServoController

leds = LEDController()
bat = packSensor()
servos = ServoController()

leds.red_led.on()
leds.green_led.on()
leds.blue_led.on()

print("bat volt: ", bat.getPackVoltage())
print("bat %: ", bat.getPackSOC())

servos.camServo(0.45)
servos.leftMotor(hwDef.motorStop)
servos.rightMotor(hwDef.motorStop)
time.sleep(5)
servos.off()
