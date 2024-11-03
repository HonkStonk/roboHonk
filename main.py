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

servos.camServo(0.42)  # range 0.0 (down) - 1.0 (up) works
# motorMinFwd = slowest forward, 1.0 = max speed forward
# motorMinBack = slowest backward, 0.0 = max speed backward
servos.leftMotor(hwDef.motorMinBack)
servos.rightMotor(0.46)
time.sleep(5)
servos.off()
