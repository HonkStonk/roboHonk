import time
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

servos.camServo(0.15)
servos.leftMotor(0.50)
servos.rightMotor(0.50)
time.sleep(5)
servos.off()
