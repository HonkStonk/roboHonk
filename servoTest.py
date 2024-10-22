from adafruit_servokit import ServoKit
import time
import gpiod

kit = ServoKit(channels=16)

kit.servo[0].set_pulse_width_range(1000, 2000)

for i in range(10):
	kit.servo[0].fraction = i*0.1
	time.sleep(0.5)

chip = gpiod.Chip('gpiochip4', gpiod.Chip.OPEN_BY_NAME)
LED0 = chip.get_line(5)
LED0.request(consumer="blinktest", type=gpiod.LINE_REQ_DIR_OUT)
LED1 = chip.get_line(6)
LED1.request(consumer="blinktest", type=gpiod.LINE_REQ_DIR_OUT)
LED2 = chip.get_line(13)
LED2.request(consumer="blinktest", type=gpiod.LINE_REQ_DIR_OUT)

LED0.set_value(1)
LED1.set_value(1)
LED2.set_value(1)
time.sleep(1.5)
LED0.set_value(0)
time.sleep(1.5)
LED1.set_value(0)
time.sleep(1.5)
LED2.set_value(0)

print("worked")
