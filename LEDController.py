import hwDef
from gpiozero import LED


class LEDController:
    def __init__(self):
        self.red_led = LED(pin=hwDef.ledPinRed, active_high=False)
        self.green_led = LED(pin=hwDef.ledPinGreen, active_high=False)
        self.blue_led = LED(pin=hwDef.ledPinBlue, active_high=False)

        self.off()

    def off(self):
        self.red_led.off()
        self.green_led.off()
        self.blue_led.off()
