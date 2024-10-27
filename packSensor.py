import hwDef
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class packSensor:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(self.i2c)
        self.batInput = AnalogIn(self.ads, hwDef.batInputPin)

    def getPackVoltage(self):
        return self.batInput.voltage * hwDef.batDivRatio

    def getPackSOC(self):
        self.cellAverageVoltage = (self.batInput.voltage * hwDef.batDivRatio) / 2
        self.percentage = (
            (self.cellAverageVoltage - hwDef.batCellEmpty)
            / (hwDef.batCellFull - hwDef.batCellEmpty)
            * 100
        )
        return self.percentage
