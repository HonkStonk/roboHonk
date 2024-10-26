# hwDef.py

camServo = 4
camServoMin = 1000
camServoMax = 2500

leftMotor = 5
rightMotor = 6
motorMin = 1000
motorMax = 2000

ledPinRed = 5
ledPinGreen = 6
ledPinBlue = 13

# ADC (ADS1115)
r_high = 100770
r_low = 55290
batDivRatio = (r_high + r_low) / r_low
batInputPin = 0  # = P0
