# hwDef.py

# Cam servo
camServo = 4
camServoMin = 770
camServoMax = 2420

# PWM motor controller
leftMotor = 5
rightMotor = 6
motorMinPWM = 1000
motorMaxPWM = 2000
motorStop = 0.627
motorMinFwd = 0.795
motorMinBack = 0.459

# LED's
ledPinRed = 5
ledPinGreen = 6
ledPinBlue = 13

# ADC (ADS1115)
r_high = 100770
r_low = 55290
batDivRatio = (r_high + r_low) / r_low
batInputPin = 0  # = P0
batCellFull = 4.2
batCellEmpty = 2.7
