"""Bounded IMU yaw reference for short supported, native stepping turns."""
import math

def wrap(value):return math.atan2(math.sin(value),math.cos(value))
def checked_turn_degrees(value):
 value=float(value)
 if not math.isfinite(value) or not .5<=abs(value)<=10:raise ValueError('Turn request must be 0.5..10 degrees')
 return value

def turn_rate(error,gyro_rate):
 if not all(math.isfinite(v) for v in [error,gyro_rate]):raise ValueError('Nonfinite yaw feedback')
 return max(-math.radians(3),min(math.radians(3),.9*error-.1*gyro_rate))
