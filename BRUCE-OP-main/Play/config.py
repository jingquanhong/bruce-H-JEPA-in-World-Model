#!usr/bin/env python
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2026 Westwood Robotics Corporation"
__date__      = "January 21, 2026"
__project__   = "BRUCE"
__version__   = "0.0.7"
__status__    = "Product"

'''
Launch Configuration
'''

SIMULATION = True    # if in simulation or not
GAMEPAD    = False   # if using gamepad or not
ESTIMATION = False   # if running estimation or not (for simulation)

HARDWARE = not SIMULATION
if HARDWARE:
    ESTIMATION = True

if __name__ == '__main__':
    print(True)
