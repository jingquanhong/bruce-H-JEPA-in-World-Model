#!/bin/bash
<< 'COMMENT'
__author__    = "Westwood Robotics Corporation"
__email__     = "info@westwoodrobotics.io"
__copyright__ = "Copyright 2026 Westwood Robotics Corporation"
__date__      = "January 21, 2026"
__project__   = "BRUCE"
__version__   = "0.0.7"
__status__    = "Product"
COMMENT

# set password
readonly PASSWORD=khadas

# power off khadas immediately
echo $PASSWORD | sudo -S shutdown -h now