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

# read config
fixed=`python3 -m Simulation.config`

if [ $fixed = "True" ]
then
  gazebo --verbose Simulation/worlds/bruce_fixed.world
else
  gazebo --verbose Simulation/worlds/bruce.world
fi