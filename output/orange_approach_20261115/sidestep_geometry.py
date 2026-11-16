"""Direction-dependent preparation and feasible side-step nominal geometry.

Only nominal references change. The original DCM and reachability constraints
remain responsible for approving footholds; no direct motor commands here.
"""
import math


def initial_support(vx, vy, previous):
    if not all(math.isfinite(float(v)) for v in (vx,vy)):
        raise ValueError('Nonfinite planar velocity')
    if abs(vx)>.001 or abs(vy)<.005:
        return previous
    # The planner's unpublished preparation step flips stance once; the first
    # published swing is after a second flip. This is the actual support side.
    # Left motion: support right, swing left; right motion: support left.
    return 1 if vy>0 else -1


def nominal_width(command, minimum, maximum, vx, lateral_horizon, max_step_time):
    values=[float(v) for v in lateral_horizon]
    if not values or not all(math.isfinite(float(v)) for v in
            [command,minimum,maximum,vx,max_step_time,*values]):
        raise ValueError('Invalid side-step geometry')
    if not 0<minimum<=maximum or not 0<max_step_time<=.5:
        raise ValueError('Invalid reachability or timing limits')
    base=max(float(command),float(minimum))
    speed=max(abs(v) for v in values)
    if abs(vx)>.001 or speed<.005:
        return base
    stride=speed*max_step_time
    width=max(base,minimum+stride+.005)
    if width+stride>maximum:
        raise ValueError('No lateral room for both outward and closing steps')
    return width
