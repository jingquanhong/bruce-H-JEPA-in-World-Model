"""Directional landing bounds inside the original DCM reachability constraints."""
import numpy as np
from Util.crossstep_targets import landing_targets as _targets, add_target_objective, TARGET_WEIGHT

STEP_METERS=.024
MIN_PROGRESS_METERS=.018
MAX_PUBLISHED_STEPS=20


def landing_targets(*args,**kwargs):
    kwargs.setdefault('step',STEP_METERS)
    return _targets(*args,**kwargs)


def progress_bounds(lower,upper,rotation,swing_start,direction,count=3):
    rotation=np.asarray(rotation,dtype=float);start=np.asarray(swing_start,dtype=float)
    if rotation.shape!=(2,2) or start.shape!=(2,) or direction not in (-1,1):
        raise ValueError('Invalid directional landing bounds')
    if not np.isfinite(np.r_[rotation.ravel(),start]).all() or np.linalg.norm(rotation.T@rotation-np.eye(2))>.01:
        raise ValueError('Invalid landing frame')
    lo=np.asarray(lower,dtype=float).copy();hi=np.asarray(upper,dtype=float).copy()
    index=2*count+1
    forward_anchor=float(rotation[:,0]@start)
    lo[index-1]=max(lo[index-1],forward_anchor-.015)
    hi[index-1]=min(hi[index-1],forward_anchor+.015)
    anchor=float(rotation[:,1]@start)
    a=anchor+direction*MIN_PROGRESS_METERS;b=anchor+direction*STEP_METERS
    lo[index]=max(lo[index],min(a,b));hi[index]=min(hi[index],max(a,b))
    if np.any(lo[index-1:index+1]>hi[index-1:index+1]):
        raise ValueError('Directional progress conflicts with foot reachability')
    return lo,hi
