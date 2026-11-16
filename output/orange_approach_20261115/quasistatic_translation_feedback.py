"""Single-support lateral position holding for the explicit slow side-step."""
import numpy as np
from Util.translation_feedback import translation_feedback as _base, translation_weight as _weight, load_forward_weight
from Util.planar_translation_feedback import lateral_activation


def axis(phase, rotation, enabled):
    # A unit lateral velocity selects the axis; activation is intentionally
    # independent of the zero CoM velocity in a quasi-static support hold.
    if not enabled or phase not in (1, 2): return np.zeros(3), 0.
    rotation = np.asarray(rotation)
    forward = np.array([rotation[0,0],rotation[1,0],0.]);forward/=np.linalg.norm(forward)
    left = np.array([-forward[1],forward[0],0.])
    return lateral_activation(phase,rotation,left,enabled)


def translation_feedback(kp_values,kd_values,phase,body_rotation,desired_velocity,position,desired_position,enabled=False):
    kp,kd,reference=_base(kp_values,kd_values,phase,body_rotation,desired_velocity,position,desired_position,enabled)
    left,active=axis(phase,body_rotation,enabled)
    if active:
        # Slow single support uses the existing stable standing horizontal
        # feedback values. Leave vertical tracking and all QP constraints alone.
        kp[:2,:2]=50.*np.eye(2);kd[:2,:2]=10.*np.eye(2)
        error=np.asarray(reference[:2])-np.asarray(position[:2])
        norm=float(np.linalg.norm(error))
        if norm>.025:reference[:2]=np.asarray(position[:2])+error*(.025/norm)
    return kp,kd,reference


def translation_weight(base,phase,rotation,velocity,enabled,weight):
    result=_weight(base,phase,rotation,velocity,enabled,weight)
    left,active=axis(phase,rotation,enabled)
    if active:result[:2,:2]=50.*np.eye(2)
    return result


def install():
    import Util.translation_feedback as legacy
    legacy.translation_feedback=translation_feedback;legacy.translation_weight=translation_weight
