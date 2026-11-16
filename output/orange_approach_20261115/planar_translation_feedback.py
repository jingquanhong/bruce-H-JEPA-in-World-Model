"""Opt-in lateral CoM tracking for BRUCE's existing constrained WBC.

Keep legacy standing and forward behavior. For commanded side motion, add
position/velocity tracking and task weight along the horizontal body-left axis.
No actuator, serial, shared-memory, file polling or integral control here.
"""
import numpy as np
from Util.translation_feedback import (
    translation_feedback as _forward_feedback,
    translation_weight as _forward_weight,
    load_forward_weight,
)

LATERAL_KP = 4.0
LATERAL_KD_ADDITION = 5.0
LATERAL_WEIGHT = 3.0
LATERAL_POSITION_ERROR_LIMIT_M = .03


def lateral_activation(phase, rotation, velocity, enabled):
    if not enabled or phase not in (1, 2):
        return np.zeros(3), 0.
    rotation = np.asarray(rotation, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if rotation.shape != (3, 3) or velocity.shape != (3,):
        raise ValueError('Invalid rotation or velocity shape')
    if not np.isfinite(rotation).all() or not np.isfinite(velocity).all():
        raise ValueError('Nonfinite rotation or velocity')
    forward = np.array([rotation[0, 0], rotation[1, 0], 0.])
    length = np.linalg.norm(forward)
    if length < .5:
        raise ValueError('Invalid horizontal body direction')
    forward /= length
    left = np.array([-forward[1], forward[0], 0.])
    speed = float(left @ velocity)
    # Smooth gain activation near zero, with identical behavior for both signs.
    u = float(np.clip((abs(speed) - .003) / .007, 0., 1.))
    blend = u * u * (3. - 2. * u)
    return left, blend


def translation_feedback(kp_values, kd_values, phase, body_rotation,
                         desired_velocity, position, desired_position,
                         enabled=False):
    kp, kd, reference = _forward_feedback(
        kp_values, kd_values, phase, body_rotation, desired_velocity,
        position, desired_position, enabled)
    left, blend = lateral_activation(phase, body_rotation, desired_velocity, enabled)
    if blend == 0.:
        return kp, kd, reference
    position = np.asarray(position)
    if not np.isfinite(position).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite CoM reference')
    direction = np.outer(left, left)
    kp += LATERAL_KP * blend * direction
    kd += LATERAL_KD_ADDITION * blend * direction
    error = float(left @ (reference - position))
    reference += left * (np.clip(error, -LATERAL_POSITION_ERROR_LIMIT_M,
                                LATERAL_POSITION_ERROR_LIMIT_M) - error)
    return kp, kd, reference


def translation_weight(base, phase, rotation, velocity, enabled, weight):
    result = _forward_weight(base, phase, rotation, velocity, enabled, weight)
    left, blend = lateral_activation(phase, rotation, velocity, enabled)
    return result + (LATERAL_WEIGHT - 1.) * blend * np.outer(left, left)


def install():
    """Activate in this private low-level worker before its warmup starts."""
    import Util.translation_feedback as legacy
    legacy.translation_feedback = translation_feedback
    legacy.translation_weight = translation_weight
