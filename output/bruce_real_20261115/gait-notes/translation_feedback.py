"""Bounded optional forward tracking; no hardware or shared-memory access."""
import numpy as np


def translation_feedback(kp_values, kd_values, phase, body_rotation, desired_velocity,
                         position, desired_position, enabled=False):
    kp = np.diag(kp_values).copy()
    kd = np.diag(kd_values).copy()
    reference = np.asarray(desired_position).copy()
    if not enabled or phase not in (1, 2):
        return kp, kd, reference
    forward = np.asarray(body_rotation)[:, 0].copy()
    forward[2] = 0.
    length = np.linalg.norm(forward)
    if length < .5:
        raise ValueError('Invalid horizontal forward direction')
    forward /= length
    if abs(float(forward @ desired_velocity)) < .005:
        return kp, kd, reference
    direction = np.outer(forward, forward)
    kp += 2. * direction
    kd += 5. * direction
    error = float(forward @ (reference - position))
    reference += forward * (np.clip(error, -.05, .05) - error)
    return kp, kd, reference


def load_forward_weight(path):
    if path is None:
        return 1.
    import json
    from pathlib import Path
    value = float(json.loads(Path(path).read_text())['forward_weight'])
    if not np.isfinite(value) or not 1. <= value <= 5.:
        raise ValueError('Forward task weight must be within 1..5')
    return value


def translation_weight(base, phase, rotation, velocity, enabled, weight):
    result = np.asarray(base).copy()
    if not enabled or phase not in (1, 2):
        return result
    forward = np.asarray(rotation)[:, 0].copy()
    forward[2] = 0.
    length = np.linalg.norm(forward)
    if length < .5:
        raise ValueError('Invalid horizontal forward direction')
    forward /= length
    if abs(float(forward @ velocity)) >= .005:
        result += (weight - 1.) * np.outer(forward, forward)
    return result
