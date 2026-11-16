"""Hip-yaw posture reference for supervised, zero-yaw stepping only."""
import numpy as np


def hold_stance_hip_yaw(positions, velocities, phase, reference):
    """Anchor stance hip yaw; preserve the original swing-leg IK solution."""
    if phase not in (0, 1, 2):
        raise ValueError('Unexpected support phase')
    if phase in (0, 1):
        positions[0] = reference[0]
        velocities[0] = 0.
    if phase in (0, 2):
        positions[5] = reference[1]
        velocities[5] = 0.


def checked_reference(joint_positions):
    reference = np.asarray(joint_positions, dtype=float)[[0, 5]].copy()
    if not np.isfinite(reference).all() or np.any(reference < -np.pi) or np.any(reference > 0):
        raise ValueError('Invalid initial hip-yaw reference')
    return reference
