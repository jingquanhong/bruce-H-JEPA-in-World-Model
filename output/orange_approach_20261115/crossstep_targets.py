"""Liftoff-anchored side-step targets, added as a soft QP objective.

The original DCM equalities, foot reachability, timing and WBC constraints
remain authoritative. These are ankle targets, not direct motor angles.
"""
import numpy as np

STEP_METERS = .03
TARGET_WEIGHT = 400.


def landing_targets(swing_xy, support_xy, yaw_rotation, direction, count=3,
                    step=STEP_METERS, minimum_width=.10, maximum_width=.20):
    swing = np.asarray(swing_xy, dtype=float)
    support = np.asarray(support_xy, dtype=float)
    rotation = np.asarray(yaw_rotation, dtype=float)
    if (swing.shape != (2,) or support.shape != (2,) or rotation.shape != (2, 2)
            or not np.isfinite(np.r_[swing, support, rotation.ravel(), direction, step]).all()
            or direction not in (-1, 1) or not 0 < step <= .04 or count != 3
            or np.linalg.norm(rotation.T @ rotation - np.eye(2)) > .01):
        raise ValueError('Invalid explicit cross-step request')
    lateral = rotation[:, 1]
    signed_width = float(lateral @ (swing-support))
    if abs(signed_width) < minimum_width-.005 or abs(signed_width) > maximum_width:
        raise ValueError('Initial foot separation outside cross-step envelope')
    # Each foot advances once, then the first foot advances again. Outward
    # and closing steps consequently have distinct targets, even at zero yaw.
    targets = np.array([swing + direction*step*lateral,
                        support + direction*step*lateral,
                        swing + direction*2*step*lateral])
    previous = support
    expected_side = np.sign(signed_width)
    for target in targets:
        width = float(lateral @ (target-previous))
        if np.sign(width) != expected_side or not minimum_width-.000001 <= abs(width) <= maximum_width-.002:
            raise ValueError('Requested cross-step would exceed original foot-spacing bounds')
        previous = target
        expected_side *= -1
    return targets


def add_target_objective(P, q, targets, weight=TARGET_WEIGHT):
    targets = np.asarray(targets, dtype=float)
    if (targets.shape != (3, 2) or not np.isfinite(targets).all()
            or not np.isfinite(weight) or not 0 < weight <= 400.):
        raise ValueError('Invalid cross-step objective')
    result_P, result_q = np.array(P, copy=True), np.array(q, copy=True)
    if result_P.shape != (13, 13) or result_q.shape != (13,):
        raise ValueError('Expected original three-step DCM problem')
    for i, target in enumerate(targets):
        j = 4*i
        result_P[j, j] += weight
        result_P[j+1, j+1] += weight
        result_q[j:j+2] -= weight*target
    return result_P, result_q
