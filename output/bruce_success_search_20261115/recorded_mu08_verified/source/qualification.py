"""Keep trajectory completion separate from verified autonomous obstacle crossing."""
from __future__ import annotations

import numpy as np

# BRUCE_KINEMATICS_AOT.T_bj_r / T_bj_l; IQ2TORQUE=0.35, IQ_MAX=30.
BEAR_TO_JOINT_RIGHT = np.array([[-1, 0, 0, 0, 0], [0, .5, -.5, 0, 0],
                               [0, -.5, -.5, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 1, 1]])
BEAR_TO_JOINT_LEFT = BEAR_TO_JOINT_RIGHT.copy()
BEAR_TO_JOINT_LEFT[3:, 3:] *= -1


def motor_current_from_joint_torque(torque):
    torque = np.asarray(torque)
    return np.concatenate((torque[..., :5] @ BEAR_TO_JOINT_RIGHT,
                           torque[..., 5:10] @ BEAR_TO_JOINT_LEFT), axis=-1) / .35


def qualify(meta, control_forces, *, mesh_clearance_verified=False,
            landing_stability_verified=False):
    forces = np.asarray(control_forces, dtype=float)
    valid = bool(forces.ndim == 2 and forces.shape[0] > 0
                 and forces.shape[1] == 22 and np.isfinite(forces).all())
    base_peak = float(np.abs(forces[:, :6]).max()) if valid else None
    leg_peak = float(np.abs(forces[:, 6:16]).max()) if valid else None
    current_peak = float(np.abs(motor_current_from_joint_torque(forces[:, 6:16])).max()) if valid else None
    checks = {
        'joint_only_control': meta.get('control_mode') == 'joint_only'
                              and not meta.get('base_actuated', True),
        'base_force_zero': valid and base_peak < 1e-6,
        'leg_torque_within_10_5_Nm': valid and leg_peak <= 10.5 + 1e-4,
        'coupled_motor_current_within_30_A': valid and current_peak <= 30. + 1e-3,
        'trajectory_criterion_passed': bool(meta.get('reference_criterion_passed',
                                                     meta.get('success', False))),
        'no_fall': not meta.get('fell', True),
        'no_box_displacement_failure': not meta.get('kicked', True),
        'mesh_clearance_verified': bool(mesh_clearance_verified),
        'landing_stability_verified': bool(landing_stability_verified),
    }
    return {
        'autonomous_success': all(checks.values()),
        'qualification_checks': checks,
        'qualification_failures': [key for key, passed in checks.items() if not passed],
        'base_control_force_peak_abs': base_peak,
        'leg_control_torque_peak_abs_Nm': leg_peak,
        'coupled_motor_current_peak_abs_A': current_peak,
    }
