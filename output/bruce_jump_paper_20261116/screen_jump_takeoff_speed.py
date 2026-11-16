"""Offline velocity screen only: recorded flat-foot crouch geometry, no robot IO.

This is not a jump controller, optimizer, contact simulation, or hardware test.
The fixed-foot path is only one possible family of jumps and does not prove
that other trajectories are infeasible.
"""
import json
import math
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from current_model import build

HERE = Path(__file__).resolve().parent


def main():
    geometry = json.loads((HERE / 'crouch-6cm-geometry.json').read_text())
    model, _ = build(HERE / 'model-export-corrected.json')
    recorded = geometry['anchor_state']
    rotation = np.asarray(recorded['estimated']['body_rot_matrix'])
    depths = -np.array([item['body_offset_m'] for item in geometry['records']])
    joint = np.array([item['q'] for item in geometry['records']])
    joint_spline = CubicSpline(depths, joint, axis=0)
    com_heights = []
    for depth, joint_angles in zip(depths, joint):
        configuration = pin.neutral(model)
        configuration[:3] = np.asarray(recorded['estimated']['body_position']) + [0, 0, -depth]
        configuration[3:7] = Rotation.from_matrix(rotation).as_quat()
        configuration[7:] = joint_angles
        com_heights.append(pin.centerOfMass(model, model.createData(), configuration)[2])
    com_spline = CubicSpline(depths, com_heights)
    cases = []
    for depth in [0., .01, .02, .03]:
        # Flight apex 1 cm above takeoff; does not mean 1 cm foot clearance.
        target_com_speed = math.sqrt(2 * 9.81 * .01)
        sensitivity = -float(com_spline(depth, 1))
        if sensitivity <= 0:
            raise RuntimeError('Invalid recorded-geometry COM sensitivity')
        body_speed = target_com_speed / sensitivity
        joint_speed = -body_speed * joint_spline(depth, 1)
        cases.append({'takeoff_body_depth_m': depth,
                      'target_com_takeoff_vz_m_s': target_com_speed,
                      'com_speed_per_body_speed': sensitivity,
                      'body_takeoff_vz_m_s': body_speed,
                      'joint_takeoff_speed_rad_s': joint_speed.tolist(),
                      'maximum_joint_takeoff_speed_rad_s': float(abs(joint_speed).max()),
                      'same_height_ideal_flight_s': 2*target_com_speed/9.81})
    result = {'status': 'KINEMATIC_SCREEN_ONLY',
              'reference_geometry': 'recorded anchor, fixed flat feet and fixed torso orientation',
              'requested_com_rise_from_takeoff_m': .01,
              'slow_crouch_reference_joint_speed_limit_rad_s': .3,
              'cases': cases,
              'hardware_actuation': False,
              'jump_trajectory_validated': False,
              'limitations': ['Model mass distribution is not physically measured after arm removal and camera installation',
                              'Contact and impact are not simulated', 'Actuator torque/speed and transient current are not certified',
                              'This tests one kinematic path; it does not establish that all other paths are infeasible',
                              'The slow-crouch speed limit is a software limit, not a rated motor speed']}
    output = HERE / 'jump-takeoff-speed-screen.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
