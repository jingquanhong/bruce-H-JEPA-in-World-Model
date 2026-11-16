"""Coordinate/symmetry/legacy-regression checks; opens no hardware interfaces."""
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

ROOT = Path(__file__).resolve().parent
repository = Path('/home/khadas/BRUCE/BRUCE-OP')
if repository.exists():
    sys.path.insert(0, str(repository))
else:
    # Local test uses the preserved legacy helper, with no BRUCE imports.
    sys.modules['Util'] = types.ModuleType('Util')
    legacy_file = ROOT.parent/'bruce_real_20261115/gait-notes/translation_feedback.py'
    spec = importlib.util.spec_from_file_location('Util.translation_feedback', legacy_file)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
import planar_translation_feedback as new
from Util import translation_feedback as old


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


kp0 = np.array([0., 0., 150.])
kd0 = np.array([5., 5., 15.])
base = np.diag([1., 1., 100.])
cases = []
for yaw in (0., np.pi/2, -.8):
    R = rotation(yaw)
    for phase in (0, 1, 2):
        for sign in (-1., 1.):
            v = R @ np.array([0., sign*.03, 0.])
            reference = R @ np.array([0., sign*.02, 0.])
            kp, kd, target = new.translation_feedback(kp0, kd0, phase, R, v,
                                                    np.zeros(3), reference, True)
            acc = kp @ target + kd @ v
            local_acc = R.T @ acc
            weights = new.translation_weight(base, phase, R, v, True, 3.)
            assert np.all(np.linalg.eigvalsh(weights) > 0.)
            if phase == 0:
                assert np.array_equal(kp, np.diag(kp0))
                assert np.array_equal(kd, np.diag(kd0))
                assert np.array_equal(weights, base)
            else:
                assert sign*local_acc[1] > .15
                assert abs(local_acc[0]) < 1e-10 and abs(local_acc[2]) < 1e-10
                assert np.isclose((R.T@weights@R)[1, 1], 3.)
            cases.append((yaw, phase, sign, local_acc.tolist()))
    for speed in (0., .03, -.03):
        v = R @ np.array([speed, 0., 0.])
        reference = R @ np.array([.06, .04, .02])
        for phase in (0, 1, 2):
            for enabled in (False, True):
                args = (kp0, kd0, phase, R, v, np.zeros(3), reference, enabled)
                assert all(np.array_equal(a,b) for a,b in zip(
                    new.translation_feedback(*args), old.translation_feedback(*args)))
                weight_args = (base, phase, R, v, enabled, 3.)
                assert np.array_equal(new.translation_weight(*weight_args),
                                      old.translation_weight(*weight_args))

for sign in (-1., 1.):
    kp, kd, reference = new.translation_feedback(kp0, kd0, 1, np.eye(3),
        np.array([0., sign*.03, 0.]), np.zeros(3), np.array([0., sign*100., 0.]), True)
    assert np.isclose(reference[1], sign*.03)
    assert np.isclose(kp[1,1]*reference[1], sign*.12)

for invalid in (np.full((3,3),np.nan), np.zeros((3,3))):
    try:
        new.lateral_activation(1, invalid, np.zeros(3), True)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid orientation accepted')

print('PASS: left/right acceleration sign, heading invariance, positive QP weights,')
print('      bounded position term, unchanged standing/zero/forward behavior, invalid feedback rejection.')
print('These are offline checks, not proof of real robot stability or displacement.')

# Exercise the exact worker hook and ensure it delegates without recursion.
new.install()
assert old.translation_feedback is new.translation_feedback
new.translation_feedback(kp0, kd0, 1, np.eye(3), np.array([0.,.03,0.]),
                         np.zeros(3), np.array([0.,.02,0.]), True)
print('PASS: worker installation hook')
