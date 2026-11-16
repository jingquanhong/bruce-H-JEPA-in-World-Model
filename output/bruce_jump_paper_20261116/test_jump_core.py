"""Numerical contract checks; no physical or simulated robot is actuated."""
import json
from pathlib import Path
import time
import unittest
import numpy as np
from jump_core import (AccelerationTask, JumpQP, LandingGains,
                       ballistic_requirements, flight_pd, landing_targets)


def fixture():
    # Synthetic floating-base mechanics, not a BRUCE model or a jump demo.
    H = np.diag(np.r_[np.ones(3), np.full(3, 5.), np.ones(10)])
    CG = np.zeros(16)
    CG[5] = 5. * 9.81
    J = np.zeros((4, 3, 16))
    J[:, :, 3:6] = np.eye(3)
    return dict(H=H, CG=CG, Jc=J, active=np.ones(4), mu=.5,
                maximum_normal_force=np.full(4, 50.), tasks=[],
                torque_limit=np.full(10, 10.5),
                joint_torque_to_current=np.eye(10) / .35,
                current_limit=np.full(10, 30.))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.feet = np.array([[0., -.06, 0.], [0., .06, 0.]])
        self.lo = np.array([[-.1, -.15, -.01], [-.1, .02, -.01]])
        self.hi = np.array([[.1, -.02, .1], [.1, .15, .1]])
        self.gains = LandingGains(.1, .1, .2, -.2, .05)

    def landing(self, lin, ang, **kw):
        args = dict(nominal=self.feet, linear_momentum=lin,
                    angular_momentum=ang, gains=self.gains,
                    lower=self.lo, upper=self.hi, minimum_width=.04)
        args.update(kw)
        return landing_targets(**args)

    def test_ballistics(self):
        b = ballistic_requirements(.01)
        self.assertAlmostEqual(b["takeoff_vz_m_s"]**2 / (2 * 9.81), .01)
        self.assertAlmostEqual(b["same_height_flight_s"], .09030472819714618)
        for h in [0, -.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                ballistic_requirements(h)

    def test_zero_momentum_no_change(self):
        np.testing.assert_array_equal(self.landing([0, 0, 0], [0, 0, 0]), self.feet)

    def test_paper_momentum_sign_and_widening(self):
        f = self.landing([.2, .1, 0], [.01, .03, 0])
        np.testing.assert_allclose(f[:, 0], .026)
        np.testing.assert_allclose(f[:, 1], [-.057, .073])
        np.testing.assert_array_equal(f[:, 2], self.feet[:, 2])

    def test_saturation_and_crossing_rejection(self):
        f = self.landing([100, -100, 0], [0, 0, 0])
        self.assertTrue(np.all(f >= self.lo) and np.all(f <= self.hi))
        with self.assertRaises(ValueError):
            self.landing([0, 0, 0], [0, 0, 0], minimum_width=.2)
        with self.assertRaises(ValueError):
            self.landing([float("nan"), 0, 0], [0, 0, 0])

    def test_flight_zero_ff_and_coupled_current(self):
        M = np.eye(2)
        M[0] = [1, 1]
        r = flight_pd(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2),
                      np.ones(2)*10, np.zeros(2), np.ones(2)*10.5,
                      M, np.ones(2)*12)
        np.testing.assert_allclose(r["torque"], [6, 6])
        self.assertLess(r["scale"], 1)
        self.assertLessEqual(abs(r["current"]).max(), 12 + 1e-9)

    def test_qp_stance_dynamics(self):
        r = JumpQP().solve(**fixture())
        self.assertEqual(r["status"], "solved")
        self.assertLess(r["constraint_violation"], 1e-3)
        f = r["forces"]
        self.assertTrue(np.all(f[:, 2] >= -1e-5))
        self.assertTrue(np.all(np.abs(f[:, :2]).sum(axis=1) <= .5*f[:, 2] + 1e-5))

    def test_qp_flight_zero_contacts_freefall(self):
        a = fixture()
        a["active"][:] = 0
        r = JumpQP().solve(**a)
        np.testing.assert_allclose(r["forces"], 0, atol=1e-5)
        self.assertAlmostEqual(r["ddq"][5], -9.81, places=4)

    def test_total_pd_effort_is_constrained(self):
        a = fixture()
        a["pd_torque"] = np.full(10, 100.)
        a["joint_acceleration_lower"] = np.full(10, -1.)
        a["joint_acceleration_upper"] = np.full(10, 1.)
        with self.assertRaises(RuntimeError):
            JumpQP().solve(**a)

    def test_motor_map_in_qp_not_independent_joint_clipping(self):
        a = fixture()
        J = np.zeros((10, 16))
        J[:, 6:] = np.eye(10)
        a["tasks"] = [AccelerationTask(J, np.zeros(10), np.full(10, 10), np.full(10, 100))]
        a["joint_torque_to_current"][0, 1] = 1/.35
        r = JumpQP().solve(**a)
        self.assertLessEqual(np.abs(r["current"]).max(), 30. + 1e-3)
        self.assertLess(r["torque_total"][0] + r["torque_total"][1], 10.501)

    def test_update_zero_to_nonzero_sparsity(self):
        qp = JumpQP()
        a = fixture()
        qp.solve(**a)
        J = np.zeros((1, 16))
        J[0, 6] = 1
        J[0, 7] = .3
        a["tasks"] = [AccelerationTask(J, np.zeros(1), np.ones(1), np.ones(1))]
        r = qp.solve(**a)
        fresh = JumpQP().solve(**a)
        np.testing.assert_allclose(r["torque_total"], fresh["torque_total"], atol=1e-3)

    def test_infeasible_base_raises_not_stale_solution(self):
        qp = JumpQP()
        a = fixture()
        qp.solve(**a)
        a["Jc"][:] = 0
        a["H"][0, :] = 0
        a["H"][:, 0] = 0
        with self.assertRaises(np.linalg.LinAlgError):
            qp.solve(**a)
        a = fixture()
        a["Jc"][:] = 0
        # Nonzero angular/linear coupling + contradictory acceleration limits.
        a["H"][5, 6] = a["H"][6, 5] = .1
        a["joint_acceleration_lower"] = np.ones(10)
        a["joint_acceleration_upper"] = -np.ones(10)
        with self.assertRaises(ValueError):
            qp.solve(**a)

    def test_nonfinite_input_rejected(self):
        a = fixture()
        a["CG"][3] = np.nan
        with self.assertRaises(ValueError):
            JumpQP().solve(**a)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status="PASS" if result.wasSuccessful() else "FAIL",
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  seconds=time.perf_counter()-started, hardware_actuation=False,
                  simulation_executed=False, real_robot_model_used=False,
                  physical_jump_validated=False)
    Path(__file__).with_name("numerical-checks.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
