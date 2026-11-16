"""Regression checks for recording alignment, restart truncation and sensor transforms."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_bruce import EpisodeWriter, json_safe
from bruce_signals import SignalBuilder, G
from qualification import qualify


class CollectionContract(unittest.TestCase):
    def test_assisted_reference_cannot_be_autonomous_success(self):
        meta = dict(control_mode='assisted', base_actuated=True,
                    success=True, fell=False, kicked=False)
        forces = np.zeros((4, 22))
        forces[:, 2] = 120
        result = qualify(meta, forces, mesh_clearance_verified=True,
                         landing_stability_verified=True)
        self.assertFalse(result['autonomous_success'])
        self.assertIn('base_force_zero', result['qualification_failures'])

    def test_unverified_geometry_and_actual_over_torque_are_rejected(self):
        meta = dict(control_mode='joint_only', base_actuated=False,
                    success=True, fell=False, kicked=False)
        forces = np.zeros((4, 22))
        result = qualify(meta, forces)
        self.assertFalse(result['autonomous_success'])
        self.assertIn('mesh_clearance_verified', result['qualification_failures'])
        forces[:, 8] = 11
        result = qualify(meta, forces, mesh_clearance_verified=True,
                         landing_stability_verified=True)
        self.assertFalse(result['autonomous_success'])
        self.assertIn('leg_torque_within_10_5_Nm', result['qualification_failures'])

    def test_individually_bounded_joints_can_overload_coupled_motor(self):
        meta = dict(control_mode='joint_only', base_actuated=False,
                    success=True, fell=False, kicked=False)
        forces = np.zeros((1, 22))
        forces[0, 9:11] = [10., -10.]
        result = qualify(meta, forces, mesh_clearance_verified=True,
                         landing_stability_verified=True)
        self.assertTrue(result['qualification_checks']['leg_torque_within_10_5_Nm'])
        self.assertFalse(result['qualification_checks']['coupled_motor_current_within_30_A'])

    def test_restart_discards_uncommitted_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'test.h5'
            writer = EpisodeWriter(path)
            ep = {'action': np.ones((3, 10)), 'step_idx': np.arange(3), 'ep_idx': np.zeros(3, dtype=int)}
            writer.write_episode(ep, {'clearance': float('nan')})
            writer.f['action'].resize(5, axis=0)
            writer.close()
            restored = EpisodeWriter(path)
            self.assertEqual((restored.n_ep, restored.ptr), (1, 3))
            self.assertEqual(restored.f['action'].shape, (3, 10))
            self.assertIsNone(json.loads(restored.f['ep_meta'][0])['clearance'])
            restored.close()

    def test_rejects_misaligned_columns(self):
        with tempfile.TemporaryDirectory() as td:
            writer = EpisodeWriter(Path(td) / 'test.h5')
            with self.assertRaises(ValueError):
                writer.write_episode({'step_idx': [0, 1], 'action': [np.zeros(10)]}, {})
            writer.close()

    def test_body_frame_specific_force_and_action(self):
        cfg = dict(imu_acc_noise=0, imu_gyro_noise=0, imu_bias_rw=0,
                   joint_q_noise=0, joint_dq_noise=0, torque_gain_std=0)
        signals = SignalBuilder(cfg, np.random.default_rng(0))
        signals.reset_episode()
        state = dict(base_quat=[np.sqrt(.5), np.sqrt(.5), 0, 0],
                     base_acc=np.zeros(3), base_ang=np.zeros(3),
                     base_pos=np.zeros(3), base_vel=np.zeros(3),
                     dof_pos=np.zeros(22), dof_vel=np.zeros(22), dof_force=np.zeros(22),
                     foot_force=np.array([0., 5., 5.1, 20.]),
                     ankle_pos={'ankle_pitch_link_r': (0, .03), 'ankle_pitch_link_l': (0, .03)})
        action = np.arange(22, dtype=float)
        cols = signals.build(state, action)
        np.testing.assert_allclose(cols['imu_acc'], [0, G, 0], atol=1e-10)
        np.testing.assert_array_equal(cols['action'], action[6:16])
        np.testing.assert_array_equal(cols['foot_contact'], [0, 0, 1, 1])
        self.assertTrue(all(np.isfinite(v).all() for v in cols.values()))

    def test_json_missing_measurements(self):
        payload = json.dumps(json_safe({'clearances': [float('nan'), 0.01]}), allow_nan=False)
        self.assertEqual(json.loads(payload), {'clearances': [None, .01]})


if __name__ == '__main__':
    unittest.main()
