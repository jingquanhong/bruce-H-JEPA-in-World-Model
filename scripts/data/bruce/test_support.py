"""Regressions for physical sole wrenches and support-transition rejection."""
import unittest

import numpy as np
import pinocchio as pin

import bruce_expert as bx
from audit_reference import audit
from support_geometry import SoleGeometry, margin
from support_reference import transfer_screen


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        row=bx._pose({'r':(0,.025),'l':(0,.025)},-.03,.4)
        q=bx.row_to_pin(row); m=bx._PIN['model']; d=m.createData()
        pin.forwardKinematics(m,d,q);pin.updateFramePlacements(m,d)
        cls.poses=[d.oMf[m.getFrameId('ankle_pitch_link_'+s)].copy() for s in ('r','l')]
        cls.soles=SoleGeometry(bx.URDF,cls.poses)

    def test_sole_is_narrower_than_historical_support_bound(self):
        for k,pose in enumerate(self.poses):
            polygon=self.soles.polygon(k,pose)
            self.assertLess(np.ptp(polygon[:,1]),.006)
            self.assertLess(margin(pose.translation[:2]+[0,.008],polygon),0.)

    def test_force_moment_arm_at_each_actual_sole_vertex(self):
        force=np.array([6.,-4.,50.])
        for k,pose in enumerate(self.poses):
            rows=self.soles.wrench_rows(k)
            vertices=self.soles.vertices[k]@pose.rotation.T
            for vertex in vertices:
                vertex=vertex.copy();vertex[2]=np.mean(vertices[:,2])
                wrench=np.r_[force,np.cross(vertex,force)]
                self.assertLessEqual(float(np.max(rows@wrench)),1e-7)
            outside=vertices.mean(axis=0)+[0.,.02,0.]
            self.assertGreater(float(np.max(rows@np.r_[force,np.cross(outside,force)])),.1)

    def test_original_reference_requires_unsupported_contact_wrench(self):
        report=audit()
        self.assertFalse(report['reference_support_screen_passed'])
        self.assertLess(report['sample_1_2_seconds']['floor_cop_margin_m'],-.07)
        self.assertAlmostEqual(report['first_swing_lateral_com_to_right_ankle_m'],.0761,places=3)

    def test_transfer_rejects_unfinished_unloading_and_capture_error(self):
        state=dict(support_cop=[[.005,-.075,0.],[0.,.075,0.]],foot_force=[20.,24.,0.,0.])
        debug=dict(com=[.005,-.075,.35],vcom=[0.,0.,0.])
        self.assertTrue(transfer_screen(state,debug)['passed'])
        state['foot_force'][2]=2.
        self.assertFalse(transfer_screen(state,debug)['checks']['swing_unloaded'])
        state['foot_force'][2]=0.
        debug['com'][1]=-.065
        self.assertFalse(transfer_screen(state,debug)['checks']['capture_near_support'])


if __name__=='__main__':
    unittest.main()
