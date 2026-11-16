"""Experimental COM/foot task controller for unassisted support diagnostics.

This controller is not qualified for obstacle crossing or hardware deployment.
Contact wrenches are planning variables, never external simulator actuators.
"""
import numpy as np
import osqp
import pinocchio as pin
import scipy.sparse as sp

import bruce_expert as bx
from qualification import BEAR_TO_JOINT_RIGHT, BEAR_TO_JOINT_LEFT, motor_current_from_joint_torque
from support_geometry import SoleGeometry


class SupportController:
    def __init__(self, reference, *, com_weight=1000., swing_weight=100., attitude_weight=1., line_contact=False, solver_backend='osqp', momentum_weight=0.):
        self.q0 = bx.row_to_pin(reference)
        self.model = bx._PIN['model']
        self.data = self.model.createData()
        m, d = self.model, self.data
        self.mapping = [(m.joints[i].idx_q, m.joints[i].idx_v, slot)
                        for i in range(2, m.njoints)
                        if (slot := bx._ROW_SLOT.get(m.names[i])) is not None]
        self.frames = [m.getFrameId('ankle_pitch_link_' + s) for s in ('r', 'l')]
        pin.forwardKinematics(m, d, self.q0)
        pin.updateFramePlacements(m, d)
        self.feet0 = [d.oMf[f].copy() for f in self.frames]
        self.soles = SoleGeometry(bx.URDF,self.feet0)
        self.com0 = pin.centerOfMass(m, d, self.q0).copy()
        self.com_weight = com_weight
        self.swing_weight = swing_weight
        self.attitude_weight = attitude_weight
        self.line_contact = line_contact
        self.previous = None
        self.failures = 0
        self.solver_backend = solver_backend
        self.momentum_weight = momentum_weight
        self.hard_com = False
        self.momentum_target = np.zeros(3)
        self.attitude_kp=40.;self.attitude_kd=12.
        self.foot_kp=200.;self.foot_kd=28.
        self.track_swing_roll=False
        self.arm_posture_weight=100.
        self.limit_horizon=.1
        self.com_axis_weight=np.ones(3)
        self.com_kp=np.full(3,100.);self.com_kd=np.full(3,20.)
        self.point_contact=False
        self.friction=.7

    def command(self, state, com, feet, stance, com_vel=None, com_acc=None,
                foot_vel=None, foot_acc=None):
        m, d = self.model, self.data
        q = self.q0.copy()
        q[:3] = state['base_pos']
        quat = state['base_quat']
        q[3:7] = np.r_[quat[1:], quat[0]]
        R = pin.Quaternion(q[3:7]).matrix()
        v = np.zeros(m.nv)
        v[:3], v[3:6] = R.T @ state['base_vel'], R.T @ state['base_ang']
        for iq, iv, slot in self.mapping:
            q[iq], v[iv] = state['dof_pos'][6+slot], state['dof_vel'][6+slot]
        pin.computeAllTerms(m, d, q, v)
        M = np.triu(d.M) + np.triu(d.M, 1).T
        for _, iv, slot in self.mapping:
            if slot < 10:
                M[iv, iv] += .00182
        bias = d.nle.copy()
        Jcom = pin.jacobianCenterOfMass(m, d, q).copy()
        pin.centerOfMass(m, d, q, v, np.zeros(m.nv))
        actual_com, vcom, acom_bias = d.com[0].copy(), d.vcom[0].copy(), d.acom[0].copy()
        pin.computeJointJacobiansTimeVariation(m, d, q, v)
        pin.updateFramePlacements(m, d)
        actual = [d.oMf[f].copy() for f in self.frames]
        Js = [pin.getFrameJacobian(m, d, f, pin.LOCAL_WORLD_ALIGNED).copy() for f in self.frames]
        dJs = [pin.getFrameJacobianTimeVariation(m, d, f, pin.LOCAL_WORLD_ALIGNED).copy() for f in self.frames]
        contact_Js = [J.copy() for J in Js]
        contact_dJs = [J.copy() for J in dJs]
        if self.line_contact:
            for k in stance:
                cop = np.asarray(state['support_cop'][k])
                offset = cop-actual[k].translation
                omega = (Js[k]@v)[3:]
                contact_Js[k][:3] -= pin.skew(offset)@Js[k][3:]
                contact_dJs[k][:3] -= pin.skew(offset)@dJs[k][3:] + pin.skew(np.cross(omega,offset))@Js[k][3:]
        Jcontact = np.concatenate([contact_Js[k] for k in stance])
        size = m.nv + 6*len(stance)
        P = np.eye(size)*1e-5
        linear = np.zeros(size)

        def task(J, target, weight):
            A = np.zeros((len(target), size)); A[:, :m.nv] = J
            P[:] += weight * A.T @ A
            linear[:] -= weight * A.T @ target

        cv = np.zeros(3) if com_vel is None else com_vel
        ca = np.zeros(3) if com_acc is None else com_acc
        com_target_acc=ca + self.com_kp*(com-actual_com) + self.com_kd*(cv-vcom) - acom_bias
        axis_scale=np.sqrt(self.com_axis_weight)
        task(axis_scale[:,None]*Jcom, axis_scale*com_target_acc, self.com_weight)
        momentum=pin.computeCentroidalMomentum(m,d,q,v).vector.copy()
        if self.momentum_weight:
            Ag=pin.computeCentroidalMap(m,d,q).copy()
            dAg=pin.computeCentroidalMapTimeVariation(m,d,q,v).copy()
            task(Ag[3:],-20*(momentum[3:]-self.momentum_target)-(dAg@v)[3:],self.momentum_weight)
        posture = 20*pin.difference(m, q, self.q0) - 8*v
        task(np.eye(m.nv)[6:], posture[6:], .1)
        arm_axes=[iv for _,iv,slot in self.mapping if slot>=10]
        arm_target=100*pin.difference(m,q,self.q0)[arm_axes]-20*v[arm_axes]
        task(np.eye(m.nv)[arm_axes],arm_target,self.arm_posture_weight)
        # BRUCE has no ankle-roll joint; lateral weight transfer needs torso roll.
        # Keep attitude a soft objective so that it cannot override COM support.
        task(np.eye(m.nv)[3:6], self.attitude_kp*pin.log3(R.T) - self.attitude_kd*v[3:6], self.attitude_weight)
        dyn = np.c_[M, -Jcontact.T]
        bounds = np.full(m.nv, 10.5); bounds[:6] = 0.
        motor_map = np.zeros((10, m.nv))
        for _, iv, slot in self.mapping:
            if slot >= 10:
                bounds[iv] = 1.
            else:
                start = 0 if slot < 5 else 5
                transform = BEAR_TO_JOINT_RIGHT if slot < 5 else BEAR_TO_JOINT_LEFT
                motor_map[start:start+5, iv] = transform.T[:, slot-start]
        constraints = [dyn, motor_map @ dyn]
        lower = [ -bounds-bias, -10.5-motor_map@bias ]
        upper = [ bounds-bias, 10.5-motor_map@bias ]
        if self.limit_horizon:
            horizon=self.limit_horizon
            for iq,iv,slot in self.mapping:
                row=np.zeros((1,size));row[0,iv]=1.
                low=2*(m.lowerPositionLimit[iq]+.003-q[iq]-v[iv]*horizon)/horizon**2
                high=2*(m.upperPositionLimit[iq]-.003-q[iq]-v[iv]*horizon)/horizon**2
                constraints.append(row);lower.append([low]);upper.append([high])
        if self.hard_com:
            row=np.zeros((3,size));row[:,:m.nv]=Jcom
            constraints.append(row);lower.append(com_target_acc);upper.append(com_target_acc)
        for k in range(2):
            error = np.r_[feet[k]-actual[k].translation,
                          pin.log3(self.feet0[k].rotation @ actual[k].rotation.T)]
            target_vel = np.zeros(6); target_acc = np.zeros(6)
            if foot_vel is not None:
                target_vel[:3] = foot_vel[k]
            if foot_acc is not None:
                target_acc[:3] = foot_acc[k]
            target = target_acc + self.foot_kp*error - self.foot_kd*(Js[k]@v-target_vel) - dJs[k]@v
            if k in stance:
                A = np.zeros((3,size)); A[:,:m.nv] = contact_Js[k][:3]
                ct = -28*(contact_Js[k]@v)[:3] - (contact_dJs[k]@v)[:3] if self.line_contact else target[:3]
                constraints.append(A); lower.append(ct); upper.append(ct)
                # There is no ankle roll actuator. Forcing world roll to zero
                # locks lateral translation to a torso-lean strategy.
                task(Js[k][4:], target[4:], 100.)
            else:
                # Like BRUCE's original walking WBC, track foot position and
                # pitch/yaw only. There is no independent ankle-roll actuator.
                axes=list(range(6)) if self.track_swing_roll else [0,1,2,4,5]
                task(Js[k][axes], target[axes], self.swing_weight)
        for i,k in enumerate(stance):
            off = m.nv+6*i
            normal = np.zeros((1,size)); normal[0,off+2]=1
            constraints.append(normal); lower.append([0.]); upper.append([100.])
            for axis,bound in ((0,self.friction),(1,self.friction),(5,.008)):
                for sign in (-1,1):
                    row = np.zeros((1,size)); row[0,off+axis]=sign; row[0,off+2]=-bound
                    constraints.append(row); lower.append([-np.inf]); upper.append([0.])
            if self.line_contact:
                roll=np.zeros((1,size));roll[0,off+3]=1.
                constraints.append(roll);lower.append([0.]);upper.append([0.])
                if self.point_contact:
                    for axis in (4,5):
                        row=np.zeros((1,size));row[0,off+axis]=1.
                        constraints.append(row);lower.append([0.]);upper.append([0.])
                else:
                    for sign in (-1,1):
                        row=np.zeros((1,size));row[0,off+4]=sign;row[0,off+2]=-.035
                        constraints.append(row);lower.append([-np.inf]);upper.append([0.])
            else:
                # Flat-sole planning cone. This is an upper bound on support
                # when the real sole is tilted, not a contact-success check.
                for coefficients in self.soles.wrench_rows(k):
                    row=np.zeros((1,size))
                    row[0,off:off+6]=coefficients
                    constraints.append(row);lower.append([-np.inf]);upper.append([0.])
        A = np.concatenate(constraints); lo = np.concatenate(lower); hi = np.concatenate(upper)
        scale = max(self.com_weight, self.swing_weight, 1.)
        if self.solver_backend=='clarabel':
            from qp_solver import solve_interior
            try: x=solve_interior(2*P/scale,2*linear/scale,A,lo,hi)
            except RuntimeError:
                self.failures+=1
                raise
        else:
            solver = osqp.OSQP()
            solver.setup(P=sp.csc_matrix(2*P/scale), q=2*linear/scale, A=sp.csc_matrix(A),
                         l=lo,u=hi,verbose=False,eps_abs=1e-4,eps_rel=1e-4,max_iter=12000)
            if self.previous is not None and len(self.previous)==size:
                solver.warm_start(x=self.previous)
            result = solver.solve(raise_error=False)
            if result.info.status_val not in (1,2) or result.x is None:
                self.failures += 1
                raise RuntimeError(f'support QP: {result.info.status}; primal={result.info.prim_res:.3g}, dual={result.info.dual_res:.3g}')
            x = result.x
        residual = float(max(np.max(lo-A@x),np.max(A@x-hi),0.))
        if residual > .01:
            self.failures += 1
            raise RuntimeError(f'support QP residual {residual:.4g}')
        self.previous = x.copy()
        tau = dyn@x+bias
        command = np.empty(16)
        for _,iv,slot in self.mapping:
            command[slot] = tau[iv]
        command[:10]=np.clip(command[:10],-10.5,10.5)
        command[10:]=np.clip(command[10:],-1.,1.)
        peak=float(np.abs(motor_current_from_joint_torque(command[:10])).max())
        if peak>29.99:
            command[:10]*=29.99/peak
        self.debug = dict(com=actual_com.tolist(), vcom=vcom.tolist(),
                          target=np.asarray(com).tolist(), qp_residual=residual,
                          feet=[p.translation.tolist() for p in actual],
                          com_acc_predicted=(Jcom@x[:m.nv]+acom_bias).tolist(),
                          angular_momentum=momentum[3:].tolist(),
                          wrenches=x[m.nv:].reshape(-1,6).tolist(), stance=list(stance))
        return command
