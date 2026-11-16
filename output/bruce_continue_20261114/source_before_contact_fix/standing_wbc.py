"""Experimental floating-base inverse dynamics; validated for standing only.

Only the 16 physical joints receive torques. Contact wrench bounds are planning
approximations; the simulator resolves actual mesh contacts independently.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
import osqp
import pinocchio as pin

import bruce_expert as bx
from qualification import BEAR_TO_JOINT_RIGHT, BEAR_TO_JOINT_LEFT


class WholeBodyController:
    def __init__(self, initial_reference, friction=0.7, leg_armature=0.00182):
        q = bx.row_to_pin(initial_reference)
        self.model = bx._PIN['model']
        self.data = self.model.createData()
        self.mapping = [(self.model.joints[i].idx_q, self.model.joints[i].idx_v, slot)
                        for i in range(2, self.model.njoints)
                        if (slot := bx._ROW_SLOT.get(self.model.names[i])) is not None]
        self.frames = [self.model.getFrameId('ankle_pitch_link_' + s) for s in ('r', 'l')]
        self.mu = friction
        self.previous = None
        self.failures = 0
        self.leg_armature = leg_armature

    def command(self, state, reference, previous_reference=None, next_reference=None, dt=.01):
        m, d = self.model, self.data
        q = bx.row_to_pin(reference)
        q[:3] = state['base_pos']
        quat = state['base_quat']
        q[3:7] = np.r_[quat[1:], quat[0]]
        v = np.zeros(m.nv)
        R = Rotation.from_quat(q[3:7]).as_matrix()
        v[:3] = R.T @ state['base_vel']
        v[3:6] = R.T @ state['base_ang']
        for iq, iv, slot in self.mapping:
            q[iq] = state['dof_pos'][6 + slot]
            v[iv] = state['dof_vel'][6 + slot]
        qr = bx.row_to_pin(reference)
        error = pin.difference(m, q, qr)
        vref = np.zeros(m.nv)
        if previous_reference is not None:
            vref = pin.difference(m, bx.row_to_pin(previous_reference), bx.row_to_pin(next_reference)) / (2 * dt)
        desired_acc = 100 * error - 20 * (v - vref)
        desired_acc[3:6] = 160 * error[3:6] - 25 * (v[3:6] - vref[3:6])
        pin.computeAllTerms(m, d, q, v)
        M = np.triu(d.M) + np.triu(d.M, 1).T
        for _, iv, slot in self.mapping:
            if slot < 10:
                M[iv, iv] += self.leg_armature
        bias = d.nle.copy()
        pin.computeJointJacobiansTimeVariation(m, d, q, v)
        pin.updateFramePlacements(m, d)
        actual = [d.oMf[f].copy() for f in self.frames]
        Jall = [pin.getFrameJacobian(m, d, f, pin.LOCAL_WORLD_ALIGNED).copy() for f in self.frames]
        DJall = [pin.getFrameJacobianTimeVariation(m, d, f, pin.LOCAL_WORLD_ALIGNED).copy() for f in self.frames]
        pin.forwardKinematics(m, d, qr)
        pin.updateFramePlacements(m, d)
        refs = [d.oMf[f].copy() for f in self.frames]
        ref_vel = [np.zeros(6) for _ in range(2)]
        ref_acc = [np.zeros(6) for _ in range(2)]
        if previous_reference is not None:
            pin.forwardKinematics(m, d, bx.row_to_pin(previous_reference)); pin.updateFramePlacements(m, d)
            prev = [d.oMf[f].copy() for f in self.frames]
            pin.forwardKinematics(m, d, bx.row_to_pin(next_reference)); pin.updateFramePlacements(m, d)
            nxt = [d.oMf[f].copy() for f in self.frames]
            for k in range(2):
                ref_vel[k][:3] = (nxt[k].translation - prev[k].translation) / (2 * dt)
                ref_vel[k][3:] = pin.log3(nxt[k].rotation @ prev[k].rotation.T) / (2 * dt)
                ref_acc[k][:3] = np.clip((nxt[k].translation - 2*refs[k].translation + prev[k].translation) / dt**2, -20, 20)
        stance = [k for k in range(2) if refs[k].translation[2] < 0.034]
        nc = len(stance)
        J = np.concatenate([Jall[k] for k in stance], axis=0) if nc else np.zeros((0, m.nv))
        size = m.nv + 6 * nc
        weights = np.r_[np.full(3, 200.), np.full(3, 200.), np.ones(m.nv - 6)]
        P = np.diag(np.r_[weights, np.full(6 * nc, 1e-4)])
        linear = np.r_[-weights * desired_acc, np.zeros(6 * nc)]
        for k in range(2):
            pos_err = refs[k].translation - actual[k].translation
            rot_err = pin.log3(refs[k].rotation @ actual[k].rotation.T)
            target = ref_acc[k] + 200 * np.r_[pos_err, rot_err] - 28 * (Jall[k] @ v - ref_vel[k]) - DJall[k] @ v
            task = np.zeros((6, size)); task[:, :m.nv] = Jall[k]
            task_weight = 1000 if k in stance else 500
            P += task_weight * task.T @ task
            linear -= task_weight * task.T @ target
        # Dynamics gives tau = M*qdd + h - J.T*f; base tau is exactly zero.
        dyn = np.c_[M, -J.T]
        lo_tau = np.full(m.nv, -10.5); hi_tau = -lo_tau
        lo_tau[:6] = hi_tau[:6] = 0
        for iq, iv, slot in self.mapping:
            if slot >= 10:
                lo_tau[iv], hi_tau[iv] = -1.0, 1.0
        constraints = [dyn]
        lower = list(lo_tau - bias); upper = list(hi_tau - bias)
        motor_map = np.zeros((10, m.nv))
        for _, iv, slot in self.mapping:
            if slot < 10:
                start = 0 if slot < 5 else 5
                transform = BEAR_TO_JOINT_RIGHT if slot < 5 else BEAR_TO_JOINT_LEFT
                motor_map[start:start+5, iv] = transform.T[:, slot-start]
        constraints.append(motor_map @ dyn)
        motor_bias = motor_map @ bias
        lower.extend(-10.5-motor_bias)
        upper.extend(10.5-motor_bias)
        for i in range(nc):
            offset = m.nv + 6 * i
            normal = np.zeros(size); normal[offset + 2] = 1
            constraints.append(normal[None]); lower.append(0); upper.append(150)
            for axis, bound in ((0, self.mu), (1, self.mu), (3, .008), (4, .035), (5, .01)):
                for sign in (-1, 1):
                    row = np.zeros(size); row[offset + axis] = sign; row[offset + 2] = -bound
                    constraints.append(row[None]); lower.append(-np.inf); upper.append(0)
        solver = osqp.OSQP()
        solver.setup(P=sp.csc_matrix(2 * P), q=2 * linear,
                     A=sp.csc_matrix(np.concatenate(constraints)),
                     l=np.array(lower), u=np.array(upper), verbose=False,
                     eps_abs=1e-4, eps_rel=1e-4, max_iter=12000)
        if self.previous is not None and len(self.previous) == size:
            solver.warm_start(x=self.previous)
        result = solver.solve(raise_error=False)
        if result.info.status_val not in (1, 2) or result.x is None:
            self.failures += 1
            raise RuntimeError(f"Inverse dynamics failed: {result.info.status}")
        self.previous = result.x.copy()
        tau = dyn @ result.x + bias
        self.debug = dict(qdd=result.x[:m.nv].tolist(), wrench=result.x[m.nv:].tolist(), tau=tau.tolist(), acc_target=desired_acc.tolist())
        command = np.empty(10)
        for iq, iv, slot in self.mapping:
            if slot < 10:
                command[slot] = np.clip(tau[iv], -10.5, 10.5)
        self.full_command = np.empty(16)
        for iq, iv, slot in self.mapping:
            bound = 10.5 if slot < 10 else 1.0
            self.full_command[slot] = np.clip(tau[iv], -bound, bound)
        return command
