"""Experimental airborne inverse dynamics with no external contact wrench."""
import numpy as np
import pinocchio as pin

from qualification import BEAR_TO_JOINT_RIGHT, BEAR_TO_JOINT_LEFT
from qp_solver import solve_interior
from support_wbc import SupportController


class FlightController(SupportController):
    def command_flight(self,state,target):
        m,d=self.model,self.data
        q=self.q0.copy();q[:3]=state['base_pos'];q[3:7]=np.r_[state['base_quat'][1:],state['base_quat'][0]]
        R=pin.Quaternion(q[3:7]).matrix()
        v=np.zeros(m.nv);v[:3]=R.T@state['base_vel'];v[3:6]=R.T@state['base_ang']
        desired=np.zeros(m.nv);weights=np.full(m.nv,.01)
        weights[:3]=0.;weights[3:6]=5000.
        desired[3:6]=100*pin.log3(R.T)-20*v[3:6]
        bounds=np.full(m.nv,10.5);bounds[:6]=0.
        motor_map=np.zeros((10,m.nv))
        for iq,iv,slot in self.mapping:
            q[iq]=state['dof_pos'][6+slot];v[iv]=state['dof_vel'][6+slot]
            desired[iv]=100*(target[slot]-q[iq])-20*v[iv]
            weights[iv]=5. if slot in (3,8) else 1.
            if slot>=10:bounds[iv]=1.;weights[iv]=.1
            else:
                start=0 if slot<5 else 5
                transform=BEAR_TO_JOINT_RIGHT if slot<5 else BEAR_TO_JOINT_LEFT
                motor_map[start:start+5,iv]=transform.T[:,slot-start]
        pin.computeAllTerms(m,d,q,v)
        M=np.triu(d.M)+np.triu(d.M,1).T
        for _,iv,slot in self.mapping:
            if slot<10:M[iv,iv]+=.00182
        bias=d.nle.copy()
        A=np.r_[M,motor_map@M]
        lower=np.r_[-bounds-bias,-10.4965-motor_map@bias]
        upper=np.r_[bounds-bias,10.4965-motor_map@bias]
        # Anticipate joint stops rather than asking the simulator's limit
        # impulses to cancel the momentum of a commanded airborne motion.
        horizon=.08
        for iq,iv,slot in self.mapping:
            row=np.zeros((1,m.nv));row[0,iv]=1.
            lo=2*(m.lowerPositionLimit[iq]+.01-q[iq]-horizon*v[iv])/horizon**2
            hi=2*(m.upperPositionLimit[iq]-.01-q[iq]-horizon*v[iv])/horizon**2
            A=np.r_[A,row];lower=np.r_[lower,lo];upper=np.r_[upper,hi]
        x=solve_interior(np.diag(2*weights),-2*weights*desired,A,lower,upper)
        residual=max(float(np.max(lower-A@x)),float(np.max(A@x-upper)))
        if residual>.001:raise RuntimeError(f'flight dynamics residual {residual}')
        torque=M@x+bias;command=np.empty(16)
        for _,iv,slot in self.mapping:command[slot]=torque[iv]
        self.debug=dict(qdd=x.tolist(),predicted_body_angular_acc=x[3:6].tolist(),constraint_residual=residual)
        return command
