"""Bounded stance-leg IK for slow double-support crouch only; no flight."""
import json
import math
from pathlib import Path
import numpy as np
from Util.jump_preparation_profile import checked_depth, offset_at


def offset_velocity(elapsed, depth):
    checked_depth(depth)
    if 0 <= elapsed < 2:
        u=elapsed/2
        return -depth*30*u*u*(1-u)*(1-u)/2
    if 2.5 <= elapsed < 4.5:
        u=(elapsed-2.5)/2
        return depth*30*u*u*(1-u)*(1-u)/2
    return 0.


class CrouchAnchor:
    def __init__(self, kin, q, rotation, lower, upper):
        self.kin=kin;self.q0=np.asarray(q).copy();self.rotation=np.asarray(rotation).copy()
        self.lower=np.asarray(lower);self.upper=np.asarray(upper)
        if self.q0.shape!=(10,) or self.rotation.shape!=(3,3) or not np.isfinite(self.q0).all():
            raise ValueError('Invalid crouch anchor')
        if not np.isfinite(self.rotation).all() or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-5):
            raise ValueError('Invalid anchor orientation')
        if self.rotation[2,2]<math.cos(math.radians(5)):
            raise ValueError('Crouch anchor must be upright')
        fk=kin.legFK(*self.q0,*np.zeros(10))
        self.positions=[np.asarray(fk[8]).copy(),np.asarray(fk[25]).copy()]
        self.directions=[np.asarray(fk[14])[:,0].copy(),np.asarray(fk[31])[:,0].copy()]
        self.ik0=self._ik(0.)
        if np.max(abs(self.ik0-self.q0))>.02:
            raise ValueError('Initial IK branch does not match current posture')

    def _ik(self, dz):
        vertical=self.rotation.T@np.array([0.,0.,dz])
        return np.array([*self.kin.legIK_ankle(self.positions[0]-vertical,self.directions[0],1.),
                         *self.kin.legIK_ankle(self.positions[1]-vertical,self.directions[1],-1.)])

    def target(self, dz, vz):
        if not math.isfinite(dz) or not -.015001<=dz<=.000001 or not math.isfinite(vz) or abs(vz)>.02:
            raise ValueError('Crouch reference outside slow preparation bounds')
        q=self.q0+self._ik(dz)-self.ik0
        epsilon=1e-5
        dq=(self._ik(dz+epsilon)-self._ik(dz-epsilon))/(2*epsilon)*vz
        if not np.isfinite(q).all() or not np.isfinite(dq).all():raise ValueError('Invalid crouch IK')
        if np.any(q<self.lower+.01) or np.any(q>self.upper-.01):raise ValueError('Crouch IK violates joint bounds')
        if max(q[3],q[8])>-.1:raise ValueError('Crouch IK near knee extension')
        if np.max(abs(q-self.q0))>.15 or np.max(abs(dq))>.3:
            raise ValueError('Crouch joint displacement/speed bound')
        return q,dq


class PreparationReference:
    def __init__(self, path, kin, lower, upper):
        self.path=Path(path) if path else None
        self.kin=kin;self.lower=lower;self.upper=upper
        self.last_read=-1.;self.reference=None;self.anchor=None;self.started=None

    def apply(self, now, phase, q_measured, body_rotation, q_desired, dq_desired):
        if self.path is None:return
        if now-self.last_read>.05:
            self.last_read=now
            if self.path.exists():
                r=json.loads(self.path.read_text())
                start=float(r['started']);depth=checked_depth(r['depth_m'])
                if not math.isfinite(start):raise ValueError('Invalid crouch time')
                if start!=self.started:
                    age=now-start
                    if not 0<=age<=.2:raise ValueError('Late crouch reference')
                    if phase!=0:raise ValueError('Crouch is double support only')
                    self.anchor=CrouchAnchor(self.kin,q_measured,body_rotation,self.lower,self.upper)
                    self.started=start;self.reference=(start,depth)
        if self.reference is None:return
        elapsed=now-self.reference[0]
        if elapsed>6.5:return
        if phase!=0:raise ValueError('Contact phase changed during crouch')
        depth=self.reference[1]
        q,dq=self.anchor.target(offset_at(elapsed,depth),offset_velocity(elapsed,depth))
        q_desired[:]=q;dq_desired[:]=dq
