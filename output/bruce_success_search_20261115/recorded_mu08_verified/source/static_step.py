"""Guarded slow crossing experiment: establish support before each foot lift."""
import numpy as np
import bruce_env as be
from support_wbc import SupportController
from support_reference import support_cop,smooth
from dynamic_step import centroidal_state,floor_forces,quintic


class StaticStepPolicy:
    def __init__(self,initial,*,height=.02,**kwargs):
        self.controller=SupportController(initial,com_weight=10000.,swing_weight=1000.,
            attitude_weight=50.,solver_backend='clarabel')
        self.height=height;self.feet0=np.array([p.translation.copy() for p in self.controller.feet0])
        self.target_feet=self.feet0.copy();self.com0=self.controller.com0.copy()
        self.phase='prepare';self.start=0.;self.events=[];self.final_start=None
        self.com_from=self.com0.copy();self.com_to=self.com0.copy();self.velocity_from=np.zeros(3)
        self.landing_min=.235;self.debug={};self.side=0;self.ready_since=None
        self.line_single=False

    def transition(self,phase,t,com,vel):
        self.phase=phase;self.start=t;self.com_from=com.copy();self.velocity_from=vel.copy()
        self.events.append(dict(phase=phase,t=t,com=com.tolist(),velocity=vel.tolist()))

    def reference(self,t):
        s=t-self.start;feet=self.target_feet.copy();stance=[0,1]
        duration=2. if self.phase=='shift' else 1.
        com,cv,ca=quintic(self.com_from,self.velocity_from,self.com_to,duration,s)
        if self.phase in ('lift','traverse','lower'):
            stance=[self.side];other=1-self.side
            feet[other]=self.foot_from+(self.foot_to-self.foot_from)*smooth(s/self.move_time)
        return com,feet,stance,cv,ca

    def command(self,scene,t,state):
        c=self.controller;com,vel,L=centroidal_state(c,state)
        feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS]);forces=floor_forces(scene)
        cop=np.array(support_cop(scene));state=state.copy();state['support_cop']=cop
        s=t-self.start
        if self.phase=='prepare' and s>=.5:
            self.com_to=np.r_[cop[0,:2],self.com0[2]]
            self.transition('shift',t,com,vel)
        elif self.phase=='shift' and s>=2.:
            self.target_feet=feet.copy();self.com_to=np.r_[cop[self.side,:2],self.com0[2]]
            self.transition('balance',t,com,vel)
        elif self.phase=='balance':
            w=np.sqrt(9.81/com[2]);cp=com[:2]+vel[:2]/w
            ready=np.linalg.norm(cp-cop[self.side,:2])<.002 and np.linalg.norm(vel[:2])<.015 and forces[1-self.side]<2. and forces[self.side]>30.
            if s>1. and ready:
                if self.ready_since is None:self.ready_since=t
                if t-self.ready_since>=.1:
                    other=1-self.side;self.foot_from=feet[other].copy();self.foot_to=self.foot_from.copy()
                    self.foot_to[2]=.025+self.height+.065;self.move_time=.8
                    self.com_to=np.r_[cop[self.side,:2],self.com0[2]]
                    self.transition('lift',t,com,vel)
            else:self.ready_since=None
            if s>3.:raise RuntimeError('static_support_transfer_not_ready')
        elif self.phase=='lift' and s>=self.move_time:
            self.target_feet[1-self.side]=self.foot_to.copy()
            self.foot_from=self.foot_to.copy();self.foot_to[0]=self.landing_min+.015
            self.move_time=1.;self.transition('traverse',t,com,vel)
        elif self.phase=='traverse' and s>=self.move_time:
            self.target_feet[1-self.side]=self.foot_to.copy()
            self.foot_from=self.foot_to.copy();self.foot_to[2]=.015
            self.move_time=.8;self.transition('lower',t,com,vel)
        elif self.phase=='lower' and s>=self.move_time and forces[1-self.side]>5.:
            self.target_feet=feet.copy()
            if self.side==0:
                self.side=1;self.com_to=np.r_[cop[1,:2],self.com0[2]];self.transition('shift',t,com,vel)
            else:
                self.com_to=np.r_[feet[:,:2].mean(0),self.com0[2]]
                self.transition('settle',t,com,vel);self.final_start=t
        ref,targets,stance,cv,ca=self.reference(t)
        c.line_contact=self.line_single and len(stance)==1
        _,fm,_,_,_=self.reference(t-.005);_,fp,_,_,_=self.reference(t+.005)
        command=c.command(state,ref,targets,stance,cv,ca,(fp-fm)/.01,(fp-2*targets+fm)/.005**2)
        self.debug=dict(phase=self.phase,com=com.tolist(),velocity=vel.tolist(),momentum=L.tolist(),
            com_target=ref.tolist(),feet=feet.tolist(),foot_targets=targets.tolist(),ground_forces=forces.tolist(),
            qp_residual=c.debug['qp_residual'],predicted_com_acc=c.debug['com_acc_predicted'])
        return command
