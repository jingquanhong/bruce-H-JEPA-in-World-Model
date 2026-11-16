"""Experimental capture-point stepping policy. Qualification is external."""
import numpy as np
import pinocchio as pin

import bruce_env as be
from support_reference import smooth,support_cop
from support_wbc import SupportController


def centroidal_state(controller,state):
    m,d=controller.model,controller.data
    q=controller.q0.copy();q[:3]=state['base_pos'];q[3:7]=np.r_[state['base_quat'][1:],state['base_quat'][0]]
    R=be.quat_to_mat(state['base_quat']);v=np.zeros(m.nv)
    v[:3]=R.T@state['base_vel'];v[3:6]=R.T@state['base_ang']
    for iq,iv,slot in controller.mapping:
        q[iq]=state['dof_pos'][6+slot];v[iv]=state['dof_vel'][6+slot]
    pin.centerOfMass(m,d,q,v)
    com=d.com[0].copy();vel=d.vcom[0].copy()
    momentum=pin.computeCentroidalMomentum(m,d,q,v).angular.copy()
    return com,vel,momentum


def floor_forces(scene):
    forces=np.zeros(2);contacts=scene.get_contacts()
    if contacts is None:return forces
    for i,(a,b) in enumerate(zip(contacts['geom_a'],contacts['geom_b'])):
        if int(a)==0:geom=int(b);f=contacts['force_b'][i]
        elif int(b)==0:geom=int(a);f=contacts['force_a'][i]
        else:continue
        name=scene.geom_link.get(geom)
        if name in be.ANKLE_LINKS:forces[be.ANKLE_LINKS.index(name)]+=max(0.,float(f[2]))
    return forces


def quintic(p0,v0,p1,duration,t):
    p0=np.asarray(p0);v0=np.asarray(v0);p1=np.asarray(p1)
    T=duration;s=np.clip(t,0.,T)
    c0=p0;c1=v0;c2=np.zeros_like(p0)
    matrix=np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]])
    c3,c4,c5=np.linalg.solve(matrix,np.array([p1-p0-v0*T,-v0,np.zeros_like(p0)]))
    p=c0+c1*s+c3*s**3+c4*s**4+c5*s**5
    v=c1+3*c3*s*s+4*c4*s**3+5*c5*s**4
    a=6*c3*s+12*c4*s*s+20*c5*s**3
    if t>T:v=np.zeros_like(v);a=np.zeros_like(a)
    return p,v,a


class DynamicStepPolicy:
    def __init__(self,initial,*,height=.02,swing_time=.24,second_time=.28,pulse_acc=1.3,
                 lateral=-.03,swing_weight=1000.,attitude_weight=50.):
        self.controller=SupportController(initial,com_weight=10000.,swing_weight=swing_weight,
            attitude_weight=attitude_weight,solver_backend='clarabel')
        c=self.controller
        self.feet0=np.array([p.translation.copy() for p in c.feet0])
        self.height=height;self.swing_time=swing_time;self.second_time=second_time
        self.pulse_acc=pulse_acc;self.lateral=lateral
        self.phase='prepare';self.start=0.;self.debug={};self.events=[]
        self.initial_com=c.com0.copy();self.omega=np.sqrt(9.81/self.initial_com[2])
        self.target_feet=self.feet0.copy();self.final_start=None
        self.landing_min=.235

    def transition(self,phase,t,com,vel):
        self.phase=phase;self.start=t
        self.phase_com=com.copy();self.phase_vel=vel.copy()
        self.events.append(dict(phase=phase,t=t,com=com.tolist(),velocity=vel.tolist()))

    def reference(self,t):
        c=self.controller;feet=self.target_feet.copy();com=self.initial_com.copy()
        cv=np.zeros(3);ca=np.zeros(3);stance=[0,1]
        s=t-self.start;w=self.omega
        if self.phase=='prepare':
            com[1]=self.lateral*smooth((t-.3)/1.)
        elif self.phase=='accelerate':
            com=self.phase_com.copy();com[0]+=.5*self.pulse_acc*s*s
            cv[0]=self.pulse_acc*s;ca[0]=self.pulse_acc
        elif self.phase=='first_swing':
            com=self.support_point.copy()
            com[:2]+=(self.phase_com[:2]-self.support_point[:2])*np.cosh(w*s)+self.phase_vel[:2]/w*np.sinh(w*s)
            cv[:2]=w*(self.phase_com[:2]-self.support_point[:2])*np.sinh(w*s)+self.phase_vel[:2]*np.cosh(w*s)
            ca[:2]=w*w*(com[:2]-self.support_point[:2]);com[2]=self.initial_com[2]
            u=np.clip(s/self.swing_time,0.,1.)
            feet[1]=self.feet0[1]+(self.target_feet[1]-self.feet0[1])*smooth(u)
            feet[1,2]+=(self.height+.045)*(smooth(u/.25) if u<.25 else 1. if u<.7 else 1-smooth((u-.7)/.3))
            stance=[0]
        elif self.phase in ('capture','second_swing'):
            com,cv,ca=quintic(self.capture_start,self.capture_velocity,self.capture_target,.65,t-self.capture_time)
            if self.phase=='second_swing':
                u=np.clip(s/self.second_time,0.,1.)
                feet[0]=self.second_from+(self.target_feet[0]-self.second_from)*smooth(u)
                feet[0,2]+=(self.height+.045)*(smooth(u/.25) if u<.25 else 1. if u<.7 else 1-smooth((u-.7)/.3))
                stance=[1]
        elif self.phase=='settle':
            target=np.r_[np.mean(self.target_feet[:,:2],axis=0),self.initial_com[2]]
            com=self.phase_com+(target-self.phase_com)*smooth(s/1.)
        return com,feet,stance,cv,ca

    def command(self,scene,t,state):
        c=self.controller;com,vel,L=centroidal_state(c,state)
        forces=floor_forces(scene);feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS])
        state=state.copy();state['support_cop']=support_cop(scene)
        if self.phase=='prepare' and t>=1.5:
            self.transition('accelerate',t,com,vel)
        if self.phase=='accelerate':
            cop=np.asarray(support_cop(scene)[0]);cp=cop[:2]+(com[:2]-cop[:2]+vel[:2]/self.omega)*np.exp(self.omega*self.swing_time)
            if cp[0]>self.landing_min:
                if cp[0]>self.landing_min+.1 or not .04<cp[1]<.20:raise RuntimeError('capture_placement_out_of_reach')
                self.target_feet[1]=[cp[0],cp[1],.025]
                self.support_point=np.r_[cop[:2],self.initial_com[2]]
                self.transition('first_swing',t,com,vel)
            elif t-self.start>.35:raise RuntimeError('forward_capture_not_ready')
        if self.phase=='first_swing' and t-self.start>=self.swing_time and forces[1]>5.:
            self.target_feet=feet.copy()
            self.capture_time=t;self.capture_start=com.copy()
            self.capture_velocity=vel.copy()
            self.capture_target=np.r_[state['support_cop'][1][:2],self.initial_com[2]]
            self.transition('capture',t,com,vel)
        if self.phase=='capture':
            capture_y=com[1]+vel[1]/self.omega
            ready=abs(capture_y-state['support_cop'][1][1])<.004 and abs(vel[1])<.025 and forces[0]<1.5 and forces[1]>30.
            if t-self.start>.7 and ready:
                self.second_from=feet[0].copy()
                self.target_feet[0]=[max(self.landing_min+.015,self.target_feet[1,0]+.015),self.target_feet[1,1]-.1517,.025]
                self.transition('second_swing',t,com,vel)
            elif t-self.start>1.5:raise RuntimeError('leading_support_capture_not_ready')
        if self.phase=='second_swing' and t-self.start>=self.second_time and forces[0]>5.:
            self.target_feet[0]=feet[0].copy()
            self.transition('settle',t,com,vel);self.final_start=t
        ref,targets,stance,cv,ca=self.reference(t)
        cm,fm,_,_,_=self.reference(t-.005);cp,fp,_,_,_=self.reference(t+.005)
        if self.phase in ('prepare','settle'):
            cv=(cp-cm)/.01;ca=(cp-2*ref+cm)/.005**2
        command=c.command(state,ref,targets,stance,cv,ca,(fp-fm)/.01,(fp-2*targets+fm)/.005**2)
        self.debug=dict(phase=self.phase,com=com.tolist(),velocity=vel.tolist(),momentum=L.tolist(),
            com_target=ref.tolist(),feet=feet.tolist(),foot_targets=targets.tolist(),
            ground_forces=forces.tolist(),qp_residual=c.debug['qp_residual'],
            predicted_com_acc=c.debug['com_acc_predicted'])
        return command
