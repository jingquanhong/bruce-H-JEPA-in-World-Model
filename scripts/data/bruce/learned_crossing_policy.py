"""Run a trained actor on live observations with the same torque controller.

This adapter never sets poses or applies floating-base forces during a trial.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
import bruce_env as be
from dynamic_step import centroidal_state
from support_wbc import SupportController
from standing_wbc import WholeBodyController
from qualification import motor_current_from_joint_torque
from train_crossing_ppo import ActorCritic


class LearnedCrossingPolicy:
    def __init__(self,initial,*,checkpoint,height,center,thickness,stochastic=False,seed=0,**kwargs):
        torch.set_num_threads(1)
        self.checkpoint=Path(checkpoint).resolve()
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        if payload['obs_dim']!=72:raise ValueError('Unexpected actor observation schema')
        self.model=ActorCritic(72);self.model.load_state_dict(payload['model']);self.model.eval()
        self.controller=SupportController(initial)
        row=initial.copy();state=dict(base_pos=row[:3],base_quat=np.array([1.,0,0,0]),base_vel=np.zeros(3),base_ang=np.zeros(3),
            dof_pos=row,dof_vel=np.zeros(22))
        c=WholeBodyController(row);c.command(state,row);self.ff=c.full_command.copy()
        self.nominal=row[6:].copy();self.row=row;self.scales=np.array([.15,.65,.35,.8,.55]*2)
        self.kp=np.r_[np.full(10,50.),np.full(6,15.)];self.kd=np.r_[np.full(10,1.5),np.full(6,.6)]
        self.lower=np.zeros(16);self.upper=np.zeros(16)
        for iq,iv,slot in self.controller.mapping:
            self.lower[slot]=self.controller.model.lowerPositionLimit[iq]
            self.upper[slot]=self.controller.model.upperPositionLimit[iq]
        self.height=height;self.center=center;self.thickness=thickness
        self.action=np.zeros(10,dtype=np.float32);self.target=self.nominal.copy();self.last_tick=-1
        self.phase='crossing';self.events=[];self.final_start=None;self.debug={}
        self.stochastic=stochastic;self.rng=np.random.default_rng(seed)
        self.provenance=dict(checkpoint=str(self.checkpoint),sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            training_config=payload['config'],actor_observations=72,actor_hz=50,torque_controller_hz=200,
            stochastic=stochastic,policy_noise_seed=seed)

    def command(self,scene,t,state):
        tick=int(round(t/.005));R=be.quat_to_mat(state['base_quat'])
        foot_ids=[scene.link_index[name] for name in be.ANKLE_LINKS]
        feet=np.array([scene.get_link_pos(n) for n in be.ANKLE_LINKS])
        fv=be.to_np(scene.robot.get_links_vel(links_idx_local=foot_ids,relative=False))
        force=be.to_np(scene.robot.get_links_net_contact_force())[foot_ids]
        contact=force[:,2]>3.
        if tick%4==0 and tick!=self.last_tick:
            phase=(tick//4)*.02*2*np.pi*1.5
            obs=np.r_[state['base_quat'],state['base_pos'][2],R.T@state['base_vel'],.25*(R.T@state['base_ang']),
                state['dof_pos'][6:]-self.nominal,state['dof_vel'][6:]*.1,
                (feet-state['base_pos']).reshape(6),fv.reshape(6)*.5,contact.astype(float),
                [self.center-state['base_pos'][0],self.height,self.center+self.thickness/2+.1-state['base_pos'][0]],
                self.action,np.sin(phase),np.cos(phase)].astype(np.float32)
            if obs.shape!=(72,):raise ValueError(obs.shape)
            with torch.no_grad():
                distribution=self.model.distribution(torch.tensor(np.clip(obs,-20,20))[None])
                action=distribution.mean[0].numpy()
                if self.stochastic:action=action+distribution.scale[0].numpy()*self.rng.standard_normal(10)
            self.action=(.5*self.action+.5*np.clip(action,-2,2)).astype(np.float32)
            self.target=self.nominal.copy();self.target[:10]+=self.action*self.scales
            self.target=np.clip(self.target,self.lower+.01,self.upper-.01);self.last_tick=tick
        tau=self.ff+self.kp*(self.target-state['dof_pos'][6:])-self.kd*state['dof_vel'][6:]
        legs=np.clip(tau[:10],-10.5,10.5)
        motor_peak=np.max(np.abs(motor_current_from_joint_torque(legs)))*.35
        legs*=min(1.,10.46/(motor_peak+1e-6))
        command=np.r_[legs,np.clip(tau[10:],-1,1)]
        com,vel,L=centroidal_state(self.controller,state)
        good=(np.min(feet[:,0])>self.center+self.thickness/2+.0435 and
            state['base_pos'][0]>self.center+self.thickness/2+.065 and R[2,2]>.98 and
            np.linalg.norm(state['base_vel'])<.10 and contact.all())
        if good:
            if self.final_start is None:self.final_start=t
            self.phase='settle'
        else:self.final_start=None;self.phase='crossing'
        self.debug=dict(phase=self.phase,com=com.tolist(),velocity=vel.tolist(),momentum=L.tolist(),
            com_target=None,feet=feet.tolist(),foot_targets=None,ground_forces=force[:,2].tolist(),
            qp_residual=None,predicted_com_acc=None,actor_action=self.action.tolist())
        return command
