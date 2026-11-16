"""Parallel Genesis training environment; screening is not qualification.

Free floating base, original collision meshes, 0.5 ms physics, 5 ms torque
controller and 20 ms learned position residuals. No state correction during
an episode. Resets only occur after a terminated episode.
"""
import argparse,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import bruce_env as be
import bruce_expert as bx
from qualification import BEAR_TO_JOINT_RIGHT,BEAR_TO_JOINT_LEFT
from standing_wbc import WholeBodyController


def inverse_rotate(q,v):
    xyz=q[...,1:];uv=torch.cross(xyz,v,dim=-1)
    return v-2*(q[...,:1]*uv-torch.cross(xyz,uv,dim=-1))


class CrossingEnv:
    dt=.02
    def __init__(self,n_envs=256,seed=0,height=.01,center=.12,thickness=.02,friction=.8,body_height=.4,
                 reward_version=2,goal_reset_prob=0.):
        import genesis as gs
        gs.init(backend=gs.gpu,seed=seed,logging_level='warning')
        self.n=n_envs;self.device=gs.device;self.height=height;self.center=center;self.thickness=thickness
        self.reward_version=reward_version;self.goal_reset_prob=goal_reset_prob
        self.row=bx._pose({'r':(0,.025),'l':(0,.025)},-.035,body_height)
        self.scene=gs.Scene(show_viewer=False,show_FPS=False,
            sim_options=gs.options.SimOptions(dt=.005,substeps=10),
            rigid_options=gs.options.RigidOptions(dt=.0005))
        self.floor=self.scene.add_entity(morph=gs.morphs.Plane(),material=gs.materials.Rigid(friction=friction))
        self.box=self.scene.add_entity(morph=gs.morphs.Box(pos=(center,0,height/2),size=(thickness,.5,height),fixed=True),
            material=gs.materials.Rigid(friction=friction))
        self.robot=self.scene.add_entity(morph=gs.morphs.URDF(file=str(be.DEFAULT_URDF),default_armature=0.),
            material=gs.materials.Rigid(friction=friction))
        self.scene.build(n_envs=n_envs)
        self.robot.set_dofs_armature(np.full(10,.00182),dofs_idx_local=np.arange(6,16))
        self.robot.set_dofs_kp(np.zeros(22));self.robot.set_dofs_kv(np.zeros(22))
        limits=np.r_[np.zeros(6),np.full(10,10.5),np.ones(6)]
        self.robot.set_dofs_force_range(-limits,limits)
        self.foot_ids=[next(i for i,l in enumerate(self.robot.links) if l.name==name) for name in be.ANKLE_LINKS]
        self.joint_ids=np.arange(6,22)
        self.nominal=torch.tensor(self.row[6:],device=self.device,dtype=torch.float32)
        self.initial=torch.tensor(np.r_[self.row[:3],[1.,0,0,0],self.row[6:]],device=self.device,dtype=torch.float32)
        names=[name for name,slot in sorted(bx._ROW_SLOT.items(),key=lambda item:item[1])]
        xml=ET.parse(be.DEFAULT_URDF).getroot();byname={j.get('name'):j for j in xml.findall('joint')}
        self.lower=torch.tensor([float(byname[n].find('limit').get('lower')) for n in names],device=self.device)
        self.upper=torch.tensor([float(byname[n].find('limit').get('upper')) for n in names],device=self.device)
        self.scales=torch.tensor([.15,.65,.35,.8,.55]*2,device=self.device)
        self.kp=torch.tensor([50.]*10+[15.]*6,device=self.device)
        self.kd=torch.tensor([1.5]*10+[.6]*6,device=self.device)
        self.motor_r=torch.tensor(BEAR_TO_JOINT_RIGHT,device=self.device,dtype=torch.float32)
        self.motor_l=torch.tensor(BEAR_TO_JOINT_LEFT,device=self.device,dtype=torch.float32)
        state=dict(base_pos=self.row[:3],base_quat=np.array([1.,0,0,0]),base_vel=np.zeros(3),base_ang=np.zeros(3),
            dof_pos=np.r_[self.row[:6],self.row[6:]],dof_vel=np.zeros(22))
        c=WholeBodyController(self.row);c.command(state,self.row)
        self.ff=torch.tensor(c.full_command,device=self.device,dtype=torch.float32)
        self.action=torch.zeros((n_envs,10),device=self.device)
        self.elapsed=torch.zeros(n_envs,device=self.device,dtype=torch.int32)
        self.hold=torch.zeros(n_envs,device=self.device,dtype=torch.int32)
        self.hit=torch.zeros(n_envs,device=self.device,dtype=torch.bool)
        self.full_start=torch.ones(n_envs,device=self.device,dtype=torch.bool)
        self.ep_reward=torch.zeros(n_envs,device=self.device)
        self.total_episodes=0;self.total_screen_success=0
        self.reset(torch.arange(n_envs,device=self.device));self.observe()
        self.obs_dim=self.obs.shape[1]

    def reset(self,ids):
        if not len(ids):return
        q=self.initial.expand(len(ids),-1).clone()
        # Small starts only; no mid-episode perturbation or assist.
        q[:,7:17]+=.008*torch.randn((len(ids),10),device=self.device)
        near_goal=torch.rand(len(ids),device=self.device)<self.goal_reset_prob
        q[:,0]+=near_goal*(self.center+self.thickness/2+.1-float(self.row[0]))
        self.full_start[ids]=~near_goal
        self.robot.set_qpos(q,envs_idx=ids,zero_velocity=True)
        self.action[ids]=0.;self.elapsed[ids]=0;self.hold[ids]=0;self.hit[ids]=False;self.ep_reward[ids]=0.

    def observe(self):
        qpos=self.robot.get_qpos();q=self.robot.get_dofs_position()[:,6:];dq=self.robot.get_dofs_velocity()[:,6:]
        pos=qpos[:,:3];quat=qpos[:,3:7]
        vel=self.robot.get_links_vel(links_idx_local=[0],relative=False)[:,0]
        ang=self.robot.get_links_ang(links_idx_local=[0])[:,0]
        foot=self.robot.get_links_pos(links_idx_local=self.foot_ids,relative=False)
        fv=self.robot.get_links_vel(links_idx_local=self.foot_ids,relative=False)
        force=self.robot.get_links_net_contact_force()[:,self.foot_ids]
        contact=force[:,:,2]>3.
        phase=self.elapsed.float()*self.dt*2*torch.pi*1.5
        goal=torch.stack((self.center-pos[:,0],torch.full_like(phase,self.height),self.center+self.thickness/2+.1-pos[:,0]),dim=1)
        self.obs=torch.cat((quat,pos[:,2:3],inverse_rotate(quat,vel),inverse_rotate(quat,ang)*.25,
            q-self.nominal,dq*.1,(foot-pos[:,None,:]).flatten(1),fv.flatten(1)*.5,contact.float(),
            goal,self.action,torch.sin(phase)[:,None],torch.cos(phase)[:,None]),dim=1).clamp(-20,20)
        self.state=(pos,quat,vel,ang,q,dq,foot,fv,contact)
        return self.obs

    @torch.no_grad()
    def step(self,action):
        old_pos=self.state[0][:,0].clone();old_min=self.state[6][:,:,0].min(1).values.clone()
        previous=self.action.clone();self.action=.5*self.action+.5*action.clamp(-2.,2.)
        target=self.nominal.expand(self.n,-1).clone();target[:,:10]+=self.action*self.scales
        target=torch.minimum(torch.maximum(target,self.lower+.01),self.upper-.01)
        contacts_now=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        effort=torch.zeros(self.n,device=self.device)
        for _ in range(4):
            q=self.robot.get_dofs_position()[:,6:];dq=self.robot.get_dofs_velocity()[:,6:]
            tau=self.ff+self.kp*(target-q)-self.kd*dq
            legs=tau[:,:10].clamp(-10.5,10.5)
            motors=torch.cat((legs[:,:5]@self.motor_r,legs[:,5:]@self.motor_l),dim=1)
            scale=(10.46/(motors.abs().amax(1,keepdim=True)+1e-6)).clamp(max=1.)
            tau=torch.cat((legs*scale,tau[:,10:].clamp(-1.,1.)),dim=1)
            self.robot.control_dofs_force(tau,dofs_idx_local=self.joint_ids)
            self.scene.step()
            c=self.robot.get_contacts(with_entity=self.box,is_padded=True)
            contacts_now|=c['valid_mask'].any(1)
            effort+=tau.square().sum(1)*.25
        self.hit|=contacts_now;self.elapsed+=1;self.observe()
        pos,quat,vel,ang,q,dq,foot,fv,contact=self.state
        up=1-2*(quat[:,1].square()+quat[:,2].square())
        foot_clear=foot[:,:,0].min(1).values>self.center+self.thickness/2+.0435
        goal_reached=foot_clear&(pos[:,0]>self.center+self.thickness/2+.065)
        desired=torch.where(goal_reached,0.,.18)
        if self.reward_version>=2:desired=((self.center+self.thickness/2+.1-pos[:,0])*2).clamp(-.1,.20)
        good=goal_reached&(up>.98)&(vel.norm(dim=1)<.10)&contact.all(1)&(~self.hit)
        self.hold=torch.where(good,self.hold+1,torch.zeros_like(self.hold))
        terminal_success=self.hold>=50
        screen_success=terminal_success&self.full_start
        fell=(pos[:,2]<.23)|(up<.65)|(pos[:,1].abs()>.32)|(pos[:,0]<-.25)|(pos[:,0]>1.)|(foot[:,:,1].abs().amax(1)>.24)
        timeout=self.elapsed>=300
        reward=10*(pos[:,0]-old_pos).clamp(-.025,.025)+5*(foot[:,:,0].min(1).values-old_min).clamp(-.03,.03)
        if self.reward_version>=2:
            goal_x=self.center+self.thickness/2+.1
            reward=8*((old_pos-goal_x).abs()-(pos[:,0]-goal_x).abs()).clamp(-.03,.03)
            reward+=4*(foot[:,:,0].min(1).values.clamp(max=goal_x+.035)-old_min.clamp(max=goal_x+.035)).clamp(-.03,.03)
            clean_goal=goal_reached&(~self.hit)
            reward+=clean_goal.float()*(.06*torch.exp(-((pos[:,0]-goal_x)/.04).square())+.10*torch.exp(-vel.square().sum(1)/.025))
            reward-=.02*(pos[:,0]-goal_x).abs()
            fell|=pos[:,0]>goal_x+.35
        reward+=.025+.025*up+.04*torch.exp(-((vel[:,0]-desired)/.25).square())
        reward+=.02*torch.exp(-((pos[:,2]-float(self.row[2]))/.08).square())
        reward-=.01*vel[:,1].square()+.005*ang.square().sum(1)
        reward-=.003*(self.action-previous).square().sum(1)+.00005*effort
        reward-=.01*(fv[:,:,:2].square().sum(2)*contact).sum(1)
        reward-=.25*contacts_now.float()+3*fell.float()
        reward+=(20 if self.reward_version>=2 else 80)*terminal_success.float()
        self.ep_reward+=reward;done=fell|timeout|terminal_success
        ids=done.nonzero().flatten()
        info=dict(episodes=int(len(ids)),screen_successes=int(screen_success.sum().item()),
            curriculum_successes=int((terminal_success&~self.full_start).sum().item()),
            reward_sum=float(self.ep_reward[ids].sum().item()) if len(ids) else 0.,
            duration_sum=float(self.elapsed[ids].sum().item())*self.dt if len(ids) else 0.,
            max_x=float(pos[:,0].max().item()),mean_x=float(pos[:,0].mean().item()),
            timeouts=int((timeout&~fell).sum().item()))
        self.total_episodes+=info['episodes'];self.total_screen_success+=info['screen_successes']
        self.reset(ids)
        if len(ids):self.observe()
        return self.obs,reward,done,info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--envs',type=int,default=128);p.add_argument('--steps',type=int,default=100)
    args=p.parse_args();torch.set_num_threads(1);env=CrossingEnv(args.envs)
    start=time.perf_counter()
    for i in range(args.steps):
        _,r,d,info=env.step(torch.zeros((args.envs,10),device=env.device))
        if i%20==0:print(i,float(r.mean()),info,flush=True)
    print(json.dumps(dict(envs=args.envs,steps=args.steps,seconds=time.perf_counter()-start,obs_dim=env.obs_dim,
        screen_successes=env.total_screen_success,qualified_successes=0)),flush=True)
