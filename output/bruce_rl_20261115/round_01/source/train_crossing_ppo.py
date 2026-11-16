#!/usr/bin/env python3
"""Train a torque-limited crossing policy. PPO screens never certify data."""
import argparse,hashlib,json,random,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from rl_crossing_env import CrossingEnv


class ActorCritic(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        def mlp(out):return nn.Sequential(nn.Linear(obs_dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,out))
        self.actor=mlp(10);self.critic=mlp(1);self.log_std=nn.Parameter(torch.full((10,),-.7))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)

    def distribution(self,obs):return Normal(self.actor(obs),self.log_std.clamp(-2.5,.5).exp())
    def value(self,obs):return self.critic(obs).squeeze(-1)


def train(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'progress.jsonl').exists():raise FileExistsError('Use a new output directory')
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    source=out/'source';source.mkdir()
    for name in ('train_crossing_ppo.py','rl_crossing_env.py','bruce_env.py','bruce_expert.py','qualification.py','standing_wbc.py','support_geometry.py'):
        shutil.copy2(Path(__file__).parent/name,source/name)
    env=CrossingEnv(args.envs,args.seed,args.height,args.center,args.thickness,args.friction,args.body_height)
    model=ActorCritic(env.obs_dim).to(env.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5)
    if args.resume:
        data=torch.load(args.resume,map_location=env.device,weights_only=False)
        model.load_state_dict(data['model'])
        if args.resume_optimizer:opt.load_state_dict(data['optimizer'])
    T,N=args.rollout,args.envs;obs=env.obs.clone()
    ob=torch.zeros((T,N,env.obs_dim),device=env.device);ac=torch.zeros((T,N,10),device=env.device)
    lp=torch.zeros((T,N),device=env.device);val=torch.zeros_like(lp);rew=torch.zeros_like(lp);done=torch.zeros_like(lp)
    started=time.perf_counter();total_steps=0;best_reward=-float('inf');best_success=-1
    metrics=[]
    def save(name,iteration):
        target=out/name;temp=target.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),iteration=iteration,
            environment_steps=total_steps,config=cfg,obs_dim=env.obs_dim,
            screened_successes=env.total_screen_success,qualified_successes=0,
            torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state()),temp)
        temp.replace(target)
    try:
      for iteration in range(args.iterations):
        stats=dict(episodes=0,screen_successes=0,reward_sum=0.,duration_sum=0.,timeouts=0,max_x=-1.)
        with torch.no_grad():
            for t in range(T):
                ob[t]=obs;dist=model.distribution(obs);action=dist.sample()
                ac[t]=action;lp[t]=dist.log_prob(action).sum(-1);val[t]=model.value(obs)
                obs,reward,terminal,info=env.step(action)
                rew[t]=reward;done[t]=terminal.float()
                for key in ('episodes','screen_successes','reward_sum','duration_sum','timeouts'):stats[key]+=info[key]
                stats['max_x']=max(stats['max_x'],info['max_x'])
            next_value=model.value(obs);gae=torch.zeros(N,device=env.device);advantages=torch.zeros_like(rew)
            for t in reversed(range(T)):
                next_v=next_value if t==T-1 else val[t+1]
                mask=1.-done[t];delta=rew[t]+args.gamma*mask*next_v-val[t]
                gae=delta+args.gamma*.95*mask*gae;advantages[t]=gae
            returns=advantages+val
        bobs=ob.flatten(0,1);baction=ac.flatten(0,1);blogp=lp.flatten();badv=advantages.flatten();bret=returns.flatten()
        total_steps+=T*N
        if stats['screen_successes']>0:save(f'candidate_rollout_{iteration+1:05d}.pt',iteration+1)
        badv=(badv-badv.mean())/(badv.std()+1e-8)
        kl=loss_value=entropy=0.;updates=0
        for epoch in range(args.epochs):
            permutation=torch.randperm(T*N,device=env.device)
            for indices in permutation.chunk(args.minibatches):
                dist=model.distribution(bobs[indices]);newlp=dist.log_prob(baction[indices]).sum(-1)
                ratio=(newlp-blogp[indices]).exp()
                policy_loss=-torch.minimum(ratio*badv[indices],ratio.clamp(.8,1.2)*badv[indices]).mean()
                value_loss=(model.value(bobs[indices])-bret[indices]).square().mean()
                ent=dist.entropy().sum(-1).mean()
                loss=policy_loss+.5*value_loss-args.entropy*ent
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                with torch.no_grad():kl+=float((blogp[indices]-newlp).mean());loss_value+=float(value_loss);entropy+=float(ent)
                updates+=1
            if kl/max(updates,1)>.04:break
        n=max(1,stats['episodes']);mean_reward=stats['reward_sum']/n
        record=dict(iteration=iteration+1,environment_steps=total_steps,seconds=time.perf_counter()-started,
            rollout_reward=float(rew.mean()),episode_reward=mean_reward,episode_seconds=stats['duration_sum']/n,
            **stats,screen_successes_total=env.total_screen_success,qualified_successes=0,
            entropy=entropy/max(updates,1),value_loss=loss_value/max(updates,1),kl=kl/max(updates,1))
        metrics.append(record)
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if iteration%5==0:print(json.dumps(record),flush=True)
        if stats['episodes'] and mean_reward>best_reward:
            best_reward=mean_reward;save('best_reward.pt',iteration+1)
        if stats['screen_successes']>best_success:
            best_success=stats['screen_successes'];save('best_screen.pt',iteration+1)
        if (iteration+1)%20==0 or stats['screen_successes']>0:save(f'checkpoint_{iteration+1:05d}.pt',iteration+1)
        if (iteration+1)%5==0:save('latest.pt',iteration+1)
    finally:
        save('latest.pt',len(metrics))
        (out/'summary.json').write_text(json.dumps(dict(iterations=len(metrics),environment_steps=total_steps,
            episodes=env.total_episodes,screened_successes=env.total_screen_success,qualified_successes=0,
            elapsed_seconds=time.perf_counter()-started,config=cfg),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--envs',type=int,default=256);p.add_argument('--iterations',type=int,default=400)
    p.add_argument('--rollout',type=int,default=32);p.add_argument('--epochs',type=int,default=5);p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--entropy',type=float,default=.003);p.add_argument('--gamma',type=float,default=.995)
    p.add_argument('--height',type=float,default=.01);p.add_argument('--center',type=float,default=.12);p.add_argument('--thickness',type=float,default=.02)
    p.add_argument('--friction',type=float,default=.8);p.add_argument('--body-height',type=float,default=.4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--resume',type=Path);p.add_argument('--resume-optimizer',action='store_true')
    train(p.parse_args())
