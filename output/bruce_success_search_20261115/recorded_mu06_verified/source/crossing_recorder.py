"""Aligned 50 Hz RGB/action records plus a separate 2 kHz physics trace.

Every output remains experimental until an independent success qualification.
Reset transients in the first 20 ms are excluded from the learning rows.
"""
import hashlib
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

import bruce_env as be


class CrossingRecorder:
    def __init__(self,out,index,scene,box):
        self.stem=Path(out)/f'attempt_{index:03d}'
        self.scene=scene;self.box=box;self.pending=None
        self.columns={k:[] for k in ('timestamp','pixels','proprio','action','sim_base_pos',
            'sim_base_quat','sim_control_force','sim_box_pos','sim_foot_force','sim_foot_pos',
            'sim_joint_pos','sim_joint_vel')}
        self.trace={k:[] for k in ('timestamp','link_pos','link_quat','box_pos','box_quat',
            'control_force','robot_box_contact_count')}
        self.writer=imageio.get_writer(str(self.stem.with_suffix('.mp4')),fps=25,codec='libx264',quality=7)
        self.start_time=None

    def observation(self,t,state,phase):
        if t<.02-1e-10:return
        if self.pending is not None:raise RuntimeError('unfinished 20 ms action block')
        rgb=self.scene.camera.render()[0]
        R=be.quat_to_mat(state['base_quat'])
        proprio=np.r_[state['dof_pos'][6:16],state['dof_vel'][6:16],
            R.T@(state['base_acc']+[0,0,9.81]),R.T@state['base_ang'],state['dof_force'][6:16]]
        if self.start_time is None:self.start_time=t
        self.pending=dict(timestamp=t-self.start_time,
            pixels=np.asarray(Image.fromarray(rgb).resize((64,64))),proprio=proprio,
            sim_base_pos=state['base_pos'],sim_base_quat=state['base_quat'],
            sim_box_pos=self.scene.box_pos(self.box),sim_foot_force=state['foot_force'],
            sim_foot_pos=np.array([self.scene.get_link_pos(n) for n in be.ANKLE_LINKS]),
            sim_joint_pos=state['dof_pos'][6:],sim_joint_vel=state['dof_vel'][6:])
        self.commands=[];self.forces=[]
        if len(self.columns['timestamp'])%2==0:
            im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,43),fill='#172332')
            draw.text((8,5),'OBSTACLE ATTEMPT - NOT A QUALIFIED SUCCESS',fill='white')
            draw.text((8,23),f't={t:.2f}s | {phase} | free base, physical mesh feet',fill='white')
            self.writer.append_data(np.asarray(im))

    def action(self,command,actual):
        if self.pending is None:return
        self.commands.append(np.asarray(command).copy());self.forces.append(actual.copy())
        if len(self.commands)==4:
            self.pending['action']=np.asarray(self.commands).reshape(64)
            self.pending['sim_control_force']=np.asarray(self.forces)
            for key,value in self.pending.items():self.columns[key].append(value)
            self.pending=None

    def physics_step(self,t,actual,contact_count):
        scene=self.scene
        values=dict(timestamp=t,link_pos=be.to_np(scene.robot.get_links_pos(relative=False)),
            link_quat=be.to_np(scene.robot.get_links_quat(relative=False)),
            box_pos=scene.box_pos(self.box),box_quat=be.to_np(self.box.get_quat()),
            control_force=actual,robot_box_contact_count=contact_count)
        for key,value in values.items():self.trace[key].append(np.asarray(value).copy())

    def finish(self,result):
        self.writer.close()
        trace_path=self.stem.with_suffix('.physics.npz')
        np.savez_compressed(trace_path,**{k:np.asarray(v) for k,v in self.trace.items()})
        n=len(self.columns['timestamp'])
        urdf=Path(self.scene.urdf)
        meta={k:v for k,v in result.items() if k!='history'}
        meta.update(trial='crossing_attempt',recorded_frames=n,record_hz=50,control_hz=200,physics_hz=2000,
            warmup_seconds=.02,autonomous_crossing_success=False,visual_hjepa_ready=False,
            control_mode='joint_only',base_actuated=False,urdf=str(urdf),
            urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
            action_kind='four_consecutive_16_joint_torque_commands_Nm',
            action_order='right_leg_5,left_leg_5,right_arm_3,left_arm_3',
            action_timing='observation precedes four 5 ms commands; final incomplete block excluded',
            proprio_kind='q10,dq10,body_specific_force3,body_gyro3,executed_leg_torque10',
            pixels_kind='64x64 RGB fixed external camera; simulator truth, not onboard vision',
            physics_trace=trace_path.name,
            physics_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            link_names=self.scene.link_names,leg_armature_kg_m2=.00182,
            purpose='experimental_obstacle_attempt_not_success_demonstration')
        h5path=self.stem.with_suffix('.h5')
        if n:
            with h5py.File(h5path,'x') as f:
                for key,values in self.columns.items():f.create_dataset(key,data=np.asarray(values),compression='gzip')
                for key,values in dict(ep_len=[n],ep_offset=[0],ep_idx=np.zeros(n,dtype=int),step_idx=np.arange(n)).items():
                    f.create_dataset(key,data=values)
                f.attrs['metadata']=json.dumps(meta,ensure_ascii=False)
                f.attrs['purpose']=meta['purpose']
        self.stem.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        return dict(hdf5=h5path.name if n else None,video=self.stem.with_suffix('.mp4').name,
            physics_trace=trace_path.name,recorded_frames=n)
