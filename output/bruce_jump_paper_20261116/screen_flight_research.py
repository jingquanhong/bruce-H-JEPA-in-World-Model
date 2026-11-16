"""Exploratory free-flight replay of a model-only push; no robot IO.

No impact/contact solver or real actuator dynamics. This cannot validate a
physical jump; synthetic PD gains here are numerical examples only.
"""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from current_model import build
from integrate_push_research import run
from jump_core import flight_pd

HERE=Path(__file__).resolve().parent


def main():
    model,description=build(HERE/'model-export-corrected.json')
    current_map=np.asarray(json.loads((HERE/'khadas-model-screen.json').read_text())['current_map_A_per_Nm'])
    push=run(6.,.10288333333333333,.06,.001)
    if push['status']!='INTEGRATED_MODEL_ONLY':raise RuntimeError('Push integration rejected')
    start=push['samples'][0];end=push['samples'][-1]
    q0=np.asarray(start['model_configuration']);q_release=np.asarray(end['model_configuration']);v_release=np.asarray(end['model_velocity'])
    frames=[model.getFrameId(n) for n in ('right_toe','right_heel','left_toe','left_heel')]
    initial=model.createData();pin.framesForwardKinematics(model,initial,q0)
    ground=np.array([initial.oMf[f].translation[2] for f in frames])
    cases=[]
    for name,kp,kd in [('zero_joint_torque',0.,0.),('hypothetical_joint_hold',5.,.1)]:
        q=q_release.copy();v=v_release.copy();q_target=q[7:].copy();rows=[];failure=None
        for k in range(201):
            t=k*.001;data=model.createData();pin.framesForwardKinematics(model,data,q)
            heights=np.array([data.oMf[f].translation[2] for f in frames])-ground
            pin.centerOfMass(model,data,q,v);comz=float(data.com[0][2]);comvz=float(data.vcom[0][2])
            pd=flight_pd(q[7:],v[6:],q_target,np.zeros(10),np.full(10,kp),np.full(10,kd),
                         np.full(10,description['torque_max']),current_map,np.full(10,description['current_max']))
            rows.append({'time_s':t,'foot_heights_m':heights.tolist(),'com_z_m':comz,'com_vz_m_s':comvz,
                         'max_joint_torque_Nm':float(abs(pd['torque']).max()),'torque_scale':pd['scale']})
            if heights.min()<-.002:
                failure={'time_s':t,'reason':'foot penetrated original floor plane by over 2 mm in no-contact model'};break
            if t>.01 and comvz<0 and heights.min()<=.002:
                failure={'time_s':t,'reason':'foot returned near floor; landing impact is not modeled'};break
            tau=np.r_[np.zeros(6),pd['torque']]
            acceleration=pin.aba(model,data,q,v,tau)
            v_next=v+acceleration*.001;q=pin.integrate(model,q,v_next*.001);v=v_next
        cases.append({'name':name,'hypothetical_pd_joint_kp_Nm_per_rad':kp,'hypothetical_pd_kd_Nm_s_per_rad':kd,
                      'failure':failure,'samples':rows})
    result={'status':'FREE_FLIGHT_SCREEN_ONLY','hardware_actuation':False,'landing_simulated':False,
            'physical_jump_validated':False,'cases':cases,
            'limitations':['Synthetic joint torques do not model the actual actuator/controller',
                           'No contact, collision or impact response after liftoff',
                           'Assumed model mass/inertia differ from unmeasured physical robot',
                           'Foot plane crossing and nominal CoM rise are not physical jumping evidence']}
    (HERE/'flight-research-screen.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([{'name':c['name'],'failure':c['failure'],'samples':len(c['samples']),
                       'max_min_foot_height_m':max(min(r['foot_heights_m']) for r in c['samples']),
                       'max_com_z_m':max(r['com_z_m'] for r in c['samples'])} for c in cases],indent=2))


if __name__=='__main__':main()
