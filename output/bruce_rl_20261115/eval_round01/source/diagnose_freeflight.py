#!/usr/bin/env python3
"""Isolate model agreement in free fall, away from contacts and joint stops."""
import argparse
import json
from pathlib import Path
import numpy as np
import pinocchio as pin

import bruce_env as be
import bruce_expert as bx
from flight_wbc import FlightController


def run(args):
    row=bx._pose({'r':(0,.025),'l':(0,.025)},-.03,.43)
    scene=be.BruceScene(backend='gpu');scene.set_gains(np.zeros(22),np.zeros(22))
    scene.reset_robot(1.5,row[6:],row[0])
    fc=FlightController(row)
    for _ in range(10):scene.step()
    state=scene.read_state();q=fc.q0.copy();q[:3]=state['base_pos'];q[3:7]=np.r_[state['base_quat'][1:],state['base_quat'][0]]
    for iq,iv,slot in fc.mapping:q[iq]=state['dof_pos'][6+slot]
    pin.computeAllTerms(fc.model,fc.data,q,np.zeros(fc.model.nv))
    mp=np.triu(fc.data.M)+np.triu(fc.data.M,1).T
    permutation=np.zeros((22,22));permutation[:6,:6]=np.eye(6)
    for _,iv,slot in fc.mapping:
        permutation[iv,6+slot]=1.
        if slot<10:mp[iv,iv]+=.00182
    mp=permutation.T@mp@permutation
    mg=be.to_np(scene.robot.get_mass_mat())
    def schur(M):return M[6:,6:]-M[6:,:6]@np.linalg.solve(M[:6,:6],M[:6,6:])
    com_gen=be.to_np(scene.robot.get_links_pos(ref=scene.gs.link_ref_frame.link_COM))
    masses=be.to_np(scene.robot.get_links_mass())
    pin.centerOfMass(fc.model,fc.data,q)
    target=bx._pose({'r':(0,.025),'l':(0,.025)},-.03,.3)[6:]
    command=fc.command_flight(state,target)
    predicted=np.array(fc.debug['predicted_body_angular_acc'])
    scene.robot.control_dofs_force(command,dofs_idx_local=np.arange(6,22))
    scene.step();after=scene.read_state()
    actual=(after['base_ang']-state['base_ang'])/.0005
    report=dict(schur_max_abs_error=float(np.abs(schur(mp)-schur(mg)).max()),
        mass_matrix_max_abs_error=float(np.abs(mp-mg).max()),
        engine_com=np.average(com_gen,weights=masses,axis=0).tolist(),
        pin_com=fc.data.com[0].tolist(),
        predicted_body_angular_acc=predicted.tolist(),actual_body_angular_acc=actual.tolist(),
        command=command.tolist(),pin_schur=schur(mp).tolist(),engine_schur=schur(mg).tolist())
    args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if 'schur' not in k or k=='schur_max_abs_error'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',type=Path,required=True)
    run(p.parse_args())
