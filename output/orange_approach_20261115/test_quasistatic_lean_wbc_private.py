"""Run BRUCE's actual QP with recorded state and private command memory only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ['BRUCE_SUPPORTED_TRANSLATION_TRACKING']='1'
os.environ['BRUCE_STATIONARY_ATTITUDE_SCALE']='.5'
os.environ['BRUCE_STATIONARY_HIP_YAW_HOLD']='1'
import sys,json,time,hashlib,contextlib
from pathlib import Path
sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
import numpy as np
import Play.Walking.low_level as low
import Settings.BRUCE_data as rds
import Util.translation_feedback as feedback
from Util.control_preflight import warm_low_level
from Util import quasistatic_translation_feedback as candidate

fixture=json.loads(Path('/home/khadas/BRUCE/diagnostics/control-warmup-fixture.json').read_text())
original_update=rds.BRUCE.update_plan_status
original_cost=low.cost_update
original_feedback=feedback.translation_feedback
original_weight=feedback.translation_weight
reports=[]
try:
    for profile in ('legacy','quasistatic'):
        if profile=='quasistatic':candidate.install()
        for phase in (0,1,2):
            for sign in (-1.,1.):
                diagnostics=[];commands=[]
                def update(robot):
                    original_update(robot)
                    robot.mode=0 if phase==0 else 1;robot.phase=phase
                    robot.des_p_wg=robot.p_wg+robot.R_yaw@np.array([0.,sign*.02,0.])
                    robot.des_v_wg=np.zeros(3)
                    robot.des_R_wb=(robot.R_yaw @ low.MF.Rx(.034906585*(1 if phase==1 else -1))) if phase in (1,2) else robot.R_wb.copy();robot.des_w_bb=np.zeros(3)
                    for leg in ('r','l'):
                        setattr(robot,'des_p_wf_'+leg,getattr(robot,'p_wa_'+leg).copy())
                        if (leg=='l' and phase==1) or (leg=='r' and phase==2):
                            setattr(robot,'des_p_wf_'+leg,getattr(robot,'p_wa_'+leg)+robot.R_yaw@np.array([0.,sign*.04,.015]))
                        setattr(robot,'des_v_wf_'+leg,np.zeros(3))
                        setattr(robot,'des_R_wf_'+leg,getattr(robot,'R_wf_'+leg).copy())
                        setattr(robot,'des_w_ff_'+leg,np.zeros(3))
                def cost(*args):
                    target_accel=args[10]@(args[13]-args[12])+args[11]@(args[15]-args[14])
                    diagnostics.append(dict(target_acceleration=target_accel.tolist(),
                        kp=np.diag(args[10]).tolist(),kd=np.diag(args[11]).tolist(),
                        weights=np.diag(args[9]).tolist()))
                    P,q=original_cost(*args)
                    diagnostics[-1]['qp_cost_sha256']=hashlib.sha256(P.tobytes()+q.tobytes()).hexdigest()
                    return P,q
                rds.BRUCE.update_plan_status=update;low.cost_update=cost
                started=time.monotonic()
                with (Path(__file__).resolve().parent/'quasistatic-lean-private-wbc.log').open('a') as log:
                    with contextlib.redirect_stdout(log):
                        count=warm_low_level(low,fixture,capture=commands)
                assert count==3 and len(diagnostics)>=3
                torques=np.asarray([r['goal_torques'] for r in commands])
                assert np.isfinite(torques).all()
                if phase in (1,2) and profile=='quasistatic':
                    assert sign*diagnostics[-1]['target_acceleration'][1]>0
                reports.append(dict(profile=profile,phase=phase,sign=sign,
                    private_commands=count,max_abs_torque_Nm=float(np.max(np.abs(torques))),
                    last_torque_Nm=torques[-1].tolist(),last_objective=diagnostics[-1],
                    elapsed_s=time.monotonic()-started))
                print('PRIVATE_QP_CASE',profile,phase,sign,flush=True)
    # Warm-started approximate QP solutions can differ with iteration count.
    # Compare the actual standing objective, rather than requiring bit-identical
    # torques from solvers stopped at different times after the 0.5 s warmup.
    Path(__file__).with_name('quasistatic-lean-private-wbc-computed.json').write_text(json.dumps(reports,indent=2))
    for phase in (0,):
        for sign in (-1.,1.):
            old=next(r for r in reports if r['profile']=='legacy' and r['phase']==phase and r['sign']==sign)
            new=next(r for r in reports if r['profile']=='quasistatic' and r['phase']==phase and r['sign']==sign)
            assert old['last_objective']['qp_cost_sha256']==new['last_objective']['qp_cost_sha256']
finally:
    rds.BRUCE.update_plan_status=original_update;low.cost_update=original_cost
    feedback.translation_feedback=original_feedback;feedback.translation_weight=original_weight

destination=Path(__file__).resolve().parent/'quasistatic-lean-private-wbc-results.json'
destination.write_text(json.dumps(dict(
    result='PASS_PRIVATE_QP_ONLY',hardware_actuation=False,
    note='Recorded-state QP computations; not a dynamics simulation or real motion validation.',
    cases=reports),indent=2))
print('PASS 12 recorded-state/private-memory QP cases; standing objective unchanged; no motor interface used.')
print(destination)
