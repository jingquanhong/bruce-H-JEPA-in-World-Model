"""Compare actual DCM QPs at recorded step starts; no live memory writes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast, inspect, sys, json, time
from pathlib import Path
sys.path.insert(0,'/home/khadas/BRUCE/BRUCE-OP')
import numpy as np
from scipy import sparse
from Util import crossstep_high_level as m
from Util.crossstep_targets import landing_targets, add_target_objective

tree=ast.parse(inspect.getsource(m.main_loop));fn=tree.body[0]
idx=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='confirm' for t in n.targets))
fn.name='private_setup';fn.body=fn.body[:idx]+[ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))]
ast.fix_missing_locations(tree);namespace=dict(m.__dict__)
exec(compile(tree,'<private-DCM-setup>','exec'),namespace)
d=namespace['private_setup']()
root=Path(__file__).resolve().parent
data=json.loads((root/'lateral-right-235505/observations.json').read_text())
summary=json.loads((root/'lateral-right-235505/summary.json').read_text())
starts=[];previous=0
for r in data:
    if r['monotonic']<summary['request_sent_monotonic']:continue
    phase=r['planned_phase']
    if phase in (1,2) and phase!=previous:starts.append(r)
    previous=phase

rows=[]
for record in starts[:8]:
    phase=int(record['planned_phase']);leg=1 if phase==1 else -1
    rotation=np.array(record['body_rotation']);f=rotation[:2,0];f/=np.linalg.norm(f)
    Rt=np.column_stack([f,[-f[1],f[0]]])
    right=np.array(record['estimated_right_foot_position'])[:2]
    left=np.array(record['estimated_left_foot_position'])[:2]
    support=right if leg==1 else left;swing=left if leg==1 else right
    direction=-1
    try:targets=landing_targets(swing,support,Rt,direction)
    except ValueError as e:
        rows.append(dict(phase=phase,rejected=str(e)));continue
    Lox,Woy,box,boy1,boy2=m.get_DCM_nom(np.zeros(3),np.full(3,-.05),np.full(3,.119),m.Ts,d['eTs'])
    box-=.002;boy1-=.006
    q=m.cost_update(d['q0'],d['W_L'],d['W_b'],d['w_s'],leg,Rt,support,d['eTs'],.119,3,Lox,Woy,box,boy1,boy2)
    b=np.array(record['estimated_com_position'])[:2]+np.array(record['estimated_com_velocity'])[:2]/d['omg']-support
    lower_r=np.array([-.2,.1]);upper_r=np.array([.2,.2]);lower_l=np.array([-.2,-.2]);upper_l=np.array([.2,-.1])
    AA,ll,uu=m.constraint_update(d['A0'],d['l0'],d['u0'],leg,Rt,0.,support,b,d['omg'],d['eTs'],3,
        (lower_r+lower_l)/2,(lower_r-lower_l)/2,(upper_r+upper_l)/2,(upper_r-upper_l)/2)
    results={}
    for label in ('baseline','explicit'):
        P,qq=(d['P0'],q) if label=='baseline' else add_target_objective(d['P0'],q,targets)
        qp=m.osqp.OSQP();qp.setup(P=sparse.triu(sparse.csc_matrix(P),format='csc'),q=qq,A=sparse.csc_matrix(AA),l=ll,u=uu,
            verbose=False,eps_abs=1e-4,eps_rel=1e-4,max_iter=1000,check_termination=1,adaptive_rho_interval=50,scaling=10)
        t=time.perf_counter();sol=qp.solve();elapsed=time.perf_counter()-t
        assert sol.info.status=='solved',sol.info.status
        results[label]=dict(lateral_landing_change_m=float(Rt[:,1]@(sol.x[:2]-swing)),
            target_error_m=float(np.linalg.norm(sol.x[:2]-targets[0])),solve_elapsed_s=elapsed)
    rows.append(dict(phase=phase,elapsed_s=record['monotonic']-summary['request_sent_monotonic'],
        target_lateral_change_m=-.03,solutions=results))
(root/'crossstep-recorded-qp-results.json').write_text(json.dumps(dict(cases=rows,
    note='Foot-center XY approximates ankle XY; replay is not a dynamics simulation or external displacement measurement.'),indent=2))
print(json.dumps(rows,indent=2))
