"""Search a small idealized flight/landing grid without robot IO.

Preserve the best numerical artifacts only; this does not validate hardware.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent


def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    results=[];best=None
    with tempfile.TemporaryDirectory() as temp:
        f=Path(temp)/'flight.json';i=Path(temp)/'impact.json'
        for weight in [.0015,.002,.0025,.003,.004,.005,.006]:
            for down in [.05,.10,.15,.20]:
                cmd=[sys.executable,str(HERE/'flight_retraction_research.py'),'--push-accel','3',
                     '--push-time','.12147','--taper','.04','--depth','.03','--flight-period','.0404',
                     '--heights','.002','--landing-speed',str(down),'--body-weight',str(weight),'--out',str(f)]
                subprocess.run(cmd,env=env,stdout=subprocess.DEVNULL,check=True)
                flight=json.loads(f.read_text());case=flight['cases'][0];samples=case['samples']
                ending=case['ending'];item={'body_weight':weight,'nominal_foot_downward_speed_m_s':down,
                    'ending':ending,'max_body_tilt_deg':max(x['body_tilt_deg'] for x in samples),
                    'max_min_foot_clearance_m':max(min(x['foot_heights_m']) for x in samples)}
                if ending and 'reached floor' in ending['reason']:
                    subprocess.run([sys.executable,str(HERE/'landing_impulse_research.py'),'--flight',str(f),'--out',str(i)],
                                   env=env,stdout=subprocess.DEVNULL,check=True)
                    impact=json.loads(i.read_text());first=next(c for c in impact['cases'] if c['impact_contact_assumption']=='first_point')
                    sliding=next(c for c in impact['cases'] if c['impact_contact_assumption']=='first_point_sliding')
                    allfeet=next(c for c in impact['cases'] if c['impact_contact_assumption']=='four_points')
                    item.update(sticking_status=first['solver_status'],sliding_status=sliding['solver_status'],
                                all_points_status=allfeet['solver_status'])
                    if sliding['solver_status']=='solved':
                        v=sliding['postimpact_contact_velocities_m_s'][0]
                        item['sliding_postimpact_speed_m_s']=(v[0]*v[0]+v[1]*v[1])**.5
                        score=item['sliding_postimpact_speed_m_s']+.02*max(0,item['max_body_tilt_deg']-5)
                        if best is None or score<best[0]:best=(score,flight,impact,item)
                results.append(item)
    report={'status':'MODEL_ONLY_GRID_SEARCH','cases':results,'hardware_actuation':False,'physical_jump_validated':False,
            'landing_control_validated':False,'limitations':['Assumed friction and unmeasured mass/inertia',
            'No motor torque-speed, latency, thermal or impact-compliance model','Weight grid is not a proof of global feasibility',
            'A sticking or sliding impact snapshot is not controlled landing recovery']}
    (HERE/'flight-landing-grid-research.json').write_text(json.dumps(report,indent=2))
    if best:
        (HERE/'flight-grid-best-research.json').write_text(json.dumps(best[1],indent=2))
        (HERE/'impact-grid-best-research.json').write_text(json.dumps(best[2],indent=2))
    print(json.dumps({'total':len(results),'reached_floor':sum('sliding_status' in x for x in results),
                      'single_point_sticking':sum(x.get('sticking_status')=='solved' for x in results),
                      'best_model_only':best[3] if best else None},indent=2))


if __name__=='__main__':main()
