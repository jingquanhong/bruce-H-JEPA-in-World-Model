"""Offline model screening over push durations; no robot or simulation IO."""
import json
from pathlib import Path
from integrate_push_research import run

HERE=Path(__file__).resolve().parent


def main():
    config=json.loads((HERE/'actuator-config-readonly.json').read_text())
    limits=[]
    for row in config['motor_reads']:
        value=row['readbacks']['get_limit_velocity_max']
        if len(value)!=1 or value[0][1]!=128:raise RuntimeError('Invalid velocity-register readback')
        limits.append(float(value[0][0][0]))
    cases=[]
    for accel in [3.,4.,5.,6.]:
        for taper in [.04,.06,.08,.10]:
            # Quintic ramps average half their endpoint values.
            push=(.443+accel*.02/2-(accel-9.81)*taper/2)/accel
            if not .04<=push<=.3:continue
            candidate=run(accel,push,taper,.001)
            rows=candidate['samples']
            if not rows:continue
            peak_speed=[max(abs(x['motor_speeds_rad_s'][i]) for x in rows) for i in range(10)]
            cases.append({'accel_m_s2':accel,'push_s':push,'taper_s':taper,'status':candidate['status'],
                'failure':candidate['failure'],'vz_m_s':rows[-1]['com_vz_m_s'],
                'ideal_apex_above_takeoff_m':rows[-1]['com_vz_m_s']**2/(2*9.81),
                'peak_motor_speeds_rad_s':peak_speed,
                'motor_ids_exceeding_velocity_register':[i+1 for i in range(10) if peak_speed[i]>limits[i]],
                'peak_motor_acceleration_rad_s2':max(x['max_motor_acceleration_rad_s2'] for x in rows),
                'peak_motor_current_A':max(x['max_motor_current_A'] for x in rows),
                'max_foot_drift_m':max(x['max_foot_drift_m'] for x in rows),
                'max_constraint_violation':max(x['constraint_violation'] for x in rows)})
    report={'status':'OFFLINE_PARAMETER_SEARCH_ONLY','cases':cases,'configured_motor_velocity_registers_rad_s':limits,
            'hardware_actuation':False,'physical_jump_validated':False,
            'limitations':['Velocity register applies in motor modes 1 and 2, not force mode 3; comparison is informational only',
              'Configuration is not a certified physical torque-speed envelope',
              'Push only: no flight, landing, thermal or PD actuator dynamics',
              'Mass, friction and contact model are assumed, not measured']}
    (HERE/'integrated-push-parameter-search.json').write_text(json.dumps(report,indent=2))
    ok=[x for x in cases if x['status']=='INTEGRATED_MODEL_ONLY' and not x['motor_ids_exceeding_velocity_register']]
    print(json.dumps({'candidates':len(cases),'integrated_without_register_exceedance':len(ok),
                      'lowest_model_motor_acceleration':sorted(ok,key=lambda x:x['peak_motor_acceleration_rad_s2'])[:2]},indent=2))


if __name__=='__main__':main()
