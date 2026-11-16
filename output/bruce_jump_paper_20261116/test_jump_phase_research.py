"""Synthetic failure cases and replay of the actual slow-crouch log; no IO."""
from dataclasses import replace
import json
from pathlib import Path
from jump_phase_research import JumpPhaseReplay, Sample

base = Sample(0., (1, 1, 1, 1), 0., 1., .01, .01, .01)


def feed(classifier, t, **updates):
    return classifier.update(replace(base, time_s=t, **updates))


machine = JumpPhaseReplay()
machine.start_push(0.)
assert feed(machine, .01) == 'push'
assert feed(machine, .02, contacts=(0,0,0,0), com_vz_m_s=.2) == 'push'
assert feed(machine, .03, contacts=(1,0,1,0), com_vz_m_s=.2) == 'push'  # dropout rejected
assert feed(machine, .04, contacts=(0,0,0,0), com_vz_m_s=.2) == 'push'
assert feed(machine, .06, contacts=(0,0,0,0), com_vz_m_s=.2) == 'flight'
assert feed(machine, .08, contacts=(0,0,0,0), com_vz_m_s=-.2) == 'flight'
assert feed(machine, .09, contacts=(1,0,0,0), com_vz_m_s=-.2) == 'landing'
assert feed(machine, .10, contacts=(1,0,1,0), com_vz_m_s=-.2) == 'landing'
assert feed(machine, .16, contacts=(1,0,1,0)) == 'landed'

for bad, reason in [({'sense_age_s':.151}, 'stale'), ({'motor_age_s':.201}, 'stale'),
                    ({'motor_fault':True}, 'motor'), ({'emergency_stop':True}, 'emergency'),
                    ({'tilt_deg':8.1}, 'tilt'), ({'com_vz_m_s':float('nan')}, 'nonfinite')]:
    machine=JumpPhaseReplay(); machine.start_push(0.)
    assert feed(machine, .01, **bad) == 'abort' and reason in machine.abort_reason

machine=JumpPhaseReplay(); machine.start_push(0.)
for i in range(1,52):feed(machine, i*.01)
assert machine.phase == 'abort' and machine.abort_reason == 'no confirmed liftoff'

def airborne():
    c=JumpPhaseReplay(); c.start_push(0.)
    feed(c,.01,contacts=(0,0,0,0),com_vz_m_s=.3)
    feed(c,.03,contacts=(0,0,0,0),com_vz_m_s=.3)
    assert c.phase=='flight'
    return c

machine=airborne()
assert feed(machine,.04,contacts=(0,0,0,0),sense_age_s=.151)=='abort'
assert machine.abort_reason=='stale feedback'
machine=airborne()
for i in range(4,56):feed(machine,i*.01,contacts=(0,0,0,0))
assert machine.phase=='abort' and machine.abort_reason=='flight timeout'
machine=airborne();feed(machine,.04,contacts=(1,0,0,0))
for i in range(5,46):feed(machine,i*.01,contacts=(1,0,0,0))
assert machine.phase=='abort' and machine.abort_reason=='landing timeout'

log = Path(__file__).with_name('six-cm-observation-1794839767048064923.json')
if log.exists():
    records = json.loads(log.read_text())['records']
    observed_flights = 0
    for offset in range(0, len(records)-5, 5):
        window=records[offset:offset+5]
        machine=JumpPhaseReplay(); machine.start_push(window[0]['observed_monotonic'])
        previous=window[0]
        for item in window:
            dt=item['observed_monotonic']-previous['observed_monotonic']
            velocity=(item['live_com_z']-previous['live_com_z'])/dt if dt>0 else 0.
            age=item['critical_feedback_ages']
            sample=Sample(item['observed_monotonic'],tuple(item['live_contacts']),velocity,
                          item['live_tilt_deg'],age['sense_s'],age['leg_s'],age['motor_s'])
            if machine.update(sample)=='flight':observed_flights+=1
            assert machine.phase!='abort', machine.abort_reason
            previous=item
    assert observed_flights==0
    print('PASS actual supported crouch replay: no false flight in',len(records),'samples')
print('PASS synthetic contact debounce, flight/landing, timeout and fault paths; offline research only')
