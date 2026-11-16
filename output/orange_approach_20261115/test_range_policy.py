"""Controller contract checks: no hardware imports or motor commands."""
from vision_range_pid import RangePID,RangeFilter,checked_command


def main():
    pid=RangePID()
    for i in range(40):
        decision=pid.update(2.,10+i*.1)
        assert 0<=decision['speed']<=.05
    assert decision['speed']==.05
    for i in range(3):
        decision=pid.update(1.03,14+i*.1)
        assert decision['speed']==0 and decision['mode']=='balance'
    assert decision['reached']
    for d in (.7,.94,float('nan')):
        decision=pid.update(d,15.)
        assert decision['mode']=='balance' and not decision['reached']
    assert pid.update(2.,float('nan'))['mode']=='balance'

    filt=RangeFilter()
    for i in range(3):filt.add(i,10+i*.1,2.+i*.001,10+i*.1)
    assert abs(filt.read(10.2)[0]-2.001)<1e-9
    assert filt.read(11.) is None
    filt=RangeFilter()
    for _ in range(3):filt.add(1,10.,2.,10.)
    assert filt.read(10.) is None,'Repeated image is not independent evidence'
    for i in range(2,5):filt.add(i,10+i*.1,2.,10+i*.1)
    filt.add(5,10.5,1.,10.5)
    assert filt.read(10.5) is None,'Abrupt range change must invalidate history'

    command=dict(session_id='example-session',monotonic=10.,measurement_monotonic=10.,
                 forward_velocity_m_s=.05,mode='walk')
    assert checked_command(command,10.1,'example-session')==.05
    invalid=[(dict(command,monotonic=9.),10.1,'example-session'),
             (dict(command,measurement_monotonic=9.),10.1,'example-session'),
             (dict(command,forward_velocity_m_s=.051),10.1,'example-session'),
             (dict(command,forward_velocity_m_s=float('nan')),10.1,'example-session'),
             (command,10.1,'different-session'),(command,9.9,'example-session')]
    for data,now,session in invalid:
        try:checked_command(data,now,session)
        except ValueError:pass
        else:raise AssertionError('Unsafe command accepted')
    assert checked_command(dict(command,mode='balance',measurement_monotonic=0.),10.1,'example-session')==0
    print('Range policy checks passed: speed/arrival bands, stale and repeated measurements, jumps, invalid commands.')


if __name__=='__main__':main()
