"""Temperature alert and missing/faulty telemetry checks, no motor IO."""
from copy import deepcopy
from watch_knee_thermal import classify

normal={'monotonic':100.,'status':'ok','values':[
    [[16.,40.,42.,44.],128],[[16.,39.,41.,43.],128]]}
assert classify(None,100.)=='WAITING_FOR_THERMAL_SAMPLE'
assert classify(normal,100.1)=='NORMAL_SAMPLE'
assert classify(normal,125.1)=='STALE_THERMAL_SAMPLE'
fault=deepcopy(normal);fault['values'][0][1]=130
assert classify(fault,100.1)=='MOTOR_STATUS_WARNING'
hot=deepcopy(normal);hot['values'][0][0][2]=70.
assert classify(hot,100.1)=='ATTENTION_70C'
invalid=deepcopy(normal);invalid['values'][0][0][1]=float('nan')
assert classify(invalid,100.1)=='INVALID_THERMAL_SAMPLE'
failed={'monotonic':100.,'status':'extended_read_failed'}
assert classify(failed,100.1)=='THERMAL_READ_FAILED'
print('PASS thermal alert, motor warning, stale, invalid and missing samples; no motor IO')
