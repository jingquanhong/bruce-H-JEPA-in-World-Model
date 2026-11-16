"""Display optional supervisor diagnostic snapshots; never commands motors.

70 C is a conservative operator attention threshold, not a manufacturer limit.
Only knee IDs 4 and 9 are covered; the hardware warning covers more components
and all other motors. Use alongside, never instead of, existing protection.
"""
import argparse
import json
import math
from pathlib import Path
import time


def classify(snapshot, now, stale_after=25.):
    if snapshot is None:return 'WAITING_FOR_THERMAL_SAMPLE'
    try:
        age=now-float(snapshot['monotonic'])
        if not math.isfinite(age) or age < -.1:return 'INVALID_THERMAL_TIME'
        if age > stale_after:return 'STALE_THERMAL_SAMPLE'
        if snapshot['status']!='ok':return 'THERMAL_READ_FAILED'
        values=snapshot['values']
        if len(values)!=2 or any(len(row)!=2 or len(row[0])!=4 for row in values):return 'INVALID_THERMAL_SAMPLE'
        if any(row[1]!=128 for row in values):return 'MOTOR_STATUS_WARNING'
        temps=[float(v) for row in values for v in row[0][1:]]
        if not all(math.isfinite(v) for v in temps):return 'INVALID_THERMAL_SAMPLE'
        return 'ATTENTION_70C' if max(temps)>=70 else 'NORMAL_SAMPLE'
    except (KeyError,TypeError,ValueError):return 'INVALID_THERMAL_SAMPLE'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    args=parser.parse_args()
    path=args.runtime/'knee-temperatures.json';previous=None;last=0.;started=time.monotonic()
    while args.runtime.is_dir() and not (args.runtime/'finished').exists():
        now=time.monotonic()
        try:snapshot=json.loads(path.read_text())
        except (OSError,ValueError):snapshot=None
        status=classify(snapshot,now)
        if status=='WAITING_FOR_THERMAL_SAMPLE' and now-started>25:status='MISSING_THERMAL_SAMPLE'
        if status!=previous or now-last>=10:
            print(json.dumps({'status':status,'sample':snapshot},ensure_ascii=False),flush=True)
            previous=status;last=now
        time.sleep(.5)
    print('SUPERVISOR_ENDED',flush=True)


if __name__=='__main__':main()
