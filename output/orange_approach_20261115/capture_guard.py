"""Capture camera frames while observing standing; never write motor commands."""
import argparse
import datetime
import json
import pathlib
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['raw', 'sw', 'filter', 'sw12'])
parser.add_argument('--runtime', type=pathlib.Path, default=pathlib.Path('/dev/shm/bruce-step-4914'))
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent
out = root / (args.mode + '-' + datetime.datetime.now().strftime('%H%M%S'))
out.mkdir()
print('OUTPUT', out, flush=True)
rows = []
with (out / 'capture.log').open('w') as log:
    process = subprocess.Popen(['nice', '-n', '10', 'taskset', '-c', '1',
        str(root / 'target_capture'), str(out), args.mode], stdout=log, stderr=subprocess.STDOUT)
    start = time.monotonic()
    try:
        while process.poll() is None:
            motor = json.loads((args.runtime / 'motor-latest.json').read_text())
            state = json.loads((args.runtime / 'state.json').read_text())
            now = time.monotonic()
            rows.append(dict(t=now, phase=state['phase'], tilt=state.get('fused_tilt_deg'),
                motor_age_s=now-motor['monotonic'], codes=motor['codes']))
            if (now-start > 30 or state['phase'] != 'holding_balance'
                    or now-motor['monotonic'] > .2 or any(x != 128 for x in motor['codes'])):
                raise RuntimeError('Camera stopped: standing feedback changed or capture timed out')
            time.sleep(.2)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        (out / 'standing-monitor.json').write_text(json.dumps(rows))
print('EXIT', process.returncode, flush=True)
print((out / 'capture.log').read_text()[-2000:], flush=True)
if (out / 'capture.json').exists():
    print((out / 'capture.json').read_text(), flush=True)
raise SystemExit(process.returncode)
