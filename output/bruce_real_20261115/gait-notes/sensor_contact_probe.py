"""Record Pico contacts with all actuator commands disabled; opens no BEAR port."""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120:
        parser.error('seconds must be in 5..120')
    import Startups.memory_manager as mm
    from Settings.BRUCE_macros import PICO_port, PICO_baudrate
    from Library.BRUCE_SENSE.Manager import SenseManager

    def disabled():
        if mm.LEG_COMMAND.get()['BEAR_enable'][0] != 0:
            raise RuntimeError('Motor enable command is active; aborting sensor-only check')

    disabled()
    if subprocess.run(['fuser', PICO_port], capture_output=True).returncode == 0:
        raise RuntimeError('Sensor port is already in use')
    sm = SenseManager(port=PICO_port, baudrate=PICO_baudrate)
    records = []
    output = ROOT.parent / 'diagnostics' / ('contacts-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.json')
    start = last_frame = time.monotonic()
    previous = None
    try:
        sm.send_data(pico_mode='nominal')
        print('SENSOR_ONLY_NO_MOTOR_ENABLE', flush=True)
        while time.monotonic() - start < args.seconds:
            disabled()
            now = time.monotonic()
            if sm.read_data():
                last_frame = now
                contact = sm.foot_contact.tolist()
                records.append({'seconds': now-start, 'contacts': contact,
                                'raw_contact_byte': sm.data4pico[-3],
                                'accel': sm.accel.tolist(), 'omega': sm.omega.tolist()})
                pattern = tuple(contact)
                if pattern != previous:
                    if previous is None or now - last_print > .2:
                        print('CONTACT', round(now-start, 2), contact, flush=True)
                        last_print = now
                    previous = pattern
            if now - last_frame > .25:
                raise RuntimeError('No fresh Pico frames for 250 ms')
            time.sleep(.001)
    finally:
        sm.send_data(pico_mode='idle')
        sm.close_port()
        with output.open('w') as stream:
            json.dump(records, stream)
            stream.flush()
            os.fsync(stream.fileno())
        print('SAVED', str(output), 'frames', len(records), flush=True)
        print('PATTERNS', dict(collections.Counter(str(r['contacts']) for r in records)), flush=True)


if __name__ == '__main__':
    main()
