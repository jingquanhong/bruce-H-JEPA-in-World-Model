"""Supported diagnostic for BEAR 1 and 6 only; no actuation without --run.

The support must carry the robot with both feet clear. Other leg actuators
remain disabled or in damping. Original gains and hardware limits are retained.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDS = (1, 6)


def smooth_goal(initial, target, elapsed, duration):
    u = np.clip(elapsed / duration, 0., 1.)
    return initial + (target - initial) * (3 * u * u - 2 * u * u * u)


def atomic_json(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


def run():
    from Library.ACTUATORS.BEAR_controller import TimedBEAR
    from Settings.BRUCE_macros import BEAR_port, BEAR_baudrate
    assert subprocess.run(['fuser', BEAR_port], capture_output=True).returncode != 0, 'BEAR port already used'
    folder = Path('/dev/shm') / ('bruce-yaw-' + str(os.getpid()))
    folder.mkdir()
    archive = ROOT.parent / 'diagnostics' / ('yaw-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    b = None
    guard = None
    guard_log = None
    records = []
    result = 'not completed'
    started = False

    def interrupt(sig, frame):
        raise KeyboardInterrupt('Supervisor interrupted')

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    try:
        b = TimedBEAR(port=BEAR_port, baudrate=BEAR_baudrate, timeout=.005, bulk_timeout=.010)
        enables = [b.get_torque_enable(i)[0][0] for i in range(1, 11)]
        assert all(x in (0, 3) for x in enables), 'Existing active actuator'
        data = b.bulk_read(list(IDS), ['present_position', 'present_velocity', 'present_iq', 'input_voltage'])
        assert data is not None and all(x[0] is not None and x[1] in (128, 136) for x in data)
        initial = np.array([x[0][0] for x in data])
        target = np.full(2, np.pi / 2)
        assert np.isfinite(initial).all() and np.all((0 <= initial) & (initial <= np.pi))
        assert max(abs(target - initial)) < .6, 'Initial hip yaw too far from nominal; inspect alignment first'
        assert all(15 < x[0][3] < 16.9 for x in data), 'Supply not ready'
        duration = max(3., 15 * float(max(abs(target - initial))))
        atomic_json(folder / 'limits.json', {'maximum_seconds': None})
        atomic_json(folder / 'children.json', {'bear': os.getpid()})
        (folder / 'heartbeat').touch()
        guard_log = (folder / 'guard.log').open('w')
        guard = subprocess.Popen(
            [sys.executable, '-u', str(ROOT / 'Util/stationary_step_test.py'), '--worker', 'guard', '--folder', str(folder)],
            stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
        # Load the measured position before changing mode and enabling torque.
        b.set_torque_enable(*[(i, 0) for i in IDS])
        b.set_goal_position(*[(i, float(q)) for i, q in zip(IDS, initial)])
        b.set_mode(*[(i, 2) for i in IDS])
        # Mark the attempted enable before the write so exceptions also damp.
        started = True
        b.set_torque_enable(*[(i, 1) for i in IDS])
        start = time.monotonic()
        last_saved = last_print = 0.
        last_check = start
        bad_since = load_since = low_since = None
        print('YAW_ONLY_ENABLED', IDS, 'initial_motor_rad', initial.tolist(), 'target_motor_rad', target.tolist(), 'runtime', folder, flush=True)
        while not (folder / 'operator-stop').exists():
            now = time.monotonic()
            (folder / 'heartbeat').touch()
            goal = smooth_goal(initial, target, now - start, duration)
            data = b.bulk_read_write(list(IDS), ['present_position', 'present_velocity', 'present_iq', 'input_voltage'],
                                     ['goal_position'], [[float(x)] for x in goal])
            if data is None or len(data) != 2 or any(x[0] is None or len(x[0]) != 4 or x[1] < 128 for x in data):
                bad_since = now if bad_since is None else bad_since
                if now - bad_since > .05:
                    raise RuntimeError('Yaw feedback invalid for >50 ms')
                time.sleep(.005)
                continue
            bad_since = None
            if any(x[1] != 128 for x in data):
                raise RuntimeError('Yaw motor fault ' + str([x[1] for x in data]))
            values = np.array([x[0] for x in data])
            assert np.isfinite(values).all(), 'Nonfinite yaw feedback'
            if np.max(np.abs(values[:, 0] - goal)) > .15:
                raise RuntimeError('Yaw tracking error >0.15 rad')
            loaded = np.max(np.abs(values[:, 2])) > 8
            load_since = (now if load_since is None else load_since) if loaded else None
            if load_since is not None and now - load_since > .1:
                raise RuntimeError('Unexpected sustained yaw current >8 A')
            low_since = (now if low_since is None else low_since) if min(values[:, 3]) < 14.5 else None
            if low_since is not None and now - low_since > .1:
                raise RuntimeError('Sustained low voltage')
            if now - last_check > 1:
                other = {i: b.get_torque_enable(i)[0][0] for i in range(1, 11) if i not in IDS}
                if any(x not in (0, 3) for x in other.values()):
                    raise RuntimeError('Another leg motor unexpectedly enabled')
                last_check = now
            if now - last_saved > .04:
                record = {'monotonic': now, 'phase': 'holding_yaw' if now - start >= duration else 'slow_yaw_alignment',
                          'motor_ids': list(IDS), 'goal_positions': goal.tolist(), 'values': data}
                records.append(record)
                atomic_json(folder / 'state.json', record)
                last_saved = now
            if now - last_print > 2:
                print('YAW_STATE', records[-1], flush=True)
                last_print = now
            if now - start >= duration and result != 'YAW_HOLD_ACTIVE':
                result = 'YAW_HOLD_ACTIVE'
                print('NOMINAL_YAW_REACHED_KEEPING_POWER', flush=True)
            time.sleep(max(0., .01 - (time.monotonic() - now)))
        result = 'OPERATOR_ENDED_YAW_HOLD'
    except BaseException as error:
        result = type(error).__name__ + ': ' + str(error)
        traceback.print_exc()
    finally:
        if b:
            if started:
                try:
                    b.set_torque_enable(*[(i, 3) for i in IDS])
                except Exception:
                    traceback.print_exc()
            b.close()
        (folder / 'finished').touch()
        if guard:
            try:
                guard.wait(timeout=2)
            except subprocess.TimeoutExpired:
                guard.terminate()
                guard.wait(timeout=2)
        if guard_log:
            guard_log.close()
        (folder / 'result.json').write_text(json.dumps({'result': result, 'records': records}, indent=2))
        shutil.copytree(folder, archive)
        # Persist the archive before the operator can power down.
        for path in archive.iterdir():
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        directory = os.open(str(archive), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        shutil.rmtree(folder)
        print('YAW_RESULT', result, 'ARCHIVE', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    else:
        print('Non-actuating check. --run slowly aligns and holds only motors 1 and 6 with full support and feet clear.')
