#!/usr/bin/env python3
"""Load the saved supported-gait profile. Default: verify and print only."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROBOT = Path('/home/khadas/BRUCE/BRUCE-OP')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Start supported stance, without an automatic walking request.')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'metadata/source-manifest.json').read_text())
    profile_bytes = (HERE / 'validated-motion-profiles-20261114.json').read_bytes()
    saved_profile = next(item for item in manifest['files']
                         if item['path'] == 'diagnostics/validated-motion-profiles-20261114.json')
    if hashlib.sha256(profile_bytes).hexdigest() != saved_profile['sha256']:
        raise SystemExit('The validated parameter file differs from the saved version.')
    profile = json.loads(profile_bytes)
    checked, mismatched = 0, []
    for item in manifest['files']:
        name = item['path']
        if 'sha256' not in item or Path(name).suffix not in ('.py', '.so', '.sh'):
            continue
        if '__pycache__' in Path(name).parts:
            continue
        if name.startswith('BRUCE-OP/'):
            source = ROBOT.parent / name
        elif name.startswith('dependencies/pybear/'):
            source = Path('/home/khadas/.local/lib/python3.10/site-packages') / name.removeprefix('dependencies/')
        else:
            continue
        checked += 1
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            mismatched.append(str(source))
    if mismatched:
        raise SystemExit('Saved-version check failed; review changes before a motor test:\n' + '\n'.join(mismatched))
    common = profile['common']
    stationary = profile['profiles']['stationary']
    required = ('hold_hip_yaw', 'translation_tracking')
    if not all(common[key] is True for key in required):
        raise SystemExit('The validated profile flags have changed.')
    command = [
        'python3', '-u', 'Util/stationary_step_test.py', '--run', '--keep-standing',
        '--translation-tracking', '--allow-small-motion',
        '--max-motion-seconds', str(common['max_motion_seconds']),
        '--hold-hip-yaw', '--manual-landing',
        '--swing-height', str(common['swing_height_m']),
        '--attitude-scale', str(common['attitude_scale']), '--smooth-stop',
        '--foot-spacing', str(common['foot_spacing_command_m']),
        '--forward-bias', str(stationary['forward_bias_m']),
        '--lateral-bias', str(common['lateral_bias_m']),
        '--minimum-step-width', str(common['minimum_width_m']),
        '--max-step-seconds', '30', '--seconds', '0',
        '--ground-wait', '180', '--stand-seconds', '10',
    ]
    print('Saved control/config/dependency files verified:', checked, flush=True)
    print('cd ' + shlex.quote(str(ROBOT)), flush=True)
    print(shlex.join(command), flush=True)
    if args.run:
        print('Starting supported stance only; manual landing and all existing protections remain active.', flush=True)
        return subprocess.call(command, cwd=ROBOT)
    print('CHECK ONLY: no controller started; no motor port opened.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
