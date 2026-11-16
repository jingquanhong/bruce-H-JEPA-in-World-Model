#!/usr/bin/env python3
"""Verify saved code; --run starts supported stance, never automatic walking."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/home/khadas/BRUCE/BRUCE-OP')
BASE=Path('/home/khadas/BRUCE/session-save-20261115/launch_validated_stance.py')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
 subprocess.run([sys.executable,str(BASE)],check=True)
 manifest=json.loads((HERE/'source-manifest.json').read_text())
 for item in manifest['files']:
  f=Path(item['path'])
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:
   raise SystemExit('IMU candidate source changed: '+str(f))
 command=json.loads((HERE/'stance-command.json').read_text())
 print('IMU candidate source verified. Stationary behavior uses saved original parameters.',flush=True)
 print(shlex.join(command),flush=True)
 if args.run:return subprocess.call(command,cwd=REPO)
 print('CHECK ONLY: no motor enabled, no walking request sent.')
 return 0

if __name__=='__main__':raise SystemExit(main())
