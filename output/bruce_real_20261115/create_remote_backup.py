#!/usr/bin/env python3
"""Archive an inactive robot's existing files; never access actuator interfaces."""
import datetime, hashlib, json, os, pathlib, subprocess, tarfile
base=pathlib.Path('/home/khadas/BRUCE')
out=base/'session-save-20261115'
out.mkdir(exist_ok=True)
meta=out/'metadata'
meta.mkdir(exist_ok=True)
repo=base/'BRUCE-OP'
commands={
 'git-status.txt':['git','-C',str(repo),'status','--short'],
 'git-head.txt':['git','-C',str(repo),'rev-parse','HEAD'],
 'working-tree.patch':['git','-C',str(repo),'diff','--binary'],
 'uname.txt':['uname','-a'],
 'python-packages.txt':['python3','-m','pip','freeze'],
 'usb-devices.txt':['lsusb'],
 'processes.txt':['ps','-eo','pid,comm'],
}
for name,cmd in commands.items():
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (meta/name).write_bytes(p.stdout)
latency={str(p):p.read_text().strip() for p in pathlib.Path('/sys/bus/usb-serial/devices').glob('*/latency_timer')}
(meta/'usb-latency.json').write_text(json.dumps(latency,indent=2))
# Include complete diagnostics, not just favorable trials, plus all current source/config/native modules.
roots=[('BRUCE-OP',repo),('diagnostics',base/'diagnostics'),
       ('BRUCE_SERIAL_PORT',base/'BRUCE_SERIAL_PORT'),
       ('dependencies/pybear',pathlib.Path('/home/khadas/.local/lib/python3.10/site-packages/pybear'))]
entries=[]
for prefix,root in roots:
 paths=[root] if root.is_file() else sorted(root.rglob('*'))
 for p in paths:
  rel=p.relative_to(root).parts if p!=root else ()
  if prefix=='BRUCE-OP' and '.git' in rel: continue
  name=str(pathlib.PurePosixPath(prefix).joinpath(*rel))
  if p.is_symlink(): entries.append(dict(path=name,symlink=os.readlink(p)))
  elif p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
   entries.append(dict(path=name,bytes=p.stat().st_size,sha256=h.hexdigest()))
manifest=dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 robot='khadas@180.85.212.21', motor_state='No controller process running; no actuation performed for this archive.',
 excluded=['BRUCE-OP/.git (HEAD, status and binary diff saved separately)'], files=entries)
(meta/'source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
archive=out/'bruce-gait-camera-full-20261115.tar.gz'
partial=archive.with_suffix(archive.suffix+'.partial')
print('Hash manifest ready:',len(entries),'files; creating archive',flush=True)
subprocess.run(['tar','--exclude=BRUCE-OP/.git','-czf',str(partial),'-C',str(base),
 'BRUCE-OP','diagnostics','BRUCE_SERIAL_PORT','-C','/home/khadas/.local/lib/python3.10/site-packages',
 'pybear','-C',str(out),'metadata'],check=True)
partial.rename(archive)
h=hashlib.sha256()
with archive.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
(out/'SHA256SUMS').write_text(h.hexdigest()+'  '+archive.name+'\n')
with tarfile.open(archive,'r:gz') as t:
 members=t.getnames()
 required=['BRUCE-OP/Settings/BRUCE_macros.py','BRUCE-OP/Util/stationary_step_test.py',
 'diagnostics/validated-motion-profiles-20261114.json',
 'diagnostics/orbbec-20261114/capture-02/depth.raw','metadata/source-manifest.json']
 for r in required: assert r in members,r
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=h.hexdigest(),members=len(members),required_files_present=True)),flush=True)
