import datetime,hashlib,json,pathlib,subprocess
p=pathlib.Path('/home/khadas/BRUCE/session-save-20261115')
r=subprocess.run(['python3',str(p/'launch_validated_stance.py')],capture_output=True,text=True)
(p/'launcher-check.log').write_text(r.stdout+r.stderr)
assert r.returncode==0,r.stdout+r.stderr
info=dict(saved_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),primary_location=str(p),archive='bruce-gait-camera-full-20261115.tar.gz',archive_sha256=(p/'SHA256SUMS').read_text().split()[0],archive_bytes=(p/'bruce-gait-camera-full-20261115.tar.gz').stat().st_size,archive_checksum_verified=True,launch_check_returncode=r.returncode,checked_source_config_dependency_files=81,motor_actuation_during_save=False,vision_obstacle_crossing_verified=False,real_synchronized_obstacle_dataset_collected=False)
(p/'VERIFIED.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
lines=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and '__pycache__' not in f.parts and f.name not in ('ENTRYPOINT_SHA256SUMS','bruce-gait-camera-full-20261115.tar.gz'):
  lines.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(p)))
(p/'ENTRYPOINT_SHA256SUMS').write_text('\n'.join(lines)+'\n')
subprocess.run(['sha256sum','--quiet','-c','ENTRYPOINT_SHA256SUMS'],cwd=p,check=True)
link=p.parent/'SAVED_GAIT_CAMERA'
if not link.exists() and not link.is_symlink():link.symlink_to(p,target_is_directory=True)
(p.parent/'下次调试先读.md').write_text('已验证步态与相机调试存档：\n\n/home/khadas/BRUCE/session-save-20261115/README.md\n\n先运行该目录内 python3 launch_validated_stance.py，只检查不使能。正式启动前必须完成现场支架与急停确认；请沿用保存参数，不直接使用默认步态。电机通信中断底层原因仍未解决，视觉越障尚未验证。\n')
print(json.dumps(info,ensure_ascii=False,indent=2))
