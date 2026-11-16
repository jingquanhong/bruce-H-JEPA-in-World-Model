"""Apply user's earlier instruction to remove rejected lateral candidates only."""
from pathlib import Path
import subprocess,json,hashlib,shutil,datetime
base=Path('/home/khadas/BRUCE');repo=base/'BRUCE-OP';saved=base/'session-save-20261116-imu-straight';diag=base/'diagnostics/step-vision-20261116'
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=saved,check=True)
# A live candidate would have to be stopped before deletion; do not kill or enable anything.
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:c=(p/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError):continue
 if any(x.endswith(b'Util/stationary_imu_direction_test.py') for x in c):raise RuntimeError('Lateral supervisor still active')
names=['imu_direction_high_level','stationary_imu_direction_test','observe_imu_direction_bout','directional_drift_feedback','observe_imu_direction_feedback_bout','test_directional_drift_feedback']
removed=[]
for name in names:
 f=repo/'Util'/(name+'.py')
 if f.exists():f.unlink();removed.append(str(f))
 for f in (repo/'Util/__pycache__').glob(name+'.*.pyc'):f.unlink();removed.append(str(f))
for f in [base/'session-save-20261116-imu-lateral-candidate',saved/'lateral-candidate-unvalidated']:
 if f.exists():shutil.rmtree(f);removed.append(str(f))
for name in ['test_imu_direction_high_private.py','save_imu_direction_trial.py']:
 f=base/'diagnostics/orange-approach-20261115'/name
 if f.exists():f.unlink();removed.append(str(f))
# Keep the original verified forward/back source manifest unchanged.
for f in [saved/'README.md',saved/'使用说明.md',base/'机器人指令操作指南.md']:
 text=f.read_text();marker='横移候选单独在'
 if marker in text:text=text[:text.index(marker)]+'本轮未通过的横移候选已按用户要求撤下；此版本仅用于已验证的前进、后退。\n'
 f.write_text(text)
rows=[]
for line in (saved/'SHA256SUMS').read_text().splitlines():
 _,name=line.split('  ',1)
 f=saved/name
 if name.startswith('lateral-candidate-unvalidated/'):continue
 rows.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+name)
(saved/'SHA256SUMS').write_text('\n'.join(rows)+'\n')
p=base/'validated-motion-default.json';data=json.loads(p.read_text());data['lateral']='user rejected and cancelled; latest candidate code/backups removed';data['current_controller']='none running at last read-only check; verify live state before any start';p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
with (base/'下次调试先读.md').open('a') as f:f.write('\n\n2026-11-16 当前连接：按用户上次要求已删除未通过的本轮IMU横移候选源码及其候选备份，不得继续该方案。原IMU前后直行保存版本校验通过，保留为默认。当前仅检查红色书本越障方案与相机，未使能；相机USB反复掉线，书本几何未知，未执行越障或训练。详见 ~/BRUCE/diagnostics/step-vision-20261116/越障实施方案.md。\n')
(diag/'rejected-lateral-removal.json').write_text(json.dumps({'removed':removed,'reason':'user explicitly rejected lateral attempt and requested not to keep it','forward_source_manifest_unchanged':True},ensure_ascii=False,indent=2))
shutil.copy2(diag/'越障实施方案.md',base/'红色书本越障方案.md')
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=saved,check=True)
subprocess.run(['python3',str(base/'imu_walk.py'),'check'],check=True)
print('REMOVED_REJECTED_CANDIDATE_ITEMS',len(removed),'VERIFIED_FORWARD_BACK_PRESERVED')
