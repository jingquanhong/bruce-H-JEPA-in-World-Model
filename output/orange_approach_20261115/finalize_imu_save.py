from pathlib import Path
import json,shutil,hashlib,os,datetime
base=Path('/home/khadas/BRUCE');save=base/'session-save-20261116-imu-straight';diag=base/'diagnostics/orange-approach-20261115';repo=base/'BRUCE-OP'
p=save/'validated-profile.json';data=json.loads(p.read_text());data['validation']='PHYSICAL_FORWARD_AND_BACKWARD_12S_OPERATOR_CONFIRMED_STRAIGHT';data['backward']={'command_vx_m_s':-.08,'seconds':12,'forward_bias_m':-.015,'lateral_bias_m':0.,'operator_feedback':'走直且稳定','observation':str(base/'diagnostics/imu-straight-trim-observation-1794776866239153213.json')};data['limitations']=['Commanded speed is not ground-measured speed','IMU holds heading; no absolute-position observation','Lateral candidate is software-checked only, not physically validated'];p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
p=save/'backward-validation.json';data=json.loads(p.read_text());data['long_12s_operator']='走直且稳定';p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
shutil.copy2(base/'imu_walk.py',save/'imu_walk.py');shutil.copy2(diag/'IMU直行使用说明.md',save/'使用说明.md')
shutil.copy2(save/'使用说明.md',save/'README.md')
candidate=save/'lateral-candidate-unvalidated';candidate.mkdir(exist_ok=True)
for name in ['imu_direction_high_level.py','stationary_imu_direction_test.py','observe_imu_direction_bout.py']:shutil.copy2(repo/'Util'/name,candidate/name)
for name in ['test_imu_direction_high_private.py','imu-direction-private-results.json']:shutil.copy2(diag/name,candidate/name)
(candidate/'validation-status.json').write_text(json.dumps({'status':'SOFTWARE_ONLY_NOT_PHYSICALLY_ENABLED','max_lateral_speed_m_s':.03,'max_lateral_seconds':6,'original_crossstep_candidates':'cancelled; not used'},indent=2))
archive=base/'diagnostics/step-20261116-045219'
for name in ['checkpoint.json','motor-before-fault.json','motor-status.jsonl']:
 if (archive/name).exists():shutil.copy2(archive/name,save/('final-'+name))
status={'saved_at':datetime.datetime.now().astimezone().isoformat(),'default_profile':str(save/'validated-profile.json'),'control_entry':str(base/'imu_walk.py'),'forward':'user verified straight and stable at commanded .08 m/s for 12 s','backward':'user verified straight and stable at commanded -.08 m/s for 12 s','lateral':'software candidate only; no restart performed','last_exit':'motor codes136; operator confirms used emergency stop or power','archive':str(archive),'autostart':False,'current_controller':'not running; read live state before any future start'}
(base/'validated-motion-default.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
(save/'session-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
with (base/'下次调试先读.md').open('a') as f:f.write('\n\n## 2026-11-16 最新默认：IMU 前后直行已验证并保存\n用户确认前进、后退均以 8 cm/s 指令、12 s 走直且稳定，并要求以后前进后退使用这版。默认入口 ~/BRUCE/imu_walk.py；持久目录 '+str(save)+'；说明见 使用说明.md。前后移动横向偏置均 0；前进 +15 mm、后退 -15 mm 前向偏置；站立恢复 -2/-6 mm。禁止恢复失败跨步。横移新候选只做软件检查，未实机使能。最新控制器因全电机136退出，用户确认操作过急停或电源，支架承重；不能据旧运行时或本文自动使能。没有开机自启。重启后先 check，支撑就绪再 start，放地后 ground，holding_balance 后才允许运动。\n')
manifest=json.loads((save/'source-manifest.json').read_text());manifest['files'].append({'path':str(base/'imu_walk.py'),'sha256':hashlib.sha256((base/'imu_walk.py').read_bytes()).hexdigest()});(save/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
files=sorted(f for f in save.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
(save/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(save))+'\n' for f in files))
for f in files+[save/'SHA256SUMS',base/'validated-motion-default.json',base/'下次调试先读.md']:
 with f.open('rb') as stream:os.fsync(stream.fileno())
print('FINAL_SAVED',save,'files',len(files))
