from pathlib import Path
import json,hashlib,shutil,os,datetime
base=Path('/home/khadas/BRUCE');repo=base/'BRUCE-OP';runtime=Path('/dev/shm/bruce-step-13376')
save=base/'session-save-20261116-imu-straight';save.mkdir(exist_ok=True)
files=['imu_heading_hold.py','imu_heading_high_level.py','stationary_imu_heading_speed_test.py','observe_imu_straight_trim_bout.py','observe_imu_speed_bout.py']
manifest=[]
for name in files:
 src=repo/'Util'/name;shutil.copy2(src,save/name)
 manifest.append({'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
for name in ['profile.json','planner-tuning.json','translation-tuning.json','forward-stage.json','checkpoint.json','state.json']:
 if (runtime/name).exists():shutil.copy2(runtime/name,save/('snapshot-'+name))
obs=base/'diagnostics/imu-straight-trim-observation-1794776566349362137.json';shutil.copy2(obs,save/obs.name)
profile={'saved_at':datetime.datetime.now().astimezone().isoformat(),'validation':'PHYSICAL_FORWARD_12S_OPERATOR_CONFIRMED_STRAIGHT','operator_feedback':'很不错完全是直的','observation':str(obs),'motion':{'command_vx_m_s':.08,'seconds':12,'forward_bias_m':.015,'lateral_bias_m':0.,'imu_heading_hold':True},'stationary':{'forward_bias_m':-.002,'lateral_bias_m':-.006},'shared':{'swing_height_m':.015,'attitude_scale':.5,'smooth_stop_s':.8,'foot_spacing_command_m':.08,'minimum_width_m':.10,'hip_yaw_hold':True},'limitations':['Commanded speed is not ground-measured speed','IMU holds heading; no absolute-position observation','Backward and lateral not validated by this forward result'],'autostart':False}
(save/'validated-profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2))
(save/'source-manifest.json').write_text(json.dumps({'files':manifest},indent=2))
(save/'README.md').write_text('# IMU 直行已验证记录\n\n前进指令 8 cm/s、12 秒，用户确认“很不错完全是直的”。\n运动时前向 DCM 偏置 +15 mm、横向 0；站立恢复 -2 mm、-6 mm。\n保留 IMU 航向反馈、低抬脚、半强度姿态反馈、平滑收步及全部保护。\n此目录为持久保存，无开机自启。后退、横移须单独验证；IMU 本身不提供无漂移位置。\n')
notes=base/'下次调试先读.md'
with notes.open('a') as f:f.write('\n\n2026-11-16 IMU 直行：8 cm/s 指令、12 s，运动时横向偏置改为 0，用户确认完全走直。持久备份：'+str(save)+'。站立参数保持 -2/-6 mm。横跨步候选仍取消，禁止默认启用。后退与横移结果独立记录。\n')
for file in save.iterdir():
 if file.is_file():
  with file.open('rb') as f:os.fsync(f.fileno())
print('SAVED',save)
