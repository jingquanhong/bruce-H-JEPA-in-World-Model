"""Run one training/validation batch with explicitly synthetic fixture data."""
import os,subprocess
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parents[1];fixture=root/'output/bruce_sim_20261113/runtime_fixture';fixture.mkdir(exist_ok=True)
rng=np.random.default_rng(42)
for split in ['train','val']:
 with h5py.File(fixture/f'runtime_{split}.h5','w') as f:
  f['pixels']=rng.integers(0,256,(160,64,64,3),dtype=np.uint8)
  for key,dim in [('action',8),('xy',2),('qpos',15),('qvel',14)]:f[key]=rng.normal(size=(160,dim)).astype('float32')
  f['ep_len']=np.array([80,80],dtype='int32');f['ep_offset']=np.array([0,80],dtype='int64')
  f['ep_idx']=np.repeat(np.arange(2,dtype='int32'),80);f['step_idx']=np.tile(np.arange(80),2)
args=['/home/user/miniconda3/envs/hjepa/bin/python','main_hjepa.py','--config-name','ant_hjepa_l3',
'data.dataset.name=runtime_train','data.dataset.val_name=runtime_val','data.dataset.val_total_transitions=null',
'loader.batch_size=2','num_workers=0','loader.persistent_workers=false','loader.prefetch_factor=null',
'+trainer.fast_dev_run=true','trainer.devices=1','planning_eval.enabled=false','final_probing_decoding_eval.enabled=false',
'output_model_name=runtime_smoke','subdir=runtime_smoke','resume_every_n_steps=0']
env=os.environ.copy();env['HJEPA_HOME']=str(fixture)
env['OMP_NUM_THREADS']='2'
result=subprocess.run(args,cwd=root/'h_jepa',env=env)
raise SystemExit(result.returncode)
