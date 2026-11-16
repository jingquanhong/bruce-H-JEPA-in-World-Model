from pathlib import Path
import json,os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from PIL import Image
p=Path(__file__).resolve().parent/'newcable-sw12';meta=json.loads((p/'capture.json').read_text());k=meta['camera_param']['rgb'];h,w=480,640
y,x=np.mgrid[:h,:w];rx=(x-k['cx'])/k['fx'];ry=(y-k['cy'])/k['fy']
def load(s):
 i=s['index'];rgb=np.fromfile(p/f'frame-{i}-rgb.raw',np.uint8).reshape(h,w,3)
 z=np.fromfile(p/f'frame-{i}-depth.raw','<u2').reshape(h,w)*s['depth_scale_mm']/1000.
 return rgb,z,np.stack((rx*z,ry*z,z),axis=-1)
s=meta['samples'][-1];rgb,z,xyz=load(s);Image.fromarray(rgb).save(p/'rgb.png');Image.fromarray((z*1000).astype(np.uint16)).save(p/'depth-mm.png')
bright=rgb.max(-1);chroma=rgb.max(-1).astype(float)-rgb.min(-1)
roi=(x>=348)&(x<=451)&(y>=318)&(y<=444)
red=roi&(rgb[:,:,0].astype(float)>1.25*rgb[:,:,1])&(rgb[:,:,0].astype(float)>1.15*rgb[:,:,2])&(rgb[:,:,0]>70)
floor=(y>170)&(x>220)&(x<510)&(~roi)&(bright>50)&(chroma<35)&(z>.2)&(z<3.)
points=xyz[floor][::4];rng=np.random.default_rng(71);best=None
for i in range(250):
 a,b,c=points[rng.choice(len(points),3,replace=False)];n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
 if norm<1e-6:continue
 n/=norm
 if n[1]>0:n=-n
 if n[1]>-.4:continue
 d=-n@a;res=np.abs(points@n+d);inliers=res<.006
 if best is None or inliers.sum()>best[0]:best=(int(inliers.sum()),inliers,n,d)
inliers=points[best[1]];center=inliers.mean(0);_,_,v=np.linalg.svd(inliers-center,full_matrices=False);n=v[-1]
if n[1]>0:n=-n
d=float(-n@center);res=inliers@n+d
rows=[]
for sample in meta['samples']:
 rgb,z,xyz=load(sample)
 red=roi&(rgb[:,:,0].astype(float)>1.25*rgb[:,:,1])&(rgb[:,:,0].astype(float)>1.15*rgb[:,:,2])&(rgb[:,:,0]>70)
 valid=red&(z>.15)&(z<4.)
 heights=xyz[valid]@n+d
 rows.append({'index':sample['index'],'red_pixels':int(red.sum()),'valid_red_depth_pixels':int(valid.sum()),'valid_red_depth_fraction':float(valid.sum()/max(red.sum(),1)), 'optical_depth_median_m':float(np.median(z[valid])) if valid.any() else None,'plane_relative_height_percentiles_m':np.percentile(heights,[10,50,90]).tolist() if valid.any() else None})
report={'source':'Current camera capture after cable replacement, not historical images','pairs':len(rows),'resolution':[w,h],'last_frame_valid_depth_fraction':float((z>0).mean()),'book_roi_xyxy':[348,318,451,444],'roi_source':'Manual association of visible red book in current RGB image; not a general object detector','floor_fit':{'normal_camera':n.tolist(),'offset_m':d,'inliers':len(inliers),'residual_rms_m':float(np.sqrt(np.mean(res**2)))},'book_frames':rows,'camera_to_body_extrinsic_calibrated':False,'motion_enabled':False,'note':'Software RGB-depth alignment and plane estimate only. No external height/distance verification; optical depth is not foot-to-obstacle distance.'}
(p/'book-depth-quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'pairs':len(rows),'floor':report['floor_fit'],'book_frame_last':rows[-1]},indent=2))
