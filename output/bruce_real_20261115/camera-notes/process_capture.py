"""Process saved Orbbec captures offline. Does not access motor or camera ports."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();root=a.folder
 m=json.loads((root/'capture.json').read_text());w,h=m['depth_width'],m['depth_height']
 raw=np.fromfile(root/'depth.raw',dtype='<u2').reshape(h,w)
 depth=raw.astype(np.float32)*m['depth_scale_mm']/1000.
 valid=(raw>0)&np.isfinite(depth)
 if not valid.any():raise RuntimeError('No valid depth values')
 z=depth[valid];near,far=float(np.percentile(z,2)),float(np.percentile(z,98))
 far=max(far,near+.01)
 anchors=np.array([[253,231,37],[94,201,98],[33,145,140],[59,82,139],[68,1,84]],float)
 t=np.clip((depth-near)/(far-near),0,1)*4
 colors=np.stack([np.interp(t,np.arange(5),anchors[:,c]) for c in range(3)],axis=-1).astype('uint8');colors[~valid]=0
 Image.fromarray(np.rint(depth*1000).clip(0,65535).astype('uint16')).save(root/'depth_mm.png')
 Image.fromarray(colors).save(root/'depth_preview.png')
 if m.get('color_format')==22:
  rgb=np.fromfile(root/'color.raw',dtype='uint8').reshape(m['color_height'],m['color_width'],3);image=Image.fromarray(rgb)
 elif m.get('color_format')==5:image=Image.open(root/'color.raw').convert('RGB')
 else:raise RuntimeError('Unsupported color format for this processor')
 image.save(root/'color.png')
 canvas=Image.new('RGB',(1280,590),(22,26,34));canvas.paste(image.resize((640,480)),(0,45));canvas.paste(Image.fromarray(colors),(640,85));d=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 except OSError:font=ImageFont.load_default()
 d.text((16,12),'RGB 640 x 480',font=font,fill='white');d.text((656,12),'Depth 640 x 400 - not RGB registered',font=font,fill='white')
 for x in range(500):
  c=tuple(int(np.interp(x/499*4,np.arange(5),anchors[:,i])) for i in range(3));d.line((700+x,525,700+x,544),fill=c)
 d.text((700,550),f'near {near:.2f} m',font=font,fill='white');d.text((1030,550),f'far {far:.2f} m',font=font,fill='white');d.text((16,550),'Black depth pixels: no valid measurement',font=font,fill='white');canvas.save(root/'rgb_depth_preview.png')
 roi=depth[h//2-15:h//2+15,w//2-15:w//2+15];roi_valid=roi[roi>0]
 stats={'device':m['device'],'depth_resolution':[w,h],'color_resolution':[m.get('color_width'),m.get('color_height')], 'depth_fps_observed':m['depth_frames']/m['duration_seconds'],'color_fps_observed':m['color_frames']/m['duration_seconds'],'valid_depth_fraction':float(valid.mean()),'nearest_pixel_m':float(z.min()),'depth_percentiles_m':{str(q):float(np.percentile(z,q)) for q in (2,10,50,90,98)},'center_30x30_median_m':float(np.median(roi_valid)) if roi_valid.size else None,'center_30x30_valid_fraction':float(roi_valid.size/roi.size),'pixels_under_1m':int(((depth>0)&(depth<1)).sum()),'pointcloud_points':int(valid.sum()),'depth_unit':'PNG pixel values are millimeters; 0 is invalid','note':'Depth/RGB are unregistered; frame sync not supported by this camera. Distances refer to the captured scene, not calibrated robot obstacle distances.'}
 (root/'analysis.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
