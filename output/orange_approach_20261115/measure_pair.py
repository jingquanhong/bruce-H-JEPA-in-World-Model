"""Range two visually assigned objects independently; never write robot commands."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage


def component_near(labels, seed, radius=18):
    choices = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = sl
        width, height = xs.stop-xs.start, ys.stop-ys.start
        area = int((labels[sl] == label_id).sum())
        center = np.array([(xs.start+xs.stop-1)/2, (ys.start+ys.stop-1)/2])
        distance = float(np.linalg.norm(center-np.asarray(seed)))
        if 20 <= area <= 6000 and 5 <= width <= 95 and 4 <= height <= 95 and distance < radius:
            mask = labels == label_id
            choices.append((distance, mask, center, [xs.start, ys.start, xs.stop, ys.stop]))
    choices.sort(key=lambda value: value[0])
    if not choices or (len(choices)>1 and choices[1][0]-choices[0][0] < 4):
        return None
    return choices[0]


def surface_range(component, depth, intrinsic):
    if component is None:
        return dict(valid=False, reason='missing or ambiguous color association')
    _, mask, center, bbox = component
    inner = ndimage.binary_erosion(mask, iterations=1)
    valid = inner & (depth > 0)
    values = depth[valid]
    result = dict(valid=False, center_xy=center.tolist(), bbox_xyxy=bbox,
                  inner_pixels=int(inner.sum()), depth_pixels=int(valid.sum()))
    if values.size:
        median = float(np.median(values))
        spread = float(np.percentile(values, 90)-np.percentile(values, 10))
        ray = np.array([(center[0]-intrinsic['cx'])/intrinsic['fx'],
                        (center[1]-intrinsic['cy'])/intrinsic['fy'], 1.])
        result.update(optical_depth_m=median, range_m=float(median*np.linalg.norm(ray)),
                      spread_m=spread, valid_fraction=float(valid.sum()/max(1,inner.sum())))
        result['valid'] = bool(values.size >= 10 and result['valid_fraction'] >= .7
                               and spread <= .10 and .4 < median < 4.)
    return result


def consensus(rows):
    good = [row for row in rows if row['paired_valid']]
    values = np.array([row['fruit']['range_m'] for row in good])
    span = (good[-1]['host_monotonic_ns']-good[0]['host_monotonic_ns'])/1e9 if good else 0.
    last_gap = (rows[-1]['host_monotonic_ns']-good[-1]['host_monotonic_ns'])/1e9 if good else None
    accepted = bool(len(good) >= 4 and span >= .5 and last_gap <= .5 and np.ptp(values) <= .08)
    return dict(range_accepted=accepted, distance_m=float(np.median(values)) if accepted else None,
                lower_range_m=float(values.min()-.05) if accepted else None,
                paired_valid_frames=len(good), total_frames=len(rows),
                valid_fraction=len(good)/len(rows), valid_span_s=span, last_valid_gap_s=last_gap,
                range_spread_m=float(np.ptp(values)) if len(values) else None,
                reason='Four independent, temporally separated fruit/reference readings must agree; invalid frames are excluded, not filled.')


def measure(root, fruit_seed, reference_seed):
    meta = json.loads((root/'capture.json').read_text())
    if meta['mode'] != 'sw' or not meta['alignment_requested']:
        raise ValueError('SDK software-aligned RGB-D required')
    intrinsic = meta['camera_param']['rgb']
    for key in ('fx','fy','cx','cy','width','height'):
        if abs(intrinsic[key]-meta['camera_param']['depth'][key]) > .01:
            raise ValueError('Aligned intrinsics differ')
    rows = []
    for sample in meta['samples']:
        i = sample['index']
        rgb = np.fromfile(root/f'frame-{i}-rgb.raw',np.uint8).reshape(sample['rgb_height'],sample['rgb_width'],3)
        depth = np.fromfile(root/f'frame-{i}-depth.raw','<u2').reshape(sample['depth_height'],sample['depth_width'])
        if rgb.shape[:2] != depth.shape:
            raise ValueError('Aligned image shapes differ')
        depth = depth.astype(float)*sample['depth_scale_mm']/1000
        hsv = np.asarray(Image.fromarray(rgb).convert('HSV'))
        color = (((hsv[...,0] < 30)|(hsv[...,0] > 245)) & (hsv[...,1] > 90) & (hsv[...,2] > 100))
        color = ndimage.binary_closing(color, structure=np.ones((3,3)))
        labels, _ = ndimage.label(color)
        fruit_component = component_near(labels,fruit_seed)
        ref_component = component_near(labels,reference_seed)
        distinct = (fruit_component is not None and ref_component is not None
                    and np.linalg.norm(fruit_component[2]-ref_component[2]) > 12
                    and not np.any(fruit_component[1]&ref_component[1]))
        fruit = surface_range(fruit_component,depth,intrinsic)
        reference = surface_range(ref_component,depth,intrinsic)
        paired = (distinct and fruit['valid'] and reference['valid']
                  and abs(fruit['optical_depth_m']-reference['optical_depth_m']) <= .10)
        rows.append(dict(index=i,host_monotonic_ns=sample['host_monotonic_ns'],
                         fruit=fruit,reference=reference,paired_valid=bool(paired)))
        if i in (0,8,len(meta['samples'])-1):
            Image.fromarray(rgb).save(root/f'frame-{i}-rgb.png')
    summary = consensus(rows)
    summary.update(measured_at_monotonic=time.monotonic(),
                   last_frame_monotonic_ns=rows[-1]['host_monotonic_ns'],
                   fruit_seed_xy=list(fruit_seed), reference_seed_xy=list(reference_seed),
                   target='orange fruit; printed reference is independently associated to its left',
                   range_is='camera to visible fruit surface, not base/feet or calibrated fruit center',
                   spatial_alignment='factory SDK software D2C; inspected against visible edges',
                   hardware_frame_sync=False, motion_commands_written=False, samples=rows)
    (root/'pair-range.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='samples'},ensure_ascii=False,indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--fruit-seed',nargs=2,type=float,default=[315,268])
    parser.add_argument('--reference-seed',nargs=2,type=float,default=[286,264])
    args=parser.parse_args()
    measure(args.capture,args.fruit_seed,args.reference_seed)
