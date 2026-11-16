"""Conservative color-target association and aligned-depth ranging; no robot commands."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def measure(root):
    meta = json.loads((root / 'capture.json').read_text())
    if meta['mode'] != 'sw' or not meta['alignment_requested']:
        raise ValueError('This measurement requires SDK software depth-to-color alignment')
    depth_k = meta['camera_param']['depth']
    rgb_k = meta['camera_param']['rgb']
    if any(abs(depth_k[key]-rgb_k[key]) > .01 for key in ('fx', 'fy', 'cx', 'cy', 'width', 'height')):
        raise ValueError('Aligned depth/RGB intrinsics differ')
    rows = []
    for s in meta['samples']:
        i = s['index']
        rgb = np.fromfile(root / f'frame-{i}-rgb.raw', np.uint8).reshape(s['rgb_height'], s['rgb_width'], 3)
        raw = np.fromfile(root / f'frame-{i}-depth.raw', '<u2').reshape(s['depth_height'], s['depth_width'])
        if rgb.shape[:2] != raw.shape:
            raise ValueError('Depth/RGB shapes differ')
        hsv = np.asarray(Image.fromarray(rgb).convert('HSV'))
        # Track the visually confirmed orange; this is color/geometry association,
        # not a general semantic classifier. Ambiguity blocks motion eligibility.
        mask = ((hsv[..., 0] >= 3) & (hsv[..., 0] <= 30)
                & (hsv[..., 1] > 110) & (hsv[..., 2] > 100))
        labels, count = ndimage.label(mask)
        candidates = []
        for label_id, slices in enumerate(ndimage.find_objects(labels), 1):
            if slices is None:
                continue
            ys, xs = slices
            h, w = ys.stop-ys.start, xs.stop-xs.start
            area = int((labels[slices] == label_id).sum())
            cx, cy = (xs.start+xs.stop-1)/2, (ys.start+ys.stop-1)/2
            if (15 <= area <= 3000 and 5 <= w <= 70 and 4 <= h <= 70
                    and .45 < w/h < 2.5 and 200 < cx < 430 and 210 < cy < 430):
                candidates.append((label_id, [xs.start, ys.start, xs.stop, ys.stop], area, cx, cy))
        row = dict(index=i, target_candidates=len(candidates), direct_range_valid=False,
                   distance_m=None, spatial_alignment='SDK ALIGN_D2C_SW_MODE', hardware_frame_sync=False)
        if len(candidates) == 1:
            label_id, bbox, area, cx, cy = candidates[0]
            inner = ndimage.binary_erosion(labels == label_id, iterations=1)
            valid = inner & (raw > 0)
            z = raw[valid].astype(float)*s['depth_scale_mm']/1000
            row.update(bbox_xyxy=bbox, center_xy=[cx, cy], mask_pixels=area,
                       inner_pixels=int(inner.sum()), valid_depth_pixels=int(valid.sum()),
                       valid_depth_fraction=float(valid.sum()/max(1,inner.sum())))
            if z.size:
                median = float(np.median(z))
                spread = float(np.percentile(z,90)-np.percentile(z,10))
                # A small ray-angle correction converts optical z to range.
                ray = np.array([(cx-rgb_k['cx'])/rgb_k['fx'], (cy-rgb_k['cy'])/rgb_k['fy'], 1.])
                row.update(optical_depth_m=median, depth_spread_10_90_m=spread,
                           distance_m=float(median*np.linalg.norm(ray)))
                row['direct_range_valid'] = bool(z.size >= 10 and row['valid_depth_fraction'] >= .5
                                                and spread < .10 and .4 < median < 4.)
        rows.append(row)
        Image.fromarray(rgb).save(root / f'frame-{i}-rgb.png')
        Image.fromarray(raw).save(root / f'frame-{i}-depth-mm.png')
    values = [r['distance_m'] for r in rows if r['direct_range_valid']]
    accepted = len(values) >= 3 and len(values)/len(rows) >= .75 and np.ptp(values) <= .10
    summary = dict(target='visually confirmed orange, tracked by color and geometry',
                   observation_count=len(rows), valid_direct_ranges=len(values),
                   range_accepted=bool(accepted),
                   distance_m=float(np.median(values)) if accepted else None,
                   candidate_range_median_m=float(np.median(values)) if values else None,
                   acceptance_rule='>=3 valid frames, >=75% frames valid, range spread <=0.10m; no hole filling',
                   target_range_m=1.0, range_is='camera to visible orange surface; center offset not calibrated',
                   samples=rows, motor_commands_written=False)
    (root / 'orange-range.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    measure(parser.parse_args().capture)
