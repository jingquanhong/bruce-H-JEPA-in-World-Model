#!/usr/bin/env python3
"""Create a clearly labelled video comparison, contact sheet and collection summary."""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCENES = ['box_04', 'box_06', 'box_08', 'step_10']
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def load_meta(path):
    with h5py.File(path) as f:
        return [json.loads(v) for v in f['ep_meta'][:]]


def labelled(frame, title, subtitle):
    im = Image.fromarray(frame).convert('RGB')
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.width, 58), fill='#101d30')
    draw.text((12, 5), title, fill='white', font=ImageFont.truetype(FONT_PATH, 18))
    draw.text((12, 31), subtitle, fill='#b3c7df', font=ImageFont.truetype(FONT_PATH, 14))
    return np.asarray(im)


def read_frames(path):
    reader = imageio.get_reader(str(path))
    frames = list(reader.iter_data())
    reader.close()
    return frames


def create(root, delivery=None):
    out = root / 'presentation'
    out.mkdir(exist_ok=True)
    assisted = load_meta(root / 'bruce_assisted_reference_50.h5')
    joint = load_meta(root / 'bruce_joint_trials_40.h5')
    rows = []
    for mode, metas in [('assisted', assisted), ('joint_only', joint)]:
        for sid in SCENES:
            selected = [m for m in metas if m['scene'] == sid]
            rows.append(dict(mode=mode, scene=sid, episodes=len(selected),
                             historical_criterion_passed=sum(m.get('reference_criterion_passed', m['success']) for m in selected),
                             autonomous_success=0,
                             falls=sum(m['fell'] for m in selected)))
    (out / 'summary.json').write_text(json.dumps(rows, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, label, color, shift in [('assisted', 'Assisted reference', '#259c8e', -.18),
                                       ('joint_only', 'Unactuated-base trials', '#d25b55', .18)]:
        subset = [r for r in rows if r['mode'] == mode]
        vals = [100 * r['historical_criterion_passed'] / r['episodes'] for r in subset]
        axes[0].bar(np.arange(4) + shift, vals, width=.34, label=label, color=color)
        for i, (r, value) in enumerate(zip(subset, vals)):
            axes[0].text(i + shift, value + 2, f"{r['historical_criterion_passed']}/{r['episodes']}", ha='center', fontsize=9)
    axes[0].set(xticks=np.arange(4), xticklabels=SCENES, ylabel='Completion criterion passed (%)', ylim=(0, 115))
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Legacy trajectory criterion only | verified autonomous successes: 0')
    with h5py.File(root / 'bruce_joint_trials_40.h5') as f:
        for i, m in enumerate(joint):
            off, n = int(f['ep_offset'][i]), int(f['ep_len'][i])
            axes[1].plot(np.arange(n) / m['record_hz'], f['sim_body_pos'][off:off+n, 2], color='#d25b55', alpha=.25)
    axes[1].set(xlabel='Recorded time (s)', ylabel='Base height (m)', title='Unactuated-base trials: loss of balance is retained')
    for ax in axes:
        ax.grid(alpha=.2, axis='y')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('BRUCE obstacle collection | Genesis | 100 Hz sensor records', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'overview.png', dpi=160)
    plt.close(fig)

    sheet = Image.new('RGB', (1280, 4 * 270 + 70), '#101d30')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 16), 'BRUCE | Assisted reference vs unactuated-base trials',
              fill='white', font=ImageFont.truetype(FONT_PATH, 26))
    writer = imageio.get_writer(str(out / 'comparison.mp4'), fps=25, codec='libx264', quality=7)
    for i, sid in enumerate(SCENES):
        a = read_frames(root / 'assisted_videos' / f'episode_{i:03d}.mp4')
        b = read_frames(root / 'joint_videos' / f'episode_{i:03d}.mp4')
        am = assisted[i]
        for k in range(max(len(a), len(b))):
            left = labelled(a[min(k, len(a)-1)], f'{sid} | EXTERNAL BASE ACTUATION',
                            'INVALID as autonomous crossing | 8cm collision confirmed' if sid == 'box_08'
                            else 'Motion reference ONLY | not an autonomous success')
            right = labelled(b[min(k, len(b)-1)], f'{sid} | JOINT-ONLY PILOT',
                             'Leg targets only | trial FAIL | last frame held after termination' if k >= len(b)
                             else 'Leg targets only | no hidden base actuation')
            writer.append_data(np.concatenate((left, right), axis=1))
        li = labelled(a[min(int(len(a)*.65), len(a)-1)], sid + ' | ASSISTED',
                      'Base servo ON; motion reference only')
        ri = labelled(b[-1], sid + ' | JOINT-ONLY PILOT', 'Leg-only trial; failure retained')
        sheet.paste(Image.fromarray(li).resize((640, 270)), (0, 70 + 270 * i))
        sheet.paste(Image.fromarray(ri).resize((640, 270)), (640, 70 + 270 * i))
    writer.close()
    sheet.save(out / 'contact_sheet.png')
    for name in ['comparison.mp4', 'contact_sheet.png', 'overview.png']:
        print(out / name)
    if delivery is not None:
        target = delivery / 'BRUCE分场景视频'
        archive = target / '历史未更正视频'
        archive.mkdir(parents=True, exist_ok=True)
        names = ['4cm障碍', '6cm障碍', '8cm障碍', '10cm台阶']
        for mode, prefix, directory in [('assisted', '基座辅助', 'assisted_videos'),
                                         ('joint', '自由基座先导', 'joint_videos')]:
            for i, name in enumerate(names):
                filename = f'{prefix}_{name}.mp4'
                dest = target / filename
                if dest.exists() and not (archive / filename).exists():
                    shutil.copy2(dest, archive / filename)
                title = 'EXTERNAL BASE ACTUATION | REFERENCE ONLY' if mode == 'assisted' else 'UNACTUATED BASE | FAILED PILOT'
                subtitle = ('8cm: COLLISION CONFIRMED | NOT autonomous success' if i == 2 else
                            'INVALID as autonomous success | base servo ON') if mode == 'assisted' else 'Failure retained | no successful obstacle crossing'
                writer = imageio.get_writer(str(dest), fps=25, codec='libx264', quality=8)
                try:
                    for frame in read_frames(root / directory / f'episode_{i:03d}.mp4'):
                        writer.append_data(labelled(frame, title, subtitle))
                finally:
                    writer.close()
        comparison = delivery / 'BRUCE仿真对比.mp4'
        if comparison.exists() and not (archive / comparison.name).exists():
            shutil.copy2(comparison, archive / comparison.name)
        shutil.copy2(out / 'comparison.mp4', comparison)
        (archive / '请先阅读.txt').write_text(
            '这些是更正前的历史视频，只保留用于追溯。辅助视频不能证明自主越障；8cm片段已经碰箱失败。\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--delivery-dir', type=Path)
    args = ap.parse_args()
    create(args.directory, args.delivery_dir)
