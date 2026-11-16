#!/usr/bin/env python3
"""Validate BRUCE collection structure, action timing and actuation provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

import bruce_signals as bs
from qualification import qualify


def load_meta(f):
    return [json.loads(m.decode() if isinstance(m, bytes) else m) for m in f['ep_meta'][:]]


def validate(path):
    checks = []

    def add(name, passed, detail=''):
        checks.append(dict(name=name, passed=bool(passed), detail=str(detail)))

    with h5py.File(path, 'r') as f:
        metas = load_meta(f)
        lens, offsets = f['ep_len'][:], f['ep_offset'][:]
        total = int(lens.sum())
        add('nonempty_episodes', len(lens) > 0 and np.all(lens > 0), lens.tolist())
        expected = np.r_[0, np.cumsum(lens[:-1])]
        add('episode_offsets', np.array_equal(offsets, expected))
        add('metadata_count', len(metas) == len(lens))
        for key, dim in bs.ALL_COLUMNS.items():
            add('shape_' + key, key in f and f[key].shape == (total, dim))
        for key in f:
            if key in ('ep_meta', 'ep_len', 'ep_offset'):
                continue
            add('length_' + key, len(f[key]) == total)
            if np.issubdtype(f[key].dtype, np.number):
                add('finite_' + key, np.isfinite(f[key][:]).all())
        for i, (off, n) in enumerate(zip(offsets, lens)):
            sl = slice(int(off), int(off + n))
            add(f'episode_{i}_indices', np.array_equal(f['step_idx'][sl], np.arange(n))
                and np.all(f['ep_idx'][sl] == i))
            if 'timestamp' in f:
                dt = np.diff(f['timestamp'][sl])
                add(f'episode_{i}_timestamps', np.allclose(dt, 1 / metas[i]['record_hz'], atol=1e-9))
        cfg = json.loads(f.attrs.get('collection_config', '{}'))
        threshold = cfg.get('sensor', {}).get('foot_contact_force_thresh', 5.0)
        add('binary_contact_proxy', np.array_equal(f['foot_contact'][:],
            (f['sim_contact_force'][:] > threshold).astype(float)))
        tau = f['joint_torque_est'][:]
        add('torque_estimate_bound', np.abs(tau).max() <= bs.TORQUE_MAX + 1e-8)
        mode = str(f.attrs.get('control_mode', 'legacy_unspecified'))
        if 'sim_reference' in f:
            add('recorded_action_equals_executed_hold_target',
                np.array_equal(f['action'][:], f['sim_reference'][:, 6:16]))
        if mode == 'joint_only':
            forces = f['sim_control_force'][:]
            add('unactuated_base', np.abs(forces[:, :6]).max() < 1e-6)
            add('leg_actuator_force_bound', np.abs(forces[:, 6:16]).max() <= 10.5 + 1e-4)
        if 'sim_body_acc' in f:
            expected = np.stack([bs.quat_to_mat(q).T @ (a + [0, 0, bs.G])
                                for q, a in zip(f['sim_base_quat'][:], f['sim_body_acc'][:])])
            imu_err = np.abs(f['imu_acc'][:] - expected)
            add('imu_specific_force_transform', imu_err.max() < 0.2, f'max error {imu_err.max():.5f} m/s²')
        summary = {}
        qualifications = []
        for m, off, n in zip(metas, offsets, lens):
            sl = slice(int(off), int(off + n))
            qualifications.append(qualify(m, f['sim_control_force'][sl]))
        for m, qualification in zip(metas, qualifications):
            row = summary.setdefault(m['scene'], dict(episodes=0, success=0,
                historical_criterion_passed=0, fell=0, kicked=0, completed=0))
            row['episodes'] += 1
            row['success'] += int(qualification['autonomous_success'])
            row['historical_criterion_passed'] += int(m.get('reference_criterion_passed', m['success']))
            for k in ('fell', 'kicked', 'completed'):
                row[k] += int(m.get(k, False))
        report = dict(dataset=str(Path(path).resolve()), control_mode=mode,
                      schema_version=str(f.attrs.get('schema_version', 'legacy')),
                      episodes=len(lens), frames=total,
                      duration_seconds=sum(int(n) / m['record_hz'] for n, m in zip(lens, metas)),
                      collection_valid=all(c['passed'] for c in checks),
                      success=sum(q['autonomous_success'] for q in qualifications),
                      historical_criterion_passed=sum(m.get('reference_criterion_passed', m['success']) for m in metas),
                      autonomous_qualification=qualifications,
                      visual_hjepa_ready=False,
                      scenes=summary, checks=checks,
                      notes=['Collection validity and obstacle success are separate.',
                             'Historical success flags are trajectory criteria, not verified autonomous successes.',
                             'No episode has verified full-mesh clearance and stable landing.',
                             'est_* are synthetic truth-plus-noise proxies, not EKF output.',
                             'clearance is an ankle-based geometric proxy, not full-mesh distance.'])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='bruce_stepover_50')
    ap.add_argument('--cache-dir', default=str(Path.home() / '.stable_worldmodel'))
    ap.add_argument('--json')
    args = ap.parse_args()
    report = validate(Path(args.cache_dir) / (args.dataset + '.h5'))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('checks', 'autonomous_qualification')}, ensure_ascii=False, indent=2))
    for check in report['checks']:
        if not check['passed']:
            print('FAIL:', check['name'], check['detail'])
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['collection_valid'] else 1)


if __name__ == '__main__':
    main()
