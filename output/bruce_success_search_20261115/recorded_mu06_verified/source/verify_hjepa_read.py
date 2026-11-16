#!/usr/bin/env python3
"""验证 BRUCE 数据的 HDF5 读取与动作对齐；不等同于模型训练适配完成。

检查:
  1. HDF5Dataset 能打开 bruce_stepover_50.h5 并枚举 clip
  2. keys_to_load 白名单生效：sim_* 仿真真值列不进模型
  3. keys_to_merge 拼出的 proprio 维度正确
  4. level1 的 frameskip/num_steps 抽帧窗口形状正确
  5. 抽出的样本数值与数据集一致（action 对齐、无 NaN）

用法:
  python verify_hjepa_read.py --dataset bruce_stepover_50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from stable_worldmodel.data.dataset import HDF5Dataset  # noqa: E402

# 实物端可采集的列（sim_* 一律不进训练）
TRAIN_KEYS = [
    "action", "imu_acc", "imu_gyro", "foot_contact",
    "leg_q", "leg_dq", "joint_torque_est",
]
# proprio 只合并有实物读取链路的代理字段，不包含真值加噪的 est_yaw。
PROPRIO_SRC = ["leg_q", "leg_dq", "imu_acc", "imu_gyro", "joint_torque_est"]
PROPRIO_DIM = 10 + 10 + 3 + 3 + 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bruce_stepover_50")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".stable_worldmodel"))
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=4)
    args = ap.parse_args()

    ds = HDF5Dataset(
        name=args.dataset,
        keys_to_load=TRAIN_KEYS,
        keys_to_cache=["action"],
        keys_to_merge={"proprio": PROPRIO_SRC},
        cache_dir=args.cache_dir,
        level1={"frameskip": args.frameskip, "num_steps": args.num_steps},
    )

    print("== H-JEPA 读入验证 ==")
    print(f"  h5            : {ds.h5_path}")
    print(f"  episodes      : {len(ds.lengths)}")
    print(f"  总帧数        : {int(ds.lengths.sum())}")
    print(f"  训练窗口数    : {len(ds.clip_indices)}")
    print(f"  读入列        : {ds.column_names}")
    print(f"  proprio 维度  : {ds.get_dim('proprio')} (期望 {PROPRIO_DIM})")

    sim_leak = [k for k in ds.column_names if k.startswith(("sim_", "est_"))]
    print(f"  sim_*/est_* 列泄漏  : {sim_leak if sim_leak else '无（符合预期）'}")

    sample = ds[0]
    print("\n  单样本形状：")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"    {k:<18s} {tuple(v.shape)}")
        else:
            print(f"    {k:<18s} {type(v).__name__}")

    ok = True
    checks = [
        ("proprio 维度正确", ds.get_dim("proprio") == PROPRIO_DIM),
        ("无 sim_*/est_* 泄漏", not sim_leak),
        ("clip 窗口非空", len(ds.clip_indices) > 0),
        ("action 形状 (num_steps, frameskip*10)",
         sample["action_level1"].shape == (args.num_steps, args.frameskip * 10)),
        ("proprio 形状 (num_steps, 36)",
         sample["proprio_level1"].shape == (args.num_steps, PROPRIO_DIM)),
        ("数值无 NaN", all(np.all(np.isfinite(np.asarray(v)))
                          for v in sample.values() if hasattr(v, "shape"))),
    ]
    print()
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    # action 与数据集原始行对齐抽查（第 1 帧的 5 个动作 = 原始行 start..start+4）
    raw_action = ds._cache["action"]
    ep, start = ds.clip_indices[0]
    ok &= bool(np.allclose(np.asarray(sample["action_level1"])[0].reshape(-1, 10),
                           raw_action[ds.offsets[ep] + start:
                                      ds.offsets[ep] + start + args.frameskip]))
    print(f"  [{'PASS' if ok else 'FAIL'}] action 与原始行对齐")

    print(f"\n结论: {'HDF5 读取与动作对齐检查通过' if ok else '存在问题'}")
    print("本检查不代表视觉 H-JEPA 已完成训练适配：本数据未包含逐帧 pixels。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
