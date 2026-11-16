#!/usr/bin/env python3
"""Bruce 采集结果的静态图（matplotlib PNG，非视频）。

用法:
  python make_figures.py --dataset bruce_stepover_50 [--outdir figures]

产出:
  traj_xz.png       每场景的 x-z 剖面（基座 / 双踝 / 障碍箱 / 最小净空标注）
  timeseries.png    单条 episode 的信号时序（IMU / 足底 / 腿角 / 力矩估计）
  dataset_stats.png 逐场景成功率、净空分布、抬脚余量-净空关系、基座位移轨迹
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bruce_expert as bx

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_BASE = "#3b6ea5"
C_R = "#d1495b"
C_L = "#2a9d8f"
C_BOX = "#8d99ae"
C_OK = "#2a9d8f"
C_BAD = "#d1495b"


def load(dataset, cache):
    path = Path(cache).expanduser() / f"{dataset}.h5"
    with h5py.File(path, "r") as f:
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        cols = {k: f[k][:] for k in
                ("sim_body_pos", "sim_ankle_pos", "imu_acc", "foot_contact",
                 "leg_q", "joint_torque_est", "action", "sim_contact_force")}
        metas = [json.loads(m.decode() if isinstance(m, bytes) else str(m))
                 for m in f["ep_meta"][:]]
    return path, ep_len, ep_off, cols, metas


def ep_slice(ep_off, ep_len, ep):
    o = int(ep_off[ep])
    return slice(o, o + int(ep_len[ep]))


# ------------------------------------------------------------------ 图 1: x-z 剖面
def fig_traj_xz(path, ep_len, ep_off, cols, metas, outdir):
    scenes = []
    for m in metas:
        if m["scene"] not in scenes:
            scenes.append(m["scene"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax, sid in zip(axes.ravel(), scenes):
        ep = next((i for i, m in enumerate(metas) if m["scene"] == sid and m["success"]),
                  next(i for i, m in enumerate(metas) if m["scene"] == sid))
        m = metas[ep]
        s = ep_slice(ep_off, ep_len, ep)
        base = cols["sim_body_pos"][s]
        ak = cols["sim_ankle_pos"][s]
        ax.plot(base[:, 0], base[:, 2], "-", color=C_BASE, lw=2.0, label="base")
        ax.plot(ak[:, 0], ak[:, 1], "-", color=C_R, lw=1.6, label="ankle R")
        ax.plot(ak[:, 2], ak[:, 3], "--", color=C_L, lw=1.6, label="ankle L")
        h = float(m["h_obs"])
        cx, hx = float(m["box_cx"]), float(m["box_hx"])
        ax.add_patch(plt.Rectangle((cx - hx, 0.0), 2 * hx, h,
                                   fc=C_BOX, ec="#4a5568", alpha=0.55, lw=1.2))
        ax.axhline(h, color="#4a5568", ls=":", lw=1.0)
        # 最小净空点标注（与 ep_meta 同口径：限定在摆动窗口内）
        x_lo, x_hi = cx - hx - bx.TOE, cx + hx + bx.HEEL
        fs = float(m["record_hz"])
        times = m["t_start"] + np.arange(int(ep_len[ep])) / fs
        best, bp = np.inf, None
        for side, a, b in m["swing"]:
            xi = 0 if side == "r" else 2
            zi = 1 if side == "r" else 3
            sel = ((times >= a) & (times <= b)
                   & (ak[:, xi] >= x_lo) & (ak[:, xi] <= x_hi))
            for x, z in zip(ak[sel, xi], ak[sel, zi]):
                c = z - bx.SOLE - h
                if c < best:
                    best, bp = c, (x, z, side.upper())
        if bp is not None:
            ax.plot(bp[0], bp[1], "o", ms=7, mfc="none", mec=C_BAD, mew=2.0)
            ax.annotate(f"{bp[2]} 净空 {best*1000:+.0f}mm",
                        xy=bp[:2], xytext=(8, 12), textcoords="offset points",
                        fontsize=9, color=C_BAD,
                        arrowprops=dict(arrowstyle="-", color=C_BAD, lw=0.9))
        ax.set_title(f"{sid}   {m['kind']}   箱高 {h*100:.0f}cm   "
                     f"净空 {m['clearance']*1000:+.1f}mm  "
                     f"{'成功' if m['success'] else '失败'}",
                     fontsize=10.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Bruce 越障轨迹（踝原点；辅助模式须独立解读）", fontsize=13)
    fig.tight_layout()
    p = outdir / "traj_xz.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 图 2: 时序
def fig_timeseries(path, ep_len, ep_off, cols, metas, outdir):
    ep = next((i for i, m in enumerate(metas)
               if m["scene"] == "box_06" and m["success"]), 0)
    m = metas[ep]
    s = ep_slice(ep_off, ep_len, ep)
    fs = float(m["record_hz"])
    t = m["t_start"] + np.arange(int(ep_len[ep])) / fs
    imu = cols["imu_acc"][s]
    fc = cols["foot_contact"][s]
    q = cols["leg_q"][s]
    tau = cols["joint_torque_est"][s]

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.2), sharex=True)
    ax = axes[0]
    for i, (c, lab) in enumerate(zip(["#d1495b", "#2a9d8f", "#3b6ea5"],
                                     ["acc_x", "acc_y", "acc_z"])):
        ax.plot(t, imu[:, i], lw=1.0, color=c, label=lab)
    ax.axhline(9.81, color="#888", ls="--", lw=0.9, label="+g")
    ax.axhline(-9.81, color="#888", ls="--", lw=0.9)
    ax.set_ylabel("IMU 比力 (m/s²)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.25)
    ax.set_title(f"episode {ep}  场景 {m['scene']}  箱高 {m['h_obs']*100:.0f}cm  "
                 f"净空 {m['clearance']*1000:+.1f}mm", fontsize=11)

    ax = axes[1]
    for i, lab in enumerate(["R_TOE", "R_HEEL", "L_TOE", "L_HEEL"]):
        ax.fill_between(t, i, i + fc[:, i], step="post", alpha=0.75,
                        color=["#d1495b", "#e07a5f", "#2a9d8f", "#7fb3ae"][i],
                        label=lab)
    ax.set_ylabel("足底接触 (0/1)")
    ax.set_yticks([])
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[2]
    names = ["HipPitch_R", "Knee_R", "Ankle_R", "HipPitch_L", "Knee_L", "Ankle_L"]
    sel = [1, 3, 4, 6, 8, 9]
    for i, lab in zip(sel, names):
        ax.plot(t, q[:, i], lw=1.0, label=lab)
    ax.set_ylabel("腿关节角 (rad)")
    ax.legend(fontsize=8, ncol=6)
    ax.grid(alpha=0.25)

    ax = axes[3]
    tau_names = ["HipY_R", "HipP_R", "HipR_R", "Knee_R", "Ank_R",
                 "HipY_L", "HipP_L", "HipR_L", "Knee_L", "Ank_L"]
    for i in range(tau.shape[1]):
        ax.plot(t, tau[:, i], lw=0.9, label=tau_names[i])
    ax.set_ylabel("力矩估计 (N·m)")
    ax.set_xlabel("时间 (s)")
    ax.legend(fontsize=7, ncol=10)
    ax.grid(alpha=0.25)
    fig.suptitle("Bruce 越障采集信号时序（实物端可采集量）", fontsize=13)
    fig.tight_layout()
    p = outdir / "timeseries.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 图 3: 数据集统计
def fig_stats(path, ep_len, ep_off, cols, metas, outdir):
    scenes = []
    for m in metas:
        if m["scene"] not in scenes:
            scenes.append(m["scene"])
    clr = np.array([m["clearance"] for m in metas]) * 1000.0
    succ = np.array([m["success"] for m in metas])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))

    ax = axes[0, 0]
    rate = [succ[[m["scene"] == s for m in metas]].mean() * 100 for s in scenes]
    ax.bar(range(len(scenes)), rate,
           color=[C_OK if r >= 80 else ("#e9c46a" if r >= 50 else C_BAD)
                  for r in rate])
    for i, r in enumerate(rate):
        ax.text(i, r + 2, f"{r:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("逐场景成功率", fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    data = [clr[[m["scene"] == s for m in metas]] for s in scenes]
    bp = ax.boxplot(data, tick_labels=scenes, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f7")
        patch.set_edgecolor("#4a5568")
    for key in ("medians", "whiskers", "caps"):
        for line in bp[key]:
            line.set_color("#4a5568")
    ax.axhline(0.0, color=C_BAD, ls="--", lw=1.2, label="净空 = 0")
    ax.set_ylabel("越障净空 (mm)")
    ax.set_title("逐场景净空分布", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(axis="x", labelsize=9)

    ax = axes[1, 0]
    le = np.array([m.get("lift_extra") or 0.0 for m in metas])
    for s in scenes:
        msk = [m["scene"] == s for m in metas]
        ax.scatter(le[msk] * 1000, clr[msk],
                   c=[C_OK if v else C_BAD for v in succ[msk]],
                   s=26, alpha=0.85, label=s)
    ax.axhline(0.0, color="#888", ls="--", lw=1.0)
    ax.set_xlabel("抬脚净空余量 lift_extra (mm)")
    ax.set_ylabel("实测越障净空 (mm)")
    ax.set_title("抬脚余量 → 实际净空（绿=成功 / 红=失败）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    for i, m in enumerate(metas):
        s = ep_slice(ep_off, ep_len, i)
        bx_ = cols["sim_body_pos"][s][:, 0]
        ax.plot(bx_, np.full(len(bx_), i), "-", lw=1.2,
                color=C_OK if m["success"] else C_BAD, alpha=0.8)
    ax.set_xlabel("基座 x (m)")
    ax.set_ylabel("episode")
    ax.set_title("逐集基座位移（绿=成功 / 红=失败）", fontsize=11)
    ax.grid(alpha=0.25)

    mode = "基座辅助参考" if metas[0].get("base_actuated") else "自由基座尝试"
    fig.suptitle(f"Bruce {mode}（{len(metas)} 集，"
                 f"成功率 {succ.mean()*100:.0f}%）", fontsize=13)
    fig.tight_layout()
    p = outdir / "dataset_stats.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description="Bruce 采集结果静态图")
    ap.add_argument("--dataset", default="bruce_stepover_50")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".stable_worldmodel"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "figures"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path, ep_len, ep_off, cols, metas = load(args.dataset, args.cache_dir)
    print(f"读取 {path}  {len(metas)} 集")
    for fn in (fig_traj_xz, fig_timeseries, fig_stats):
        p = fn(path, ep_len, ep_off, cols, metas, outdir)
        print("  ->", p)


if __name__ == "__main__":
    main()
