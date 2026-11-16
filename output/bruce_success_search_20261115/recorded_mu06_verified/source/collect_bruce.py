#!/usr/bin/env python3
"""Reproducible BRUCE obstacle trials with explicit actuation and sensor provenance.

Use --config config/bruce_stepover.yaml. joint_only leaves the base unactuated;
assisted is an explicitly labelled trajectory reference, not deployable control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bruce_env as be            # noqa: E402
import bruce_expert as bx         # noqa: E402
import bruce_signals as bs        # noqa: E402
from qualification import qualify  # noqa: E402

FOOT_PARTS = be.FOOT_PARTS


def json_safe(value):
    """Missing geometric measurements are JSON null, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


# ================================================================ HDF5 写出
class EpisodeWriter:
    """复刻 world.py 的 HDF5 契约：懒建列 + 整集追加 + 断点续采。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        kwargs = {"name": str(self.path),
                  "mode": "a" if exists else "w",
                  "libver": "latest"}
        if not exists:
            kwargs.update({"fs_strategy": "page", "fs_page_size": 4 * 1024 * 1024})
        self.f = h5py.File(**kwargs)


        if "ep_len" in self.f:                      # 续采
            self.n_ep = int(self.f.attrs.get("committed_episodes", len(self.f["ep_len"])))
            self.ptr = (int(self.f["ep_offset"][self.n_ep - 1] + self.f["ep_len"][self.n_ep - 1])
                        if self.n_ep else 0)
            # 恢复时只保留已提交的完整 episode，丢弃中断留下的尾部行。
            for key, ds in self.f.items():
                size = self.n_ep if key in ("ep_len", "ep_offset", "ep_meta") else self.ptr
                if len(ds) < size:
                    raise ValueError(f"Committed dataset is truncated: {key}")
                if len(ds) > size:
                    ds.resize(size, axis=0)
            self.initialized = True
            if "ep_meta" in self.f:
                self.meta = [m.decode() if isinstance(m, bytes) else str(m)
                             for m in self.f["ep_meta"][:]]
            else:
                self.meta = []
        else:                                        # 全新文件
            self.n_ep, self.ptr, self.initialized = 0, 0, False
            self.meta = []

    # ---- 懒建列（与 world.py._init_h5_datasets 同规则）----
    def _init_datasets(self, sample_ep):
        for key, data_list in sample_ep.items():
            if key in ("ep_len", "ep_idx", "policy"):
                continue
            key = key.replace("/", "_")
            sample = np.array(data_list[0])
            shape = (0,) + sample.shape
            maxshape = (None,) + sample.shape
            if sample.ndim >= 2:
                chunks = (100,) + sample.shape
                compression = hdf5plugin.Blosc(
                    cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            else:
                chunks = (1000,) + sample.shape
                compression = None
            dtype = sample.dtype
            if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
                dtype = h5py.string_dtype()
            self.f.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype,
                                  chunks=chunks, compression=compression)
        self.f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)
        self.f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self.f.create_dataset("ep_idx", shape=(0,), maxshape=(None,), dtype=np.int32,
                              chunks=(1000,))
        self.f.create_dataset("ep_meta", shape=(0,), maxshape=(None,),
                              dtype=h5py.string_dtype())
        self.f.swmr_mode = True
        self.initialized = True

    # ---- 整集追加（与 world.py._write_episode 同逻辑）----
    def write_episode(self, ep, meta):
        ep_len = len(ep["step_idx"])
        if ep_len == 0 or any(len(values) != ep_len for values in ep.values()):
            raise ValueError("All episode columns must have the same nonzero length")
        if not self.initialized:
            self._init_datasets(ep)
        for key in ep:
            h5_key = key.replace("/", "_")
            if h5_key in ("ep_len", "policy"):
                continue
            ds = self.f[h5_key]
            curr = ds.shape[0]
            ds.resize(curr + ep_len, axis=0)
            ds[curr:] = np.array(ep[key])

        i = self.f["ep_offset"].shape[0]
        self.f["ep_offset"].resize(i + 1, axis=0)
        self.f["ep_len"].resize(i + 1, axis=0)
        self.f["ep_offset"][i] = self.ptr
        self.f["ep_len"][i] = ep_len
        self.f["ep_meta"].resize(i + 1, axis=0)
        self.f["ep_meta"][i] = json.dumps(json_safe(meta), ensure_ascii=False, allow_nan=False)

        self.ptr += ep_len
        self.n_ep += 1
        self.meta.append(json.dumps(json_safe(meta), ensure_ascii=False, allow_nan=False))
        self.f.flush()
        self.f.attrs["committed_episodes"] = self.n_ep
        self.f.flush()

    def close(self):
        self.f.close()


# ================================================================ 工具
def sample_spec(spec, rng):
    """按 {choices:[...]} / {range:[lo,hi]} / 标量 三种形式采样域随机化参数。"""
    if isinstance(spec, dict):
        if "choices" in spec:
            return float(rng.choice(np.asarray(spec["choices"], dtype=np.float64)))
        if "range" in spec:
            lo, hi = spec["range"]
            return float(rng.uniform(float(lo), float(hi)))
        raise ValueError(f"无法解析的随机化规格: {spec}")
    return float(spec)


def build_trajectory(scene, dr):
    """按场景类型生成闭式专家轨迹 (n,22) 与 meta（闭式解，无随机性）。"""
    if scene["kind"] == "step_over":
        traj, meta = bx.gen_step_over(
            h_obs=float(scene["h_obs"]),
            bx0=dr["bx0"],
            lift_extra=dr["lift_extra"],
            T_lipm=dr["T_lipm"])
        meta = bx.with_over(meta)
    elif scene["kind"] == "step_up":
        traj, meta = bx.gen_step_up(
            h_plat=float(scene["h_plat"]),
            x_plat=float(scene.get("x_plat", 0.14)),
            T_a=float(scene.get("T_a", 1.0)),
            T_b=float(scene.get("T_b", 0.8)),
            T_c=float(scene.get("T_c", 1.0)),
            T_d=float(scene.get("T_d", 0.5)))
    else:
        raise ValueError(f"未知场景类型: {scene['kind']}")
    return traj, meta


def swing_windows(scene, meta):
    """该场景需要复核净空的摆动窗口 [(side, t0, t1), ...]。"""
    if scene["kind"] == "step_over":
        return [(side, float(t0), float(t1)) for (_bi, side, t0, t1) in meta["over"]]
    sw = meta["swing"]
    return [("l", float(sw["l"][0]), float(sw["l"][1])),
            ("r", float(sw["r"][0]), float(sw["r"][1]))]


def interp_target(traj, t):
    """按 DT_TRAJ 线性插值 22 维目标；超出轨迹范围时取端点。"""
    n = len(traj)
    x = t / bx.DT_TRAJ
    if x <= 0.0:
        return traj[0]
    if x >= n - 1:
        return traj[-1]
    i = int(np.floor(x))
    f = x - i
    return traj[i] * (1.0 - f) + traj[i + 1] * f


# ================================================================ 采集器
class BruceCollector:
    def __init__(self, cfg, seed=0, backend="gpu"):
        self.cfg = cfg
        self.seed = int(seed)
        self.control_mode = cfg.get("control_mode", "joint_only")
        self.video_dir = Path(cfg["video_dir"]) if cfg.get("video_dir") else None
        self.video_episodes = int(cfg.get("video_episodes", 0))
        self.rng = np.random.default_rng(self.seed)

        sim = cfg["sim"]
        self.sim_dt = float(sim["dt"])
        self.record_hz = float(cfg["record_hz"])
        ratio = 1.0 / (self.record_hz * self.sim_dt)
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("record period must be an integer multiple of simulation dt")
        self.steps_per_frame = int(round(ratio))
        self.episode_seconds = float(cfg["episode_seconds"])
        self.n_frames = int(round(self.episode_seconds * self.record_hz))
        self.warmup_steps = int(cfg.get("warmup_steps", 10))

        self.scene = None
        # 障碍箱：每个场景一个实体，运行时把不用的挪走
        self.scenes = cfg["scenes"]
        specs = []
        for sc in self.scenes:
            b = sc["box"]
            h = float(sc.get("h_obs", sc.get("h_plat", 0.06)))
            specs.append({"half": (float(b["hx"]), float(b["hy"]), h / 2.0),
                          "pos": (float(b["cx"]), 0.0, h / 2.0)})
        self.box_specs = specs
        self.scene = be.BruceScene(sim_dt=self.sim_dt,
                                   substeps=int(sim.get("substeps", 1)),
                                   seed=self.seed,
                                   backend=backend,
                                   box_specs=specs,
                                   control_mode=self.control_mode,
                                   camera=self.video_dir is not None,
                                   default_armature=float(sim.get('default_armature', 0.)),
                                   leg_armature=float(sim.get('leg_armature', .00182)))
        self.boxes = self.scene.boxes

        self.kp = np.asarray(bx.KP, dtype=np.float64)
        self.kv = np.asarray(bx.KV, dtype=np.float64)

        sensor = cfg.get("sensor", {})
        self.sb = bs.SignalBuilder(sensor, rng=self.rng)

    # ---- 单条 episode ----
    def run_episode(self, ep_index):
        cfg = self.cfg
        episode_seed = self.seed + int(ep_index)
        self.rng = np.random.default_rng(episode_seed)
        self.sb = bs.SignalBuilder(cfg.get("sensor", {}), rng=self.rng)
        dr_cfg = cfg.get("domain_randomization", {})
        sc_idx = ep_index % len(self.scenes)
        sc = self.scenes[sc_idx]
        dr = {k: sample_spec(v, self.rng) for k, v in dr_cfg.items()
              if k != "box_x_jitter"}
        jitter = float(dr_cfg.get("box_x_jitter", 0.0))
        box_cx = float(sc["box"]["cx"]) + float(self.rng.uniform(-jitter, jitter))

        traj, meta = build_trajectory(sc, dr)
        # Arms hold a fixed posture: the varying policy action contains only leg targets.
        traj[:, 16:] = traj[0, 16:]
        self.n_frames = max(int(round(self.episode_seconds * self.record_hz)),
                            int(np.ceil((meta["T"] + 0.2) * self.record_hz)))

        # 摆箱 + 参数随机化
        self.scene.park_boxes(self.boxes, keep=sc_idx)
        box = self.boxes[sc_idx]
        h = float(sc.get("h_obs", sc.get("h_plat", 0.06)))
        self.scene.reset_box(box, (box_cx, 0.0, h / 2.0))
        friction = dr.get("friction", 1.0)
        self.scene.set_friction(friction)
        self.scene.set_box_friction(box, friction)
        kp_scale = dr.get("kp_scale", 1.0)
        self.scene.set_gains(self.kp * kp_scale, self.kv * kp_scale)

        # 复位机器人到轨迹起点
        base_z = float(traj[0][2])
        base_x = float(traj[0][0])
        self.scene.reset_robot(base_z, dof_pos=traj[0][6:22], base_x=base_x)

        # 复位后先空跑 warmup 步：set_qpos 之后 solver 的加速度缓存还是脏值
        # （实测首帧 get_links_acc 给到 2.8e3 m/s^2），空跑后状态与目标对齐再开录
        t = 0.0
        for _ in range(self.warmup_steps):
            self.scene.apply_target(interp_target(traj, t))
            self.scene.step()
            t += self.sim_dt
        warmup_t0 = t

        # 逐帧采集
        self.sb.reset_episode()
        buf = {k: [] for k in bs.ALL_COLUMNS}
        buf["step_idx"] = []
        for key in ("timestamp", "sim_reference", "sim_control_force", "sim_body_acc"):
            buf[key] = []
        video = None
        if self.video_dir is not None and ep_index < self.video_episodes:
            import imageio.v2 as imageio
            self.video_dir.mkdir(parents=True, exist_ok=True)
            video = imageio.get_writer(str(self.video_dir / f"episode_{ep_index:03d}.mp4"),
                                       fps=25, codec="libx264", quality=7)
        rec = {"t": [], "ankle_r": [], "ankle_l": []}
        box_start = self.scene.box_pos(box).copy()
        kicked = False
        fell = False
        finite = True
        min_base_z = 1e9

        for f in range(self.n_frames):
            t_rec = t
            target_rec = interp_target(traj, t_rec)

            # 观测真值（100 Hz）+ 该时刻踝位置（净空复核用）
            st = self.scene.read_state()
            cols = self.sb.build(st, target_rec)
            finite &= bool(np.all(np.isfinite(st["dof_pos"])))
            min_base_z = min(min_base_z, float(st["base_pos"][2]))
            rec["t"].append(t_rec)
            rec["ankle_r"].append(st["ankle_pos"].get("ankle_pitch_link_r", (0.0, 0.0)))
            rec["ankle_l"].append(st["ankle_pos"].get("ankle_pitch_link_l", (0.0, 0.0)))

            # 行 t 保存 o_t 与随后整个记录周期实际保持的目标 a_t，无需循环移位。
            for k, v in cols.items():
                buf[k].append(v)
            buf["step_idx"].append(f)
            buf["timestamp"].append(t_rec)
            buf["sim_reference"].append(target_rec.copy())
            buf["sim_control_force"].append(st["dof_force"].copy())
            buf["sim_body_acc"].append(st["base_acc"].copy())
            if video is not None and f % max(1, int(round(self.record_hz / 25))) == 0:
                video.append_data(self.scene.camera.render()[0])

            # 推进仿真；帧中点再采一次踝轨迹（净空复核 200 Hz）
            for k in range(self.steps_per_frame):
                self.scene.apply_target(target_rec)
                self.scene.step()
                t += self.sim_dt
                if k == self.steps_per_frame // 2 - 1:
                    ap = self.scene.read_ankle_pos()
                    rec["t"].append(t_rec + 0.5 / self.record_hz)
                    rec["ankle_r"].append(ap.get("ankle_pitch_link_r", (0.0, 0.0)))
                    rec["ankle_l"].append(ap.get("ankle_pitch_link_l", (0.0, 0.0)))

            # 箱体是否被撞移/踢倒（m6 判据）
            bp = self.scene.box_pos(box)
            if (np.linalg.norm(bp - box_start) > 0.02
                    or bp[2] < h / 2.0 - 0.01):
                kicked = True

            R = be.quat_to_mat(st["base_quat"])
            if st["base_pos"][2] < 0.22 or R[2, 2] < 0.5:
                fell = True
                break
        if video is not None:
            video.close()

        buf["ep_idx"] = [ep_index] * len(buf["step_idx"])

        # ---- 净空与成败判定 ----
        top = h
        front = box_cx - float(sc["box"]["hx"])
        x_lo = front - bx.TOE
        x_hi = box_cx + float(sc["box"]["hx"]) + bx.HEEL
        clearances = []
        for side, t0, t1 in swing_windows(sc, meta):
            c = bx.clearance_over(rec, side, top, x_lo, x_hi, t0, t1)
            clearances.append(float(c))
        finite &= bool(np.all(np.isfinite(np.asarray(rec["t"]))))
        valid_clearances = [c for c in clearances if np.isfinite(c)]
        clr = min(valid_clearances) if valid_clearances else float("nan")
        fell = bool(fell or min_base_z < 0.30)
        final_state = self.scene.read_state()
        final_ankles = final_state["ankle_pos"]
        rear = box_cx + float(sc["box"]["hx"])
        if sc["kind"] == "step_over":
            completed = all(p[0] - bx.HEEL > rear for p in final_ankles.values())
        else:
            completed = all(p[0] - bx.HEEL > front and p[1] - bx.SOLE > h - 0.015
                            for p in final_ankles.values())
        success = bool(finite and (not kicked) and (not fell)
                       and completed and np.isfinite(clr) and clr > 0.0)

        ep_meta = {
            "episode": ep_index,
            "seed": episode_seed,
            "control_mode": self.control_mode,
            "base_actuated": self.control_mode == "assisted",
            "urdf": self.scene.urdf,
            "leg_armature_diagonal_kg_m2": float(self.cfg['sim'].get('leg_armature', .00182)),
            "completed": bool(completed),
            "n_frames": len(buf["step_idx"]),
            "clearance_kind": "ankle_based_proxy_not_mesh_distance",
            "estimator_kind": "truth_plus_noise_not_hardware_EKF",
            "action_hold": "zero_order_hold_one_record_period",
            "arm_targets": traj[0, 16:].tolist(),
            "scene": sc["id"],
            "kind": sc["kind"],
            "h_obs": float(sc.get("h_obs", sc.get("h_plat", 0.0))),
            "box_cx": box_cx,
            "box_hx": float(sc["box"]["hx"]),
            "swing": [[s, float(a), float(b)] for s, a, b in swing_windows(sc, meta)],
            "t_start": float(warmup_t0),
            "record_hz": float(self.record_hz),
            "bx0": dr.get("bx0"),
            "lift_extra": dr.get("lift_extra"),
            "T_lipm": dr.get("T_lipm"),
            "friction": friction,
            "kp_scale": kp_scale,
            "clearance": clr,
            "clearances": clearances,
            "kicked": bool(kicked),
            "fell": fell,
            "min_base_z": min_base_z,
            "finite": bool(finite),
            "success": success,
            "base_dx": float(self.scene.read_state()["base_pos"][0] - base_x),
            "traj_T": float(meta["T"]),
        }
        ep_meta['reference_criterion_passed'] = success
        ep_meta.update(qualify(ep_meta, buf['sim_control_force']))
        ep_meta['success'] = ep_meta['autonomous_success']
        return buf, ep_meta


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser(description="Bruce 越障数据采集（Genesis → H-JEPA HDF5）")
    ap.add_argument("--config", required=True, help="采集配置 YAML")
    ap.add_argument("--episodes", type=int, default=None, help="目标总集数（覆盖 YAML）")
    ap.add_argument("--out", default=None, help="数据集名（覆盖 YAML）")
    ap.add_argument("--cache-dir", default=None, help="输出目录（默认 ~/.stable_worldmodel）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（覆盖 YAML）")
    ap.add_argument("--control-mode", choices=["joint_only", "assisted"], default=None)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--video-episodes", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="删除已有数据集重新采集")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.control_mode:
        cfg["control_mode"] = args.control_mode
    cfg["video_dir"] = args.video_dir
    cfg["video_episodes"] = args.video_episodes
    out_cfg = cfg.get("output", {})
    dataset_name = args.out or out_cfg.get("dataset_name", "bruce_stepover")
    episodes = int(args.episodes if args.episodes is not None
                   else out_cfg.get("episodes", 10))
    seed = int(args.seed if args.seed is not None else out_cfg.get("seed", 0))

    cache_dir = Path(args.cache_dir or out_cfg.get("cache_dir")
                     or os.environ.get("HJEPA_HOME", Path.home() / ".stable_worldmodel"))
    cache_dir = Path(cache_dir).expanduser()
    path = cache_dir / f"{dataset_name}.h5"
    meta_path = cache_dir / f"{dataset_name}.meta.jsonl"

    if args.fresh and path.exists():
        path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    t0 = time.time()
    # 先开 writer 拿到「已录集数」，再据此顺延 seed（world.py 同款）
    writer = EpisodeWriter(path)
    if writer.n_ep and writer.f.attrs.get('schema_version') != 'bruce_obstacle_v3_qualified_outcome':
        writer.close()
        raise ValueError('Legacy dataset is read-only for this collector; use a new output name')
    config_json = json.dumps(cfg, sort_keys=True)
    if "collection_config" in writer.f.attrs and writer.f.attrs["collection_config"] != config_json:
        writer.close()
        raise ValueError("Resume configuration differs from the saved dataset")
    writer.f.attrs["collection_config"] = config_json
    writer.f.attrs["schema_version"] = "bruce_obstacle_v3_qualified_outcome"
    writer.f.attrs["sensor_provenance"] = "simulated sensor proxies; est_* is truth plus noise"
    writer.f.attrs["control_mode"] = cfg.get("control_mode", "joint_only")
    writer.f.attrs["seed"] = seed
    if writer.n_ep >= episodes:
        print(f"[collect] Already complete: {writer.n_ep} episodes in {path}")
        writer.close()
        return
    try:
        ep_seed = seed
        collector = BruceCollector(cfg, seed=ep_seed,
                                    backend=cfg.get("sim", {}).get("backend", "gpu"))

        print(f"[collect] 数据集: {path}")
        print(f"[collect] 目标 {episodes} 集，已录 {writer.n_ep} 集，seed={ep_seed}")
        print(f"[collect] 每集 {collector.n_frames} 帧 @ {collector.record_hz} Hz"
              f"（{collector.steps_per_frame} 仿真步/帧）")

        n_ok = sum(bool(json.loads(m).get("success")) for m in writer.meta)
        with meta_path.open("a", encoding="utf-8") as sidecar:
            for ep in range(writer.n_ep, episodes):
                buf, meta = collector.run_episode(ep)
                writer.write_episode(buf, meta)
                sidecar.write(json.dumps(json_safe(meta), ensure_ascii=False, allow_nan=False) + "\n")
                sidecar.flush()
                n_ok += int(meta["success"])
                print(f"  ep {ep:4d} | {meta['scene']:<8s} | "
                      f"clr {meta['clearance']*1000:+7.1f}mm | "
                      f"{'OK ' if meta['success'] else 'FAIL'}"
                      f"{'(踢箱)' if meta['kicked'] else ''}"
                      f"{'(摔倒)' if meta['fell'] else ''} | "
                      f"dx {meta['base_dx']:+.3f} | 总集 {writer.n_ep}")
    finally:
        writer.close()

    dt = time.time() - t0
    print(f"[collect] 完成: {writer.n_ep} 集, 成功 {n_ok} 集, 用时 {dt:.1f}s")
    print(f"[collect] 侧车元数据: {meta_path}")


if __name__ == "__main__":
    main()
