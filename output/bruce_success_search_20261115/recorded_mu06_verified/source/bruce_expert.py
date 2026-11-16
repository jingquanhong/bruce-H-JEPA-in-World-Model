"""Analytic BRUCE reference trajectories (LIPM and inverse kinematics).

Row layout: base [x,y,z,roll,pitch,yaw], right leg 5, left leg 5, arms 6.
Samples are DT_TRAJ seconds apart. Base motion is a reference, not an actuator.
Imported from the existing m5_modal_gen/m6_scenes reference implementation."""

import os
from pathlib import Path

import numpy as np

# ---- 资产路径（仓库内 URDF；可用环境变量 BRUCE_URDF 覆盖）----
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_URDF = (_REPO_ROOT / "BRUCE-OP-main" / "Simulation" / "models" / "bruce"
                  / "urdf" / "bruce_physical.urdf")
URDF = os.environ.get(
    "BRUCE_URDF",
    str(_DEFAULT_URDF),
)

SIM_DT = 5e-4
DT_TRAJ = 0.01                     # 轨迹采样周期（重放时线性插值到仿真步）

# ---- 运动学常数（calib_leg.py 标定）----
L1, L2 = 0.2049, 0.1999
HIP_X, HIP_Z = 0.0292, -0.0398
MASS = 4.524

# ---- 傀儡 PD 增益（与 manual_control 行走模式一致）----
KP = np.array([1000.0, 1000.0, 5000.0, 60.0, 60.0, 60.0] + [100.0] * 10 + [60.0] * 6)
KV = np.array([30.0, 30.0, 60.0, 12.0, 12.0, 12.0] + [3.0] * 10 + [2.5] * 6)

FOOT_Y = 0.0759                    # 髋横向偏距
H_STAND = 0.46                     # 站立基座悬高
Z_STANCE = 0.025                   # 原始脚部网格的支撑高度
SOLE = 0.025                       # 踝原点到脚底网格面
TOE = 0.056                        # 踝前方脚趾长（mesh y+）
HEEL = 0.0385                      # 踝后方脚跟长（mesh y-）


def lift_prof(s, rise=0.13, fall=0.15):
    """摆动腿抬升平顶梯形: 快速上升到1 -> 保持 -> 平滑下降。
    比单 sin(πs) 快速起抬，保证脚趾到达障碍前缘时脚底已高于障碍顶。"""
    if s < rise:
        return smooth(s / rise)
    if s <= 1.0 - fall:
        return 1.0
    return smooth((1.0 - s) / fall)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def ik(x, z):
    """踝相对髋 pitch 原点目标 (x, z)。返回 (hip, knee, ankle)，膝负弯曲。"""
    d = -z
    r = np.hypot(x, d)
    r = min(r, L1 + L2 - 1e-4)
    r = max(r, abs(L1 - L2) + 1e-4)
    a = np.arccos(np.clip((L1 * L1 + L2 * L2 - r * r) / (2 * L1 * L2), -1, 1))
    knee = a - np.pi
    b = np.arccos(np.clip((L1 * L1 + r * r - L2 * L2) / (2 * L1 * r), -1, 1))
    hip = np.arctan2(x, d) + b
    return hip, knee, -(hip + knee)


def smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def _pose(feet, bx, H, sh=(0.0, 0.0)):
    """由双脚世界目标 (x,z) 与基座 (bx,H) 算 22 维目标行。feet = {'r':(x,z),'l':(x,z)}

    行格式为 dof 空间: [x,y,z,roll,pitch,yaw + 右腿5 + 左腿5 + 右臂3 + 左臂3]。
    """
    q = np.zeros(22)
    q[:3] = [bx, 0.0, H]                        # roll/pitch/yaw 恒 0（dof 空间）
    for side, o in (("r", 6), ("l", 11)):
        fx, fz = feet[side]
        h, k, a_ = ik(fx - bx - HIP_X, fz - H - HIP_Z)
        q[o:o + 5] = [0.0, h, 0.0, k, a_]
    q[16], q[19] = sh                            # 肩 pitch 反相摆
    q[18] = q[21] = 0.25                         # 双肘微屈
    return q


# ---- 22 维 dof 行 -> Pinocchio q ----
# 行内关节顺序 [右腿,左腿,右臂,左臂] 与 URDF 文档顺序一致；
# Pinocchio 解析出的顺序为 [左腿,左臂,右腿,右臂]，必须按关节名重映射。
_ROW_SLOT = {"hip_yaw_r": 0, "hip_pitch_r": 1, "hip_roll_r": 2, "knee_pitch_r": 3,
             "ankle_pitch_r": 4, "hip_yaw_l": 5, "hip_pitch_l": 6, "hip_roll_l": 7,
             "knee_pitch_l": 8, "ankle_pitch_l": 9, "shoulder_pitch_r": 10,
             "shoulder_roll_r": 11, "elbow_pitch_r": 12, "shoulder_pitch_l": 13,
             "shoulder_roll_l": 14, "elbow_pitch_l": 15}
_PIN = {"model": None}


def rpy_to_quat(rpy):
    """欧拉 XYZ -> 四元数 xyzw（Pinocchio 布局）。"""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r / 2), np.sin(r / 2), np.cos(p / 2),
                              np.sin(p / 2), np.cos(y / 2), np.sin(y / 2))
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def row_to_pin(row):
    """22 维 dof 行 -> Pinocchio q（欧拉转四元数 + 关节名重映射，双腿/双臂 L/R 顺序对齐）。

    外部依赖：本函数惰性 import ``pinocchio`` 并从模块常量 ``URDF`` 构建模型
    （原始实现即为此写法）。仅用于离线 Pinocchio 校验，轨迹生成不依赖它。
    """
    import pinocchio as pin
    if _PIN["model"] is None:
        _PIN["model"] = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    model = _PIN["model"]
    q = np.zeros(model.nq)
    q[:3] = row[:3]
    q[3:7] = rpy_to_quat(row[3:6])
    for i in range(1, model.njoints):
        name = model.names[i]
        if name in _ROW_SLOT:
            q[model.joints[i].idx_q] = row[6 + _ROW_SLOT[name]]
    return q


# ---------------------------------------------------------------- 三模态生成器
def gen_squat(H1=0.30, T_down=1.0, T_hold=0.4, T_up=1.0):
    """下蹲：双脚原地不动，基座竖直 H0->H1->H0（准静态，ZMP=质心投影）。

    质心比基座原点靠前约 37mm（Pinocchio 实测），基座后移 50mm 使投影落在支撑内。
    """
    feet = {"r": (0.0, Z_STANCE), "l": (0.0, Z_STANCE)}
    bx = -0.05
    T = T_down + T_hold + T_up
    n = int(T / DT_TRAJ) + 1
    traj = np.zeros((n, 22))
    for i in range(n):
        t = i * DT_TRAJ
        if t < T_down:
            H = H_STAND + (H1 - H_STAND) * smooth(t / T_down)
        elif t < T_down + T_hold:
            H = H1
        else:
            H = H1 + (H_STAND - H1) * smooth((t - T_down - T_hold) / T_up)
        traj[i] = _pose(feet, bx, H)
    checks = [(t, [(0.0, FOOT_Y), (0.0, -FOOT_Y)]) for t in
              (0.0, T_down * 0.5, T_down, T_down + T_hold, T - 0.01)]
    return traj, {"name": "下蹲(准静态/VHIP)", "T": T, "checks": checks}


def gen_step_over(h_obs=0.06, stride=0.22, T_prep=0.3, T_lipm=1.0, T_land=0.5, T_follow=1.2,
                  bx0=-0.03, lift_extra=0.045):
    """跨越：LIPM 闭式轨道前移 + 左摆腿越障 + 右脚跟进，H=0.45（后蹬相可达性）。

    bx0: 预备位基座前后偏移（质心前偏补偿量，可调）；lift_extra: 抬脚净空余量
    （需覆盖快速抬升段 kp=100 的 PD 跟踪滞后 ~2cm，默认 0.045 实测充足）。
    步幅 0.22 保证脚（长9.45cm）能整体越过 5cm 厚的箱子；
    抬脚用平顶梯形 profile，脚趾到达箱前缘时脚底已升至障碍顶上。
    """
    H = 0.45
    w = np.sqrt(9.81 / (H - 0.04))               # LIPM 固有频率（质心高≈H-0.04）
    c = stride / 2
    T = T_prep + T_lipm + T_land + T_follow
    n = int(T / DT_TRAJ) + 1
    traj = np.zeros((n, 22))
    lift = h_obs + lift_extra
    for i in range(n):
        t = i * DT_TRAJ
        if t < T_prep:                           # 预备：双脚站立（基座后移补偿质心前偏）
            bx = bx0
            feet = {"r": (0.0, Z_STANCE), "l": (0.0, Z_STANCE)}
            sh = (0.0, 0.0)
        elif t < T_prep + T_lipm:                # LIPM 单支撑 + 左摆腿越障
            s = (t - T_prep) / T_lipm
            bx = bx0 + c * np.sinh(w * s * T_lipm) / np.sinh(w * T_lipm)  # 闭式轨道
            feet = {"r": (0.0, Z_STANCE),
                    "l": (stride * s, Z_STANCE + lift * lift_prof(s))}
            sh = (0.35 * np.sin(np.pi * s), -0.35 * np.sin(np.pi * s))
        elif t < T_prep + T_lipm + T_land:       # 落地双支撑
            bx = bx0 + c
            feet = {"r": (0.0, Z_STANCE), "l": (stride, Z_STANCE)}
            sh = (0.0, 0.0)
        else:                                    # 右脚跟进（同样越过障碍）
            s = (t - T_prep - T_lipm - T_land) / T_follow
            feet = {"r": (stride * smooth(s), Z_STANCE + lift * lift_prof(s)),
                    "l": (stride, Z_STANCE)}
            sh = (-0.35 * np.sin(np.pi * s), 0.35 * np.sin(np.pi * s))
            bx = bx0 + c + c * smooth(s)
        traj[i] = _pose(feet, bx, H, sh)
    checks = [(T_prep * 0.5, [(0.0, FOOT_Y), (0.0, -FOOT_Y)]),
              (T_prep + T_lipm + T_land * 0.5, [(0.0, -FOOT_Y), (stride, FOOT_Y)]),
              (T - 0.01, [(stride, FOOT_Y), (stride, -FOOT_Y)])]
    ds_windows = [(0.0, T_prep, [(0.0, FOOT_Y), (0.0, -FOOT_Y)]),
                  (T_prep + T_lipm, T_prep + T_lipm + T_land,
                   [(0.0, -FOOT_Y), (stride, FOOT_Y)]),
                  (T - 0.01, T, [(stride, FOOT_Y), (stride, -FOOT_Y)])]
    return traj, {"name": "跨越(LIPM闭式轨道)", "T": T, "checks": checks,
                  "ds_windows": ds_windows, "bx0": bx0, "lift_extra": lift_extra,
                  "omega": w, "stride": stride, "h_obs": h_obs,
                  "swing": {"l": (T_prep, T_prep + T_lipm),
                            "r": (T_prep + T_lipm + T_land, T)}}


def gen_step_up(h_plat=0.10, x_plat=0.14, T_a=1.0, T_b=0.8, T_c=1.0, T_d=0.5):
    """上台阶：左脚上平台 -> 质心平移并微蹲(H 0.46->0.44) -> 右脚跟上 -> 站直。

    x_plat=0.14 使落脚全脚掌位于平台面上（平台前缘 x=0.09）；
    抬升前段加快（s/0.35），脚趾触及平台前缘时脚底已高于台面。
    """
    H_hi, H_lo = 0.46, 0.44
    T = T_a + T_b + T_c + T_d
    n = int(T / DT_TRAJ) + 1
    traj = np.zeros((n, 22))
    for i in range(n):
        t = i * DT_TRAJ
        if t < T_a:                              # 左脚上平台，基座前移少许
            s = t / T_a
            feet = {"r": (0.0, Z_STANCE),
                    "l": (x_plat * smooth(s), Z_STANCE + h_plat * smooth(min(1.0, s / 0.35))
                          + 0.05 * np.sin(np.pi * s))}
            bx = 0.02 * smooth(s)
            H = H_hi
        elif t < T_a + T_b:                      # 质心平移到两脚之间（微蹲增可达性）
            s = (t - T_a) / T_b
            bx = 0.02 + 0.03 * smooth(s)
            H = H_hi + (H_lo - H_hi) * smooth(s)
            feet = {"r": (0.0, Z_STANCE), "l": (x_plat, Z_STANCE + h_plat)}
        elif t < T_a + T_b + T_c:                # 右脚跟上平台
            s = (t - T_a - T_b) / T_c
            feet = {"r": (x_plat * smooth(s), Z_STANCE + h_plat * smooth(min(1.0, s / 0.35))
                          + 0.05 * np.sin(np.pi * s)),
                    "l": (x_plat, Z_STANCE + h_plat)}
            bx = 0.05 + 0.02 * smooth(s)
            H = H_lo
        else:                                    # 平台上站直
            s = (t - T_a - T_b - T_c) / T_d
            feet = {"r": (x_plat, Z_STANCE + h_plat), "l": (x_plat, Z_STANCE + h_plat)}
            bx = 0.07
            H = H_lo + (H_hi - H_lo) * smooth(s)
        traj[i] = _pose(feet, bx, H)
    # 静态校验只覆盖双支撑窗口（单支撑窗口归摆动动力学，超出闭式静态范围）
    checks = [(T_a + T_b * 0.5, [(0.0, -FOOT_Y), (x_plat, FOOT_Y)]),
              (T - 0.01, [(x_plat, FOOT_Y), (x_plat, -FOOT_Y)])]
    return traj, {"name": "上台阶(准静态攀爬)", "T": T, "checks": checks,
                  "h_plat": h_plat, "x_plat": x_plat,
                  "swing": {"l": (0.0, T_a), "r": (T_a + T_b, T_a + T_b + T_c)}}


# ---------------------------------------------------------------- 闭式校验
def static_zmp_margin(com_xy, feet, half=(0.045, 0.025), margin=0.01):
    """支撑多边形（脚矩形并集的外包矩形）内缩 margin 后，质心投影到边界的最小距离。"""
    xs = [f[0] for f in feet]
    ys = [f[1] for f in feet]
    lo_x, hi_x = min(xs) - half[0] + margin, max(xs) + half[0] - margin
    lo_y, hi_y = min(ys) - half[1] + margin, max(ys) + half[1] - margin
    dx = min(com_xy[0] - lo_x, hi_x - com_xy[0])
    dy = min(com_xy[1] - lo_y, hi_y - com_xy[1])
    return float(min(dx, dy))


def clearance_over(rec, side, top, x_lo, x_hi, t_lo=None, t_hi=None):
    """摆动经过 [x_lo,x_hi] 区间时踝对高度 top 的实测最小净空（含脚底偏置）。

    t_lo/t_hi 限定摆动时间段：reset 落地暂态中踝会扫过窗口前缘下方，
    不限时段会把暂态误计为越障净空。

    外部依赖：``rec`` 为实测记录字典（由采集端从仿真器回读并注入），需含
    ``"t"``（时间列表）与 ``"ankle_r"/"ankle_l"``（踝 (x,z) 列表）。
    """
    vals = [z - SOLE - top for i, (x, z) in enumerate(rec["ankle_" + side])
            if x_lo <= x <= x_hi
            and (t_lo is None or rec["t"][i] >= t_lo)
            and (t_hi is None or rec["t"][i] <= t_hi)]
    return float(min(vals)) if vals else float("nan")


# ================================================================ m6 场景扩展
C_MIN = 0.005            # 闭式净空基准阈值: 5mm (依据 +2.5mm 候选实机刮蹭卡滞)
GAMMA_LIFT = 0.5         # 执行裕度随抬脚增长系数: C_req = C_MIN + 0.5*max(0, lift-0.105)
# 物理依据: 闭式SDF不含PD跟踪滞后, 抬脚越大摆速越快滞后越大 —
#   6cm箱+lift0.105 闭式6~9mm -> 实机+42mm 通过(基准档);
#   8cm箱+lift0.16  闭式+6.2mm -> 实机踢翻箱体(-14mm) (C_req=32.5mm 正确拒绝);
#   12cm箱+lift0.20 闭式+72mm -> 实机踢翻箱体 (同规则拒绝, 与平衡裕度>=0.10同理)
MARGIN_NORM = 0.10                 # 项目硬约束: 归一化平衡裕度阈值（源自 m6_pareto）


def make_box(cx, h_obs, cy=0.0, hx=0.025):
    """障碍箱: 高 h_obs, x 向半宽 hx, y 向半宽 0.15。"""
    return {"center": np.array([cx, cy, h_obs / 2]),
            "half": np.array([hx, 0.15, h_obs / 2])}


SCENES = [
    {"id": "A", "name": "矮箱4cm", "h_obs": 0.04, "boxes": [make_box(0.10, 0.04)]},
    {"id": "B", "name": "标准箱6cm", "h_obs": 0.06, "boxes": [make_box(0.10, 0.06)]},
    {"id": "C", "name": "高箱8cm", "h_obs": 0.08,
     "boxes": [make_box(0.10, 0.08)]},
    {"id": "D", "name": "双箱连续6+6cm", "h_obs": 0.06,
     "boxes": [make_box(0.10, 0.06), make_box(0.32, 0.06)], "double": True},
]


def gen_double_step(h_obs=0.06, dx2=0.22, bx0=-0.03, lift_extra=0.045, T_lipm=0.8):
    """双箱连续跨越: 两段 gen_step_over 拼接。
    第二段整体 x 平移 dx2 — 基座与双脚同时平移时相对几何不变(关节角不变, 刚体平移),
    两段在衔接点姿态连续: 第一段末态基座 bx0+stride、双脚 x=stride 恰为第二段初态。"""
    t1, m1 = gen_step_over(h_obs=h_obs, bx0=bx0, lift_extra=lift_extra, T_lipm=T_lipm)
    t2, m2 = gen_step_over(h_obs=h_obs, bx0=bx0, lift_extra=lift_extra, T_lipm=T_lipm)
    t2[:, 0] += dx2
    traj = np.vstack([t1, t2[1:]])
    T1 = m1["T"]
    ds = list(m1["ds_windows"])
    for a, b, feet in m2["ds_windows"]:
        ds.append((a + T1, b + T1, [(x + dx2, y) for (x, y) in feet]))
    over = [(0, "l") + tuple(m1["swing"]["l"]), (0, "r") + tuple(m1["swing"]["r"]),
            (1, "l", T1 + m2["swing"]["l"][0], T1 + m2["swing"]["l"][1]),
            (1, "r", T1 + m2["swing"]["r"][0], T1 + m2["swing"]["r"][1])]
    meta = {"name": "双箱连续跨越", "T": T1 + m2["T"], "ds_windows": ds, "over": over,
            "bx0": bx0, "lift_extra": lift_extra, "h_obs": h_obs, "double": True}
    return traj, meta


def with_over(meta):
    """单箱场景补越障窗口(双箱已在 gen_double_step 内构建)。"""
    if "over" not in meta:
        meta["over"] = [(0, "l") + tuple(meta["swing"]["l"]),
                        (0, "r") + tuple(meta["swing"]["r"])]
    return meta


def clearance_multi(sm, traj, boxes):
    """全帧(10ms)自模型点云对多障碍的最小 SDF(粗采样会漏最差点)。

    外部依赖：``sm`` 为外部注入的自模型对象（鸭子类型），需提供
    ``points_world(row) -> (M,3)`` 与静态方法 ``box_sdf(pts, center, half) -> (M,)``。
    """
    best = np.inf
    for i in range(len(traj)):
        pts = sm.points_world(traj[i])
        for b in boxes:
            best = min(best, float(sm.box_sdf(pts, b["center"], b["half"]).min()))
    return best


def eval_scene(sm, sc):
    """场景内候选网格闭式评估: 碰撞 / ZMP 裕度 / 能耗 / 时间。

    外部依赖：``sm`` 为外部注入的自模型对象（鸭子类型），需提供
    ``energy(traj) -> float``、``static_margin(traj, ds_windows) -> float``、
    ``points_world(row) -> (M,3)``、``box_sdf(pts, center, half) -> (M,)``。
    """
    cands = []
    for bx0 in (-0.06, -0.03, 0.0):
        for le in (0.005, 0.045, 0.08):
            for Tl in (0.8, 1.2):
                if sc.get("double"):
                    traj, meta = gen_double_step(sc["h_obs"], bx0=bx0,
                                                 lift_extra=le, T_lipm=Tl)
                else:
                    traj, meta = gen_step_over(
                        h_obs=sc["h_obs"], bx0=bx0, lift_extra=le, T_lipm=Tl)
                    meta = with_over(meta)
                cands.append({"bx0": bx0, "lift": le, "Tl": Tl,
                              "T": meta["T"], "E": sm.energy(traj),
                              "S": sm.static_margin(traj, meta["ds_windows"]),
                              "C": clearance_multi(sm, traj, sc["boxes"]),
                              "traj": traj, "meta": meta})
    for c in cands:
        c["Sn"] = c["S"] / 0.045                        # 归一化平衡裕度
        lift = sc["h_obs"] + c["lift"]                  # 抬脚绝对高度
        c_req = C_MIN + GAMMA_LIFT * max(0.0, lift - 0.105)
        c["ok"] = (c["C"] > c_req) and (c["Sn"] >= MARGIN_NORM)
    return cands


def pareto_front(cands):
    """(T, E) 双目标最小化的非支配集（源自 m6_pareto.py，纯数学）。"""
    front = []
    for a in cands:
        if not any((b["T"] <= a["T"] and b["E"] <= a["E"] and
                    (b["T"] < a["T"] or b["E"] < a["E"])) for b in cands if b is not a):
            front.append(a)
    return sorted(front, key=lambda c: c["T"])


__all__ = [
    # 常量
    "URDF", "SIM_DT", "DT_TRAJ", "L1", "L2", "HIP_X", "HIP_Z", "MASS", "KP", "KV",
    "FOOT_Y", "H_STAND", "Z_STANCE", "SOLE", "TOE", "HEEL",
    "C_MIN", "GAMMA_LIFT", "MARGIN_NORM", "SCENES",
    # 轨迹生成
    "gen_squat", "gen_step_over", "gen_step_up", "gen_double_step",
    # 运动学 / 工具
    "ik", "smooth", "lift_prof", "_pose", "_np", "rpy_to_quat", "row_to_pin",
    # 评估
    "static_zmp_margin", "clearance_over", "clearance_multi", "eval_scene",
    "make_box", "with_over", "pareto_front",
]
