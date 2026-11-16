"""Genesis BRUCE scene, joint actuation, optional assisted reference and ground-truth reads.

Frames: base velocities/accelerations are world-frame; quaternion order is wxyz.
Foot contact groups follow BRUCE-OP: R_TOE, R_HEEL, L_TOE, L_HEEL."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = (_REPO_ROOT / "BRUCE-OP-main" / "Simulation" / "models" / "bruce"
                 / "urdf" / "bruce_physical.urdf")

# 足底接触点的排列顺序（= BRUCE-OP PICO 位图解包顺序，见 BRUCE_SENSE/Manager.py）
FOOT_PARTS = ("R_TOE", "R_HEEL", "L_TOE", "L_HEEL")
ANKLE_LINKS = ("ankle_pitch_link_r", "ankle_pitch_link_l")


def to_np(x):
    """Genesis 返回值统一转 numpy（其内部是 CUDA torch 张量）。"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------- URDF 解析
def parse_urdf_collisions(urdf_path):
    """按 XML 顺序展平 URDF 的所有碰撞体。

    Genesis 分配 global geom index 的顺序 = link 顺序 × link 内 collision 顺序，
    与本函数展平顺序一一对应，因此可以用它把 Genesis 的 geom 索引对齐回
    「哪个 link 的第几个碰撞体」。

    返回 list[dict]：{link, geom_type, origin_xyz}
    """
    root = ET.parse(str(urdf_path)).getroot()
    out = []
    for link in root.findall("link"):
        lname = link.get("name")
        for coll in link.findall("collision"):
            geo = coll.find("geometry")
            if geo is None:
                continue
            if geo.find("mesh") is not None:
                gtype = "MESH"
            elif geo.find("sphere") is not None:
                gtype = "SPHERE"
            elif geo.find("box") is not None:
                gtype = "BOX"
            elif geo.find("cylinder") is not None:
                gtype = "CYLINDER"
            else:
                gtype = "OTHER"
            org = coll.find("origin")
            xyz = (0.0, 0.0, 0.0)
            if org is not None and org.get("xyz"):
                xyz = tuple(float(v) for v in org.get("xyz").split())
            out.append({"link": lname, "geom_type": gtype, "origin_xyz": xyz})
    return out


def _foot_part_of(link_name, origin_xyz, geom_type="SPHERE"):
    """判定一个踝部碰撞体是脚趾还是脚跟（只针对两个球）。

    URDF 里踝 link 的两个球：
        toe  : origin = (0.025, +0.050, 0)   —— 前方 5cm
        heel : origin = (0.025, -0.035, 0)   —— 后方 3.5cm
    （与 bruce_expert 的 TOE=0.056 / HEEL=0.0385 常量互为印证）

    mesh 碰撞体（origin 0,0,0）不参与 toe/heel 静态归属：它的 y=0 会被
    误判成脚跟。mesh 上的接触改由「接触点相对踝原点的 y 符号」动态判定
    （见 BruceScene._mesh_contact_part）。
    """
    if geom_type != "SPHERE":
        return None
    y = float(origin_xyz[1])
    if link_name == "ankle_pitch_link_r":
        return "R_TOE" if y > 0 else "R_HEEL"
    if link_name == "ankle_pitch_link_l":
        return "L_TOE" if y > 0 else "L_HEEL"
    return None


# ---------------------------------------------------------------- 场景封装
class BruceScene:
    """Genesis 场景：地面 + 障碍箱 + BRUCE 机器人。

    一条约束：**gs.init 全进程只能调用一次**，因此本类设计为「建一次场景、
    每 episode 用 set_pos / set_qpos 复用途此」，不重建 scene。
    """

    def __init__(self, urdf=None, sim_dt=5e-4, substeps=1, seed=0,
                 backend="gpu", show_viewer=False, box_specs=None,
                 control_mode="joint_only", camera=False, default_armature=0.0,
                 leg_armature=0.00182):
        """box_specs: list[{"half": (hx,hy,hz), "pos": (x,y,z), "friction": float}]

        Genesis 1.4.4 **不允许 build 之后再加实体**，所以所有障碍箱必须在
        建场时一次性加好（每个场景一个，运行时把不用的挪走）。
        """
        import genesis as gs

        self.gs = gs
        self.urdf = str(urdf or os.environ.get("BRUCE_URDF", DEFAULT_URDF))
        self.sim_dt = float(sim_dt)
        self.substeps = int(substeps)
        self.seed = int(seed)
        if control_mode not in ("joint_only", "assisted"):
            raise ValueError(f"Unknown control mode: {control_mode}")
        self.control_mode = control_mode

        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu, seed=self.seed)

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=self.substeps),
            rigid_options=gs.options.RigidOptions(dt=self.sim_dt),
            show_FPS=False,
        )
        self.scene.add_entity(morph=gs.morphs.Plane())

        # 障碍箱：建场时一次加完（顺序即 box_specs 顺序）
        self.boxes = []
        for spec in (box_specs or []):
            half = np.asarray(spec["half"], dtype=np.float64)
            pos = np.asarray(spec["pos"], dtype=np.float64)
            box = self.scene.add_entity(
                morph=gs.morphs.Box(lower=tuple(pos - half), upper=tuple(pos + half)))
            if spec.get("friction") is not None:
                box.set_friction(float(spec["friction"]))
            self.boxes.append(box)

        self.robot = self.scene.add_entity(morph=gs.morphs.URDF(
            file=self.urdf, default_armature=default_armature))
        self.camera = self.scene.add_camera(
            res=(640, 480), pos=(1.15, -1.5, 0.85),
            lookat=(0.14, 0.0, 0.26), fov=42, GUI=False,
        ) if camera else None
        self.scene.build()
        # BRUCE_macros.REFLECTED_INERTIA；对角近似，尚未建模电机差动耦合惯量。
        self.robot.set_dofs_armature(np.full(10, leg_armature),
                                     dofs_idx_local=np.arange(6, 16))

        self._init_geom_maps()
        self._init_link_indices()

    # ---- 初始化辅助 ----
    def _init_geom_maps(self):
        """建立 global geom index -> (link 名, 足底部位) 的映射。

        做法：解析 URDF 得到 collision 展平顺序，与 Genesis 的 geom 列表逐项对齐
        （长度、link 名、几何类型三项都必须一致，否则说明顺序假设失效，直接报错）。
        """
        urdf_rows = parse_urdf_collisions(self.urdf)
        geoms = self.scene.rigid_solver.geoms
        gs_rows = []
        for gi in range(self.robot.geom_start, self.robot.geom_end):
            g = geoms[gi]
            gtype = str(g.type).rsplit(".", 1)[-1].replace("GEOM_TYPE.", "")
            gs_rows.append((gi, g.link.name, gtype))
        if len(urdf_rows) != len(gs_rows):
            raise RuntimeError(
                f"URDF collision 数({len(urdf_rows)}) != Genesis geom 数({len(gs_rows)})，"
                "URDF/Genesis 对齐失败")
        for u, (gi, lname, gtype) in zip(urdf_rows, gs_rows):
            if u["link"] != lname or u["geom_type"] != gtype:
                raise RuntimeError(
                    f"URDF/Genesis 碰撞体顺序不一致: URDF={u} vs Genesis={(gi, lname, gtype)}")
        # geom index -> 足底部位 index（0..3，对应 FOOT_PARTS）
        self.geom_part = {}
        # geom index -> 所属 link 名（mesh 接触归属判定要用）
        self.geom_link = {}
        # 踝部的 mesh 碰撞体（origin y=0，不能静态归 toe/heel，按接触点 y 动态判）
        self.ankle_mesh_geoms = set()
        for u, (gi, lname, _t) in zip(urdf_rows, gs_rows):
            self.geom_link[gi] = lname
            part = _foot_part_of(u["link"], u["origin_xyz"], u["geom_type"])
            if part is not None:
                self.geom_part[gi] = FOOT_PARTS.index(part)
            elif lname in ANKLE_LINKS and u["geom_type"] == "MESH":
                self.ankle_mesh_geoms.add(gi)

    def _init_link_indices(self):
        """机器人内部 link 索引：以 Genesis 自己建立的 link 列表为准（权威）。

        Genesis 会合并 URDF 里的 fixed link，其 link 顺序/数量与 URDF XML
        未必一致，所以不能拿 URDF 的 link 顺序去索引 links_idx_local。
        """
        self.link_names = [l.name for l in self.robot.links]
        self.link_index = {name: i for i, name in enumerate(self.link_names)}
        self.ankle_idx = {n: self.link_index[n] for n in ANKLE_LINKS if n in self.link_index}
        missing = [n for n in ANKLE_LINKS if n not in self.ankle_idx]
        if missing:
            raise RuntimeError(f"URDF 中缺少踝部 link: {missing}")

    # ---- 障碍箱 ----
    def park_boxes(self, boxes, keep=0):
        """把除 keep 之外的所有箱体挪到远处，只留当前场景的箱子在场内。"""
        for i, b in enumerate(boxes):
            if i == keep:
                continue
            self.reset_box(b, (0.0, 5.0, -1.0))

    def set_box_friction(self, box, value):
        box.set_friction(float(value))

    def reset_box(self, box, pos):
        """把箱体位置与姿态复位到 (pos) 并清零速度（用于每 episode 重新摆障碍）。"""
        n_qs = box.n_qs
        q = np.zeros(n_qs)
        q[0:3] = pos
        if n_qs >= 7:
            q[3] = 1.0
        box.set_qpos(q, zero_velocity=True)

    def box_pos(self, box):
        return to_np(box.get_pos())

    # ---- 机器人控制 ----
    def set_gains(self, kp, kv):
        kp, kv = np.array(kp, dtype=float), np.array(kv, dtype=float)
        if self.control_mode == "joint_only":
            # 浮动基座不可驱动；限制的是实际执行力矩，而非只裁剪观测值。
            kp[:6] = kv[:6] = 0.0
            limits = np.array([0.0] * 6 + [10.5] * 10 + [3.0] * 6)
            self.robot.set_dofs_force_range(-limits, limits)
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)

    def set_friction(self, value):
        self.robot.set_friction(float(value))

    def reset_robot(self, base_z, dof_pos=None, base_x=0.0, base_y=0.0):
        """浮动基座复位：基座置于 (base_x, base_y, base_z)，四元数 w=1，速度清零。

        dof_pos 为 16 个关节角（rad）；None 表示全零（直腿）。
        """
        n_qs = self.robot.n_qs
        q = np.zeros(n_qs)
        q[0] = float(base_x)
        q[1] = float(base_y)
        q[2] = float(base_z)
        q[3] = 1.0
        if dof_pos is not None:
            q[7:7 + len(dof_pos)] = np.asarray(dof_pos, dtype=np.float64)
        self.robot.set_qpos(q, zero_velocity=True)

    def apply_target(self, target22):
        """辅助模式允许基座控制；正式模式仅驱动 16 个实体关节。"""
        target = np.asarray(target22, dtype=np.float64)
        if self.control_mode == "joint_only":
            self.robot.control_dofs_position(target[6:], dofs_idx_local=np.arange(6, 22))
        else:
            self.robot.control_dofs_position(target)

    def step(self):
        self.scene.step()

    # ---- 状态回读 ----
    def read_state(self):
        """回读一帧传感器真值（全部 numpy，世界系除非注明）。"""
        robot = self.robot
        base_pos = to_np(robot.get_links_pos(links_idx_local=[0], relative=False))[0]
        base_quat = to_np(robot.get_links_quat(links_idx_local=[0], relative=False))[0]
        base_vel = to_np(robot.get_links_vel(links_idx_local=[0], relative=False))[0]
        base_acc = to_np(robot.get_links_acc(links_idx_local=[0], relative=False))[0]
        base_ang = to_np(robot.get_links_ang(links_idx_local=[0]))[0]
        dof_pos = to_np(robot.get_dofs_position())
        dof_vel = to_np(robot.get_dofs_velocity())
        # 真实力矩只能取 control_force：get_dofs_force 对关节 dof 返回 ~1e-5（不可用）
        dof_force = to_np(robot.get_dofs_control_force())
        return {
            "base_pos": base_pos,          # (3,) m
            "base_quat": base_quat,        # (4,) wxyz
            "base_vel": base_vel,          # (3,) m/s 世界系
            "base_acc": base_acc,          # (3,) m/s^2 经典线加速度（不含重力）
            "base_ang": base_ang,          # (3,) rad/s 世界系角速度
            "dof_pos": dof_pos,            # (22,) 基座6 + 关节16
            "dof_vel": dof_vel,            # (22,)
            "dof_force": dof_force,        # (22,) PD 实际施加的广义力（N / N·m）
            "foot_force": self.read_foot_forces(),
            "ankle_pos": self.read_ankle_pos(),
        }

    def get_contacts(self):
        """回读当前所有接触对（已排除自碰撞）。无接触时返回 None。"""
        try:
            c = self.robot.get_contacts(exclude_self_contact=True)
        except Exception:
            return None
        if c is None or len(c["geom_a"]) == 0:
            return None
        return {k: to_np(v) for k, v in c.items()}

    def _ankle_frame(self, link_name):
        """踝 link 的 (世界原点, 局部 y 轴在世界系的方向)。"""
        idx = self.ankle_idx[link_name]
        p = to_np(self.robot.get_links_pos(links_idx_local=[idx], relative=False))[0]
        q = to_np(self.robot.get_links_quat(links_idx_local=[idx], relative=False))[0]
        return p, quat_to_mat(q)[:, 1]

    def _mesh_contact_part(self, geom_idx, p_world):
        """踝 mesh 上的接触点归属到脚趾还是脚跟：按接触点相对踝原点的 y 符号。"""
        link_name = self.geom_link.get(int(geom_idx))
        if link_name not in self.ankle_idx:
            return None
        origin, y_axis = self._ankle_frame(link_name)
        dy = float(np.dot(np.asarray(p_world) - origin, y_axis))
        if link_name == "ankle_pitch_link_r":
            return 0 if dy > 0 else 1      # R_TOE / R_HEEL
        return 2 if dy > 0 else 3          # L_TOE / L_HEEL

    def read_foot_forces(self):
        """足底 4 点法向力（N），顺序 FOOT_PARTS = (R_TOE, R_HEEL, L_TOE, L_HEEL)。

        接触对的法向力大小 = |dot(作用在该 geom 上的力, normal)|。
        机器人自身接触（自碰撞）被排除。

        默认模型仅使用原始脚部网格，按踝局部 y 符号拆分接触。
        兼容历史模型附加球的静态分组。
        """
        out = np.zeros(len(FOOT_PARTS))
        c = self.get_contacts()
        if c is None:
            return out
        ga, gb = c["geom_a"], c["geom_b"]
        fa, fb = c["force_a"], c["force_b"]
        nrm = c["normal"]
        pos = c.get("position")
        g0, g1 = self.robot.geom_start, self.robot.geom_end
        for i in range(len(ga)):
            if g0 <= ga[i] < g1:
                force, gi = fa[i], ga[i]
            elif g0 <= gb[i] < g1:
                force, gi = fb[i], gb[i]
            else:
                continue
            gi = int(gi)
            if gi in self.ankle_mesh_geoms:
                if pos is None:
                    continue
                part = self._mesh_contact_part(gi, pos[i])
            else:
                part = self.geom_part.get(gi)
            if part is None:
                continue
            out[part] += abs(float(np.dot(force, nrm[i])))
        return out

    def read_ankle_pos(self):
        """左右踝 link 原点世界位置 (x, z)，用于越障净空复核。"""
        out = {}
        idxs = [self.ankle_idx[n] for n in ANKLE_LINKS if n in self.ankle_idx]
        names = [n for n in ANKLE_LINKS if n in self.ankle_idx]
        if not idxs:
            return out
        pos = to_np(self.robot.get_links_pos(links_idx_local=idxs, relative=False))
        for name, p in zip(names, pos):
            out[name] = (float(p[0]), float(p[2]))
        return out

    def get_link_pos(self, name):
        idx = self.ankle_idx.get(name)
        if idx is None:
            return None
        return to_np(self.robot.get_links_pos(links_idx_local=[idx], relative=False))[0]


def quat_to_mat(quat_wxyz):
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵（body -> world）。"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
