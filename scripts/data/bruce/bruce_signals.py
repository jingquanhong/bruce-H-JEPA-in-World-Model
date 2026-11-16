"""Bruce 越障采集的信号保真层：把 Genesis 仿真真值翻译成「实物端可采集」的量。

本模块纯 numpy，不依赖 Genesis，可单独单测。

三类列：
  REAL     —— 实物端直接可测（含实物已固化的换算，如 BEAR 电流→力矩）
  DERIVED  —— 实物端由传感器 + 模型推导（此处用「真值 + 估计级噪声」作 EKF 替身）
  SIM-ONLY —— 只有仿真拿得到，**训练禁止使用**，仅用于仿真质检，不能据此估计真实 sim2real gap

实物链路依据（BRUCE-OP 仓库）：
  LEG_JOINT_LIST / IQ2TORQUE / IQ_MAX / TORQUE_MAX : Settings/BRUCE_macros.py:62-63, 317-330
  足底 4 路布尔接触顺序 [R_TOE,R_HEEL,L_TOE,L_HEEL] : Library/BRUCE_SENSE/Manager.py:43,90
  exp_filter(h, present, w) = h*w + present*(1-w)     : Library/Util/math_function.py:165-170
  BEAR 三路读数都用 w=0.90                            : Startups/run_bear.py:254-256
  IMU 比力模型 a1 = Rm @ (am - ba1) - [0,0,g]         : Library/STATE_ESTIMATION/BRUCE_ESTIMATION_AOT.py:155
  Qa=1e-2 (加速度噪声) / Qba=1e-4 (零偏随机游走)      : BRUCE_ESTIMATION_AOT.py HARDWARE 常量
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 关节与常量
# BRUCE-OP Settings/BRUCE_macros.py:62-63 的 LEG_JOINT_LIST（顺序 = 本数据 action 顺序）
LEG_JOINT_LIST = (
    "HIP_YAW_R", "HIP_PITCH_R", "HIP_ROLL_R", "KNEE_PITCH_R", "ANKLE_PITCH_R",
    "HIP_YAW_L", "HIP_PITCH_L", "HIP_ROLL_L", "KNEE_PITCH_L", "ANKLE_PITCH_L",
)
ACTION_DIM = len(LEG_JOINT_LIST)          # 10

# Genesis 22 维 dof 布局: [基座6 | 右腿5 | 左腿5 | 右臂3 | 左臂3]
# 腿部 10 关节 = dof[6:16]，顺序与 LEG_JOINT_LIST 完全一致
LEG_DOF_SLICE = slice(6, 16)

IQ2TORQUE = 0.35          # BEAR 电流(A) → 力矩(N·m) 线性系数
IQ_MAX = 30.0             # 电流上限
TORQUE_MAX = 10.5         # 关节力矩上限 (N·m)
TORQUE_QUANT_STEP = IQ2TORQUE / IQ_MAX   # 仿真代理量化步长，非硬件标定值
IQ_STEP = TORQUE_QUANT_STEP / IQ2TORQUE  # 仿真电流量化步长
EXP_FILTER_W = 0.90       # BEAR 侧一阶低通权重
G = 9.81
FOOT_PARTS = ("R_TOE", "R_HEEL", "L_TOE", "L_HEEL")


# ---------------------------------------------------------------- 工具
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
        [2 * (x * z + y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_yaw(quat_wxyz):
    """四元数 (w,x,y,z) -> 偏航角 yaw (rad)，ZYX 欧拉序。"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def exp_filter(history, present, weight=EXP_FILTER_W):
    """BRUCE-OP Util/math_function.py:165-170 的一阶低通。"""
    return history * weight + present * (1.0 - weight)


def quantize(x, step):
    """对称量化（就近取整），零步长时原样返回。"""
    if step <= 0:
        return np.asarray(x, dtype=np.float64)
    return np.round(np.asarray(x, dtype=np.float64) / step) * step


# ---------------------------------------------------------------- IMU 噪声模型
class ImuModel:
    """PICO 级 IMU 噪声：白噪声 + 零偏随机游走。

    实物链路：Library/BRUCE_SENSE/Manager.py:31（串口帧 acc/gyro）
    噪声幅值由采集配置指定，未用实机测量标定。
    """

    def __init__(self, acc_noise=0.01, gyro_noise=0.01, bias_rw=1e-4, rng=None):
        self.acc_noise = float(acc_noise)
        self.gyro_noise = float(gyro_noise)
        self.bias_rw = float(bias_rw)
        self.rng = rng or np.random.default_rng()

    def reset(self):
        self.bias_acc = np.zeros(3)
        self.bias_gyro = np.zeros(3)

    def step(self):
        """零偏随机游走 + 本帧白噪声。返回 (acc_noise_vec, gyro_noise_vec)。"""
        self.bias_acc = self.bias_acc + self.rng.normal(0.0, self.bias_rw, 3)
        self.bias_gyro = self.bias_gyro + self.rng.normal(0.0, self.bias_rw, 3)
        return (self.bias_acc + self.rng.normal(0.0, self.acc_noise, 3),
                self.bias_gyro + self.rng.normal(0.0, self.gyro_noise, 3))


# ---------------------------------------------------------------- 关节编码器模型
class JointEncoderModel:
    """BEAR 电机侧读数：编码器量化 + 一阶低通。

    实物链路：Startups/run_bear.py:221-250（present_position / present_velocity）
    滤波口径：run_bear.py:254-256 对 position/velocity/iq 三路都用 weight=0.90
    """

    def __init__(self, q_noise=0.002, dq_noise=0.002, weight=EXP_FILTER_W):
        self.q_noise = float(q_noise)
        self.dq_noise = float(dq_noise)
        self.weight = float(weight)

    def reset(self):
        self.q_filt = None
        self.dq_filt = None

    def step(self, q, dq):
        """q/dq: (10,) 真值。返回 (量化+低通后的 q, dq)。"""
        q = quantize(q, self.q_noise)
        dq = quantize(dq, self.dq_noise)
        if self.q_filt is None:
            self.q_filt, self.dq_filt = q.copy(), dq.copy()
        else:
            self.q_filt = exp_filter(self.q_filt, q, self.weight)
            self.dq_filt = exp_filter(self.dq_filt, dq, self.weight)
        return self.q_filt.copy(), self.dq_filt.copy()


# ---------------------------------------------------------------- 力矩估计模型
class TorqueEstimateModel:
    """实物没有力矩传感器，只有 BEAR 电流；按实物链路复刻「电流→力矩」估计。

    链路：真实力矩 → /IQ2TORQUE 得电流 → 按 IQ_MAX 量化 → ×IQ2TORQUE
          → 每关节固定 ±5% 增益误差 → 饱和到 ±TORQUE_MAX
    """

    def __init__(self, gain_std=0.05, rng=None):
        self.gain_std = float(gain_std)
        self.rng = rng or np.random.default_rng()

    def reset(self):
        # 增益误差是硬件标定常量，一集内固定（不是逐帧噪声）
        self.gain = 1.0 + self.rng.normal(0.0, self.gain_std, ACTION_DIM)

    def step(self, tau_true):
        """tau_true: (10,) 真实力矩 (N·m)。返回实物侧的力矩估计 (10,)。"""
        iq = np.clip(np.asarray(tau_true, dtype=np.float64) / IQ2TORQUE,
                     -IQ_MAX, IQ_MAX)
        iq = quantize(iq, IQ_STEP)        # 电流 ADC 量化 (0.0333 A/LSB)
        tau = iq * IQ2TORQUE              # → 力矩步长 0.011667 N·m
        tau = tau * self.gain
        return np.clip(tau, -TORQUE_MAX, TORQUE_MAX)


# ---------------------------------------------------------------- 状态估计噪声模型
class EstimatorNoiseModel:
    """DERIVED 列的「EKF 替身」：真值 + 估计级噪声。

    实物侧实机跑 BRUCE_ESTIMATION_AOT 的 EKF（500 Hz），仿真侧无参考系，
    故第一版用带噪真值代替；Stage B 可换成真实 EKF。
    """

    def __init__(self, pos_sigma=0.01, vel_sigma=0.05, acc_sigma=0.10,
                 yaw_sigma=np.deg2rad(2.0), rng=None):
        self.pos_sigma = float(pos_sigma)
        self.vel_sigma = float(vel_sigma)
        self.acc_sigma = float(acc_sigma)
        self.yaw_sigma = float(yaw_sigma)
        self.rng = rng or np.random.default_rng()

    def step(self, pos, vel, acc, yaw):
        return (np.asarray(pos) + self.rng.normal(0, self.pos_sigma, 3),
                np.asarray(vel) + self.rng.normal(0, self.vel_sigma, 3),
                np.asarray(acc) + self.rng.normal(0, self.acc_sigma, 3),
                float(yaw + self.rng.normal(0, self.yaw_sigma)))


# ---------------------------------------------------------------- 列定义
# REAL 列：名字 -> 维度
REAL_COLUMNS = {
    "action": ACTION_DIM,
    "imu_acc": 3,
    "imu_gyro": 3,
    "foot_contact": 4,
    "leg_q": ACTION_DIM,
    "leg_dq": ACTION_DIM,
    "joint_torque_est": ACTION_DIM,
}
# DERIVED 列
DERIVED_COLUMNS = {
    "est_body_pos": 3,
    "est_body_vel": 3,
    "est_body_acc": 3,
    "est_yaw": 1,
}
# SIM-ONLY 列（训练禁止使用）
SIM_COLUMNS = {
    "sim_body_pos": 3,
    "sim_body_vel": 3,
    "sim_base_quat": 4,
    "sim_contact_force": 4,
    "sim_joint_torque": ACTION_DIM,
    "sim_ankle_pos": 4,        # [右踝x, 右踝z, 左踝x, 左踝z]，越障净空离线复核用
}
ALL_COLUMNS = {**REAL_COLUMNS, **DERIVED_COLUMNS, **SIM_COLUMNS}


class SignalBuilder:
    """把一帧仿真真值转成所有数据列。跨帧状态（滤波器/零偏）在本类内部维护。

    用法：
        sb = SignalBuilder(cfg, rng)
        sb.reset_episode()
        cols = sb.build(state, action22)
    """

    def __init__(self, cfg=None, rng=None):
        cfg = cfg or {}
        self.rng = rng or np.random.default_rng()
        self.imu = ImuModel(
            acc_noise=cfg.get("imu_acc_noise", 0.01),
            gyro_noise=cfg.get("imu_gyro_noise", 0.01),
            bias_rw=cfg.get("imu_bias_rw", 1e-4),
            rng=self.rng,
        )
        self.encoder = JointEncoderModel(
            q_noise=cfg.get("joint_q_noise", 0.002),
            dq_noise=cfg.get("joint_dq_noise", 0.002),
            weight=cfg.get("exp_filter_w", EXP_FILTER_W),
        )
        self.torque = TorqueEstimateModel(
            gain_std=cfg.get("torque_gain_std", 0.05), rng=self.rng)
        self.estimator = EstimatorNoiseModel(
            pos_sigma=cfg.get("est_pos_sigma", 0.01),
            vel_sigma=cfg.get("est_vel_sigma", 0.05),
            acc_sigma=cfg.get("est_acc_sigma", 0.10),
            yaw_sigma=cfg.get("est_yaw_sigma", np.deg2rad(2.0)),
            rng=self.rng,
        )
        self.foot_thresh = float(cfg.get("foot_contact_force_thresh", 5.0))

    def reset_episode(self):
        self.imu.reset()
        self.encoder.reset()
        self.torque.reset()

    def build(self, state, action22):
        """state: bruce_env.read_state() 的字典；action22: (22,) 本帧下发的 dof 目标。

        返回 {列名: numpy 值}，维度见 REAL_COLUMNS / DERIVED_COLUMNS / SIM_COLUMNS。
        """
        quat = state["base_quat"]
        R = quat_to_mat(quat)                       # body -> world

        # ---- IMU：世界系动力学加速度（含重力）→ 机体系比力 ----
        acc_world = np.asarray(state["base_acc"], dtype=np.float64)
        specific_force_world = acc_world + np.array([0.0, 0.0, G])
        acc_body = R.T @ specific_force_world
        gyro_body = R.T @ np.asarray(state["base_ang"], dtype=np.float64)
        n_acc, n_gyro = self.imu.step()
        imu_acc = acc_body + n_acc
        imu_gyro = gyro_body + n_gyro

        # ---- 足底接触：仿真接触力阈值化为布尔（PICO 只有开关，没有力） ----
        f = np.asarray(state["foot_force"], dtype=np.float64)
        foot_contact = (f > self.foot_thresh).astype(np.float32)

        # ---- 腿关节角 / 角速度：编码器量化 + 一阶低通 ----
        dof_pos = np.asarray(state["dof_pos"], dtype=np.float64)
        dof_vel = np.asarray(state["dof_vel"], dtype=np.float64)
        leg_q, leg_dq = self.encoder.step(dof_pos[LEG_DOF_SLICE], dof_vel[LEG_DOF_SLICE])

        # ---- 力矩估计：真值 -> 电流 -> 量化 -> 增益误差 -> 饱和 ----
        tau_true = np.asarray(state["dof_force"], dtype=np.float64)[LEG_DOF_SLICE]
        joint_torque_est = self.torque.step(tau_true)

        # ---- DERIVED：真值 + 估计级噪声 ----
        yaw = quat_to_yaw(quat)
        est_pos, est_vel, est_acc, est_yaw = self.estimator.step(
            state["base_pos"], state["base_vel"], acc_world, yaw)

        # ---- action：腿部关节位置目标（= 下发给实机的指令） ----
        action = np.asarray(action22, dtype=np.float64)[LEG_DOF_SLICE].copy()

        # ---- SIM-ONLY：踝原点 xz（净空复核 / 剖面图离线用） ----
        ankle = state.get("ankle_pos", {}) or {}
        ar = ankle.get("ankle_pitch_link_r", (np.nan, np.nan))
        al = ankle.get("ankle_pitch_link_l", (np.nan, np.nan))
        sim_ankle_pos = np.array([ar[0], ar[1], al[0], al[1]], dtype=np.float64)

        return {
            # REAL
            "action": action,
            "imu_acc": imu_acc,
            "imu_gyro": imu_gyro,
            "foot_contact": foot_contact,
            "leg_q": leg_q,
            "leg_dq": leg_dq,
            "joint_torque_est": joint_torque_est,
            # DERIVED
            "est_body_pos": est_pos,
            "est_body_vel": est_vel,
            "est_body_acc": est_acc,
            "est_yaw": np.array([est_yaw], dtype=np.float64),
            # SIM-ONLY
            "sim_body_pos": np.asarray(state["base_pos"], dtype=np.float64),
            "sim_body_vel": np.asarray(state["base_vel"], dtype=np.float64),
            "sim_base_quat": np.asarray(quat, dtype=np.float64),
            "sim_contact_force": f,
            "sim_joint_torque": tau_true,
            "sim_ankle_pos": sim_ankle_pos,
        }


__all__ = [
    "LEG_JOINT_LIST", "ACTION_DIM", "LEG_DOF_SLICE", "IQ2TORQUE", "IQ_MAX",
    "TORQUE_MAX", "TORQUE_QUANT_STEP", "IQ_STEP", "EXP_FILTER_W", "G", "FOOT_PARTS",
    "quat_to_mat", "quat_to_yaw", "exp_filter", "quantize",
    "ImuModel", "JointEncoderModel", "TorqueEstimateModel", "EstimatorNoiseModel",
    "REAL_COLUMNS", "DERIVED_COLUMNS", "SIM_COLUMNS", "ALL_COLUMNS",
    "SignalBuilder",
]
