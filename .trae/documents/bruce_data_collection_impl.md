# Bruce 越障数据采集器实施计划（Stage 1 + Stage 2）

## Context

上一轮已完成《bruce_obstacle_data_collection_plan.md》（已获批），并把采集器所依赖的**全部物理语义与接口**用 Genesis 1.4.4 实测探针摸清。本轮用户要求：**真正把数据采出来**，同时**呈现采集源码**，且**呈现方式不能是视频**（用静态图 / 表格 / 内联可视化）。

当前仓库状态：`scripts/data/bruce/` 下只有两个已就绪的模块，采集器本体（信号层 / 采集主程序 / 质检脚本 / 配置）**尚未创建**，一行数据都没落盘。

已确定的关键事实（探针实测，是本设计的前提）：

1. **free joint 6 dof 会被 `control_dofs_position` 驱动** → m5/m6 的 `replay_collect` 对含基座的 22 维全量下发，本质是「傀儡模式」。采集器必须复刻，否则越障行为与已验证结果不一致。
2. **`get_dofs_force` 对关节 dof 返回 ≈1e-5（不可用）**；真实力矩必须取 **`get_dofs_control_force`**（站立时腿部 ≈2.0~2.25 N·m，基座 dof[2] = −258 N）。
3. **接触落在 ankle mesh + toe 球上，不是 heel 球**：站立时接触对 `(plane, geom7/8/14/15)`，4 个法向力 42.8/108.3/42.8/108.4 N（合计 ≈302 N ≈ 44 N 自重 + 258 N 基座 PD 弹簧）。现有 `bruce_env.geom_part` 只映射球 → **会丢掉 mesh 上的 85.6 N**，必须修。
4. 基座 PD 会被地面反压出 ~258 N 额外向下压力 → 接触力量级由基座弹簧主导，这是「傀儡模式」的固有特征，须在报告中声明，不能声称是纯自由站立动力学。

已核对的数据契约（`stable_worldmodel/world.py` / `data/dataset.py`）：

- `ep_offset`(int64) / `ep_len`(int32) / `ep_idx`(int32) 必写；`step_idx` 为帧号列。
- 建列规则：`ndim>=2` → chunks `(100,)+shape` + `hdf5plugin.Blosc(lz4, clevel=5, SHUFFLE)`；1D → chunks `(1000,)+shape`，不压缩。
- 文件：`mode='a' if exists else 'w'`、`libver='latest'`、新建时 `fs_strategy='page'` + `fs_page_size=4MB`、`swmr_mode=True`、每集 `f.flush()`。
- 续采指针 = `ep_offset[-1] + ep_len[-1]`；seed 顺延 `seed + n_ep_recorded`。
- **action 左移**：buffer 里 action 比观测晚一拍，收尾时 `pop(0)` + `append(原值)` → 落盘第 t 行 = `(o_t, a_t)`。

---

## 实施步骤

### Step 1 — 修 `scripts/data/bruce/bruce_env.py`（唯一需要改动的既有文件）

1. **足底力归属修正**：`_init_geom_maps` 额外记录 `self.geom_link: {gi: link_name}` 与 ankle 的 mesh geom 集合；
   `read_foot_forces` 中遇到 ankle mesh 的接触时，用 `get_contacts` 的 `position` 减去该踝 link 原点、投影到该 link 的**局部 y 轴**（由 `get_links_quat` + `quat_to_mat` 得到），`y>0 → TOE`、`y<0 → HEEL`；toe/heel 球沿用原映射。保持 4 路输出语义与 BRUCE-OP `BRUCE_SENSE/Manager.py:43,90` 的 PICO 位图顺序 `[R_TOE, R_HEEL, L_TOE, L_HEEL]` 一致。
2. **`read_state` 的力矩列改为 `get_dofs_control_force`**（真实力矩来源），保留 `dof_force` 字段名不变但语义改为「PD 实际施加的关节力矩」，杜绝下游误用旧接口。
3. **新增多箱管理**：`add_boxes(specs)` 一次性建多个箱（每个场景一个：4cm/6cm/8cm/10cm 台阶）并只 `scene.build()` 一次；`park_boxes()` 把非当前场景的箱体挪到 `y=+5 m`；`set_box_friction()`。箱体保持**动态刚体**（可被踢翻 → 失败样本入库，用户已确认）。
4. 清理两处未使用变量告警（`add_box` 的 `size`、`read_state` 的 `qpos`）。
5. `read_state` 增补 `base_ang`（已有）、`dof_control_force`，并暴露 `read_contact_positions()` 供净空复核。

### Step 2 — 新建 `scripts/data/bruce/bruce_signals.py`

把「仿真真值」翻译成「实物可采集」的信号，纯 numpy，无 Genesis 依赖，可单测。

| 类/函数 | 职责 | 实物依据 |
|---|---|---|
| 常量 | `LEG_JOINT_LIST`（= `BRUCE_macros.py:62-63`，10 维序 = URDF 前 10 关节 = Genesis dof 6..15）、`ACTION_DIM=10`、`IQ2TORQUE=0.35`、`IQ_MAX=30`、`TORQUE_MAX=10.5`、`EXP_FILTER_W=0.90`、`G=9.81` | 同左 |
| `ImuModel` | 机体系原始读数：`acc_body = Rᵀ(a_world + [0,0,g])`、`gyro_body = Rᵀ ω_world`；叠加 σ=0.01 白噪（`Qa`）与 σ=1e-4 零偏随机游走（`Qba`） | `BRUCE_ESTIMATION_AOT.py:155` |
| `JointEncoderModel` | 关节角/角速度：0.002 rad 量化 + 一阶低通 `h = 0.90·h_prev + 0.10·present` | `run_bear.py:254-256` |
| `TorqueEstimateModel` | 真实力矩 → `/0.35` 得电流 → 按 `IQ_MAX=30` 量化（步长 0.35/30=0.01167）→ `×0.35` → 每关节固定 ±5% 增益误差 → 饱和 ±10.5 N·m | `BRUCE_macros.py:317-330` |
| `EstimatorNoiseModel` | `est_*` = 真值 + 估计级噪声（位置 1 cm/1σ、姿态 2°/1σ、速度 0.05 m/s/1σ） | 替代实机 EKF 的替身 |
| `SignalBuilder` | 输入 `bruce_env.read_state()` 的原始字典 → 输出 REAL / DERIVED / SIM-ONLY 三组列 | — |

产出列（与方案 §2.5 一一对应）：
- **REAL**：`action(10)`、`imu_acc(3)`、`imu_gyro(3)`、`foot_contact(4)`、`leg_q(10)`、`leg_dq(10)`、`joint_torque_est(10)`
- **DERIVED**：`est_body_pos(3)`、`est_body_vel(3)`、`est_body_acc(3)`、`est_yaw(1)`
- **SIM-ONLY**（训练禁止使用）：`sim_body_pos(3)`、`sim_body_vel(3)`、`sim_base_quat(4)`、`sim_contact_force(4)`、`sim_joint_torque(10)`

### Step 3 — 新建 `scripts/data/collect_bruce.py` + `config/bruce_stepover.yaml`

**argparse + PyYAML，零 `stable_worldmodel` 依赖**（跑在 `nudt` 环境 py3.10）。

- 场景表：`box_04 / box_06 / box_08 / step_10`（几何取自 `bruce_expert.make_box`，`h_obs` 与专家 `gen_step_over(h_obs=…)` 严格一致）。
- 域随机化（YAML 可调）：`box_x_jitter 0.02`、`bx0 ∈ {-0.06,-0.03,0}`、`lift_extra ∈ {0.005,0.045,0.08}`、`T_lipm ∈ {0.8,1.2}`、`friction ∈ [0.6,1.2]`、`kp_scale ∈ [0.85,1.15]`。
- 每条 episode：
  1. 采样场景 + DR 参数 → `bruce_expert.gen_step_over` / `gen_step_up` 生成闭式轨迹（**纯解析解，零训练零 RL**）；
  2. `reset_robot(base_z=H, dof_pos=traj[0][6:22])` + 摆箱 + 设摩擦/增益；
  3. **2000 Hz 重放 / 100 Hz 记录**：每帧 20 个仿真步；每步线性插值 22 维目标 → `control_dofs_position` → `scene.step()`；每 20 步回读一帧真值并生成所有列；
  4. 用 `bruce_expert.clearance_over` 复核越障净空；按 m6 判据检测箱体被撞移/踢倒（位移 > 0.02 m 或 z 下沉 > 0.01 m）；
  5. 写 `ep_meta`（场景 id / 箱高 / 箱位 / bx0 / lift_extra / T_lipm / 实测净空 / 成功与否 / 箱体是否被踢）+ 侧车 `<name>.meta.jsonl`。
- **HDF5 写出严格复刻 `world.py` 契约**：列创建规则、Blosc 压缩、chunks、`ep_offset/ep_len/ep_idx`、`swmr_mode=True`、每集 `flush()`、续采指针与 seed 顺延；**action 左移**按 `world.py:937-949` 原样实现（先让 action 晚一拍入 buffer，收尾 `pop(0)+append(原值)`），保证落盘第 t 行是 `(o_t, a_t)`。
- 附加列：`step_idx`（帧号）、`episode_meta` 走 `ep_meta` 数据集 + 侧车 JSONL。
- 参数：`--config --episodes --out --seed --resume`。

### Step 4 — 新建 `scripts/data/bruce/check_dataset.py`

质检（全部自动化，终端表格 + JSON 报告）：
列齐全性/维度/dtype、`ep_len` 单调且 `sum(ep_len)==总帧数`、`ep_offset` 与 `ep_len` 自洽、无 NaN、**action[t] == 第 t 帧下发的腿部目标**（对齐校验）、`foot_contact` 与 `sim_contact_force` 阈值一致、逐场景成功率/净空分布/箱体被踢率、REAL vs SIM 差值统计。

### Step 5 — 跑冒烟 + 小批量采集（本轮目标：5 + 50 条）

1. **冒烟 5 ep**：`~/miniconda3/envs/nudt/bin/python scripts/data/bruce/collect_bruce.py --config ... --episodes 5`。
   重点核对：站立段 `imu_acc_z ≈ +9.81`（验证重力符号与 `get_links_acc` 语义，若不是则改用「经典加速度 + 重力补偿」的另一支公式）；`foot_contact` 支撑腿常 1 / 摆动腿 0；`leg_q` 在限位内；`action` 与下发目标逐帧一致。
2. **小批量 50 ep**（用户已确认本轮规模），产出 `bruce_stepover_50.h5`。
3. `check_dataset.py` 全绿。

### Step 6 — 非视频呈现（本轮硬性验收项）

1. **matplotlib 静态图**（`scripts/data/bruce/figures/`，Agg 后端，无视频）：
   - `traj_xz.png`：x–z 剖面 —— 基座轨迹 + 左右踝轨迹 + 箱体轮廓 + 净空标注（4 个场景子图）；
   - `timeseries.png`：`imu_acc_z`、`foot_contact`、`leg_q`、`joint_torque_est` 典型 episode 时序；
   - `dataset_stats.png`：逐场景成功率柱状图、净空分布直方图、步长/抬脚参数分布。
2. **终端表格**：`check_dataset.py` 的质检摘要直接贴进对话。
3. **内联可视化**：用 `dynamic-ui` skill + `PureShowWidget` 渲染一张数据总览（成功率/净空/时间轴示意），随回复直接可见。
4. **采集源码呈现**：最终回复中给出 `collect_bruce.py` / `bruce_signals.py` / `bruce_env.py` 的关键源码片段 + 可点击文件链接。

---

## 涉及文件

| 文件 | 动作 |
|---|---|
| `scripts/data/bruce/bruce_env.py` | 修改（足底力归属 + 力矩源 + 多箱 API + 清理告警） |
| `scripts/data/bruce/bruce_signals.py` | 新建 |
| `scripts/data/bruce/collect_bruce.py` | 新建 |
| `scripts/data/bruce/check_dataset.py` | 新建 |
| `scripts/data/bruce/config/bruce_stepover.yaml` | 新建 |
| `scripts/data/bruce/figures/*.png` | 新建（采集后生成） |

复用而不重写的既有实现：`bruce_expert.py`（闭式专家，`gen_step_over` / `gen_step_up` / `clearance_over` / `make_box` / `SCENES`）、`bruce_env.py` 的 `BruceScene` 骨架、`world.py` 的 HDF5 契约。

## 验证

1. `~/miniconda3/envs/nudt/bin/python scripts/data/bruce/collect_bruce.py --config scripts/data/bruce/config/bruce_stepover.yaml --episodes 5` → 生成合法 `.h5`。
2. `~/miniconda3/envs/nudt/bin/python scripts/data/bruce/check_dataset.py --dataset bruce_stepover_50` → 全部检查项 PASS。
3. **H-JEPA 读入验证**（nudt 环境已有 torch/omegaconf/gymnasium，可直接 import）：
   `HDF5Dataset(name='bruce_stepover_50', keys_to_load=[action, imu_acc, ..., est_yaw], keys_to_merge={'proprio': [imu_acc, imu_gyro, foot_contact, leg_q, leg_dq]}, level1={'frameskip':5,'num_steps':4})` → 确认 `proprio_dim=30`、窗口可枚举、`sim_*` 列未进入。
4. 续采验证：同一 `--out` 再跑一次 `--episodes 10`，确认 `ep_len` 追加、`ep_offset` 连续、无重复 seed。
5. 图像产物人工核对：箱体轮廓与踝轨迹的相对位置符合「跨过箱子」的预期，净空标注数值与 `clearance_over` 输出一致。
