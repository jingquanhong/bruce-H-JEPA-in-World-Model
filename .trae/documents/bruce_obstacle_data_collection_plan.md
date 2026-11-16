# Bruce 越障数据采集方案（仿真 → H-JEPA 数据契约）

> 状态：待用户审阅。本方案只覆盖 **本期仿真采集**（不含实机采集脚本）。
> 目标：在 Genesis 中让 BRUCE 越过一个台阶/箱体，产出与 H-JEPA `HDF5Dataset` 完全兼容的 HDF5 数据集；
> 同时严格区分「实物可采集 / 实物可推导 / 仿真特权」三类信号，训练只允许使用前两类。

---

## 1. 结论先行（决策摘要）

| 决策项 | 结论 | 依据 |
|---|---|---|
| 仿真平台 | **Genesis 1.4.4**（不是 Gazebo） | 本机无 gazebo classic（仅 gz sim 8.11）；BRUCE-OP 官方行走控制器二进制不在仓库；`Simulation/worlds/bruce.world` 只有地面+太阳，无地形/障碍；无相机 sensor |
| 机器人模型 | 用 `sdf2urdf.py` 从 `model.sdf` 生成 **仓库内 URDF**（mesh 指向仓库内 meshes），消除对 `~/桌面/nudt` 的外部依赖 | `nudt/bruce_ws/sdf2urdf.py` 已验证；仓库内 `BRUCE-OP-main/Simulation/models/bruce/meshes/` 有 18 个 STL |
| 越障语义 | 主任务：**跨过一个箱子**（step-over，箱高 4/6/8 cm），次任务：上台阶（0.10 m 平台），可选 | 用户「能够跨过一个台阶都行」；闭式 LIPM 专家已验证 `gen_step_over` / `gen_step_up` |
| 专家策略 | **闭式解析解**（LIPM 轨道 + 解析 IK），零训练、零 RL、零在线试错 | `m5_modal_gen.gen_step_over` / `gen_step_up`；`m6_scenes` 已给出净空/平衡裕度判据 |
| action | **10 维腿部关节位置目标**（右腿5+左腿5，顺序=BRUCE-OP `LEG_JOINT_LIST`），上肢不入 action | 用户决定；与实物 `BRUCE_data.set_command_leg_positions` 的 `goal_positions(10)` 完全一致 |
| proprio | **30 维，仅由实物可直采信号拼成**：IMU acc(3)+IMU gyro(3)+足底接触(4)+腿关节角(10)+腿关节角速度(10) | 实物链路：PICO 串口 + BEAR 电机 |
| 记录频率 | 仿真 2000 Hz（dt=5e-4），**数据集统一 100 Hz 存储**；训练 `level1.frameskip=5` → 有效 20 Hz | 与 H-JEPA ant 配置一致（action 存高频、其余列读取时抽帧） |
| 依赖取舍 | 采集脚本**零 `stable_worldmodel` 依赖**，只依赖 genesis+numpy+h5py+hdf5plugin+PyYAML；跑在 `nudt`(py3.10) 环境；训练在 py3.11 环境读同一 `.h5` | `nudt` 环境缺 gymnasium/omegaconf/hydra；`genesis-world 1.4.4` 要求 Python ≥3.10,<3.14，与 py3.11 训练栈不冲突但当前无该环境 |
| 数据落盘 | `<cache_dir>/bruce_stepover_<n>.h5`，`cache_dir` 默认 `~/.stable_worldmodel`（`HJEPA_HOME` 未设置） | `stable_worldmodel/data/utils.py:5-9` |

---

## 2. BRUCE-OP 信号可采集性判定表（本次方案的事实依据）

来源文件均在 `BRUCE-OP-main/`。判定口径：

- **REAL**：实物直接测得（含「电流→力矩换算」这类已被实物固化的换算）
- **DERIVED**：实物上由传感器 + 模型推导得到（实物链路已存在，只是依赖模型/滤波）
- **SIM-ONLY**：只有仿真能拿到（或仿真真值与实物观测量语义不同）

### 2.1 REAL —— 实物可直接采集

| 信号 | 维度 | 实物链路 | 代码依据 |
|---|---|---|---|
| IMU 加速度 | 3 | PICO 串口帧 | `Library/BRUCE_SENSE/Manager.py:31`；`Startups/memory_manager.py:37` |
| IMU 角速度 | 3 | 同上 | `BRUCE_SENSE/Manager.py:31`；`memory_manager.py:38` |
| 足底接触（布尔） | 4 `[R_TOE, R_HEEL, L_TOE, L_HEEL]` | PICO 位图解包，**无接触力** | `BRUCE_SENSE/Manager.py:43,90`；`memory_manager.py:39` |
| 腿关节位置 | 10 | BEAR `present_position` → `bear2joint` | `Startups/run_bear.py:221-250,258`；`memory_manager.py:74` |
| 腿关节速度 | 10 | BEAR `present_velocity` → `bear2joint` | `run_bear.py:221-250,260`；`memory_manager.py:75` |
| 腿关节力矩 | 10 | BEAR `present_iq` × `IQ2TORQUE=0.35`（**电流估计，非力矩传感器**） | `run_bear.py:262`；`Settings/BRUCE_macros.py:318` |
| 膝关节温度 / 电压 | 2 | BEAR `winding_temperature` / `input_voltage` | `run_bear.py:177-193`；`memory_manager.py:77-78` |
| 臂关节位置 / 速度 | 6+6 | Dynamixel（**无臂力矩**） | `memory_manager.py:91-94` |
| 手柄 / 用户指令 | 22 键 | 游戏手柄 | `memory_manager.py:43-69` |
| 关节指令（本数据集 action） | 10 | `set_command_leg_positions` 下发的 `goal_positions` | `Settings/BRUCE_data.py:286-304` |

### 2.2 DERIVED —— 实物可由传感器+模型推导

| 信号 | 维度 | 实物链路 | 代码依据 |
|---|---|---|---|
| EKF 机体位姿/速度/加速度/偏航/角速度 | 3/3/3/1/3 | 实机上 `BRUCE_ESTIMATION_AOT` 500 Hz 跑 IMU+足底接触+FK | `Library/STATE_ESTIMATION/BRUCE_ESTIMATION_AOT.py:24,61-161`；`Startups/run_estimation.py:151-173,190-244` |
| COM 位姿/速度、角动量 | 3/3/3 | 估计器 + `robot_ID` 动力学 | `run_estimation.py:207-210`；`Library/ROBOT_MODEL/BRUCE_DYNAMICS_AOT.py:51` |
| H / CG / AG / dAGdq | 16/16/6×16/6 | `robot_ID` | `BRUCE_DYNAMICS_AOT.py:51`；`run_estimation.py:211-214` |
| 足/趾/跟/踝 位姿·速度·雅可比 | — | `leg_Jacobian` + `base2task` | `Library/ROBOT_MODEL/BRUCE_KINEMATICS_AOT.py:47,201,271`；`run_estimation.py:176-244` |
| 闭式腿 IK / FK | — | BRUCE-OP 自带解析解 | `BRUCE_KINEMATICS_AOT.py:89,47` |
| 关节↔电机换算（BEAR 减速比 9.0 / IQ↔力矩） | — | `joint2bear` / `bear2joint` | `BRUCE_KINEMATICS_AOT.py:330,355`；`BRUCE_macros.py:318,355-357` |

> 仿真侧注意：`run_estimation.py:167` 在仿真里用 `Bruce.imu_rot_matrix` 真值代替 EKF 姿态估计，**仿真与实机的估计器行为并不等价**——这正是本方案要用「噪声注入 + 双列并存」去逼近的差距。

### 2.3 SIM-ONLY —— 仅仿真可得（**禁止进训练输入**）

| 信号 | 维度 | 说明 | 代码依据 |
|---|---|---|---|
| 机体真值位姿/速度/欧拉角 | 3/3/3 | Gazebo `BODY_POSE` 真值，实物无绝对参考 | `Library/BRUCE_GYM/GAZEBO_INTERFACE/Manager.py:80-84,177-189` |
| 真值关节力矩 | 16 | `JOINT_STATE.force` 是真实力矩；实物只有电流估计 | `GAZEBO_INTERFACE/Manager.py:62,171`；`run_simulation.py:129,138` |
| 外力/外力矩注入 | 3/3 | `BODY_FORCE` / `BODY_TORQUE` | `GAZEBO_INTERFACE/Manager.py:95,98,285-291` |
| 逐接触法向力/冲量 | — | 实物足底只有布尔开关 | `BRUCE_SENSE/Manager.py:43,90` |
| 场景/障碍几何真值 | — | 实物需外部感知 | — |
| 仿真步进/暂停、PID 增益与限位写入 | — | 标定通道 | `GAZEBO_INTERFACE/Manager.py:68,74,209-225,246-275` |
| 仿真真值 `imu_rot_matrix` | 3×3 | 实机走 EKF | `run_estimation.py:167`；`memory_manager.py:40` |

### 2.4 跨模态缺口（决定「像素」只能是可选附加列）

- **实机与仿真模型都没有相机**：`BRUCE-OP-main/Simulation/models/bruce/model.sdf` 全文只有 4 个 `type='contact'` sensor（L320/349/640/669），无 camera；`Simulation/worlds/bruce.world` 仅 ground_plane + sun。
- 实物 IMU 走 PICO 串口，只传 加速度/角速度/足底接触（`BRUCE_SENSE/Manager.py:31`），**不传图像**。
- 结论：`pixels` 在实物端**没有对应传感器**。按用户决策，pixels 作为**可选附加列**（默认关闭），只用于打通 H-JEPA 的视觉通路做算法调试，**不构成可上实机的观测**。

### 2.5 落地到数据列（每列必须打上来源标签）

| HDF5 列名 | 维度 | 来源标签 | 说明 |
|---|---|---|---|
| `action` | (T,10) | REAL(指令) | 腿部关节位置目标 rad，顺序=BRUCE-OP `LEG_JOINT_LIST` |
| `imu_acc` | (T,3) | REAL | 仿真真值 + PICO 级噪声/零偏 |
| `imu_gyro` | (T,3) | REAL | 同上 |
| `foot_contact` | (T,4) | REAL | 由仿真接触力阈值化 → 布尔（顺序 R_TOE,R_HEEL,L_TOE,L_HEEL） |
| `leg_q` | (T,10) | REAL | 腿关节角 |
| `leg_dq` | (T,10) | REAL | 腿关节角速度 |
| `joint_torque_est` | (T,10) | REAL(估计) | 真实力矩→电流→×0.35→量化，模拟实物电流估计 |
| `est_body_pos` / `est_body_vel` / `est_body_acc` / `est_yaw` | (T,3)/(T,3)/(T,3)/(T,1) | DERIVED | 真值 + 估计级噪声（第一版）；标注为「EKF 替身」 |
| `est_body_pos_e` 等 | | DERIVED | 备选：直接接入 BRUCE EKF（Stage B，可选） |
| `sim_body_pos` / `sim_body_vel` / `sim_base_quat` | (T,3)/(T,3)/(T,4) | **SIM-ONLY** | 训练禁止使用，仅用于量化 sim2real gap |
| `sim_contact_force` | (T,4) | **SIM-ONLY** | 真值接触力 |
| `sim_joint_torque` | (T,10) | **SIM-ONLY** | 真值关节力矩 |
| `pixels` | (T,H,W,3) uint8 | 无实物对应 | 可选附加列，默认关闭 |

**训练时 `keys_to_load` 只允许出现 REAL + DERIVED 列**；`sim_*` 列留在文件里做对照分析。

---

## 3. 目标平台现状与需补齐项

| 项 | 现状 | 动作 |
|---|---|---|
| Genesis | ✅ `nudt` 环境已装 1.4.4（py3.10） | 直接用 |
| BRUCE URDF | ⚠️ 只有 `~/桌面/nudt/bruce_ws/bruce.urdf`（mesh 指向仓库外） | 生成仓库内 `bruce_genesis.urdf` |
| h5py / hdf5plugin | ❌ `nudt` 缺（`pip install --dry-run` 已验证可装 3.16.0 / 7.1.0） | `pip install h5py hdf5plugin` |
| PyYAML | ✅ 6.0.3 | — |
| hydra / omegaconf / gymnasium | ❌ `nudt` 缺 | 采集脚本**不使用**这三者（argparse + PyYAML） |
| H-JEPA 训练环境 (py3.11 + torch 2.7.1) | ⚠️ 本机不存在（无 uv、无 .venv、conda 仅 llm-course/nudt） | 本期只产出数据 + 读入校验；训练环境另立任务 |
| GPU | RTX 3060 12GB | Genesis GPU backend 可用（`gs.init(backend=gs.gpu)`，失败回退 cpu） |

---

## 4. 实施步骤

### Stage 0 — 资产与依赖（0.5 h）

0.1 `pip install h5py hdf5plugin`（`nudt` 环境）。
0.2 用 `sdf2urdf.py` 生成 `BRUCE-OP-main/Simulation/models/bruce/urdf/bruce_genesis.urdf`：
   - 输入 `BRUCE-OP-main/Simulation/models/bruce/model.sdf`
   - mesh 目录 `BRUCE-OP-main/Simulation/models/bruce/meshes`（绝对路径，消除外部依赖）
   - 校验：16 个转动关节、总质量与 `model.sdf` 一致（当前 model.sdf 求和 ≈ 4.30 kg；注意与 `BRUCE_macros.MASS_TOT=4.4497`、`nudt` 用的 4.524 **三者不一致**，必须统一，见 §6）
0.3 记录质量/惯量口径决议，写进 `bruce_signals.py` 常量。

### Stage 1 — 采集器实现

新增目录 `scripts/data/bruce/`（与 `scripts/data/collect_antmaze.py` 同级，风格对齐）：

| 文件 | 职责 |
|---|---|
| `bruce_expert.py` | 闭式专家：从 `nudt/bruce_ws/m5_modal_gen.py` + `m6_scenes.py` **移植**（不 import 外部路径）：`ik()`、`lift_prof()`、`_pose()`、`gen_step_over()`、`gen_step_up()`、`gen_double_step()`、`static_zmp_margin()`、`clearance_over()`、场景表 `SCENES` |
| `bruce_env.py` | Genesis 封装：`build_scene()`（Plane + Box + URDF + camera 可选）、`reset()`（浮动基零速 + 基座悬高）、`step_targets()`、`read_state()`（IMU 真值、关节角/角速、力矩、接触力与接触对）、`foot_contact_bools()`（阈值化，顺序 R_TOE,R_HEEL,L_TOE,L_HEEL）、`camera_render()` |
| `bruce_signals.py` | 信号映射与保真：REAL 列生成（噪声/零偏/量化）、DERIVED 列（估计噪声模型）、SIM-ONLY 列（真值）、单位与关节顺序常量（`LEG_JOINT_LIST` 序） |
| `collect_bruce.py` | 主程序：argparse + YAML 配置；场景采样→闭式专家→Genesis 重放→逐列记录→HDF5 写出；断点续采；质检摘要 |
| `check_dataset.py` | 数据质检：列齐全性/维度/`ep_len`/`ep_offset` 一致性/NaN/时间轴对齐/成功率分布/REAL-vs-SIM 差异统计 |
| `config/bruce_stepover.yaml` | 采集配置（见 §5） |

**HDF5 写出严格复刻 `stable_worldmodel/world.py` 的契约**（`_init_h5_datasets` L428-489 / `_write_episode` L685-719）：
- resizable dataset，`maxshape=(None,)+sample.shape`
- `ndim>=2` → `hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE)`，chunks `(100,)+shape`；否则 chunks `(1000,)+shape`
- `ep_len`(int32) / `ep_offset`(int64) 必写；额外写 `ep_idx`(int32, 每步所属 episode)
- `swmr_mode=True`；新建文件加 `fs_strategy='page', fs_page_size=4MB`
- 续采：若 `ep_len` 已存在，从 `ep_offset[-1]+ep_len[-1]` 继续追加

### Stage 2 — 闭环冒烟 + 质检（0.5 h）

2.1 单场景 5 episode 冒烟：跑通 → 检查 `foot_contact` 布尔序列是否符合预期（支撑腿常 1、摆动腿 0）、`leg_q` 是否在限位内、`action` 是否等于下发的目标。
2.2 用 `bruce_expert.clearance_over` 复核每条轨迹的越障净空 > 0（沿用 `m6_scenes` 的 `C_req = 5mm + 0.5*max(0, lift-0.105)` 判据）。
2.3 跑 `check_dataset.py`，产出成功率/净空/接触统计。

### Stage 3 — 正式采集

3.1 冒烟通过后分批采集（支持中断续采）：先 500 episode 试采，再扩到 **3000 episode**（约 3000×300 帧 = 0.9M transitions @100 Hz）。
3.2 场景配比：箱高 4/6/8 cm 各约 1/3；8 cm 属能力域边界，**保留失败样本**（世界模型需要学会失败分布）；另加 ~10% 上台阶样本。
3.3 每条 episode 写 `episode_meta`（场景 id、箱高、箱位、bx0、lift_extra、T_lipm、成功与否、净空），便于分层评估。

### Stage 4 — H-JEPA 接入（配置 + 读入验证）

新增：
- `h_jepa/config/train/base/bruce.yaml`：`keys_to_load=[action, imu_acc, imu_gyro, foot_contact, leg_q, leg_dq, joint_torque_est, est_body_pos, est_body_vel, est_body_acc, est_yaw]`；`keys_to_merge: proprio: [imu_acc, imu_gyro, foot_contact, leg_q, leg_dq]` → **proprio_dim = 30**；`keys_to_cache=[action]`；`level1.frameskip=5, num_steps=history_size+rollout_n`；`level1.wm: action_dim=10, proprio_dim=30, use_proprio=true`；`env: bruce`。
- `h_jepa/config/train/bruce_hjepa_l2.yaml`（继承 base + `num_levels: 2`）、`h_jepa/config/eval/bruce_l2.yaml`。
- `keys_to_load` **不列** `sim_*` 与 `pixels`（若开像素则另出 `bruce_pixel` 配置）。

读入验证（不需要 py3.11 训练环境）：
```
pip install omegaconf gymnasium==1.2.3      # nudt 环境，仅为 import stable_worldmodel
python -c "import stable_worldmodel as swm; d=swm.data.HDF5Dataset(name='bruce_stepover_3000', keys_to_load=[...], keys_to_merge={'proprio':[...]}, level1={'frameskip':5,'num_steps':4}); print(d.clip_indices[:3], d.get_col_data('proprio').shape)"
```
（`stable_worldmodel/__init__.py` 会连带 import gymnasium/torch/omegaconf，故需这三样；`nudt` 已有 torch。）

---

## 5. 数据集契约（与 H-JEPA 对齐的硬约束）

1. **文件名/位置**：`<cache_dir>/<name>.h5`，`cache_dir` = `HJEPA_HOME` 或 `~/.stable_worldmodel`。
2. **必备索引**：`ep_len`(int32)、`ep_offset`(int64)（`HDF5Dataset.__init__` L42-46 强制读取）；额外 `ep_idx`(int32)。
3. **时间轴**：所有列同一 100 Hz 时间轴，长度 = episode 帧数。
4. **action 语义**：`action[t]` = 在 `o_t` 观测下发出的关节位置目标；读取时 `HDF5Dataset` 把 action reshape 成 `(frames, frameskip, action_dim)`（L190-192），因此 action 必须按 100 Hz 存满、其余列由读取侧抽帧（L221-222）。
5. **proprio 合并**：`keys_to_merge` 用 `merge_col` 沿最后一维拼接（`dataset.py:302-315`），各源列必须是 `(T, d)`，拼出 `(T, 30)`；`level1.wm.proprio_dim` 必须等于 30。
6. **normalizer**：`h_jepa/data.py:154-172` 只对 `keys_to_load/keys_to_cache/keys_to_merge` 中的非 `pixels` 列做 z-score → `sim_*` 列天然不会被归一化（因不在列表内），进一步保证不进模型。
7. **长度**：`level1.span = num_steps×frameskip = (3+1)×5 = 20` 帧 ≤ episode 帧数（≈300）✔。

### 采集 YAML（`config/bruce_stepover.yaml` 关键字段）

```yaml
sim_dt: 5.0e-4          # 2000 Hz
record_hz: 100          # 数据集存储频率
episode_seconds: 3.2    # 闭式轨迹 3.0s + 前后静置
scenes:                 # 箱高 / 箱位 / 箱厚
  - {id: box_04, h: 0.04, x: 0.10, hx: 0.025}
  - {id: box_06, h: 0.06, x: 0.10, hx: 0.025}
  - {id: box_08, h: 0.08, x: 0.10, hx: 0.025}
  - {id: step_10, h: 0.10, x: 0.34, hx: 0.25, platform: true}
domain_randomization:   # 域随机化
  box_x_jitter: 0.02
  bx0: [-0.06, -0.03, 0.0]
  lift_extra: [0.005, 0.045, 0.08]
  T_lipm: [0.8, 1.2]
  friction: [0.6, 1.2]
  kp_scale: [0.85, 1.15]
  imu_acc_noise: 0.01       # = Qa HARDWARE σ
  imu_bias_rw: 1.0e-4       # = Qba
  joint_q_noise: 0.002      # 编码器量化
  torque_quant_step: 0.0117 # 0.35 / IQ_MAX(30)
sensor:
  foot_contact_force_thresh: 5.0   # N
output:
  dataset_name: bruce_stepover_3000
  episodes: 3000
  pixels: false            # 可选附加列
```

---

## 6. 实物一致性（sim2real）对齐手段

| 差距 | 对齐做法 |
|---|---|
| 质量口径不一致（4.30 / 4.4497 / 4.524 kg） | 以 `model.sdf` 为真值源（URDF 即由此生成），统一到一处；再做 ±10% 质量随机化覆盖实物差异 |
| IMU 理想 vs PICO 有噪声 | 加白噪声 σ=0.01 m/s²（= `BRUCE_ESTIMATION_AOT.py:45` HARDWARE `Qa`）+ 零偏随机游走 σ=1e-4（`:46` `Qba`） |
| 足底只有布尔 | 仿真接触力 > 5 N 阈值化为布尔；**真值接触力另存 `sim_contact_force`** |
| 真实力矩 vs 电流估计 | `sim_joint_torque`（真值）→ `/0.35` 得电流 → 按 IQ_MAX=30 量化 → `×0.35` → `joint_torque_est`（含 ±5% 增益误差） |
| 关节角/角速度 | 加编码器量化噪声（0.002 rad）+ 一阶低通（BEAR 侧已有 `exp_filter(0.90)`，`run_bear.py:254-255`） |
| 机体位姿：仿真真值 vs 实物 EKF | 第一版：`est_*` = 真值 + 估计级噪声（位置 1 cm/1σ、姿态 2°/1σ、速度 0.05 m/s/1σ）；Stage B 可选直接接 BRUCE EKF |
| 实物力矩上限 | BEAR `TORQUE_MAX=10.5 Nm`（`BRUCE_macros.py:325`）；Genesis 位置控制时读实际力矩并按 10.5 Nm 饱和，暴露为 `joint_torque_est` 的裁剪证据 |
| PD 增益 | 仿真 `KP/KV` 用 `m5` 标定值（`KP_BASE=[1000,1000,5000,60,60,60]`, `KV_BASE=[30,30,60,12,12,12]`），实物 `pos_kp=60/pos_kd=1.0`（`BRUCE_macros.py`）→ 统一为位置环 + ±15% 随机化 |
| 摩擦 | `model.sdf` 中 μ=10（过高，仿真内不易打滑）→ Genesis 摩擦设 0.6~1.2 并随机化 |
| 相机缺失 | pixels 默认关闭；若开启，仅作视觉通路调试，不宣称可上实机 |
| 悬浮 vs 自由站立 | 本方案为**自由站立**仿真；实物若为悬挂测试架需在报告中声明该差异 |

---

## 7. sim2real 风险清单（需在 Stage 3 报告中量化）

1. **浮动基动力学差异**：实物落地接触刚度/阻尼未知，Genesis 用默认 solver 参数 → 净空与落足时刻可能有 ±1 帧级偏差。
2. **BEAR 减速比与带宽**：实物 BEAR 位置环带宽有限（`pos_kp=60`），仿真位置控制是理想 PD → 跟踪滞后差异（`m5` 已实测滞后约 2 cm，`lift_extra` 默认 0.045 即为此补偿）。
3. **质量与惯量三项不一致**（§4 Stage 0.3）→ 影响动力学与接触分布。
4. **足底碰撞几何**：URDF 用 5 mm 碰撞球（`bruce.urdf` ankle 的两个 sphere），真实鞋底形变/摩擦未建模。
5. **力矩估计误差**：`IQ2TORQUE=0.35` 是线性近似，低电流段误差大（BEAR 静摩擦区）。
6. **8 cm 箱为能力域边界**：闭式净空仅毫米级，域内可能踢翻箱体 → 失败样本占比对分布影响大，需按场景分层报告。
7. **视觉模态无法闭环**：无相机 → 若最终要视觉规划，需外接相机并单独做感知标定，属另一条工作线。
8. **数据分布偏窄**：闭式专家生成的轨迹多样性有限（4 个参数维度）→ 世界模型可能过拟合到「单一步态族」，需靠域随机化与失败样本补偿。

---

## 8. 验收标准

- [ ] `collect_bruce.py` 在 `nudt` 环境一条命令跑通 5 episode，生成合法 `.h5`
- [ ] `check_dataset.py` 全部通过：列名/维度、`ep_len`/`ep_offset` 单调且 `sum(ep_len)==总帧数`、无 NaN、`action` 与下发目标一致、时间轴长度一致
- [ ] `HDF5Dataset` 能读入并产出 `proprio.shape == (T,30)`、`action` reshape 为 `(frames,5,10)`
- [ ] 越障净空 > 0 的 episode 占比 ≥ 90%（6 cm 箱）；8 cm 箱按能力域边界如实统计（不修饰）
- [ ] `sim_*` 列不出现在任何训练配置的 `keys_to_load/keys_to_cache/keys_to_merge` 中
- [ ] 冒烟 500 episode → 正式 3000 episode 可中断续采
- [ ] 输出一份采集报告：各场景成功率、平均净空、REAL-vs-SIM 差异统计（EKF 替身 vs 真值）

---

## 9. 明确不做（本期）

- 实机采集脚本（PICO 串口 + BEAR 读数 → 同一 HDF5 列结构）
- 接入真实 BRUCE EKF 替代「估计替身」（Stage B 可选）
- H-JEPA 训练/评估跑通（py3.11 训练环境需单独建立）
- 像素视觉数据正式采集（默认关闭）
