# BRUCE 仿真采集

2026-11-15 最新结果见 [多场景报告](../../../output/bruce_collection_20261115/README.md)：新增 32 条越障尝试、4244 帧 RGB，均有实际动作、视频和 2 kHz 物理轨迹；合格成功仍为 0。`probe_dynamic_step.py --record` 提供多场景实验采集，`curate_crossing_attempts.py` 复核并合并失败尝试；新 QP 路径另依赖 Clarabel 0.11.1。`bash BRUCE仿真命令.sh crossings` 会采六种高度，当前无合格成功时退出码为 2。该数据集与以下历史 10 维位置动作数据接口不同。

2026-11-14 最新复核见 [续查报告](../../../output/bruce_continue_20261114/README.md)。已收紧脚底支撑范围并增加 `validate_physics.py --trial support` 抬脚前必要条件检查；新采 3 个 episode、471 帧。8 cm 仍未通过，不能将这些调试记录标为成功示范。新验证额外依赖 Pinocchio、OSQP、SciPy、trimesh；继续使用现有 `nudt` 环境即可。

本轮物理修正与新验证见 [BRUCE仿真说明](../../../BRUCE仿真说明.md)。新默认模型恢复原始脚部网格并限制硬件关节范围。新验证含 RGB 和力矩动作，但只通过了 5 秒站立；8 cm 越障未通过。下述 `collect_bruce.py` 是解析位置参考的调试采集接口，不能输出已验证的自主成功示范。

在仓库根目录运行。Genesis 采集使用已有 `nudt` Python 环境，H-JEPA 检查使用 `hjepa` 环境。

```bash
/home/user/miniconda3/envs/nudt/bin/python scripts/data/bruce/collect_bruce.py \
  --config scripts/data/bruce/config/bruce_stepover.yaml \
  --control-mode joint_only --episodes 40 --out bruce_joint_trials \
  --cache-dir output/my_bruce_run --video-dir output/my_bruce_run/videos

/home/user/miniconda3/envs/nudt/bin/python scripts/data/bruce/check_dataset.py \
  --dataset bruce_joint_trials --cache-dir output/my_bruce_run \
  --json output/my_bruce_run/report.json
```

无显示器环境可在命令前设置 `PYOPENGL_PLATFORM=egl`。依赖为 Genesis、NumPy、h5py、hdf5plugin、PyYAML；视频另需 imageio、imageio-ffmpeg，绘图另需 matplotlib、Pillow。

默认 `joint_only` 禁止驱动浮动基座，腿关节执行力矩限制为 ±10.5 N·m，手臂保持固定姿态。当前解析轨迹在此模式下会失衡。`assisted` 会驱动基座，适合参考动作和采集接口演示，不代表实物可执行控制；不能和自由基座模式合并解释。

每集按 `seed + episode_index` 独立采样参数，记录配置、时间戳、模式和完成判据。动作按记录周期零阶保持，行 t 保存当前观测和随后实际执行的目标，不做循环移位。续采检查配置，并截去未提交的尾部数据；历史版本数据禁止追加，换模型或参数时请使用新数据集名。

字段分三类：

- `imu_acc/imu_gyro/foot_contact/leg_q/leg_dq/joint_torque_est`：有 BRUCE-OP 实物链路依据的仿真传感器代理。
- `est_*`：当前为真值加噪声，不是真实 EKF 输出，默认读取白名单排除。
- `sim_*`：仿真质检真值，默认读取白名单排除。

`action` 为 10 维腿关节位置目标，右腿后左腿，每腿顺序为 yaw、pitch、roll、knee、ankle。噪声、量化和足底力阈值尚未经实机标定；实物力矩需要电流换算与机构映射，不能将仿真代理当作实机测量。

```bash
/home/user/miniconda3/envs/nudt/bin/python scripts/data/bruce/test_collection.py
/home/user/miniconda3/envs/hjepa/bin/python scripts/check_hjepa_runtime.py
/home/user/miniconda3/envs/hjepa/bin/python scripts/check_hjepa_entry.py
```

第三条命令会创建单独标记的合成图像/动作样本，只跑一次训练与验证，用于入口冒烟检查。它不使用 BRUCE 数据训练。历史位置参考采集的 HDF5 没有逐帧 pixels；本次另存的物理验证文件含 pixels、36维 proprio 和64维力矩块动作，需单独适配配置，数据量也不足以形成正式训练集。

本轮交付与限制见 [采集报告](../../../output/bruce_sim_20261113/README.md)。
