# 2026-11-14：8 cm 问题续查与真实采集

**原因进一步查清，部分约束已修正；8 cm 自主越障仍未实现，成功样本仍为 0。**

## 查清了什么

1. 旧辅助视频的问题仍成立：它驱动了浮动基座，且有碰箱、超出当前腿力矩参考上限的记录，不能作为自主成功示范。此次没有修改历史 HDF5。
2. `validate_physics.py` 原 8 cm 参考在抬左脚时没有横向重心转移。1.2 秒时，重心比右踝偏左 76.10 mm。考虑参考加速度和角动量变化后的平面受力中心要求仍在支撑多边形外约 74.05 mm，而不只是静态重心超界。完整筛查见 `reference_audit_v2.json`。这是有限差分、平面刚体接触近似下的参考筛查，不等于真实接触仿真。
3. 原控制器采用横向 ±8 mm 的力矩支撑范围。原始脚部网格的最低 0.1 mm 高度带投影呈狭长三角形，前端仅约 5.47 mm 宽，并向脚跟收窄。它不是整只脚 21 mm 的最大外形宽度，也不是完整平坦脚掌。新控制器按该网格投影约束平面受力中心，并计入从脚底到踝原点的力矩臂。曲面脚底、脚倾斜后的实际接触仍需单独验证。

## 本次落地的修改

- `support_geometry.py`：从物理 URDF 碰撞网格读取近脚底投影及平面接触力矩约束。
- `standing_wbc.py`：替换原横向 ±8 mm、纵向 ±35 mm 的矩形支撑假设，保留基座无驱动和力矩/电流限制。
- `support_wbc.py`、`support_reference.py`：独立实验控制器及重心转移、单脚抬起诊断；不是已验证的越障控制器。BRUCE 没有踝侧摆关节，支撑脚姿态和身体姿态不能随意同时锁死。
- `validate_physics.py --trial support`：在抬脚前检查捕获点、横向速度及双脚载荷。未满足必要条件就终止，不把未完成的转移执行成抬脚。检查通过也不保证安全或越障成功。
- `audit_reference.py`、`test_support.py`：可重跑的参考支撑筛查及回归检查。

## 当前可复查产物

以 `verified/` 为本次最终记录；每项均包含 MP4、HDF5、指标 JSON、独立记录完整性 JSON。

| 记录 | 时长 / 帧数 | 实际结果 |
|---|---:|---|
| `verified/stand.*` | 5.00 s / 250 | 站立通过；基座力为 0；腿力矩峰值 1.24 N·m |
| `verified/box08.*` | 1.42 s / 71 | 未通过；收紧接触约束后，原参考的逆动力学 QP 达到迭代上限，终止执行。不能将此写成已证明物理不可行或越障成功 |
| `verified/support.*` | 3.00 s / 150 | `support_transfer_not_ready`；抬脚前终止，尚未实现稳定单脚支撑 |

重心转移检查在 3 秒处测到：捕获点横向坐标约 −74.45 mm，右脚实际受力中心约 −84.58 mm，相差 10.13 mm；左脚仍未满足卸载阈值。因此没有进入抬脚阶段。这解决了该诊断中“未就绪仍继续抬脚”的流程问题，没有解决单脚平衡控制本身。

本次共 3 个 episode、471 帧，均含 64×64 RGB、36 维本体观测及 64 维力矩块动作。`support.h5` 另含执行动作前的仿真重心、速度、接触中心和足底载荷。动作行对应随后四个 5 ms 控制周期，记录周期为 20 ms。它们可用于接口验证和失败动力学研究，**不是成功越障训练集**，也未进行模型训练。

基座外力峰值均为 0；实际执行力矩、耦合电流及动作对齐均通过检查。`regression.log` 的 11 项测试通过；`hjepa_read.log` 确认三份文件均可被项目真实 `HDF5Dataset` 读取，图像 / 本体 / 动作窗口分别为 `(4,3,64,64)`、`(4,36)`、`(4,128)`。视频经抽帧检查，终止原因保留在结尾状态卡中。

项目根目录也保存了便于打开的视频：`BRUCE物理复核_20261114_站立.mp4`、`BRUCE物理复核_20261114_8cm未通过.mp4`、`BRUCE物理复核_20261114_重心转移未就绪.mp4`。

## 重跑

在项目根目录执行，选择未使用的输出目录：

```bash
BRUCE_VALIDATION_OUTPUT=output/bruce_next bash BRUCE仿真命令.sh audit
BRUCE_VALIDATION_OUTPUT=output/bruce_next bash BRUCE仿真命令.sh validate
BRUCE_VALIDATION_OUTPUT=output/bruce_next bash BRUCE仿真命令.sh support
BRUCE_VALIDATION_OUTPUT=output/bruce_next bash BRUCE仿真命令.sh trial08
```

`support` 未就绪、`trial08` 未通过时退出码为 2；不是脚本已经越障成功。

```bash
/home/user/miniconda3/envs/nudt/bin/python -m unittest discover -s scripts/data/bruce -p 'test_*.py' -v
/home/user/miniconda3/envs/hjepa/bin/python scripts/data/bruce/verify_physics_hjepa_read.py output/bruce_continue_20261114/verified --names stand box08 support
```

`support_01/` 至 `support_11/` 是不同控制假设的中间参数探针；`records/` 是本次修正主控制器接触范围前的对照记录。它们不是最终成功数据，部分探针仍会侧倒或遇到求解失败。`source_before_contact_fix/` 保留修改前的相关源码；`manifest.json` 记录本次最终源码和数据哈希。

剩余工作是实现接触约束下的可靠重心转移、单脚支撑、摆腿和落脚，再验证完整网格净空与落地稳定性。新控制器仍使用仿真真值、对角电机反射惯量及未实机标定的接触/电流近似，不能直接作为实机策略。
