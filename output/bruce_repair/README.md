# 当前物理修正产物

最终记录位于 `validation_v4/`，以项目根目录 `BRUCE仿真说明.md` 为当前说明。

- `stand.h5/mp4/json`：5秒站立检查通过，250帧，基座控制力为零。
- `box08.h5/mp4/json`：8cm尝试在1.56秒失衡，78帧，自主成功为0。
- `*_integrity.json`：两份记录均通过观测/动作对齐、力矩/耦合电流限制、基座无驱动和模型哈希检查。
- `hjepa_read.json`：真实 HDF5Dataset 能读取 RGB、本体观测和动作，未进行模型训练。
- `manifest_current.json`：位于本目录，记录当前源码和最终HDF5的哈希；也确认两份历史HDF5未被修改。
- `assisted_qualification.json`、`joint_qualification.json`：历史数据重新资格复核，自主越障成功均为0。

`validation_v1/` 到 `validation_v3/`、`collector_smoke/` 及本目录其他探针日志/轨迹/视频均为开发中间结果，不能当作最终成功示范。其中部分试验使用旧的模型假设、初始距离或控制器参数；以最终v4记录内的参数与哈希为准。

旧辅助样本不能用于学习由腿部动作决定的自主转移动力学。新文件使用固定外部相机、仿真真值反馈和力矩块动作，仍不是已标定、可实机迁移的训练集。当前没有合格自主越障成功数据。
