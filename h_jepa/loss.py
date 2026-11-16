# 导入 PyTorch 主包：提供 Tensor、autograd 以及神经网络所需的基础运算
import torch
# 导入 torch.nn 模块：提供 nn.Module 基类和 nn.Linear 等网络层
import torch.nn as nn
# 导入 torch.nn.functional（别名 F）：提供 mse_loss 这类无状态函数式算子
import torch.nn.functional as F


# ---- 段落标题：通用初始化函数——把线性层权重置为截断正态、偏置置零 ----
def init_module_weights(m, std=0.02):
    # 只对全连接层 nn.Linear 生效；其它层（LayerNorm 等）保持框架默认初始化
    if isinstance(m, nn.Linear):
        # 用截断正态分布（均值 0、标准差 std=0.02，截去 ±2σ 外的值）初始化权重，训练更稳定
        nn.init.trunc_normal_(m.weight, std=std)
        # 若该层存在偏置参数（不带偏置时 m.bias 为 None）
        if m.bias is not None:
            # 把偏置全部初始化为 0
            nn.init.constant_(m.bias, 0)


# ---- 段落标题：SIGReg——用"经验特征函数"做各向同性高斯正则，防止表征坍缩 ----
class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (ECF means averaged across DDP ranks)"""

    def __init__(self, knots=17, num_proj=1024):
        # 调用父类构造函数，完成 Module 内部的参数/缓冲区注册机制初始化
        super().__init__()
        # 随机投影方向的个数：把 D 维特征投影到 num_proj 个随机方向上
        self.num_proj = num_proj
        # 频率网格 t：在 [0, 3] 上均匀取 knots=17 个采样点，作为特征函数的频率坐标
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        # 相邻频率间距 dt = (3-0)/(17-1)，供下面构造梯形积分权重使用
        dt = 3 / (knots - 1)
        # 梯形积分权重：先给所有节点填 2*dt（内部节点的标准权重）
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        # 首尾两个端点权重减半为 dt（梯形法端点修正）
        weights[[0, -1]] = dt
        # 高斯窗 exp(-t^2/2)：压低高频分量，对应各向同性高斯的特征函数形状
        window = torch.exp(-t.square() / 2.0)
        # 注册频率网格 t 为缓冲区：随模型 .to(device) 一起搬运，但不是可训练参数
        self.register_buffer("t", t)
        # 注册目标曲线 phi = 高斯窗：正则项希望特征函数逼近的"理想值"
        self.register_buffer("phi", window)
        # 注册最终权重 = 梯形权重 × 高斯窗，用于对逐点误差加权求和（即带窗积分）
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # ---- 段落标题：SIGReg 前向——随机投影 -> 估计 ECF -> 与目标比误差 ----
        # 生成 (D, num_proj) 的高斯随机矩阵 A，每列是一个随机投影方向
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        # 按列（dim=0 遍历 D 维）求 L2 范数并就地除掉，把每个方向归一成单位向量
        A = A.div_(A.norm(p=2, dim=0))
        # 投影并乘频率：(T,B,D)@(D,P) -> (T,B,P)，unsqueeze(-1) 扩出末维再乘 t，
        # 广播得到 (T, B, P, knots)，即每个样本每个投影方向在各频率上的相位值
        x_t = (proj @ A).unsqueeze(-1) * self.t
        # 对 x_t 的第 -3 维（(T,B,P,knots) 中对应输入第 1 维，即批次 B 维）取平均，
        # 得到余弦/正弦经验特征函数分量，形状 (T, P, knots)
        cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
        # 取参与平均的样本数：对文档约定的 (T,B,D) 输入，size(-2) 即批次维 B
        n = proj.size(-2)
        # 若启用了分布式训练（多进程 DDP 已初始化）
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # untracked by autograd: each rank backprops its local ECF, DDP's grad average
            # then yields the gradient of the global statistic
            # ---- 段落标题：跨 DDP 进程把 ECF 平均成全局统计，并换算全局样本数 ----
            # 读取全局进程（卡）数
            world_size = torch.distributed.get_world_size()
            # 对余弦、正弦两个统计量依次做跨卡规约
            for ecf in (cos, sin):
                # all_reduce 默认 SUM：把各卡上的张量按元素求和（写回原张量）
                torch.distributed.all_reduce(ecf)
                # 用 .data 就地除以卡数求平均；.data 绕过 autograd，该规约不产生梯度
                ecf.data.div_(world_size)
            # 样本数从单卡批量扩大为全局批量（单卡 B × 卡数）
            n = n * world_size
        # 逐点平方误差：余弦分量与目标 phi 的偏差平方 加上 正弦分量的平方
        err = (cos - self.phi).square() + sin.square()
        # 用预存权重对频率维加权求和（带窗积分），再乘样本数 n 得到逐点统计量
        statistic = (err @ self.weights) * n
        # 对剩余所有维（T、P）取均值，输出标量正则损失
        return statistic.mean()


# ---- 段落标题：逆动力学模型 IDM——由相邻两帧隐状态预测中间执行的动作 ----
class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 初始化父类 Module
        super().__init__()
        # 用 Sequential 串起 3 层 MLP：输入 state_dim*2（两帧隐状态沿特征维拼接的宽度）
        # -> ReLU -> hidden_dim 隐层 -> ReLU -> 输出 action_dim 维的连续动作
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 对模型内所有子模块套用上面的 init_module_weights 做统一初始化
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        # 把 t 帧与 t+1 帧的隐状态沿 dim=1（特征维）拼接成 (B, 2*D) 再送入 MLP
        return self.model(torch.cat([state_t, state_t_plus_1], dim=1))


# ---- 段落标题：IDM 损失——预测动作与真实动作的均方误差（目标端停梯度） ----
class InverseDynamicsLoss(nn.Module):
    """MSE between the IDM action predicted from (z_t, z_t+1) and the raw action a_t."""

    def __init__(self, idm):
        # 调用父类构造函数
        super().__init__()
        # 保存逆动力学模型（作为子模块注册，其参数会被优化器统一管理）
        self.idm = idm

    def forward(self, emb, action):
        """
        emb: (B, T, D), action: (B, T, A) or (B, T-1, A)
        """
        # ---- 段落标题：取相邻帧对 -> IDM 预测动作 -> 与真实动作算 MSE ----
        # 前 T-1 帧与后 T-1 帧各自 flatten(0,1) 合并批次和时间维成 (B*(T-1), D)，
        # 送入 IDM 得到每一步的预测动作（切片 : 表示当前帧、1: 表示下一帧）
        pred = self.idm(emb[:, :-1].flatten(0, 1), emb[:, 1:].flatten(0, 1))  # [B*(T-1), A]
        # 真实动作同样取前 T-1 步并展平成 (B*(T-1), A)，与预测动作逐行对齐
        target = action[:, : emb.size(1) - 1].flatten(0, 1)  # [B*(T-1), A]
        # 计算 MSE；target.detach() 停止梯度：只训练 IDM，不让梯度经动作标签回传
        return F.mse_loss(pred, target.detach())
