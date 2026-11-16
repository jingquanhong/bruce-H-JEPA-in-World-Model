"""Module Implementations for Transformer-based Models"""

# =============================================================================
# 文件总览（读代码前先看这里）
# -----------------------------------------------------------------------------
# 本文件是 H-JEPA（分层联合嵌入预测世界模型）里"网络积木"的集合，不含训练循环：
#   1) build_projector : 按配置构建"投影头"，把编码器隐藏维映射到嵌入维
#   2) modulate / FeedForward / Attention / ConditionalBlock / Block / Transformer
#      : Transformer 主体（含 AdaLN-zero 条件调制，用于让预测器"看"到动作条件）
#   3) Embedder        : 把低维信号（动作、状态向量）嵌入到高维空间的 MLP
#   4) MLP / ResidualLatentMLP : 通用前馈网络，H-JEPA 各层的 latent 编码器常用
#   5) CLSDecoder      : 把一个 CLS 嵌入向量解码回像素图像，用于重建/可视化评估
#
# 训练侧的 LightningModule、optimizer/scheduler、checkpoint 钩子并不在本仓库里：
# 它们在外部库 spt 中，本仓库的入口脚本 h_jepa/main_hjepa.py 只负责把
# optimizer_factory / scheduler_factory 注册给 Lightning 框架（见该文件 308-350 行）。
# =============================================================================

# ---- 段落标题：依赖导入 ----
# einops 的 rearrange：按 "b t (h d) -> b h t d" 这样的模式重排张量形状，
# 比手写 view/permute 更直观，也不容易把维度写错
from einops import rearrange
# PyTorch 主包：提供 torch.Tensor、torch.nn 等基础能力
import torch
# torch.nn.functional（别名 F）：无状态函数式算子，
# 这里主要用 F.scaled_dot_product_attention（内部可走 FlashAttention 融合内核）
from torch.nn import functional as F
# torch.nn（别名 nn）：神经网络层的容器，如 Linear / LayerNorm / Dropout / ModuleList
import torch.nn as nn

# 从 loss 模块引入统一的权重初始化函数（截断正态初始化 + 偏置置零），
# 该模块本身用 apply() 递归作用于当前模块及其所有子模块
from loss import init_module_weights


# ---- 段落标题：投影头构建——把编码器隐藏维映射到目标嵌入维的小网络 ----
def build_projector(projector_cfg, *, input_dim: int, output_dim: int) -> nn.Module:
    # 从配置字典里取 "type" 字段（缺省 "mlp"），统一转成小写字符串再比较
    projector_type = str(projector_cfg.get("type", "mlp")).lower()

    # 分支一：恒等映射——不做任何变换，也不引入可学习参数
    if projector_type == "identity":
        # 恒等投影不能改变特征维度：输入维与输出维必须完全一致，否则立即报错
        if int(input_dim) != int(output_dim):
            # 抛出 ValueError，并在消息里带上具体维度，方便定位是哪个 level 的配置写错了
            raise ValueError(
                "Identity projector cannot change feature size: "
                f"got input_dim={input_dim}, output_dim={output_dim}."
            )
        # 维度一致时返回恒等层：原样透传张量
        return nn.Identity()

    # 分支二：只做 LayerNorm 归一化，不做线性维度变换
    if projector_type == "layernorm":
        # 在 output_dim 维上做层归一化；注意它要求输入最后一维就是 output_dim
        return nn.LayerNorm(output_dim)

    # 分支三：MLP 投影头（默认分支），最常用的可学习映射
    if projector_type == "mlp":
        # 构造 MLP：先升维到 2048，经过 BatchNorm1d + GELU，再降维到 output_dim
        return MLP(
            input_dim=input_dim,       # 输入维：上游编码器的隐藏维
            output_dim=output_dim,     # 输出维：这一层 JEPA 使用的嵌入维
            hidden_dim=2048,           # 中间隐藏维：先升维再降维，容量更大、表达力更强
            norm_fn=nn.BatchNorm1d,    # 归一化方式：批归一化（对每个特征通道在批内统计）
        )

    # 兜底：配置里写了不认识的结构类型，直接报错，避免静默用错模型
    raise ValueError(f"Unsupported projector type '{projector_cfg.get('type')}'")


# ---- 段落标题：AdaLN-zero 调制算子——用条件向量生成的 shift/scale 改写特征 ----
def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    # 对（已归一化的）特征做仿射变换：先乘 (1 + scale) 缩放，再加 shift 平移。
    # 写成 "1 + scale" 而不是 "scale" 是因为 AdaLN-zero 初始化时 scale=0，
    # 此时调制等价于恒等变换（不破坏预训练主干），训练中再逐渐学会调制强度
    return x * (1 + scale) + shift


# ---- 段落标题：Transformer 的前馈网络（FFN）——逐位置的非线性变换 ----
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        # 调用父类 nn.Module 的构造函数，建立内部的参数注册表（parameters/buffers）
        super().__init__()
        # 用 nn.Sequential 顺序堆叠：归一化 -> 升维线性 -> GELU -> Dropout -> 降维线性 -> Dropout
        self.net = nn.Sequential(
            nn.LayerNorm(dim),              # 先做层归一化，稳定 FFN 的输入分布
            nn.Linear(dim, hidden_dim),     # 升维到隐藏维（通常是 dim 的 4 倍）
            nn.GELU(),                       # 高斯误差线性单元，Transformer 常用的平滑激活
            nn.Dropout(dropout),             # 训练时随机丢弃一部分神经元，防过拟合
            nn.Linear(hidden_dim, dim),     # 降维回到原始维度，保证能和残差分支相加
            nn.Dropout(dropout),             # 输出端再做一次 dropout
        )

    def forward(self, x):
        # 直接按顺序执行上面定义的层堆叠，输入输出形状相同：(..., dim)
        return self.net(x)


# ---- 段落标题：多头自注意力——缩放点积注意力，支持因果掩码 ----
class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # 内层维度 = 单头维度 × 头数；Q/K/V 各自都要这么多通道
        inner_dim = dim_head * heads
        # 只有"单头且单头维 == 总维"时才跳过输出投影，此时 to_out 用 Identity 省参数
        project_out = not (heads == 1 and dim_head == dim)
        # 保存头数，forward 里用它把通道维拆成 (heads, dim_head)
        self.heads = heads
        # 缩放因子 1/sqrt(dim_head)：除以它可避免点积随维度变大而数值爆炸，
        # 导致 softmax 梯度饱和（标准 Transformer 的 "scaled" 就是指这个）
        self.scale = dim_head**-0.5
        # 保存 dropout 概率，训练/推理两种模式取不同值
        self.dropout = dropout
        # pre-norm 结构：在计算注意力之前先做层归一化
        self.norm = nn.LayerNorm(dim)
        # 对注意力权重在最后一维（key 的位置维）做 softmax，归一化成概率分布
        self.attend = nn.Softmax(dim=-1)
        # 一次性线性投影出 Q、K、V 三份：输出通道 = inner_dim * 3，且不使用偏置
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # 输出投影：把多头拼接后的结果映射回原始维度（并接一个 dropout）
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        # pre-norm：先归一化输入，再送入 QKV 投影
        x = self.norm(x)
        # 只有训练模式才启用 dropout；评估/推理时用 0.0 保证输出确定
        drop = self.dropout if self.training else 0.0
        # 一次线性投影后在最后一维切成 3 份：q, k, v，每份形状 (B, T, inner_dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        # 把每个头的通道维度单独拆出来：t: (B, T, inner_dim) -> (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        # 缩放点积注意力：is_causal=True 时使用上三角掩码，
        # 使第 i 个位置看不到 j>i 的信息（自回归预测必需）
        # 注意 softmax 已包含在 F.scaled_dot_product_attention 内部，self.attend 实际未被用到
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        # 把多头结果合并回 (B, T, inner_dim)：把 (heads, dim_head) 重新拼回一个通道维
        out = rearrange(out, "b h t d -> b t (h d)")
        # 经输出投影回到 (B, T, D)
        return self.to_out(out)


# ---- 段落标题：带 AdaLN-zero 条件调制的 Transformer Block ----
class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        # 第一个子层：多头自注意力
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # 第二个子层：前馈网络
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        # 注意力分支前的 LayerNorm：elementwise_affine=False 表示不设可学习的缩放/偏置，
        # 因为缩放/偏置改由条件调制网络按样本动态生成
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # FFN 分支前的 LayerNorm，同样不带可学习仿射参数
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 条件调制网络：把条件向量 c（形状 (B, T, dim)）映射成 6 组参数
        # ——注意力分支的 shift/scale/gate 三组，FFN 分支的 shift/scale/gate 三组
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # AdaLN-zero 初始化：把调制层最后一个 Linear 的权重全部置 0
        # 于是初始时 shift/scale/gate 全为 0，整个 block 等价于恒等映射，
        # 这样可以安全地把 block 插入到已训练好的主干上而不破坏其表征
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # 同样把该层的偏置也置 0，保证初始输出严格为 0
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # 由条件 c 生成 6 个调制量，每个形状为 (B, T, dim)（可广播到 x 的形状）
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # 注意力子层：先 norm1 归一化，再做 AdaLN 调制，最后用 gate 加门控后残差相加
        # 初始时 gate_msa=0，所以训练初期这一支路等价于"什么都不做"
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # FFN 子层：同样的"归一化 -> 调制 -> 门控 -> 残差"结构
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # 返回形状与输入一致：(B, T, dim)
        return x


# ---- 段落标题：标准（无条件）Transformer Block —— DiT/原始 Transformer 的那种 ----
class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        # 多头自注意力子层
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # 前馈网络子层
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        # 注意力分支的 pre-norm LayerNorm（不带可学习仿射参数）
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # FFN 分支的 pre-norm LayerNorm（不带可学习仿射参数）
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        # 残差连接 + 注意力子层输出
        x = x + self.attn(self.norm1(x))
        # 残差连接 + FFN 子层输出
        x = x + self.mlp(self.norm2(x))
        # 返回 (B, T, dim)
        return x


# ---- 段落标题：Transformer 主体——输入投影 + N 个 Block + 输出投影 ----
class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,            # 输入特征维度（如上一时刻的嵌入维度）
        hidden_dim,           # Transformer 内部宽度（Block 里所有层的维度）
        output_dim,           # 输出维度（如预测出的下一时刻嵌入维度）
        depth,                # Block 堆叠层数
        heads,                # 注意力头数
        dim_head,             # 每个注意力头的维度
        mlp_dim,              # FFN 中间层维度
        dropout=0.0,          # dropout 概率
        block_class=Block,    # Block 类型：Block（标准）或 ConditionalBlock（条件调制）
    ):
        super().__init__()
        # 输出前的 LayerNorm：稳定最后一层 Block 的输出分布
        self.norm = nn.LayerNorm(hidden_dim)
        # 用 ModuleList 保存 depth 个 Block，
        # 这样子模块会被正确注册，optimizer 才能看到它们的参数
        self.layers = nn.ModuleList([])

        # 输入投影：把 input_dim 映射到 hidden_dim；两者相同时直接用恒等层省参数
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 条件向量投影：条件 c 的维度同样是 input_dim（与主干输入同源），也要映射到 hidden_dim
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 输出投影：把 hidden_dim 映射到 output_dim（例如输出"下一时刻嵌入"的预测）
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        # 循环 depth 次，逐个构造 Block 并加入列表
        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        # 输入投影：(B, T, input_dim) -> (B, T, hidden_dim)
        x = self.input_proj(x)

        # 条件向量 c 非空时才做投影（用标准 Block 时可以完全不传 c）
        if c is not None:
            c = self.cond_proj(c)

        # 逐层前向：标准 Block 只吃 x；条件 Block 需要额外吃 c 作为条件信号
        # （用 isinstance 判断，因为 ConditionalBlock 并不是 Block 的子类）
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        # 最后一层归一化
        x = self.norm(x)
        # 输出投影：(B, T, hidden_dim) -> (B, T, output_dim)
        x = self.output_proj(x)
        # 返回前向结果
        return x


# ---- 段落标题：Embedder——把低维信号（动作/状态向量）嵌入到高维空间的 MLP ----
class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,        # 输入特征维度（level1 通常是 frameskip × action_dim）
        smoothed_dim=10,     # 1x1 卷积（patch_embed）之后的通道数
        emb_dim=10,          # 最终嵌入维度：需与该层的状态嵌入维度一致，便于融合
        mlp_scale=4,         # MLP 中间维度的放大倍数
        patch_embed=True,    # 是否先做 1x1 卷积做通道混合
    ):
        super().__init__()
        # 1x1 卷积等价于"逐时间位置的全连接"，作用是把多通道观测混合平滑成 smoothed_dim 维；
        # kernel_size=1, stride=1 表示不改变时间长度
        self.patch_embed = (
            nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
            if patch_embed
            else None
        )
        # 决定后面 MLP 的输入维度：用了卷积就是 smoothed_dim，否则就是原始 input_dim
        feature_dim = smoothed_dim if patch_embed else input_dim
        # 两层 MLP：升维到 mlp_scale * emb_dim -> SiLU 激活 -> 降维到 emb_dim
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D) or (B, T, C, D)
        """
        # 强制转成 float32：避免整型/半精度输入带来类型或数值问题
        x = x.float()
        # 只有配置了 patch_embed 才走卷积平滑这一段
        if self.patch_embed is not None:
            # 情况一：3D 输入 (B, T, D)
            if x.dim() == 3:
                # permute 把特征维放到最后，适配 Conv1d 期望的 (B, C, L) 布局
                x = x.permute(0, 2, 1)
                # 1x1 卷积：(B, T, D) -> (B, smoothed_dim, T)，时间长度不变
                x = self.patch_embed(x)
                # 再 permute 回 (B, T, smoothed_dim) 的"时间在前"布局
                x = x.permute(0, 2, 1)
            # 情况二：4D 输入 (B, T, C, D)，需要把 (C, D) 一起当作特征
            elif x.dim() == 4:
                # 解包形状：批大小、时间步、通道数、每通道的特征维
                b, t, c, d = x.shape
                # 合并时间与通道维：(B, T, C, D) -> (B, T*C, D)
                x = x.reshape(b, t * c, d)
                # 转到 (B, D, T*C) 布局，适配卷积
                x = x.permute(0, 2, 1)
                # 1x1 卷积混合通道：(B, D, T*C) -> (B, smoothed_dim, T*C)
                x = self.patch_embed(x)
                # 转回 (B, T*C, smoothed_dim)
                x = x.permute(0, 2, 1)
                # 拆回时间与通道维：(B, T*C, smoothed_dim) -> (B, T, C, smoothed_dim)
                x = x.reshape(b, t, c, -1)
            # 其他维度数直接报错，避免静默产生错误形状的下游结果
            else:
                raise ValueError(f"Expected 3D or 4D input, got shape {tuple(x.shape)}")
        # 最终的 MLP 投影：输出形状 (..., emb_dim)
        x = self.embed(x)
        # 返回嵌入结果
        return x


# ---- 段落标题：通用 MLP 编码器——H-JEPA 各层的 latent 编码器常用 ----
class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        final_ln=False,
    ):
        super().__init__()
        # 输出维缺省等于输入维（保持维度不变的"同构"编码）
        out_dim = output_dim or input_dim
        # 实例化归一化层：norm_fn 为 None 时用 Identity（等价于不做归一化）
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        # 主体四层：升维线性 -> 归一化 -> 激活 -> 降维线性
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )
        # 可选的输出端 LayerNorm；final_ln=False 时就是恒等层（不改变数值）
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

    # 前向：把输入展平成 (B*T, D) 送进 MLP，再接（可选的）末端 LayerNorm
    def forward(self, x):
        """
        x: (B*T, D)
        """
        # 先过主体 MLP，再过（可选的）末端归一化
        return self.final_norm(self.net(x))


# ---- 段落标题：ResidualLatentMLP——用于"精炼"潜在表征的两层 MLP ----
class ResidualLatentMLP(nn.Module):
    """MLP for refining latent vectors."""

    def __init__(
        self,
        input_dim,
        hidden_dim=384,
        output_dim=None,
        act_fn=nn.GELU,
        final_ln=True,
        trunc_normal_init=False,
    ):
        super().__init__()
        # 输出维缺省等于输入维
        out_dim = output_dim or input_dim
        # 输入端 LayerNorm：先归一化再进 MLP，稳定深层训练
        self.input_norm = nn.LayerNorm(input_dim)
        # 第一层全连接：input_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 激活函数实例（GELU）
        self.act = act_fn()
        # 第二层全连接：hidden_dim -> out_dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # 输出端 LayerNorm；final_ln=False 时为恒等层
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()
        # 可选：套用统一初始化（截断正态 + 偏置置零），与仓库其它模块保持一致
        if trunc_normal_init:
            self.apply(init_module_weights)

    def forward(self, x):
        # 输入归一化
        x = self.input_norm(x)
        # 第一层线性
        x = self.fc1(x)
        # 非线性激活
        x = self.act(x)
        # 第二层线性
        x = self.fc2(x)
        # 输出端（可选）归一化后返回
        return self.final_norm(x)


# ---- 段落标题：CLSDecoder——把 CLS 嵌入向量解码回像素图像（重建/可视化评估用） ----
class CLSDecoder(nn.Module):
    def __init__(
        self,
        cls_dim=384,          # 输入 CLS 嵌入的维度
        img_size=224,         # 目标图像边长
        patch_size=16,        # 每个 patch 的边长
        dim=256,              # decoder 内部宽度
        heads=8,              # 交叉注意力头数
        depth=3,              # 交叉注意力层数
    ):
        super().__init__()

        # 图像被切成 (img_size // patch_size)^2 个 patch
        self.num_patches = (img_size // patch_size) ** 2
        # 单个 patch 展平后的像素维度：patch_size*patch_size 个像素 × 3 通道
        patch_dim = patch_size * patch_size * 3

        # Learnable queries with better initialization
        # 可学习的 patch 查询向量，形状 (1, num_patches, dim)：作为交叉注意力的 Query
        self.queries = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        # 用小标准差正态初始化（沿用 ViT 的 0.02 初始化惯例），
        # 避免全零导致各 query 的梯度完全对称、无法彼此分化
        nn.init.normal_(self.queries, std=0.02)

        # Project CLS token
        # 把 CLS 嵌入（cls_dim 维）投影到 decoder 内部维度 dim
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # Multiple cross-attention layers
        # 多层交叉注意力；用 ModuleList 保存，参数会被正确注册
        self.layers = nn.ModuleList()

        # 逐层构造，每层包含：交叉注意力 + 两个 LayerNorm + 一个 MLP
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        # 交叉注意力：Query 来自可学习 query，Key/Value 来自 CLS 嵌入
                        "cross_attn": nn.MultiheadAttention(
                            dim, heads, batch_first=True
                        ),
                        # 注意力残差之后的 LayerNorm
                        "norm1": nn.LayerNorm(dim),
                        # 前馈网络：dim -> 4*dim -> dim
                        "mlp": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        # MLP 残差之后的 LayerNorm
                        "norm2": nn.LayerNorm(dim),
                    }
                )
            )

        # Final projection to pixels
        # 最终把特征映射到像素值：dim -> patch_dim
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

        # 记住 patch 大小，forward 做形状变换时要用
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (B*T, D) with T optional
        """
        # 批大小（若输入是 (B*T, D)，这里 B 实际就是 B*T）
        B = x.size(0)
        # patch 数量
        P = self.num_patches

        # Project and expand CLS token
        # 把 CLS 嵌入投影到 dim，再插入一个长度为 1 的维度 -> (B, 1, dim)
        # 它作为交叉注意力的 Key/Value，是所有 patch query 共享的信息源
        kv = self.cls_proj(x).unsqueeze(1)  # (B, 1, h)
        # 把可学习 query 沿批维广播：(1, P, dim) -> (B, P, dim)，不复制内存
        q = self.queries.expand(B, -1, -1)  # (B, P, h)

        # Cross-attention layers with residuals
        # 逐层做：交叉注意力 -> 残差 -> LayerNorm -> MLP -> 残差 -> LayerNorm
        for layer in self.layers:
            # 交叉注意力输出（MultiheadAttention 返回 (attn_output, attn_weights)，取第 0 项）
            # 形状 (B, P, dim)
            attn_out = layer["cross_attn"](q, kv, kv)[0]
            # 残差相加后做 LayerNorm
            q = layer["norm1"](q + attn_out)
            # 前馈网络输出
            mlp_out = layer["mlp"](q)
            # 再次残差相加并做 LayerNorm
            q = layer["norm2"](q + mlp_out)

        # Project to pixels
        # 映射到像素空间：(B, P, patch_dim)
        patches = self.to_pixels(q)  # (B, P, patch_dim)
        # 把每个 patch 拆成 (patch_size, patch_size, 3) 的像素网格
        patches = patches.reshape(B, P, self.patch_size, self.patch_size, 3)

        # Reshape to image
        # 由 patch 数量反推出图像的高和宽（要求图像是方形网格）
        H = W = int(self.num_patches**0.5)
        # 把 P 维拆成 H×W：(B, P, p, p, 3) -> (B, H, W, p, p, 3)
        patches = patches.reshape(B, H, W, self.patch_size, self.patch_size, 3)
        # permute 把通道维提到前面：索引含义依次是 (B, H, W, p, p, 3) -> (B, 3, H, p, W, p)
        img = img.permute(0, 5, 1, 3, 2, 4)
        # 合并 (H, p) 与 (W, p) 两对相邻维度，得到 (B, 3, H*p, W*p) 的图像张量
        img = img.reshape(B, 3, H * self.patch_size, W * self.patch_size)

        # 返回重建出的图像
        return img
