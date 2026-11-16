"""JEPA Model Implementation"""

# ---- 段落标题：本文件实现 JEPA（Joint-Embedding Predictive Architecture，联合嵌入预测架构）----
# JEPA 的核心思想：不直接预测像素，而是预测"未来状态的潜在表示(latent)"。
#   1) 编码器 encoder：观测 x（像素或下层嵌入） -> 潜在嵌入 z
#   2) 预测器 predictor：以 (历史嵌入 z_hist, 动作嵌入 a) 为条件 -> 预测的未来嵌入 z_hat
#   3) 训练时让 z_hat 逼近（stop-grad 的）真实嵌入 z_next，避免逐像素重建导致的模糊解
#
# 本文件注释中通用的维度记号：
#   B     : batch 大小
#   T     : 时间步长度（序列长度）
#   S     : 规划时候选动作序列的数量（sample 维）
#   D     : 状态嵌入维度（embed_dim）
#   A_emb : 动作嵌入维度

import torch
# PyTorch 核心：张量运算、autograd、cat / index_select 等
import torch.nn.functional as F
# 函数式接口：F.mse_loss 等
from einops import rearrange
# rearrange：用声明式语法一次完成 reshape / permute 的组合
from torch import nn
# torch.nn：所有可训练模块的基类


# ---- 段落标题：通用小工具 ----
def detach_clone(v):
    # detach() 切断梯度计算图；clone() 复制一份独立存储，
    # 这样后续 in-place 修改既不会污染原张量，也不会把梯度带回上游
    # 若不是张量（例如 None、bool、int），直接原样返回
    return v.detach().clone() if torch.is_tensor(v) else v


# ---- 段落标题：编码器调用的统一封装 ----
def _run_encoder(encoder, x):
    # 输入维数 >= 4：通常是图像序列 (B, T, C, H, W)，需要 ViT 的位置编码插值
    if x.ndim >= 4:
        # interpolate_pos_encoding=True：图像分辨率/序列长度与预训练不一致时，
        # 对 ViT 的位置编码做插值，保证编码器能处理非 224x224 的输入
        return encoder(x, interpolate_pos_encoding=True)
    # 输入维数 < 4：纯向量序列 (B, T, D)，普通前向即可
    return encoder(x)


def _split_encoder_output(
    encoder_output,
    allow_cls_token: bool = True,
) -> torch.Tensor:
    # 有的 encoder 返回 HuggingFace 风格的输出对象（带 last_hidden_state 属性），
    # 有的直接返回张量；这里统一取出"隐藏状态"张量
    hidden = (
        encoder_output.last_hidden_state
        if hasattr(encoder_output, "last_hidden_state")
        else encoder_output
    )

    # 2 维：(N, D) —— 已经压成"每个样本一个向量"，直接返回
    if hidden.ndim == 2:
        return hidden
    # 3 维：(B, T, D) 且允许取 CLS token：取时间维第 0 个 token 作为整段序列的总结
    if hidden.ndim == 3 and allow_cls_token:
        return hidden[:, 0]

    # 3 维但不允许取 CLS：说明输入是被"分块"的序列，必须由 encoder 自行压成 2 维
    if hidden.ndim == 3:
        raise ValueError(
            "Chunked temporal encoder inputs must be collapsed by the encoder "
            "to rank-2 shape (B*T, D); got rank-3 output "
            f"with shape {tuple(hidden.shape)}"
        )

    # 其余维数（1 维或 4 维以上）都不符合预期，直接报错
    raise ValueError(
        "Encoder output must be rank-2 or rank-3, "
        f"got shape {tuple(hidden.shape)}"
    )


# ---- 段落标题：ProjectedEncoder = 基础编码器 + 投影头 ----
class ProjectedEncoder(nn.Module):
    """Encoder followed by its projection head."""

    def __init__(self, encoder, projector=None):
        # 调用 nn.Module 初始化，建立参数注册表
        super().__init__()
        # 基础编码器（ViT / MLP / SequenceEncoder 等），作为子模块注册，参数会被优化器看到
        self.encoder = encoder
        # 投影头：把编码器隐层维度映射到最终 embed_dim；未提供时用恒等映射
        self.projector = projector or nn.Identity()

    def forward(self, x, *, allow_cls_token: bool = True):
        # 第一步：跑基础编码器，得到隐藏状态（可能是 (B, T, D) 或 (B, D)）
        output = _run_encoder(self.encoder, x)
        # 第二步：挑出需要的 token（CLS 或整段），得到 (B, D) / (B*T, D)
        hidden = _split_encoder_output(output, allow_cls_token=allow_cls_token)
        # 第三步：过投影头，输出维度变成 embed_dim，形状仍是 (B, D) / (B*T, D)
        return self.projector(hidden)

    def encode_info(self, info, *, key: str, allow_cls_token: bool = True):
        # 从 info 字典取出该 key 的张量并转成 float32（uint8 像素必须先转浮点）
        # "b t ... -> (b t) ..."：把 (B, T, ...) 合并成 (B*T, ...)，让编码器逐帧独立处理
        x = rearrange(info[key].float(), "b t ... -> (b t) ...")
        # 返回字典：键固定为 "embed"，值形状 (B*T, D)
        return {"embed": self(x, allow_cls_token=allow_cls_token)}


# ---- 段落标题：FusionEncoder = 图像分支 + 本体感分支融合 ----
class FusionEncoder(nn.Module):
    """Encode pixel/proprio streams and fuse them into the model embedding."""

    def __init__(
        self,
        pixel_encoder,
        proprio_encoder,
        projector=None,
        proprio_key: str = "proprio",
    ):
        # 建立参数注册表
        super().__init__()
        # 图像分支编码器（ViT 等），输出 pixel_embed
        self.pixel_encoder = pixel_encoder
        # 本体感（机器人自身状态）分支编码器（MLP 等），输出 proprio_embed
        self.proprio_encoder = proprio_encoder
        # 融合投影头：输入是 [pixel | proprio] 拼接后的宽度，输出是统一的 embed_dim
        self.projector = projector or nn.Identity()
        # 本体感数据在 info 字典里的键名：level1 用 "proprio"，高层用 "proprio_input"
        self.proprio_key = proprio_key

    def encode_info(self, info, *, key: str, allow_cls_token: bool = True):
        # 融合编码器必须能拿到本体感流，否则无法融合
        if self.proprio_key not in info:
            raise KeyError(
                "FusionEncoder requires a proprio stream, but "
                f"'{self.proprio_key}' is missing from input"
            )

        # 图像分支：(B*T, D_pixel)
        pixel_embed = self.pixel_encoder.encode_info(
            info,
            key=key,
            allow_cls_token=allow_cls_token,
        )["embed"]
        # 本体感分支：(B*T, D_proprio)
        proprio_embed = self.proprio_encoder.encode_info(
            info,
            key=self.proprio_key,
            allow_cls_token=allow_cls_token,
        )["embed"]
        # 沿最后一维拼接：(B*T, D_pixel) ⊕ (B*T, D_proprio) -> (B*T, D_pixel + D_proprio)
        # 再过投影头，得到统一维度 (B*T, embed_dim)
        embed = self.projector(torch.cat([pixel_embed, proprio_embed], dim=-1))
        # 同时保留两路原始嵌入，便于按分量（pixel / proprio）分别计算损失
        return {
            "embed": embed,
            "pixel_embed": pixel_embed,
            "proprio_embed": proprio_embed,
        }


# ---- 段落标题：ProjectedPredictor = 基础预测器 + 投影头 ----
class ProjectedPredictor(nn.Module):
    """Predictor followed by its projection head."""

    def __init__(self, predictor, projector=None):
        # 建立参数注册表
        super().__init__()
        # 基础预测器（ARPredictor / CausalTransformerPredictor 等）
        self.predictor = predictor
        # 投影头：把预测器的输出空间映射回 embed_dim，
        # 只有维度对齐才能和 encoder 产生的目标嵌入算 MSE
        self.projector = projector or nn.Identity()

    def forward(self, emb, act_emb):
        # 预测器前向：输入历史嵌入 (B, T, D) 与动作嵌入 (B, T, A_emb)
        # 输出预测的嵌入 (B, T, D_hidden)
        preds = self.predictor(emb, act_emb)
        # 把 (B, T, D_hidden) 摊平成 (B*T, D_hidden) 再过投影头
        # （投影头里常用 BatchNorm1d，要求 2D 输入）
        preds = self.projector(rearrange(preds, "b t d -> (b t) d"))
        # 再把 batch 维拆回去：(B*T, D) -> (B, T, D)
        return rearrange(preds, "(b t) ... -> b t ...", b=emb.size(0))


# ---- 段落标题：JEPA 单层模型（一个层级的基本单元）----
class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_embed,
        action_encoder=None,
        level:int=1,
        action_queue_size: int = 0,
        temporal_stride: int = 1,
        temporal_window_size: int = 1,
        target_length: int | None = None,
    ):
        # 建立参数注册表
        super().__init__()

        # 状态编码器：观测 -> 嵌入，(B, T, ...) -> (B, T, D)
        self.encoder = encoder
        # 预测器：历史嵌入 + 动作嵌入 -> 未来嵌入，(B, T, D) -> (B, T, D)
        self.predictor = predictor
        # 动作嵌入模块：把动作向量（level1 是原始动作块，高层是 latent action）映射到 A_emb 维
        self.action_embed = action_embed
        # 动作编码器：把下层的一段动作压成"宏观动作"，高层才用；缺省为恒等映射
        self.action_encoder = action_encoder or nn.Identity()
        # 层级编号（1 = 最细粒度层，数值越大时间尺度越粗）
        self.level = level
        # 时间下采样步长：每隔 stride 帧取一个时间步（高层的时间轴更稀疏）
        self.temporal_stride = int(temporal_stride)
        # 时间窗口大小：高层的一个 token 覆盖下层多少个连续时间步
        self.temporal_window_size = int(temporal_window_size)
        # 该层目标序列长度（= history_size + rollout_n）；None 表示不裁剪
        self.target_length = None if target_length is None else int(target_length)
        # 初始化"潜在动作队列"（给规划器当初值先验用）
        self.configure_action_queue(action_queue_size)

    def _state_windows(
        self,
        x: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        """Subsample or chunk a time sequence using full state windows only."""
        # window_size == 1：只需按下标抽取（抽稀），不做窗口拼接
        if window_size == 1:
            # index_select 沿时间维取 starts 指定的下标：(B, T, ...) -> (B, N, ...)
            # 其中 N = starts.numel()，即窗口个数
            return x.index_select(1, starts)

        # 构造窗口内的相对偏移 0, 1, ..., window_size-1
        offsets = torch.arange(window_size, device=x.device)
        # 外积得到每个窗口起点对应的 window_size 个连续下标，形状 (N, window_size)
        indices = starts[:, None] + offsets[None, :]
        # 摊平成一维 (N*window_size,)，便于一次性批量 index_select
        flat = indices.reshape(-1)
        # 按下标取出所有窗口内的帧：(B, T, ...) -> (B, N*window_size, ...)
        chunks = x.index_select(1, flat)
        # 还原成带窗口维的结构：(B, N, window_size, ...)，最后一维保持原特征维
        return chunks.reshape(x.size(0), starts.numel(), window_size, *x.shape[2:])

    def _action_windows(
        self,
        x: torch.Tensor,
        starts: torch.Tensor,
        state_window_size: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunk actions with zero padding.

        The first action in chunk T aligns with the last state in chunk T.
        The returned mask is True for padded positions.
        """
        # 动作序列的总长度 T
        seq_len = x.size(1)
        # 每个窗口的第一个动作下标 = 窗口内最后一个状态的下标
        # （"从该状态出发执行的那个动作"与该状态对齐，这是分层动作对齐的关键约定）
        action_starts = starts + state_window_size - 1
        # 动作块长度等于 stride（与状态窗口长度 window_size 相互独立）
        offsets = torch.arange(stride, device=x.device)
        # 得到 (N, stride) 的动作下标矩阵
        indices = action_starts[:, None] + offsets[None, :]
        # 标记哪些下标落在有效范围内
        valid = indices < seq_len
        # 无效位置即"填充"，mask 中 True 表示 padding
        padding_mask = ~valid
        # clamp 把越界下标压回合法范围，避免 index_select 越界报错（随后会被乘 0 抹掉）
        indices = indices.clamp(max=seq_len - 1)

        # 摊平下标用于批量取样
        flat = indices.reshape(-1)
        # 取动作：(B, T, act_dim) -> (B, N*stride, act_dim)
        chunks = x.index_select(1, flat)
        # 还原成块结构：(B, N, stride, act_dim)
        chunks = chunks.reshape(x.size(0), starts.numel(), stride, *x.shape[2:])

        # 把有效掩码从 (N, stride) 提升为 (1, N, stride)，便于向特征维广播
        valid = valid.unsqueeze(0)
        # 继续在末尾补 1，直到维数与 chunks 一致（所有特征维都靠广播处理）
        while valid.ndim < chunks.ndim:
            valid = valid.unsqueeze(-1)
        # 生成 (B, N, stride) 的 padding mask，True 表示该动作是补出来的
        mask = padding_mask.unsqueeze(0).expand(x.size(0), -1, -1)
        # 把填充位置乘 0（清零），并返回 mask 供上层使用
        return chunks * valid.to(dtype=chunks.dtype), mask

    def _chunk_temporal_info(
        self,
        info: dict,
        key: str,
        stride: int,
        window_size: int,
    ) -> dict:
        # 统一转成 int，避免配置里是 numpy 标量/字符串导致比较异常
        stride = int(stride)
        window_size = int(window_size)
        # 步长必须 >= 1
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        # 窗口必须 >= 1
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        # 必须存在作为时间长度基准的 key
        if key not in info:
            raise KeyError(f"Input key '{key}' not found in info")

        # 取基准张量（通常是像素流或下层嵌入流）
        ref = info[key]
        # 必须是至少 3 维的 (B, T, ...) 张量
        if not torch.is_tensor(ref) or ref.ndim < 3:
            raise ValueError(
                f"Expected '{key}' to be a tensor with shape (B, T, ...), "
                f"got {type(ref)} with shape {getattr(ref, 'shape', None)}"
            )

        # 序列总长度 T
        seq_len = ref.size(1)
        # 序列太短，凑不出一整个窗口
        if seq_len < window_size:
            raise ValueError(
                f"Sequence length {seq_len} is shorter than window_size={window_size}"
            )
        # 窗口起点：0, stride, 2*stride, ...，保证每个起点后面都有完整的 window_size 帧
        starts = torch.arange(0, seq_len - window_size + 1, stride, device=ref.device)
        # 浅拷贝 info 字典，后续只替换被分块的键（不改动调用方的原字典）
        chunked = dict(info)

        # 遍历 info 中所有张量流，统一做分块
        for info_key, value in info.items():
            # 非张量或维数不足 3 的（标量、字典等）保持原样
            if not torch.is_tensor(value) or value.ndim < 3:
                continue
            # batch 维或时间长度与基准对不齐的流跳过（无法与基准同步分块）
            if value.size(0) != ref.size(0) or value.size(1) != seq_len:
                continue

            # 动作流：单独按 stride 分块（动作块长度 = stride），并额外产出 padding mask
            if info_key == "action":
                action_chunks, action_mask = self._action_windows(
                    value,
                    starts=starts,
                    state_window_size=window_size,
                    stride=stride,
                )
                # 分块后的动作：(B, N, stride, act_dim)
                chunked[info_key] = action_chunks
                # padding mask：(B, N, stride)，True = 填充
                chunked["action_mask"] = action_mask
            # 其它以 "action" 开头的键（如 action_1、action_mask）保持不变
            elif not info_key.startswith("action"):
                # 状态流：按 window_size 分块：(B, T, ...) -> (B, N, window_size, ...)
                chunked[info_key] = self._state_windows(
                    value,
                    starts=starts,
                    window_size=window_size,
                )

        # 返回分块后的 info 副本
        return chunked

    def configure_action_queue(self, queue_size: int = 0) -> None:
        """Configure latent action queue used to initialize planner priors."""
        # 转成 int 并记录队列长度
        queue_size = int(queue_size)

        # 保存队列长度配置（0 表示禁用）
        self.action_queue_size = queue_size
        # 长度为 0：直接禁用队列
        if queue_size == 0:
            self.latent_action_queue = None
            return

        # 还没注册过这个 buffer（首次调用）：注册一个空的 (0, 0) buffer
        if "latent_action_queue" not in self._buffers:
            self.register_buffer(
                "latent_action_queue", torch.empty(0, 0), persistent=True
            )
        # 已注册但被置成 None：重新放一个空 buffer 回去
        elif self.latent_action_queue is None:
            self._buffers["latent_action_queue"] = torch.empty(0, 0)

        # 取当前队列
        queue = self.latent_action_queue
        # 若已有内容（例如从 checkpoint 恢复），只保留最后 queue_size 条
        if torch.is_tensor(queue) and queue.ndim == 2 and queue.numel() > 0:
            self._buffers["latent_action_queue"] = queue[-queue_size:]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # latent_action_queue grows during training, so a fresh model's empty (0,0)
        # buffer can't receive a checkpoint's (queue_size, A_emb) tensor under strict
        # load. Resize it to the saved shape before the parent copies into it.
        # checkpoint 中的完整键名 = prefix + buffer 名
        key = prefix + "latent_action_queue"
        # 只有 checkpoint 里有该 buffer、且当前模型也注册了它时才需要特殊处理
        if key in state_dict and "latent_action_queue" in self._buffers:
            # 取出保存下来的队列张量，形状 (queue_size, A_emb)
            saved = state_dict[key]
            # 预先把 buffer 调整成同样的形状，否则 strict 加载会因形状不匹配报错
            self._buffers["latent_action_queue"] = saved.new_empty(saved.shape)
        # 调用父类完成正常的参数 / buffer 拷贝
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _update_latent_action_queue(self, latent_actions: torch.Tensor) -> None:
        """latent_actions: (B, T, A_emb)"""

        # 只有训练模式、且队列长度 > 0 时才维护队列
        if not self.training or self.action_queue_size <= 0:
            return

        # 断言输入形状 (B, T, A_emb)
        assert latent_actions.ndim == 3, "latent_actions must be (B, T, A_emb)"

        # 只取每个样本的第一个时间步（T=0 通常代表"当前"时刻），并断梯度
        # samples 形状：(B, A_emb)
        samples = latent_actions[:, 0].detach()

        # 当前队列
        queue = self.latent_action_queue
        # 目标队列长度
        queue_size = int(self.action_queue_size)

        # 队列为空：直接用新样本初始化
        if queue.numel() == 0:
            updated = samples
        else:
            # 旧队列在前、新样本在后，沿 batch 维拼接：(old_len + B, A_emb)
            updated = torch.cat(
                [queue.to(device=samples.device, dtype=samples.dtype), samples],
                dim=0,
            )

        # 只保留最近 queue_size 条（先进先出的滑动窗口）
        self._buffers["latent_action_queue"] = updated[-queue_size:]

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Return latent action queue, or None if disabled/empty."""
        # 兼容"该属性根本没被创建"的情况
        queue = getattr(self, "latent_action_queue", None)
        # 未启用或为空时返回 None
        if not torch.is_tensor(queue) or queue.numel() == 0:
            return None

        # 返回队列本身，形状 (queue_size, A_emb)
        return queue

    def encode(
        self,
        info,
        key="pixels",
        temporal_stride: int | None = None,
        chunk_temporal_inputs: bool = False,
    ):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        # 标记输入是否被"分块"：分块后序列里没有 CLS token 可取
        chunked_state_input = False
        # 需要对时间维做分块/抽稀（高层 JEPA 编码下层嵌入时）
        if chunk_temporal_inputs:
            # 未显式给步长时，使用本层配置的 temporal_stride
            if temporal_stride is None:
                temporal_stride = getattr(self, "temporal_stride", 1)
            # 取本层的时间窗口大小
            temporal_window_size = getattr(self, "temporal_window_size", 1)
            # 窗口 > 1 说明是真正的"窗口化"输入（而不是简单抽稀）
            chunked_state_input = int(temporal_window_size) > 1

            # 对 info 中所有时间流做分块/抽稀，返回新的 info 副本
            info = self._chunk_temporal_info(
                info,
                key=key,
                stride=temporal_stride,
                window_size=temporal_window_size,
            )

        # batch 维大小 B（分块不会改变 B）
        B = info[key].size(0)
        # 调用编码器，得到 {"embed": (B*T, D)} 之类的字典
        encoder_outputs = self.encoder.encode_info(
            info,
            key=key,
            allow_cls_token=not chunked_state_input,
        )
        # 把编码结果写回 info，键名加后缀 "_0"，并把 (B*T, D) 还原成 (B, T, D)
        for output_key, value in encoder_outputs.items():
            info[f"{output_key}_0"] = rearrange(
                value,
                "(b t) ... -> b t ...",
                b=B,
            )

        # 输入里带动作时，才编码动作
        if "action" in info:
            # 取分块时生成的 padding mask（可能不存在）
            action_mask = info.get("action_mask")
            # 没有 mask：直接编码动作张量 (B, T, act_dim)
            if action_mask is None:
                latent_actions = self.action_encoder(info["action"])
            else:
                # 有 mask：动作编码器会忽略 padding 位置
                latent_actions = self.action_encoder(info["action"], action_mask)
            # 把 latent action 写进队列（供规划器当作动作先验）
            self._update_latent_action_queue(latent_actions)
            # 动作嵌入：latent action -> (B, T, A_emb)，与状态嵌入在时间上对齐
            info["action_0"] = self.action_embed(latent_actions)

        # 返回带 embed_0 / action_0 等键的 info
        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        # 直接调用预测器：历史嵌入 + 动作嵌入 -> 预测的未来嵌入 (B, T, D)
        return self.predictor(emb, act_emb)

    def parallel_unroll(self, emb, act_emb, nsteps):
        """nsteps passes over the band: each pass predicts frames 1..T-1 from the previous pass (the
        encoder band on pass 0, detached after), with ground-truth frame 0 re-injected on the left.
        Returns every pass stacked: (nsteps, B, T, D), aligned with emb.
        """
        # 需要预测的帧数：总长 T+1，第 0 帧是作为条件的真实帧
        T = emb.size(1) - 1
        # 第 0 轮的输入：前 T 帧真实嵌入（最后一帧没有后继，先丢掉）
        pred_input = emb[:, :T]
        # 收集每一轮 rollout 的预测结果
        passes = []
        # 循环 nsteps 轮
        for _ in range(nsteps):
            # 用当前输入预测后 T 帧，再在最左边拼上真实的第 0 帧作为条件
            # predicted 形状：(B, T+1, D)，其中第 0 帧是真实条件帧
            predicted = torch.cat([emb[:, :1], self.predict(pred_input, act_emb[:, :T])], dim=1)
            # 保存这一轮的完整预测带
            passes.append(predicted)
            # 下一轮输入：真实第 0 帧 + 上一轮预测的第 1..T-1 帧（detach 断梯度）
            # 这样多轮之间不会串成一条长梯度链
            pred_input = torch.cat([emb[:, :1], predicted[:, 1:T].detach()], dim=1)
        # 把 nsteps 轮结果堆叠：(nsteps, B, T+1, D)
        return torch.stack(passes)

    def rollout(self, info, action_sequence, history_size: int = 1):
        """Rollout the model given an initial info dict and action sequence."""

        # 必须提供像素观测作为初始状态
        assert "pixels" in info, "pixels not in info_dict"
        # 图像高度（用于切分"当前/历史动作"与"未来动作"）
        H = info["pixels"].size(2)
        # 候选动作序列形状 (B, S, T, act_dim)：B 个样本、S 条候选、T 个时间步
        B, S, T = action_sequence.shape[:3]
        # 沿时间维切分：前 H 个是"当前/历史"动作，其余是未来要执行的动作
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        # 把历史动作写进 info，供编码器使用
        info["action"] = act_0
        # 需要向未来预测的步数
        n_steps = T - H

        # 取每个候选序列的第 0 帧观测，构造初始状态字典
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        # 若还没有 embed_0（初始嵌入），就现场编码一次（推理不需要梯度）
        if "embed_0" not in _init:
            with torch.no_grad():
                _init = self.encode(_init)
        # 把初始嵌入广播到 S 条候选：(B, D) -> (B, S, D)
        emb = info["embed_0"] = _init["embed_0"].unsqueeze(1).expand(B, S, -1, -1)
        # 把初始字典里的张量都 detach+clone，避免污染调用方持有的数据
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # 合并 batch 维与候选维：(B, S, D) -> (B*S, D)，每条候选当成独立样本
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        # 历史动作同样合并：(B, S, T0, act_dim) -> (B*S, T0, act_dim)
        act = rearrange(act_0, "b s ... -> (b s) ...")
        # 未来动作：(B, S, T-H, act_dim) -> (B*S, T-H, act_dim)
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # 历史窗口长度：预测器一次看多少帧历史
        HS = history_size
        # 逐步向前预测 n_steps 步
        for t in range(n_steps):
            # 当前动作序列的嵌入：(B*S, T0+t, A_emb)
            act_emb = self.action_embed(act)
            # 取最近 HS 帧的历史嵌入：(B*S, HS, D)
            emb_trunc = emb[:, -HS:]
            # 取最近 HS 帧对应的动作嵌入：(B*S, HS, A_emb)
            act_trunc = act_emb[:, -HS:]
            # 预测下一帧嵌入，只保留最后一帧：(B*S, 1, D)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            # 把预测帧接到嵌入序列末尾：(B*S, T0+t+1, D)
            emb = torch.cat([emb, pred_emb], dim=1)

            # 取出这一步要执行的真值动作：(B*S, 1, act_dim)
            next_act = act_future[:, t : t + 1, :]
            # 动作序列同步增长（长度 +1）
            act = torch.cat([act, next_act], dim=1)

        # predict the last state
        # 循环结束后再预测最后一帧（把预测覆盖到时间轴末端）
        act_emb = self.action_embed(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # 把合并后的 (B*S, ...) 还原成 (B, S, ...)
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        # 写回 info，键名与编码器侧对称（predicted_embed_0）
        info["predicted_embed_0"] = pred_rollout

        # 返回带预测嵌入的 info
        return info

    def criterion(self, info_dict: dict):
        """Planning cost: MSE between the last cost_last_n predicted embeddings and the goal."""
        # 取所有预测嵌入，去掉第 0 帧（它是真实条件帧）：(B, S, W, D)
        pred_emb = info_dict["predicted_embed_0"][:, :, 1:, :]
        # 只对最后 cost_last_n 帧计算代价，默认 1 帧
        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        # 实际使用的窗口长度（不能超过可用帧数）
        cost_window = min(cost_last_n, pred_emb.shape[-2])

        # 截取最后 cost_window 帧：(B, S, cost_window, D)
        pred_emb = pred_emb[:, :, -cost_window:, :]
        # 目标嵌入广播成与预测同形状，并 detach（目标不参与反传）
        goal_emb = info_dict["goal_embed_0"].expand_as(pred_emb).detach()
        # 逐元素平方误差，不做归约：(B, S, cost_window, D)
        cost = F.mse_loss(pred_emb, goal_emb, reduction="none")

        # 中间帧的权重（默认 1.0 表示不加权）
        intermediate_weight = float(info_dict.get("intermediate_cost_weight", 1.0))
        # 权重不为 1 且窗口内确实有多帧时才加权
        if intermediate_weight != 1.0 and cost.shape[2] > 1:
            # 构造 (cost_window,) 的权重向量
            weights = cost.new_ones(cost.shape[2])
            # 最后一帧（真正的终点）权重保持 1，前面各帧用 intermediate_weight
            weights[:-1] = intermediate_weight
            # 广播到 (B, S, cost_window, D)：权重只沿时间维变化
            cost = cost * weights.view(1, 1, -1, 1)

        # 把 batch 维与时间维都求和，只保留候选维 S -> (B, S)
        return cost.sum(dim=tuple(range(2, cost.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # 若还没算过目标嵌入，就必须要提供原始目标图像
        if "goal_embed_0" not in info_dict:
            assert "goal" in info_dict, "goal not in info_dict"

        # 模型参数所在设备（所有张量都要搬到同一设备上）
        device = next(self.parameters()).device
        # 把 info_dict 里所有张量搬到该设备
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # 目标嵌入仍然缺失时，现场编码目标图像
        if "goal_embed_0" not in info_dict:
            # 取每个候选序列第 0 帧的状态（观测），作为编码输入的模板
            _goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            # 用目标图像替换观测像素
            _goal["pixels"] = _goal["goal"]

            # 处理 "goal_xxx" 形式的键：去掉前缀后当作普通状态流（例如 goal_pixels -> pixels）
            for k in info_dict:
                if k.startswith("goal_"):
                    _goal[k[len("goal_"):]] = _goal.pop(k)

            # 目标不需要动作
            _goal.pop("action")
            # 编码目标图像，得到目标嵌入（推理不需要梯度）
            with torch.no_grad():
                _goal = self.encode(_goal)

            # keep the sample axis, (B, 1, T, D): expand_as against (B, S, W, D) is wrong for B > 1
            # 保留候选轴（长度为 1），形状 (B, 1, T, D)：
            # 注意不能直接 expand_as 到 (B, S, W, D)，那样在 B > 1 时会错位
            info_dict["goal_embed_0"] = _goal["embed_0"].unsqueeze(1)

        # 对每条候选动作序列做 rollout，得到预测的潜在轨迹
        info_dict = self.rollout(info_dict, action_candidates)

        # 计算规划代价：(B, S)，S 条候选各自的代价
        return self.criterion(info_dict)
