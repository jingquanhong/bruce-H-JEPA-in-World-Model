"""Gradient-based solver for model-based planning."""
# 本模块：基于梯度的规划求解器 —— 把"动作序列"当成可优化参数，
# 对世界模型的代价函数求反向传播，用梯度下降迭代出低成本动作

# 导入 time：记录单次 solve 的墙钟耗时
import time
# 从 typing 导入 Any：configure 的 config 参数类型标注
from typing import Any

# 导入 Gymnasium：动作空间 gym.Space 用于确定动作维度
import gymnasium as gym
# 导入 NumPy：均值/有限性判断等小工具
import numpy as np
# 导入 PyTorch：动作张量、autograd 梯度、优化器都基于它
import torch

# 从同级 solver.py 导入 Costable 协议（模型接口）与 get_planning_horizon（递减视野调度）
from .solver import Costable, get_planning_horizon


# ---------------------------------------------------------------------------
# 段落标题：GradientSolver —— 对每个并行环境维护一条动作序列 (num_samples, horizon, action_dim)
#           一次 solve 的整体流程：
#           1) init_action() 构造带噪声的初始动作张量（并注册为可优化参数）
#           2) 按 batch_size 分批，对每个 batch 跑 n_steps 轮：算代价→反传→更新→加噪→裁剪
#           3) 早停判定（连续 patience 轮无显著提升就停）
#           4) 用最终代价在 num_samples 个候选里选出最优动作，返回动作 + 模型预测
# ---------------------------------------------------------------------------
class GradientSolver(torch.nn.Module):
    """Gradient-based solver using backpropagation through the world model.

    Args:
        model: World model implementing the Costable protocol.
        n_steps: Number of gradient descent iterations.
        min_its: Minimum iterations to run before early stopping can trigger.
        batch_size: Number of environments to process in parallel.
        var_scale: Initial variance scale for action perturbations.
        num_samples: Number of action samples to optimize in parallel.
        action_noise: Noise added to actions during optimization.
        early_stop_patience: Consecutive non-improving steps before stopping.
        early_stop_rel_delta: Minimum relative improvement required to reset patience.
        action_clip_sigma: Clip actions to prior mean ± sigma·std each step; None uses queue quantiles.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        optimizer_cls: PyTorch optimizer class to use.
        optimizer_kwargs: Keyword arguments for the optimizer.
    """

    def __init__(
        self,
        model: Costable,
        n_steps: int,
        min_its: int = 30,
        batch_size: int | None = None,
        var_scale: float = 1,
        num_samples: int = 1,
        action_noise: float = 0.0,
        early_stop_patience: int = 5,
        early_stop_rel_delta: float = 1e-2,
        action_clip_sigma: float | None = None,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        # 调用 torch.nn.Module 的构造函数（本类继承 Module，为的是能保存参数/用标准接口）
        super().__init__()
        # 世界模型：提供 get_cost / rollout，且必须满足 Costable 协议
        self.model = model
        # GD 的最大迭代轮数（每轮对整批动作做一次 求导→更新）
        self.n_steps = n_steps
        # 最少迭代轮数：未跑满这个数之前，早停判定不生效
        self.min_its = max(0, int(min_its))
        # 每批同时处理的环境数；None 表示所有环境一次性处理（可能吃很多显存）
        self.batch_size = batch_size
        # 每个环境并行优化的动作候选条数（最后从 num_samples 条里挑代价最低的）
        self.num_samples = num_samples
        # 初始扰动方差缩放：动作初始化噪声 = 先验标准差 × var_scale
        self.var_scale = var_scale
        # 迭代过程中额外加的高斯探索噪声标准差（0 表示不加）
        self.action_noise = action_noise
        # 早停耐心值：连续多少轮无显著提升就提前结束 GD
        self.early_stop_patience = max(0, int(early_stop_patience))
        # 判定"显著提升"的相对阈值（相对最好代价的比例）
        self.early_stop_rel_delta = max(0.0, float(early_stop_rel_delta))
        # 动作裁剪的 sigma 系数：裁到 先验均值 ± sigma×标准差；None 则用历史队列分位数
        self.action_clip_sigma = action_clip_sigma
        # 计算设备（'cpu' 或 cuda 设备），张量都会搬到这里
        self.device = device
        # 独立的随机数生成器并固定种子，保证扰动/噪声可复现
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        # 记录使用的优化器类（默认 SGD，也可换 Adam 等）
        self.optimizer_cls = optimizer_cls
        # 优化器超参数；未提供时默认 {'lr': 1.0}（动作空间尺度小，步长大点没关系）
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1.0}
        )

        # 以下三个是 configure() 才会填充的运行时字段，先置 None
        self._n_envs = None
        self._action_dim = None
        self._config = None
        # Per-solve compute metadata, appended on every solve() call; collected by
        # the eval to compute planner FLOPs (see planning_eval.pop_solve_records).
        # 每次 solve 追加一条计算元数据（迭代数、耗时等），供评测统计规划器算力/FLOPs
        self.solve_records = []

    def configure(
        self,
        *,
        action_space: gym.Space,
        n_envs: int,
        config: Any,
        level: int = 1,
    ) -> None:
        """Configure the solver with environment specifications."""
        # 保存动作空间（形状形如 (n_envs, ...action_dim...)，用于推导动作维度）
        self._action_space = action_space
        # 记录并行环境数，动作张量第 0 维就是它
        self._n_envs = n_envs
        # 保存规划配置对象（horizon、action_block、cost_last_n 等都从它读）
        self._config = config
        # 记录当前求解器服务于哪一层（1=底层动作，>1=高层隐动作）
        self.level = int(level)
        # 第 1 层（原始动作）：动作空间形状去掉第 0 维（环境批量维）后求乘积 = 单步动作维度
        if level == 1:
            self._action_dim = int(np.prod(action_space.shape[1:]))
        # 更高层（隐动作）：直接从配置里读取隐动作维度
        else:
            self._action_dim = int(config.action_dim)

    def _expand_info_dict_for_samples(
        self,
        info_dict: dict,
        start_idx: int,
        end_idx: int,
        num_samples: int,
    ) -> dict:
        """Slice env batch and add sample dimension expected by world models."""
        # 本批的环境数 = 结束索引 - 起始索引
        current_bs = end_idx - start_idx
        # 存放展开后的 info 字典
        expanded_infos = {}

        # 遍历 info 中的每个键值对
        for k, v in info_dict.items():
            # 张量：先切出本批环境的切片并搬到计算设备
            if torch.is_tensor(v):
                v_batch = v[start_idx:end_idx].to(self.device)
                # unsqueeze(1) 在第 1 维插入"采样"维 → (bs, 1, ...)；
                # expand 把该维复制成 num_samples 份（共享内存、不实际复制数据）
                v_batch = v_batch.unsqueeze(1).expand(
                    current_bs, num_samples, *v_batch.shape[1:]
                )
            # numpy 数组：同样先切片
            elif isinstance(v, np.ndarray):
                v_batch = v[start_idx:end_idx]
                # 在新的第 1 维上真复制 num_samples 份，得到 (bs, num_samples, ...)
                v_batch = np.repeat(v_batch[:, None, ...], num_samples, axis=1)
            # 其他标量/字符串（如 cost_last_n、variation 名）：原样透传，不批量
            else:
                v_batch = v

            # 写回展开后的字典
            expanded_infos[k] = v_batch

        # 返回带"采样维"的 info，世界模型按 (bs, num_samples, ...) 对齐动作候选
        return expanded_infos

    @staticmethod
    def _has_significant_improvement(
        score: float,
        best_score: float,
        rel_delta: float,
    ) -> bool:
        """Return True when `score` improves enough over `best_score`."""
        # 历史最好值不是有限数（第一次比较，正无穷）→ 视为显著提升
        if not np.isfinite(best_score):
            return True

        # 提升量 = 历史最好代价 - 当前代价（代价越小越好，正数表示变好）
        improvement = best_score - score
        # 显著阈值 = 相对比例 × |历史最好值|，用 1e-12 防止 0 附近除法退化
        threshold = rel_delta * max(abs(best_score), 1e-12)
        # 只有提升量超过阈值才算"显著改善"
        return improvement > threshold

    @torch.inference_mode()
    def _rollout_final_actions(self, info_dict: dict, actions: torch.Tensor) -> dict:
        """Roll out the model once with optimized actions and return predictions."""
        # 把最优动作搬到设备，并在第 1 维插入"采样"维 → (n_envs, 1, horizon, action_dim)
        action_candidates = actions.to(self.device).unsqueeze(1)
        # 展开 info 到与候选匹配的形状（此处只要 1 个采样，因为动作已定）
        info_dict = self._expand_info_dict_for_samples(
            info_dict, start_idx=0, end_idx=self.n_envs, num_samples=1
        )

        # 用世界模型向前推演整条动作序列（inference_mode 下不建梯度图，省显存）
        rollout_info = self.model.rollout(
            info_dict,
            action_candidates,
        )

        # 收集所有 'predicted*' 前缀的预测结果（预测的下一帧/隐状态等）
        predictions: dict[str, torch.Tensor] = {}

        # 遍历 rollout 返回的信息
        for k, v in rollout_info.items():
            # 只要键以 'predicted' 开头且是张量的项，其他元数据跳过
            if not (k.startswith('predicted') and torch.is_tensor(v)):
                continue

            # 断开计算图并搬到 CPU，方便外部保存/比较
            value = v.detach().cpu()
            # 若第 1 维是大小为 1 的采样维（ndim>1 且 shape[1]==1），压掉它
            if value.ndim > 1 and value.shape[1] == 1:
                value = value[:, 0]
            # 存入预测字典
            predictions[k] = value

        # 返回 {预测键: CPU 张量}
        return predictions

    def _get_action_prior_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Estimate mean/std prior from queued latent actions, if available."""
        # 向模型索要"历史隐动作队列"；模型没实现该方法时取 None
        queue = getattr(self.model, 'get_latent_action_queue', lambda: None)()
        # 队列不是非空张量（无先验数据）→ 返回 None，调用方退回标准正态先验
        if not (torch.is_tensor(queue) and queue.numel() > 0):
            return None

        # 队列必须是二维：(历史动作条数, 动作维度)
        assert queue.ndim == 2

        # 断开梯度并搬到计算设备
        queue = queue.detach().to(self.device)
        # 沿第 0 维（样本维）求均值，得到每维动作的先验均值 (action_dim,)
        mean = queue.mean(dim=0)
        # 求总体标准差（unbiased=False），并用 clamp_min 防止方差为 0 导致除零
        std = queue.std(dim=0, unbiased=False).clamp_min(1e-4)

        # 返回 (均值, 标准差)，用于初始化动作与裁剪边界
        return mean, std

    def _get_action_clip_bounds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Estimate per-dimension clipping bounds from queued latent actions."""
        # 配置了 sigma 裁剪模式：用 先验均值 ± sigma×标准差 作为边界
        if self.action_clip_sigma is not None:
            # 取先验统计量
            prior_stats = self._get_action_prior_stats()
            # 没有先验时退回标准正态：均值 0、标准差 1
            if prior_stats is None:
                mean = torch.zeros(self.action_dim, device=self.device)
                std = torch.ones(self.action_dim, device=self.device)
            # 有先验就用历史动作算出的均值/标准差
            else:
                mean, std = prior_stats
            # 下界 = mean - sigma*std，上界 = mean + sigma*std，返回 (lower, upper)
            return mean - self.action_clip_sigma * std, mean + self.action_clip_sigma * std

        # 未配置 sigma：退化为"分位数裁剪"模式，同样先取历史动作队列
        queue = getattr(self.model, 'get_latent_action_queue', lambda: None)()
        # 队列不可用 → 返回 None，solve 中将不做任何裁剪
        if not (torch.is_tensor(queue) and queue.numel() > 0):
            return None

        # 断言二维结构 (条数, 维度)
        assert queue.ndim == 2

        # 断梯度、搬设备
        queue = queue.detach().to(self.device)
        # 每一维的 2% 分位数作为下界（去掉历史动作的极端低值）
        lower = torch.quantile(queue, 0.02, dim=0)
        # 每一维的 98% 分位数作为上界（去掉极端高值）
        upper = torch.quantile(queue, 0.98, dim=0)

        # 返回分位数边界
        return lower, upper

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        # 透传 configure 时记录的并行环境数（动作张量第 0 维）
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        # 单步动作维度 × action_block：一次优化的"宏动作"总维度
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        # 规划视野：一次求解要优化多少个时间步（来自配置）
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        # 让实例可像函数一样调用，直接等价于 solve(...)
        return self.solve(*args, **kwargs)

    def init_action(
        self,
        actions: torch.Tensor | None = None,
        planning_horizon: int | None = None,
    ) -> None:
        """Initialize the action tensor for optimization."""
        # 未指定本次规划视野时，退回配置的默认 horizon
        if planning_horizon is None:
            planning_horizon = self.horizon

        # 打印当前层级与本次使用的规划视野，便于调试/阅读日志
        print(f"level {self.level} planning horizon: {planning_horizon }")

        # 从模型的历史隐动作队列估计先验均值/标准差（可能为 None）
        prior_stats = self._get_action_prior_stats()
        # 第 2 层及以上的高层规划必须有隐动作先验，否则无从初始化
        if self.model.level > 1:
            assert prior_stats is not None, (
                'Higher-level GD requires latent action priors from the model.'
            )

        # 没有先验：用标准正态（均值 0、标准差 1）作为动作初始化分布
        if prior_stats is None:
            base_mean = torch.zeros(self.action_dim, device=self.device)
            base_std = torch.ones(self.action_dim, device=self.device)
        # 有先验：直接用历史动作的均值与标准差
        else:
            base_mean, base_std = prior_stats
        # 用 var_scale 放大/缩小初始扰动幅度（探索强度旋钮）
        base_std = base_std * self.var_scale

        # ---- 段落：构造初始动作张量 (n_envs, planning_horizon, action_dim) ----
        # 没有外部传入的热启动动作：全部初始化为先验均值
        if actions is None:
            # 从标量形状 (1,1,-1) 广播成 (n_envs, horizon, action_dim) 再 clone 出独立副本
            actions = base_mean.view(1, 1, -1).expand(
                self._n_envs, planning_horizon, -1
            ).clone()
        # 有热启动动作（如上一次求解结果，可做时域继承）
        else:
            # 搬到计算设备
            actions = actions.to(self.device)
            # 已有动作比本次视野短：需要补齐到 planning_horizon
            if actions.shape[1] < planning_horizon:
                # 计算还缺多少步
                remaining = planning_horizon - actions.shape[1]
                # 缺的部分用先验均值填充（全零/均值尾巴）
                new_actions = base_mean.view(1, 1, -1).expand(
                    self._n_envs, remaining, -1
                ).clone()
                # 拼接到已有动作序列末尾，凑满本次视野
                actions = torch.cat([actions, new_actions], dim=1)
            # 已有动作更长：截断到前 planning_horizon 步即可
            else:
                actions = actions[:, :planning_horizon]

        # 在第 1 维插入采样维并复制 num_samples 份 → (n_envs, num_samples, horizon, action_dim)
        actions = actions.unsqueeze(1).repeat_interleave(
            self.num_samples, dim=1
        )
        # 给第 0 个采样之外的所有候选加高斯扰动（第 0 条保持"干净"作为对照）；
        # 噪声幅度 = 先验标准差，按 (1,1,1,action_dim) 广播到每个时间步
        actions[:, 1:] += (
            torch.randn(
                actions[:, 1:].shape,
                generator=self.torch_gen,
                device=self.device,
            )
            * base_std.view(1, 1, 1, -1)
        )

        # 若上次的 self.init 已存在且形状完全一致：原地拷贝覆盖（复用 Parameter 对象）
        if hasattr(self, 'init') and self.init.shape == actions.shape:
            self.init.copy_(actions)
        # 否则新建一个 nn.Parameter —— 这是关键：Parameter 默认 requires_grad=True，
        # 后续 get_cost 才能对它反向传播求出动作的梯度
        else:
            self.init = torch.nn.Parameter(actions)

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        planning_horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Solve the planning problem using gradient descent."""
        # 记录整次求解的开始时刻，用于最后打印总耗时
        start_time = time.time()
        # 只对"最近若干步"的转移计算代价（至少为 1）
        cost_last_n = max(1, int(self._config.cost_last_n))
        # 中间步代价的权重（终态代价之外的逐步代价加权和）
        intermediate_cost_weight = float(self._config.intermediate_cost_weight)
        # 未显式指定本次视野时，按递减调度表算出本次应使用的 horizon
        if planning_horizon is None:
            planning_horizon = get_planning_horizon(
                configured_horizon=self.horizon,
                # 重规划间隔 = 重视界步数 × 动作块大小（每个宏动作占 action_block 个环境步）
                replanning_interval=int(self._config.receding_horizon) * int(self._config.action_block),
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
        # 最终输出字典（actions + predictions）
        outputs = {}

        # 构造初始动作属于"准备数据"，不需要建梯度图
        with torch.no_grad():
            self.init_action(init_action, planning_horizon=planning_horizon)
        # 计算本次迭代要用的动作裁剪边界（None 表示不裁剪）
        clip_bounds = self._get_action_clip_bounds()

        # Determine batch size (default to all envs if not specified which can cause memory issues)
        # 未指定 batch_size 则一次处理所有环境（环境多时可能爆显存）
        batch_size = (
            self.batch_size if self.batch_size is not None else self.n_envs
        )
        # 环境总数
        total_envs = self.n_envs

        # Lists to hold results from each batch to be concatenated later
        # 收集每个 batch 选出的最优动作，最后拼成完整结果
        batch_top_actions_list = []

        # --- Outer Loop: Iterate over batches ---
        # 记录每个环境实际跑了几轮 GD、所在 batch 的耗时（每环境一条，便于评测统计）
        per_env_iters = []
        per_env_time = []
        # 外层循环：按 batch_size 把环境切成若干批，逐批优化（控制显存峰值）
        for start_idx in range(0, total_envs, batch_size):
            # 本批结束索引（最后一批可能不足 batch_size）
            end_idx = min(start_idx + batch_size, total_envs)
            # 本批环境数
            current_bs = end_idx - start_idx
            # 记录本批开始时刻
            batch_t0 = time.time()

            # 从全局 init 中切出本批初始动作，clone+detach 得到独立的普通张量
            # （detach 切断与全局 Parameter 的图关系，本批各自建图）
            batch_init = self.init[start_idx:end_idx].clone().detach()
            # 打开梯度开关：这才是"可优化参数"
            batch_init.requires_grad = True

            # We initialize the optimizer class passed in __init__ with the kwargs
            # 用配置的优化器类（默认 SGD）包住这个张量，后续 optim.step() 会按梯度更新它
            optim = self.optimizer_cls([batch_init], **self.optimizer_kwargs)

            # Prepare Batch Infos
            # Slice the input info_dict and then expand dimensions
            # 把 info 切片并加上采样维，与动作候选的 (bs, num_samples, ...) 对齐
            expanded_infos = self._expand_info_dict_for_samples(
                info_dict, start_idx, end_idx, num_samples=self.num_samples
            )

            # Perform Gradient Descent for this batch
            # 本批目前见过的最好（最低）代价，初始化为正无穷
            best_batch_score = float('inf')
            # 连续"无显著提升"的轮数计数（早停用）
            stalled_steps = 0

            # ---- 内层循环：GD 的一次迭代 = 算代价 → 反向传播 → 更新动作 → 加噪 → 裁剪 ----
            for step in range(self.n_steps):
                # 浅拷贝 info，避免污染原始 expanded_infos
                current_info = expanded_infos.copy()
                # 把"只看最近几步代价"的参数注入 info，传给模型的代价函数
                current_info['cost_last_n'] = cost_last_n
                # 注入中间步代价权重
                current_info['intermediate_cost_weight'] = intermediate_cost_weight

                # Calculate cost using the batch parameter
                # 核心：模型对 batch_init（当前动作）算代价 —— 图会记录从动作到代价的全部运算
                costs = self.model.get_cost(current_info, batch_init)
                for k in ('embed_0', 'goal_embed_0'):  # encode the observation and goal once per solve
                    # 第一次迭代时模型已把观测/目标编码缓存在 current_info 中；
                    # setdefault 把它们回填到 expanded_infos，之后每轮复用、只编码一次
                    expanded_infos.setdefault(k, current_info[k])

                # 断言代价必须是 torch.Tensor（否则没法反传）
                assert isinstance(costs, torch.Tensor), (
                    f'Got {type(costs)} cost, expect torch.Tensor'
                )
                # 断言形状必须是 (本批环境数, 采样条数)：每个环境的每条候选一个代价
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples
                ), (
                    f'Cost should be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                )
                # 断言代价连着计算图（requires_grad），否则 backward 会报错
                assert costs.requires_grad, (
                    'Cost must requires_grad for GD solver.'
                )

                cost = costs.sum()  # Sum cost for this batch
                # 反向传播：算出 ∂cost/∂batch_init，梯度写入 batch_init.grad
                cost.backward()
                # 优化器按梯度更新动作（SGD：动作 -= lr × 梯度）
                optim.step()
                # 清空梯度（set_to_none 更省内存），下一轮重新累积
                optim.zero_grad(set_to_none=True)

                # Add noise
                # 配置了探索噪声：直接在动作数据上加高斯扰动（相当于注入式探索）
                if self.action_noise > 0:
                    batch_init.data += (
                        # 用固定种子的生成器取噪声，形状与动作完全一致
                        torch.randn(
                            batch_init.shape,
                            generator=self.torch_gen,
                            device=batch_init.device,
                        )
                        # 乘以噪声强度
                        * self.action_noise
                    )

                # 有裁剪边界时：把动作夹在 [lower, upper] 内，防止跑出先验支持范围
                if clip_bounds is not None:
                    lower, upper = clip_bounds
                    # clamp_ 原地裁剪；边界按 (1,1,1,action_dim) 广播到 (bs, samples, horizon, dim)
                    batch_init.data.clamp_(
                        min=lower.view(1, 1, 1, -1),
                        max=upper.view(1, 1, 1, -1),
                    )

                # ---- 段落：早停判定 —— 连续 patience 轮无显著提升且已跑满 min_its 就 break ----
                if self.early_stop_patience > 0:
                    # 每个环境在 num_samples 条候选中的最低代价（取第 1 维最小值）
                    total_best_costs = costs.min(dim=1).values.detach().cpu().tolist()
                    # 对本批所有环境求平均，得到一个标量"本批得分"（越低越好）
                    batch_score = float(np.mean(total_best_costs))
                    # 判断相对历史最好成绩是否有显著改善
                    if self._has_significant_improvement(
                        batch_score,
                        best_batch_score,
                        self.early_stop_rel_delta,
                    ):
                        # 有改善：刷新最好成绩，耐心计数清零
                        best_batch_score = batch_score
                        stalled_steps = 0
                    # 没有显著改善
                    else:
                        # 连续停滞轮数 +1
                        stalled_steps += 1
                        # 只有跑满最少迭代次数、且连续停滞达到耐心上限，才真正提前终止
                        if (
                            step + 1 >= self.min_its
                            and stalled_steps >= self.early_stop_patience
                        ):
                            # 跳出 GD 循环
                            break

            iters_run = step + 1  # actual GD iterations for this batch (early-stop aware)

            # Update the global self.init with the optimized batch values
            with torch.no_grad():
                # 把本批优化后的动作写回全局 init，供后续/下次求解参考
                self.init[start_idx:end_idx] = batch_init

            with torch.no_grad():
                # 最后用优化完的动作再算一次精确代价（不建图）
                final_info = expanded_infos.copy()
                # 同样注入最近步数与中间代价权重
                final_info['cost_last_n'] = cost_last_n
                final_info['intermediate_cost_weight'] = intermediate_cost_weight
                # 得到最终的 (bs, num_samples) 代价矩阵
                final_costs = self.model.get_cost(final_info, batch_init)

            # 对每个环境的 num_samples 条候选按代价升序排序，取第 0 个 = 最优候选索引
            top_idx = torch.argsort(final_costs, dim=1)[:, 0]
            # 生成每个环境在本批内的行号 0..current_bs-1，用于成对索引
            batch_indices = torch.arange(current_bs, device=batch_init.device)

            # 成对索引 (行=环境, 列=该环境最优候选) → 取出每个环境的最优动作序列
            top_actions_batch = batch_init[batch_indices, top_idx]
            # 搬到 CPU 存起来，避免占用显存
            batch_top_actions_list.append(top_actions_batch.detach().cpu())

            # 本批每个环境都记录相同的迭代轮数与批耗时（batch_size=1 时即逐 episode 记录）
            per_env_iters.extend([int(iters_run)] * current_bs)
            per_env_time.extend([time.time() - batch_t0] * current_bs)

        # Concatenate all batch results
        # 沿第 0 维拼接各批结果 → (n_envs, horizon, action_dim) 的最终动作
        outputs['actions'] = torch.cat(batch_top_actions_list, dim=0)
        # 用最优动作做一次完整前向推演，拿到预测结果（预测观测/隐状态等）
        outputs['predictions'] = self._rollout_final_actions(
            info_dict, outputs['actions']
        )
        end_time = time.time()
        # 打印整次 solve 的墙钟耗时
        print(
            f'GradientSolver.solve completed in {end_time - start_time:.4f} seconds.'
        )

        # 记录本次求解的计算元数据，供评测统计规划器迭代量与耗时（FLOPs 估算用）
        self.solve_records.append({
            'level': int(getattr(self, 'level', 1)),
            'horizon': int(planning_horizon),
            'num_samples': int(self.num_samples),
            'n_iters': per_env_iters,          # one entry per env (batch_size=1 -> per episode)
            'wall_clock': end_time - start_time,
            'wall_clock_per_env': per_env_time,
        })

        # 返回动作与预测：调用方（控制循环）取 actions 的第 0 步执行（receding horizon 控制）
        return outputs
