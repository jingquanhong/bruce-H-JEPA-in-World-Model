# 导入标准库 math：本文件用 math.ceil 做向上取整（计算规划调用次数）
import math
# 从 typing 导入类型工具：Any（任意类型）、Protocol（结构化类型协议）、runtime_checkable（让协议支持运行时 isinstance 检查）
from typing import Any, Protocol, runtime_checkable

# 导入 Gymnasium：动作空间 gym.Space 是 Solver.configure 的入参之一
import gymnasium as gym
# 导入 PyTorch：动作序列与代价都是 torch.Tensor
import torch


# ---------------------------------------------------------------------------
# Costable 协议：规定"世界模型"必须提供的代价计算接口
# （Protocol = 鸭子类型：模型无需显式继承，只要实现这两个方法就算符合）
# ---------------------------------------------------------------------------
class Costable(Protocol):
    """Protocol for world model cost functions."""

    # criterion：输入环境状态信息字典 + 一批候选动作张量，输出每个候选动作的代价值
    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute the cost criterion for action candidates.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        # 协议方法只有 docstring，没有实现体（... 表示由具体模型提供）
        ...

    # get_cost：与 criterion 同型的接口，GradientSolver 迭代中实际调用的就是它
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        # 协议占位体，具体实现在世界模型类中
        ...


# ---------------------------------------------------------------------------
# 段落标题：重规划（receding horizon）调度工具函数
# 评估总步数 eval_budget、每隔 replanning_interval 步重新规划一次，
# 据此推算"一共能调用多少次求解器""当前是第几次调用""每次调用看多远的视野"
# ---------------------------------------------------------------------------
def num_planning_calls(*, eval_budget: int, replanning_interval: int) -> int:
    """Return how many solver calls fit in the evaluation budget."""
    # 总步数必须为正，否则没有可规划的余地
    if eval_budget <= 0:
        raise ValueError(f'Eval budget must be positive, got {eval_budget}.')
    # 重规划间隔也必须为正，否则一次都没法规划
    if replanning_interval <= 0:
        raise ValueError(
            f'Replanning interval must be positive, got {replanning_interval}.'
        )
    # 调用次数 = 总步数 ÷ 重规划间隔，向上取整（最后一次不足一个间隔也算一次规划）
    return math.ceil(int(eval_budget) / int(replanning_interval))


def planning_call_index(
    *,
    replanning_interval: int,
    steps_taken: int | None = None,
) -> int:
    """Return the current zero-based planning call index."""
    # 间隔必须为正
    if replanning_interval <= 0:
        raise ValueError(
            f'Replanning interval must be positive, got {replanning_interval}.'
        )
    # 还没走任何步 → 第 0 次规划调用；否则用整除算出"当前是第几次"（0 基）
    return 0 if steps_taken is None else int(steps_taken) // int(replanning_interval)


def decreasing_horizon_schedule(
    *,
    num_calls: int,
    configured_horizon: int,
) -> list[int]:
    """Build a monotone decreasing per-call horizon schedule.

    When there are enough calls to represent every horizon value, each value
    greater than 1 receives the same number of calls and horizon 1 receives any
    remainder. If there are fewer calls than horizon values, keep the configured
    horizon as the first call and horizon 1 as the final call.
    """
    # 统一转成 int，防止传入 numpy 整型等
    num_calls = int(num_calls)
    configured_horizon = int(configured_horizon)
    # 调用次数必须为正
    if num_calls <= 0:
        raise ValueError(
            f'Number of planning calls must be positive, got {num_calls}.'
        )
    # 规划视野（horizon）必须为正
    if configured_horizon <= 0:
        raise ValueError(
            f'Planning horizon must be positive, got {configured_horizon}.'
        )
    # 视野恒为 1：每次规划都只看 1 步，整个调度就是 num_calls 个 1
    if configured_horizon == 1:
        return [1] * num_calls
    # 只有一次规划机会：直接用配置的完整视野
    if num_calls == 1:
        return [configured_horizon]
    # 调用次数 < 视野取值个数（没法逐级递减）：用线性插值在 [configured, 1] 之间均匀取样
    if num_calls < configured_horizon:
        # 视野总共要跨过的幅度：从 configured_horizon 降到 1，跨度 = configured_horizon - 1
        span = configured_horizon - 1
        # 插值分母 = 调用次数 - 1（首尾两次调用分别对应最大视野和最小视野）
        denom = num_calls - 1
        # 从 idx=0（最大视野）到 idx=num_calls-1（=1）线性递减，ceil 保证单调且不低于 1
        return [
            max(1, math.ceil(1 + span * (num_calls - 1 - idx) / denom))
            for idx in range(num_calls)
        ]

    # 调用次数 ≥ 视野取值个数：先构造完整递减序列 [H, H-1, ..., 2, 1]
    horizon_values = list(range(configured_horizon, 0, -1))
    # 每个视野值平均能分到的调用次数（整除）
    calls_per_horizon = num_calls // len(horizon_values)
    # 除不尽的余数个调用
    extra_calls = num_calls % len(horizon_values)
    # 先给每个视野值分配基础次数
    counts = [calls_per_horizon] * len(horizon_values)
    # 把余数个额外调用加给序列末尾的几个视野值（较小的视野多分 1 次）
    for idx in range(len(horizon_values) - extra_calls, len(horizon_values)):
        counts[idx] += 1

    # 按分配次数展开成最终的逐次调度表
    schedule = []
    # 严格一一配对：每个视野值对应它的调用次数
    for horizon, count in zip(horizon_values, counts, strict=True):
        # 重复 horizon 共 count 次追加进调度表
        schedule.extend([horizon] * count)
    # 返回如 [5,5,4,4,3,3,2,2,1,1] 的单调不增序列
    return schedule


def get_planning_horizon(
    *,
    configured_horizon: int,
    replanning_interval: int,
    steps_taken: int | None = None,
    eval_budget: int,
) -> int:
    """Return this call's horizon from the decreasing schedule over the eval budget."""
    # 先算总共有多少次规划调用（eval_budget / replanning_interval 向上取整）
    schedule = decreasing_horizon_schedule(
        num_calls=num_planning_calls(
            eval_budget=int(eval_budget),
            replanning_interval=int(replanning_interval),
        ),
        configured_horizon=int(configured_horizon),
    )
    # 再算当前是第几次调用（已走步数 // 重规划间隔）
    call_idx = planning_call_index(
        replanning_interval=replanning_interval,
        steps_taken=steps_taken,
    )
    # 索引越界说明调度表比预期短（步数超出预算），直接报错
    if call_idx >= len(schedule):
        raise ValueError(
            'Planning call index exceeds resolved horizon schedule: '
            f'got call {call_idx} for schedule length {len(schedule)}.'
        )
    # 从调度表中取出本次规划应使用的视野长度
    return schedule[call_idx]


# ---------------------------------------------------------------------------
# Solver 协议：所有规划求解器（如 GradientSolver）需要对外暴露的统一接口
# runtime_checkable 使其可用 isinstance(x, Solver) 做运行时检查
# ---------------------------------------------------------------------------
@runtime_checkable
class Solver(Protocol):
    """Protocol for model-based planning solvers."""

    # configure：用环境信息（动作空间、并行环境数、规划配置）初始化求解器
    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment and planning specifications.

        Args:
            action_space: The action space of the environment.
            n_envs: Number of parallel environments.
            config: Planning configuration object.
        """
        # 协议占位体，具体实现在各求解器中
        ...

    # 属性 action_dim：展平后的动作维度（含 action_block 分组后的总维度）
    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        ...

    # 属性 n_envs：需要并行规划的环境数量（批量维大小）
    @property
    def n_envs(self) -> int:
        """Number of parallel environments being planned for."""
        ...

    # 属性 horizon：规划视野长度（要一次优化多少个时间步的动作）
    @property
    def horizon(self) -> int:
        """Planning horizon length in timesteps."""
        ...

    # solve：核心方法——求解规划问题，返回优化后的动作序列
    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Solve the planning optimization problem to find optimal actions.

        Args:
            info_dict: Dictionary containing environment state information.
            init_action: Optional initial action sequence to warm-start the solver.
            steps_taken: Number of environment steps already consumed.
            eval_budget: Total environment-step budget for evaluation.

        Returns:
            Dictionary containing optimized actions and other solver-specific info.
        """
        # 协议占位体：调用方（如评测循环）只依赖这个签名，不关心内部实现
        ...
