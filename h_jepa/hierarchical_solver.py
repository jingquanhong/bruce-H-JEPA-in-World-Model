
"""Hierarchical planner: one gradient solver per level of a hierarchical world model, planned top-down."""

# ==== 模块总览：分层（层级式）世界模型的规划器，自顶向下规划 ====
# H-JEPA 世界模型由多个 level 组成：level1 最细（直接输出真实环境动作），
# level 越高时间步越粗 —— 高层的一步预测对应低层更多的帧（宏观动作）。
# 规划策略是自顶向下：
#   1) 最高层先在自己的隐空间里用梯度下降（GD）规划一条子目标（waypoint）隐状态序列；
#   2) 高层预测出的第 1 个隐状态，作为低一层的"目标"逐层下发；
#   3) 一直传到 level1，由 level1 解出可执行的真实动作序列交给环境执行。
# 本文件三大组成部分：
#   build_hierarchical_solver —— 工厂函数：给每一层配一个梯度求解器；
#   HierarchicalSolver        —— 顶层编排器：观测/goal 逐层编码、子目标下发、地平线调度；
#   LevelCostModel            —— 单层代价模型：在本层空间 rollout，再把预测投影到目标所在层算 cost。

# ---- 标准库导入 ----
# Mapping：键值映射的类型标注接口（dict 也满足），用于 level_solvers 参数类型
from collections.abc import Mapping
# Any：泛型类型标注，表示"任意类型"，仅给类型检查器看，不影响运行
from typing import Any

# ---- 第三方库导入 ----
# gymnasium：强化学习环境标准库（Gym 的继任者），这里主要用 gym.Space 表示动作空间
import gymnasium as gym
# hydra：配置驱动的工具库；hydra.utils.instantiate 能按 yaml 里的 _target_ 字段动态构建对象
import hydra
# numpy：数值计算库，用于对动作空间 shape 求积得到动作维度等
import numpy as np
# torch：张量与自动微分框架，规划中的梯度下降优化全部依赖它
import torch
# torch.nn.functional：函数式算子集合，本文件用其中的 F.mse_loss 计算动作代价
import torch.nn.functional as F
# torch.nn：神经网络基类模块，下面的 LevelCostModel 继承 nn.Module
from torch import nn

# ---- 从本仓库 stable_worldmodel 导入的三个"规划调度"辅助函数 ----
# num_planning_calls(eval_budget, replanning_interval)：整段评测一共能做几次规划 = ceil(总步数预算 / 重规划间隔)
# planning_call_index(replanning_interval, steps_taken)：由已走步数算出"当前是第几次规划调用"（0 基）
# decreasing_horizon_schedule(num_calls, configured_horizon)：生成一条单调递减的地平线序列（先长后短，末尾降到 1）
from stable_worldmodel.solver.solver import (
    num_planning_calls,
    planning_call_index,
    decreasing_horizon_schedule,
)


# ==== 工厂函数：为分层模型的每一层分别构建一个梯度求解器 ====
def build_hierarchical_solver(cfg: Any, model: Any) -> "HierarchicalSolver":
    """Build one GradientSolver per level from cfg.solver.solvers.levelN."""
    # 按 1..num_levels 依次取出每层的子世界模型；list 索引 0 对应 level1
    level_models = [model.get_level(level) for level in range(1, model.num_levels + 1)]
    # 用来收集 {层号(1 基): 该层 solver 实例} 的字典
    level_solvers = {}
    # enumerate(..., start=1) 让 level 从 1 编号；level_models[level:] 是"本层之上的全部更高层模型"
    for level, level_model in enumerate(level_models, start=1):
        # 用 LevelCostModel 包装本层模型：rollout 仍在本层空间进行，
        # 但代价会按需投影到"目标所在层"的隐空间再计算
        adapted_model = LevelCostModel(
            model=level_model,
            upper_models=level_models[level:],
        )
        # hydra 按 cfg.solver.solvers['levelN'] 中的 _target_ 实例化该层的梯度求解器（GradientSolver），
        # 并把上面包装好的代价模型注入为它的 model 参数
        level_solvers[level] = hydra.utils.instantiate(
            cfg.solver.solvers[f'level{level}'], model=adapted_model
        )

    # 把所有层的 solver 与模型交给顶层编排器
    return HierarchicalSolver(level_solvers=level_solvers, level_models=level_models)


# ==== 类：HierarchicalSolver —— 分层规划的顶层编排器 ====
class HierarchicalSolver:
    """Hierarchical wrapper that orchestrates one child solver per model level.

    Args:
        level_solvers: Mapping from 1-based level index to initialized solver.
        level_models: Per-level JEPA models of the hierarchical model, level 1 first.
    """

    def __init__(
        self,
        level_solvers: Mapping[int, Any],
        level_models: list[Any],
    ) -> None:
        # 把外部映射固化成普通 dict：键是 1 基层号，值是该层的子 solver
        self.level_solvers = dict(level_solvers)
        # 各层 JEPA 子模型组成的 list（索引 0 = level1），编码观测/goal 时逐层使用
        self.level_models = level_models

        # 下面三个是延迟初始化的运行时状态，等 configure() 被调用后才真正赋值
        # 并行环境个数（配置前为 None）
        self._n_envs: int | None = None
        # 基础动作维度（配置前为 None；不含 action_block 展开）
        self._action_dim: int | None = None
        # 整份分层规划配置（含 receding_horizon、level1/level2/... 子配置）
        self._config: Any = None

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure all child solvers with shared environment settings."""
        # 记录并行环境数与分层规划配置，供后续属性与调度函数使用
        self._n_envs = n_envs
        self._config = config
        # 动作维度 = 动作空间 shape 除掉最前面的批维后求乘积（例如 Box(2,) 得 2），对任意动作形状通用
        self._action_dim = int(np.prod(action_space.shape[1:]))

        # 按层号升序逐层配置子 solver，让它们共享同一个动作空间与环境数
        for level in sorted(self.level_solvers):
            # 传入该层专属的规划配置（cfg.hierarchical_plan_config.levelN）和层号，
            # 子 solver 据此知道自己的 horizon/action_block/action_dim 等
            self.level_solvers[level].configure(
                action_space=action_space,
                n_envs=n_envs,
                config=self._level_plan_config(level),
                level=level,
            )

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        # 并行环境个数（在 evaluate 时用于把数据切给每个 env）
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        # 展平后的动作维度 = 单步动作维度 × level1 的 action_block（一次规划输出 action_block 步动作）
        return self._action_dim * self._level_plan_config(1).action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        # 对外公布的规划地平线取最底层 level1 配置的 horizon
        return self._level_plan_config(1).horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        # 让实例可像函数一样调用，直接转发到 solve()
        return self.solve(*args, **kwargs)

    def solve(
        self,
        info_dict: dict,
        steps_taken: int,
        eval_budget: int,
    ) -> dict:
        """Plan top-down: each level's predicted latents become the goals of the level below.

        A level with planning horizon 1 (other than level 1) is skipped. Each planned level then
        takes its goal from one of two sources:
          - no plan from the level above (top level, or every level above was skipped): the real
            goal, measured at this level or, through skipped levels with
            ``horizon_one_goal_cost_space='upper'``, at the highest of them;
          - the level above planned: its first predicted latent (and macro-action), measured in
            the upper level's space.
        """
        # ==== 段落 1：初始化"由上层下发给本层"的子目标状态 ====
        # 上一层规划出的隐状态序列 [B, T_sub, D]；None 表示上层没有规划结果（本层要用真实 goal）
        subgoal_latents = None
        # 上一层规划出的宏动作序列，用作低层的动作目标（action_cost_weight > 0 时生效）
        subgoal_actions = None
        # 若上层因 horizon=1 被跳过：记录"真实 goal 应取自哪一层的编码"；None 表示没有跳过层
        skipped_goal_level = None
        # Tracks whether the level above was anchored to the true goal at its last step.
        # True for the top level (trivially), and propagates down only when the substitution
        # is performed — so lower levels only anchor when the entire chain above is consistent.
        # 中文补充：upper_anchored 表示"上一层的最后一个预测是否被替换成了真实 goal"。
        # 顶层默认 True；只有发生替换时才逐层向下传递，因此低层只有在整条上层链都锚定
        # 在真实 goal 上时才允许自己也做替换（保证优化目标语义一致）。
        upper_anchored = True
        # 生成每一层在"本次评测各次规划调用"中应使用的地平线调度表 {层号: [horizon1, horizon2, ...]}
        horizon_schedules = self._horizon_schedules(eval_budget=eval_budget)
        # 当前是第几次规划调用 = 已走环境步数 // 重规划间隔（0 基），用于查调度表
        call_idx = planning_call_index(
            replanning_interval=self._replanning_interval(),
            steps_taken=steps_taken,
        )
        # 把当前观测图像 pixels 逐层编码，得到 {embed_1, embed_2, ...}（每层只留最后一帧）
        obs_embeddings = self._encode_all_levels(info_dict=info_dict, key='pixels')
        # 同样把真实目标图像 goal 逐层编码，得到各层的目标隐状态
        goal_embeddings = self._encode_all_levels(info_dict=info_dict, key='goal')

        # ==== 段落 2：自顶向下逐层规划（从最高层遍历到 level1） ====
        # range(层数, 0, -1) -> [N, N-1, ..., 1]，高层先规划，其预测成为低层的目标
        for level in range(len(self.level_models), 0, -1):
            # 取该层的规划配置（horizon、num_subgoals、action_cost_weight、
            # horizon_one_goal_cost_space 等）
            level_cfg = self._level_plan_config(level)
            # 该层在"第 call_idx 次规划"时使用的地平线长度（随评测进程递减）
            planning_horizon = horizon_schedules[level][call_idx]

            if level > 1 and planning_horizon == 1:
                # A one-step upper plan produces no useful subgoal for the lower
                # level, so skip that solve and pass through an actual goal.
                # 'upper' keeps the goal of the highest skipped level so a chain of
                # skipped levels projects the lower plan all the way up to it.
                # 中文补充：非底层却只规划 1 步时，其预测对低层毫无信息量，
                # 于是跳过该层的 solve，把"真实 goal"直接透传下去；
                # 若配置 horizon_one_goal_cost_space='upper'，则记住这条跳过链中
                # 最高那一层的层号，让低层的代价统一算在那个高层空间里。
                skipped_goal_level = (
                    (skipped_goal_level or level)
                    if level_cfg.horizon_one_goal_cost_space == 'upper' else None
                )
                # 清空上层子目标：表示"上层没有可用的规划结果"
                subgoal_latents = None
                subgoal_actions = None
                # 结束本层迭代，继续向更低层遍历
                continue

            # 浅拷贝 info_dict（不动调用方的数据），往里注入本层专用字段
            level_info = dict(info_dict)
            # 本层当前观测的隐状态 embed_0（来自上面的逐层编码，形状 [B, 1, D]）
            level_info['embed_0'] = obs_embeddings[f'embed_{level}']
            # 本层动作代价权重：>0 时把"预测动作 vs 动作目标"的误差加进总代价
            level_info['action_cost_weight'] = level_cfg.action_cost_weight

            if subgoal_latents is None:
                # No plan from the level above: this is the top level, or every level above
                # was skipped (horizon 1). Aim at the real goal. It is measured at this level,
                # unless skipped levels with horizon_one_goal_cost_space='upper' passed down a
                # higher level; then every predicted step is encoded up to that level
                # (stride 1) and compared with the real goal there. This is also how the
                # _project configs measure level-1 cost in a level-k latent.
                # 中文补充：没有上层子目标 —— 说明本层是顶层，或上面所有层都因 horizon=1
                # 被跳过。此时目标就是"真实 goal"：默认在本层空间比较；若有跳过链传下来的
                # goal_level（更高层），本层预测会以 stride=1 逐帧向上编码到那一层再比较。
                # goal 层号（没有跳过链时就等于本层）
                goal_level = skipped_goal_level or level
                # 真实 goal 在 goal_level 层的隐状态 [B, 1, D]
                level_info['goal_embed_0'] = goal_embeddings[f'embed_{goal_level}']
                # 记录 goal 所在层号，供 LevelCostModel 决定是否向上投影
                level_info['goal_embed_0_level'] = goal_level
                # True 表示 goal 不在本层空间，算代价前要先向上投影
                level_info['goal_embed_0_in_upper_space'] = goal_level > level
                # True 表示"稠密投影"：stride=1，把预测每一帧都编码上去逐一打分
                level_info['dense_upper_goal_projection'] = goal_level > level
                # 本层的目标直接锚定在真实 goal 上
                this_level_anchored = True
            else:
                # The level above planned: follow its plan. The goal is its first predicted
                # latent (the next waypoint), compared in the upper level's latent space after
                # encoding this level's predictions at the upper level's stride. The upper
                # plan's first macro-action is also a target (used when action_cost_weight > 0).
                # 中文补充：上层规划过 —— 本层改跟上层的计划：目标 = 上层预测序列的
                # 第 1 个隐状态（下一个路点），在上层隐空间里比较；上层的第 1 个宏动作
                # 同时作为动作目标（action_cost_weight>0 时使用）。
                # 上层每规划一步，对应本层 num_subgoals 个子目标帧
                num_subgoals = self._level_plan_config(level + 1).num_subgoals
                # 上层预测序列若带初始上下文帧（索引 0），第 1 个真正的预测帧在索引 1；否则在 0
                first_pred_idx = 1 if subgoal_latents.shape[1] > 1 else 0
                # 动作比状态早一步：取索引 first_pred_idx-1 的动作，最小钳到 0
                first_action_idx = max(0, first_pred_idx - 1)
                # 判断取窗口是否一直延伸到上层预测序列的末尾（用于后面的真实 goal 替换）
                includes_last = (first_pred_idx + num_subgoals >= subgoal_latents.shape[1])
                # 切出 [first_pred_idx, +num_subgoals) 的隐状态作为本层的子目标（clone 拿独立内存）
                next_goal = subgoal_latents[:, first_pred_idx:first_pred_idx + num_subgoals].clone()
                # 对应地切出同样长度的动作序列作为动作目标，与状态子目标一一对应
                next_action = subgoal_actions[
                    :,
                    first_action_idx:first_action_idx + num_subgoals,
                ].clone()
                # 状态子目标数与动作子目标数必须一致，否则后续代价按时间对不齐
                if next_action.shape[1] != next_goal.shape[1]:
                    raise ValueError(
                        f"Level {level}: got {next_action.shape[1]} action "
                        f"subgoals for {next_goal.shape[1]} state subgoals."
                    )
                # 默认本层不算"锚定"在真实 goal 上（除非下面发生替换）
                this_level_anchored = False
                if includes_last and upper_anchored:
                    # The upper planner's last prediction should converge to its goal,
                    # so replace it with the ground truth goal embedding directly.
                    # Only valid if the entire chain above is anchored — otherwise the
                    # upper level was not optimizing toward goal_embeddings[level+1].
                    # 中文补充：上层规划的最后一步本就应该收敛到它的真实 goal，
                    # 所以直接用 level+1 层的真实 goal 隐状态替换掉最后一个子目标，
                    # 避免低层去追一个"上层没追准"的终点。仅当上面整条链都锚定时合法。
                    next_goal[:, -1] = goal_embeddings[f'embed_{level + 1}'].squeeze(1)
                    # 替换之后，本层也被认为锚定在真实 goal 上
                    this_level_anchored = True
                # 把上层路点隐状态设为本层要比较的目标
                level_info['goal_embed_0'] = next_goal
                # 把上层宏动作设为本层的动作目标（action_cost_weight>0 时参与代价）
                level_info['goal_action'] = next_action
                # 标记目标位于"上层空间"，算代价前必须把本层预测向上投影
                level_info['goal_embed_0_in_upper_space'] = True

            # num_subgoals, upper JEPA stride, and planning_horizon are coupled:
            # after stride-subsampling the level-N rollout (T+1 frames → ceil((T+1)/stride))
            # and dropping the context frame, exactly ceil((T+1)/stride)-1 frames remain to
            # compare against the subgoals. If num_subgoals doesn't match, criterion's
            # expand_as either crashes or — if cost_last_n accidentally equals num_subgoals —
            # silently compares the wrong temporal positions.
            # 中文补充：子目标个数与"上层 stride + 规划地平线"强耦合 ——
            # 本层 rollout 出 T+1 帧，按 stride 抽帧后剩 ceil((T+1)/stride) 帧，
            # 去掉上下文帧正好是 ceil((T+1)/stride)-1 帧要和子目标逐一比较。
            # 下面的断言在运行早期就验证这个等式，防止悄悄错位比较。
            # 取出刚注入的目标（可能是真实 goal，也可能是上层子目标）
            goal_emb = level_info.get('goal_embed_0')
            if (
                level_info.get('goal_embed_0_in_upper_space', False)
                and torch.is_tensor(goal_emb)
                and goal_emb.shape[-2] > 1
            ):
                # 直接上一层的 JEPA 模型的时间步长（本层多少帧 = 上层 1 帧）
                _stride = self.level_solvers[level].model.upper_models[0].temporal_stride
                # 按公式算出期望的子目标帧数：(horizon + stride) // stride - 1
                _expected = (planning_horizon + _stride) // _stride - 1
                # 断言实际子目标数 == 期望值，不匹配立即报错（避免静默错位）
                assert goal_emb.shape[-2] == _expected, (
                    f'Level {level}: {goal_emb.shape[-2]} subgoals but upper temporal_stride={_stride} '
                    f'with planning_horizon={planning_horizon} expects {_expected}'
                )

            # ==== 段落 3：调用本层求解器，做该层的梯度下降规划 ====
            # 输入：本层观测 embed_0、目标 goal_embed_0、地平线；输出：动作与预测序列
            result = self.level_solvers[level].solve(
                info_dict=level_info,
                planning_horizon=planning_horizon,
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
            # 本层的锚定状态变成"下一层判断 upper_anchored"时看到的上层状态（逐层下传）
            upper_anchored = this_level_anchored
            # 本层正常规划过，跳过层记录清零（跳过链到此中断）
            skipped_goal_level = None
            # 本层预测出的隐状态序列，成为下一层规划时的子目标
            subgoal_latents = result['predictions']['predicted_embed_0']
            # 本层规划出的动作序列，成为下一层的动作目标
            subgoal_actions = result['actions']

        # 循环最后必然执行到 level1（level1 永不被跳过），其 actions 即真实环境动作
        return {'actions': result['actions']}

    def _level_plan_config(self, level: int) -> Any:
        """Return planning config for a specific hierarchy level."""
        # 用属性名动态取配置：cfg.level1 / cfg.level2 / ...
        return getattr(self._config, f'level{level}')

    def _replanning_interval(self) -> int:
        """Return the environment-step interval between solver calls."""
        # 两次规划之间隔多少个环境步 = 每次重规划执行的规划步数 receding_horizon
        # × 每个规划步打包的底层动作数 action_block
        return int(self._config.receding_horizon) * int(
            self._level_plan_config(1).action_block
        )

    def _pre_decay_horizon(self, level: int, *, fallback: int) -> int:
        """衰减开始前该层使用的地平线（配置缺省时回退到本层 stride）。"""
        # 取该层配置对象
        level_cfg = self._level_plan_config(level)
        # 优先用配置里显式写的 pre_decay_horizon
        configured = getattr(level_cfg, 'pre_decay_horizon', None)
        if configured is None:
            # 没写就退而取该层模型的时间步长 temporal_stride，再不行用 fallback
            configured = getattr(self.level_models[level], 'temporal_stride', fallback)

        # 转成整数地平线
        horizon = int(configured)
        if horizon <= 0:
            # 地平线必须为正，否则后面的调度/rollout 都没有意义
            raise ValueError(
                f'pre_decay_horizon must be positive for level {level}, got {horizon}.'
            )
        return horizon

    def _horizon_schedules(
        self,
        *,
        eval_budget: int,
    ) -> dict[int, list[int]]:
        """Build recursive per-level horizon schedules for one evaluation."""
        # 本次评测的总规划调用次数 = ceil(eval_budget / 重规划间隔)
        calls = num_planning_calls(
            eval_budget=int(eval_budget),
            replanning_interval=self._replanning_interval(),
        )
        # 结果表：{层号: 长度为 calls 的地平线列表}
        schedules: dict[int, list[int]] = {}
        # 最高层层号（例如 3 层就是 3）
        highest_level = len(self.level_models)

        # 从最高层往下逐层构造调度（低层调度依赖高层调度，所以必须倒序）
        for level in range(highest_level, 0, -1):
            # 该层配置里的目标 horizon
            level_cfg = self._level_plan_config(level)
            horizon = level_cfg.horizon
            if level == highest_level:
                # 最高层没有"更上层"参照，直接用通用的递减地平线序列
                schedules[level] = decreasing_horizon_schedule(
                    num_calls=calls,
                    configured_horizon=horizon,
                )
                # 进入下一层
                continue

            # 上一层（更高层）已经生成好的调度表
            upper_schedule = schedules[level + 1]
            # 本层在"上层还没开始收尾"期间保持的较长地平线
            pre_decay_horizon = self._pre_decay_horizon(level, fallback=horizon)
            try:
                # 上层调度里第一次出现地平线 1 的位置（上层开始收尾的那次规划）
                tail_start = upper_schedule.index(1)
            except ValueError:
                # 上层整个评测都不降到 1：本层全程保持 pre_decay_horizon
                schedules[level] = [pre_decay_horizon] * calls
                continue

            # 从上层收尾点开始，本层也进入递减阶段，生成剩余 calls-tail_start 次的地平线
            tail = decreasing_horizon_schedule(
                num_calls=calls - tail_start,
                configured_horizon=horizon,
            )
            # 拼接：前段保持 pre_decay_horizon，后段跟上层同步递减
            schedules[level] = [pre_decay_horizon] * tail_start + tail

        return schedules

    def _encode_all_levels(
        self,
        info_dict: dict,
        key: str = 'pixels',
    ) -> dict[str, torch.Tensor]:
        """Encode one pixel-like stream through all hierarchy levels.

        Returns a dict with HJEPA-style keys: embed_1..embed_N.
        Actions are intentionally ignored.
        """
        # 先按 level1 的输入约定取出该流（key='pixels' 或 'goal'）打包成 {'pixels': ...}
        encoded = self._level1_encode_info(info_dict, key)
        # 缺失该键直接报错，避免后面静默地用 None 规划
        if encoded.get('pixels') is None:
            raise KeyError(f"Missing '{key}' key in info_dict for hierarchical planning.")

        # 取出像素/图像数据
        goal_pixels = encoded['pixels']
        # 若是 numpy 等非张量数据，先转成 torch 张量
        if not torch.is_tensor(goal_pixels):
            goal_pixels = torch.as_tensor(goal_pixels)

        # 找到模型所在设备（取第一个参数的 device，通常是 cuda）
        model_device = next(self.level_models[0].parameters()).device
        # 把图像搬到模型设备上，保证后续编码不报设备不匹配错误
        goal_pixels = goal_pixels.to(model_device)

        # Match JEPA get_cost behavior if a sample axis is present.
        # 中文：若张量带"采样"轴（维度 >= 6，形如 [B, S, T, C, H, W]），只取第 0 个样本，
        # 与 JEPA 里 get_cost 的行为保持一致。
        if goal_pixels.ndim >= 6:
            goal_pixels = goal_pixels[:, 0]

        # 写回处理好的像素数据
        encoded['pixels'] = goal_pixels
        # 其余键（如 proprio 本体感知）做同样的搬设备/去采样轴处理
        for info_key, value in list(encoded.items()):
            # 像素上面已单独处理过，跳过
            if info_key == 'pixels':
                continue
            # 非张量先转张量
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            # 搬到模型设备
            value = value.to(model_device)
            # 维度 >= 4 视为带采样轴，同样只取第 0 个样本
            if value.ndim >= 4:
                value = value[:, 0]
            # 写回字典
            encoded[info_key] = value

        # 存放最终输出：{embed_1: ..., ..., embed_N: ...}
        outputs: dict[str, torch.Tensor] = {}

        # These observation/goal embeddings are fixed conditioning inputs for planning.
        # Detach them so repeated GD steps do not try to backprop through the same
        # encoder graph multiple times.
        # 中文：观测/goal 编码是规划的固定条件输入，规划时的梯度只对候选动作求，
        # 不应该穿回编码器 —— 所以整段编码放在 no_grad 下并 detach。
        with torch.no_grad():
            # 从 level1 到最高层依次编码
            for level in range(1, len(self.level_models) + 1):
                # 取出该层 JEPA 模型
                jepa = self.level_models[level - 1]
                # 该层的输入键：level1 用 'pixels'，更高层用上一层的 'embed_{level-1}'
                input_key = self._level_input_key(level)
                if input_key not in encoded:
                    # 输入缺失说明 info_dict 不完整，直接报错
                    raise KeyError(
                        f"Missing '{input_key}' for level {level} encoding."
                    )

                # 更高层的输入是"上一层隐状态序列"，需要截取最近的上下文窗口
                if level > 1:
                    encoded[input_key] = self._latest_context_window(
                        encoded[input_key],
                        jepa.temporal_window_size,
                    )

                # 调该层 JEPA 的编码器：得到本层的 embed_0 隐状态序列
                # chunk_temporal_inputs=True 表示把时间维按块送入（高层用）
                level_out = jepa.encode(
                    encoded,
                    key=input_key,
                    chunk_temporal_inputs=level > 1,
                )

                # 把本层输出写回 encoded，供"下一层"继续当输入使用
                encoded[f'embed_{level}'] = level_out['embed_0']
                # 输出只保留最后一帧 [:, -1:] 并 detach：规划只需要"当前时刻"的表征
                outputs[f'embed_{level}'] = level_out['embed_0'][:, -1:].detach()

        return outputs

    def _level1_encode_info(self, info_dict: dict, key: str) -> dict[str, Any]:
        """把 info_dict 中 key 对应的流取成 level1 编码器所需的 {'pixels': ..., 'proprio': ...}。"""
        if key == 'goal':
            # goal 流：像素取 info_dict['goal']
            encoded = {'pixels': info_dict.get('goal')}
            # 若数据集提供了目标时刻的本体感知，一并带上
            if 'goal_proprio' in info_dict:
                encoded['proprio'] = info_dict['goal_proprio']
        else:
            # 观测流：按 key（如 'pixels'）取
            encoded = {'pixels': info_dict.get(key)}
            # 同样带上当前时刻的本体感知
            if 'proprio' in info_dict:
                encoded['proprio'] = info_dict['proprio']

        return encoded

    @staticmethod
    def _level_input_key(level: int) -> str:
        """level1 吃原始 'pixels'，level>1 吃下一层的隐状态 'embed_{level-1}'。"""
        return 'pixels' if level == 1 else f'embed_{level - 1}'

    @staticmethod
    def _latest_context_window(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """取最近 window_size 帧；不够时用首帧在时间维左侧复制补齐。"""
        # 序列已经够长：直接切最后 window_size 帧
        if x.shape[1] >= window_size:
            return x[:, -window_size:]

        # 不够长：把首帧复制 expand 出 (window_size - 当前长度) 帧作为左前缀
        prefix = x[:, :1].expand(-1, window_size - x.shape[1], *x.shape[2:])
        # 前缀与原序列在时间维(1)拼接，凑满 window_size 帧
        return torch.cat([prefix, x], dim=1)


# ==== 类：LevelCostModel —— 单层的规划代价模型 ====
class LevelCostModel(nn.Module):
    """Cost model for one level: rolls out in the level's own latent space and scores the
    predictions against the goal in the goal's latent space.

    ``get_cost`` runs three stages:
      1. roll out this level's predictor from ``embed_0`` with the candidate actions;
      2. if the goal lives at a higher level, encode the predicted latents (and, for the
         action cost, the candidate actions) up to that level with the upper JEPA encoders.
         The goal is usually the next waypoint of the level above, compared at that level's
         stride. When every level in between was skipped at horizon 1 with
         ``horizon_one_goal_cost_space='upper'``, the goal is the real goal at the highest
         skipped level, and the predictions are encoded up the chain at stride 1 ("dense"), so
         every predicted step is scored;
      3. compare with ``goal_embed_0`` (``JEPA.criterion``), plus the weighted action cost.
    """

    def __init__(
        self,
        model: Any,
        upper_models: list[Any],
    ) -> None:
        # 先调 nn.Module 的初始化（注册参数/缓冲区机制）
        super().__init__()
        # 本层的 JEPA 世界模型（提供 rollout / criterion / action_embed 等）
        self.model = model
        # 本层之上的更高层模型列表（用于把预测向上投影到目标所在层），最高层时为空
        self.upper_models = upper_models
        # 记录本层层号（从模型上读），日志与键名拼接会用到
        self.level = self.model.level

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        """用候选动作在本层空间做一步滚动预测，返回含 predicted_embed_0 的字典。"""
        return self.model.rollout(info_dict, action_candidates)

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Forward latent-action queue access to the wrapped level model."""
        # 转发：取被包装模型维护的"隐动作队列"（用于动作先验统计）
        return self.model.get_latent_action_queue()

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """三段式计算规划代价：rollout 预测 -> 投影到目标空间 -> 与目标比较 + 动作代价。"""
        # 第 1 段：拿候选动作在本层 rollout，info_dict 被替换成含预测结果的字典
        info_dict = self.rollout(info_dict, action_candidates)

        # 本层预测出的隐状态序列 [B, S, T, D]（S 为采样维）
        pred_emb = info_dict["predicted_embed_0"]
        # 是否需要向上投影到目标所在层（默认需要）
        use_upper_space = bool(info_dict.get('goal_embed_0_in_upper_space', True))
        # 目标所在层号（缺省认为目标在"本层的上一层"）
        goal_level = int(info_dict.get('goal_embed_0_level', self.level + 1))
        # 动作代价权重（0 表示完全不计动作误差）
        action_weight = float(info_dict.get("action_cost_weight", 0.0))
        # 第 2 段：把预测（以及要参与代价的候选动作）投影到目标所在层的隐空间
        pred_for_cost, action_for_cost = self._project_to_goal_space(
            pred_emb,
            # 只有权重非 0 时才需要动作投影，省算力
            action_candidates if action_weight != 0.0 else None,
            use_upper_space=use_upper_space,
            # dense=True：stride=1 的稠密投影（跳过层链时使用）
            dense=bool(info_dict.get("dense_upper_goal_projection", False)),
            goal_level=goal_level,
        )

        # 第 3 段：构造交给 criterion 的字典（浅拷贝后替换预测字段）
        cost_info = dict(info_dict)
        cost_info["predicted_embed_0"] = pred_for_cost
        # 用本层 JEPA 的 criterion（如 1 - 余弦相似度 / MSE）算状态目标代价，越小越接近目标
        cost = self.model.criterion(cost_info)
        if action_weight != 0.0:
            # 计算"预测动作 vs 目标动作"的逐帧 MSE 代价
            action_cost = self._action_cost(action_for_cost, cost_info)
            if action_cost is not None:
                # 总代价 = 状态目标代价 + 权重 × 动作代价
                cost = cost + action_weight * action_cost
        return cost

    def _project_to_goal_space(
        self,
        pred_emb: torch.Tensor,
        action_candidates: torch.Tensor | None,
        *,
        use_upper_space: bool,
        dense: bool,
        goal_level: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode predicted latents (and candidate actions) up to the goal's level."""
        # 没有上层模型（本层是最高层）或目标就在本层空间：不用投影，原样返回
        if not self.upper_models or not use_upper_space:
            return pred_emb, None

        # Front-pad the predicted stream so that every window of the upper encoder is full.
        # Real-stride projection uses causal windows: pad window_size - 1. Dense (stride-1)
        # projection slides full windows over the stream: pad only when the stream is
        # shorter than one window. Actions get the same front pad, with zeros. With
        # window_size 1 (all paper models) the pad is 0.
        # 中文：为了让上层编码器的每个时间窗口都"填满"，需要在序列前面补帧：
        # 真实 stride 投影用因果窗口 -> 补 window_size-1 帧；
        # 稠密(stride=1)投影滑动满窗 -> 只在序列短于一个窗口时才补；
        # 论文里的模型 window_size=1，因此 pad 实际为 0。
        # 上层模型的时间窗口大小
        window_size = self.upper_models[0].temporal_window_size
        # 计算需要在时间维前面补几帧
        pad = max(0, window_size - pred_emb.shape[-2]) if dense else window_size - 1
        # 用"复制首帧"的方式在时间维前补 pad 帧（状态用首帧更合理）
        states = self._prefix_repeat_first(pred_emb, pad)

        # 把补齐后的预测状态逐层向上编码到 goal_level
        pred_for_cost = self._encode_states_up(states, dense=dense, goal_level=goal_level)
        # 动作投影结果（不需要时为 None）
        action_for_cost = None
        if action_candidates is not None:
            # 候选动作也投影上去：池化成上层的宏动作，用于动作代价
            action_for_cost = self._encode_actions_up(
                states, action_candidates, pad=pad, dense=dense
            )
        return pred_for_cost, action_for_cost

    def _encode_states_up(
        self,
        states: torch.Tensor,
        *,
        dense: bool,
        goal_level: int,
    ) -> torch.Tensor:
        """Encode the padded predicted stream with the level above, then chain up to goal_level."""
        # None: each upper encoder uses its own temporal_stride.
        # 中文：dense=True 时强制 stride=1（逐帧投影）；否则传 None 让各上层用自己的 stride。
        temporal_stride = 1 if dense else None
        # 记住 batch 维 b 与时间维 s，最后要把展平的维还原回去
        b, s = states.shape[:2]
        # 上一层编码器的输入键名：'embed_{本层层号}'
        input_key = f'embed_{self.level}'
        # 先用紧邻的上一层编码器编码：把 (b, s, T, D) 展平成 (b*s, T, D) 一起送入
        upper_out = self.upper_models[0].encode(
            {input_key: states.flatten(0, 1)},
            key=input_key,
            chunk_temporal_inputs=True,
            temporal_stride=temporal_stride,
        )
        # 若目标比"紧邻上层"还高，就继续一层层往上传（直到 goal_level）
        for upper_level in range(self.level + 2, goal_level + 1):
            # 下一个编码器的输入键名
            input_key = f'embed_{upper_level - 1}'
            # upper_models 的索引 = 相对本层的偏移 - 1
            upper_out = self.upper_models[upper_level - self.level - 1].encode(
                {input_key: upper_out['embed_0']},
                key=input_key,
                chunk_temporal_inputs=True,
                temporal_stride=temporal_stride,
            )
        # 把展平的前两维还原成 (b, s)，得到可用于比较的投影结果
        return upper_out["embed_0"].unflatten(0, (b, s))

    def _encode_actions_up(
        self,
        states: torch.Tensor,
        action_candidates: torch.Tensor,
        *,
        pad: int,
        dense: bool,
    ) -> torch.Tensor:
        """Pool the candidate actions into the upper level's macro-actions."""
        # 取紧邻上层模型（负责把动作池化成宏动作）
        upper_model = self.upper_models[0]
        # 同样记住 batch 维与时间维，结尾要还原
        b, s = states.shape[:2]
        # Embed, front-pad like the states, and pad the end to the state length so the
        # upper model chunks states and actions with the same windows.
        # 中文：先把动作嵌入成向量，再像状态一样在前面补零、在末尾补零到与状态等长，
        # 这样上层模型切分状态与动作时用的是完全相同的时间窗口。
        # 把候选动作 [B, S, T, A] 嵌入成 [B, S, T, D_a]
        action_emb = self.model.action_embed(action_candidates)
        # 时间维前面补 pad 帧零（与状态的前缀补齐长度一致）
        action_emb = self._prefix_zeros(action_emb, pad)
        # 末尾补零到与 states 相同的时间长度
        action_emb = self._pad_zeros_to_length(action_emb, states.shape[-2])
        # 上层编码器的输入键名
        input_key = f'embed_{self.level}'
        # 让上层按与状态相同的方式切块（stride：dense 用 1，否则用上层自己的 stride）
        chunked = upper_model._chunk_temporal_info(
            {input_key: states.flatten(0, 1), "action": action_emb.flatten(0, 1)},
            key=input_key,
            stride=1 if dense else upper_model.temporal_stride,
            window_size=upper_model.temporal_window_size,
        )
        # 用上层的动作编码器把每个窗口内的低层动作池化成一个宏动作
        pooled = upper_model.action_encoder(chunked["action"], chunked["action_mask"])

        # The final pooled action points beyond the final encoded upper state.
        # 中文：最后一个池化动作对应"超出最终上层状态"的时刻，与状态错一格，
        # 所以还原 (b, s) 后在时间维丢掉最后一帧 [..., :-1, :] 对齐。
        return pooled.unflatten(0, (b, s))[..., :-1, :]

    def _action_cost(
        self,
        pred_action: torch.Tensor | None,
        info_dict: dict,
    ) -> torch.Tensor | None:
        """计算预测动作与目标动作的逐帧 MSE（对特征维求和），只取最后 cost_window 帧。"""
        # 目标动作（上层下发的宏动作），没有目标就没法算动作代价
        target_action = info_dict.get("goal_action")
        if pred_action is None or target_action is None:
            # 缺任一方直接返回 None，调用方会跳过动作代价项
            return None

        # 只比较最后几步（配置 cost_last_n，默认 1 步）
        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        # 实际窗口取三者最小，防止越界
        cost_window = min(cost_last_n, pred_action.shape[-2], target_action.shape[-2])
        # 预测动作截取最后 cost_window 帧
        pred_action = pred_action[..., -cost_window:, :]
        # 目标同样截尾并对齐到预测的形状（单步目标可广播）
        target_action = target_action[..., -cost_window:, :].expand_as(pred_action)

        # 逐元素 MSE（不聚合），后续自己按需求和
        cost = F.mse_loss(
            pred_action,
            # 目标不参与梯度（detach），梯度只流向候选动作
            target_action.detach(),
            reduction="none",
        )
        # 对 batch/时间之外的所有维（特征维）求和 -> 形状 [B, S, T]
        return cost.sum(dim=tuple(range(2, cost.ndim)))

    @staticmethod
    def _prefix_repeat_first(x: torch.Tensor, count: int) -> torch.Tensor:
        """在时间维(-2)前面复制首帧 count 次作为前缀；count<=0 原样返回。"""
        if count <= 0:
            return x

        # 构造前缀形状：除时间/特征维外不变，时间维变成 count
        prefix_shape = (*x.shape[:-2], count, x.shape[-1])
        # 取首帧 expand 成前缀（零拷贝的视图）
        prefix = x[..., :1, :].expand(prefix_shape)
        # 在时间维(-2)上把前缀拼到原序列前面
        return torch.cat([prefix, x], dim=-2)

    @staticmethod
    def _prefix_zeros(x: torch.Tensor, count: int) -> torch.Tensor:
        """在时间维前面补 count 帧全零；count<=0 原样返回。"""
        if count <= 0:
            return x

        # 用 x 的 dtype/device 造全零前缀
        prefix = x.new_zeros(*x.shape[:-2], count, x.shape[-1])
        # 拼接到时间维开头
        return torch.cat([prefix, x], dim=-2)

    @staticmethod
    def _pad_zeros_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """在时间维末尾补零，使总长度达到 target_len。"""
        # 需要补的长度 = 目标长度 - 当前长度
        pad_len = target_len - x.shape[-2]
        if pad_len == 0:
            # 已经等长，原样返回
            return x

        # 造出尾部零填充
        suffix = x.new_zeros(*x.shape[:-2], pad_len, x.shape[-1])
        # 拼到时间维末尾
        return torch.cat([x, suffix], dim=-2)
