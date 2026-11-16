# ---- 段落：导入依赖 ----
from collections import deque            # 双端队列：暂存规划好的待执行动作
from dataclasses import dataclass        # 声明式配置类(frozen=创建后不可改)
from pathlib import Path                 # 路径处理
from typing import Any, Protocol         # Any=任意类型; Protocol=结构化类型协议
from collections.abc import Callable     # 可调用对象的类型标注

import numpy as np                       # numpy 数组
import torch                             # PyTorch 张量
from loguru import logger as logging     # loguru 日志器，别名成 logging 用
from torchvision import tv_tensors       # torchvision 张量类型(如 tv_tensors.Image)

import stable_worldmodel as swm          # 本仓库包(用其缓存目录工具)
from stable_worldmodel.solver import Solver  # 规划求解器协议/基类


# ---- 段落：MPC 规划循环的配置(单层) ----
@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon; each call uses a decreasing schedule derived
            from it and the eval budget.
        receding_horizon: Number of steps to execute before re-planning.
        action_block: Number of times each action is repeated (frameskip).
        action_dim: Optional explicit per-step action dimension used by solvers.
        pre_decay_horizon: Optional lower-level horizon used before its
            resolved decay segment starts in hierarchical planning.
        horizon_one_goal_cost_space: For skipped upper levels with horizon 1,
            choose lower-level goal cost space: 'lower' keeps current behavior,
            'upper' projects lower predictions to the actual upper-level goal.
    """

    horizon: int                          # 规划视野：往前推演多少步
    receding_horizon: int                 # 每次规划实际执行前多少步(滚动执行)
    action_block: int = 1                 # 每个规划动作重复执行几次 = frameskip
    action_dim: int | None = None         # 求解器用的每步动作维度(可显式指定)
    pre_decay_horizon: int | None = None  # 分层规划中 decay 段开始前用的低层 horizon
    num_subgoals: int = 1                 # 一次规划产生多少个子目标
    cost_last_n: int = 1                  # 只对最后 n 步累计 cost
    intermediate_cost_weight: float = 1.0 # 中间步 cost 的权重
    action_cost_weight: float = 0.0       # 动作幅值惩罚权重(鼓励小动作)
    horizon_one_goal_cost_space: str = 'lower'  # horizon=1 时目标 cost 空间: 'lower'/'upper'


# ---- 段落：分层 MPC 配置：每层一个 PlanConfig，从 level1 向上排 ----
@dataclass(frozen=True)
class HierarchicalPlanConfig:
    """Configuration container for hierarchical MPC planning.

    Attributes:
        receding_horizon: Number of high-level steps to execute before re-planning.
        levels: Per-level planning configs, ordered from level 1 upward. Also
            exposed as ``level1``, ``level2``, ... attributes.
    """

    receding_horizon: int                  # 高层执行多少步后重新规划一次
    levels: tuple[PlanConfig, ...] = ()    # 各层配置元组(按 level1, level2... 排序)

    def __getattr__(self, name: str) -> PlanConfig:
        # 支持 cfg.level1 这种属性访问，转成 tuple 下标取值
        index = name.removeprefix('level')   # 'level2' -> '2'
        if name.startswith('level') and index.isdigit() and 1 <= int(index) <= len(self.levels):
            return self.levels[int(index) - 1]   # 1-based 名字转 0-based 下标
        raise AttributeError(name)   # 其他属性名一律报"不存在"


# ---- 段落：可逆变换协议：归一化器/缩放器都要实现正向与逆向两个方法 ----
class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...   # 协议只约束签名，具体实现留给归一化器类

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...   # 逆变换：推理时把模型输出还原回真实量纲


# ---- 段落：策略基类——约定 set_env / get_action / reset 三大接口 ----
class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any       # 关联的环境对象，由 set_env 注入
    type: str      # 策略类型标签('base'/'random'/'world_model')

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None   # 初始未绑定环境
        self.type = 'base'   # 基类默认类型
        for arg, value in kwargs.items():
            setattr(self, arg, value)   # 把额外配置参数逐个挂成自身属性

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        # 子类必须实现：输入观测，返回一个动作；基类直接报未实现
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env   # 绑定环境，之后从 env 取动作空间等信息

    def reset(self) -> None:
        """Reset any internal policy state before a new rollout."""
        return None   # 基类无状态无需重置；子类在此清 buffer/计数器

    # ---- 段落：观测预处理——process 数值归一化 + transform 图像变换，统一成张量 ----
    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like WorldModelPolicy.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        # 先把 proprio 维度对齐到归一化器所期望的维度
        self._align_proprio_to_processor(info_dict)

        # 逐列(键)处理观测字典
        for k, v in info_dict.items():
            # 记录当前是否还是 numpy，决定最后要不要转成 torch
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            # 该列若配置了数值预处理器(如 StandardScaler)，先做归一化
            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape   # 记住原形状，处理完还要还原
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])   # (e,t,...) 展平成 (e*t,...) 逐样本处理

                # process and reshape back
                v = self.process[k].transform(v)   # 逐样本做数值归一化
                v = v.reshape(shape)               # 还原回原形状

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if hasattr(self, 'transform') and k in self.transform:
                shape = None   # None 表示末尾无需还原前两维
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape                # 记住 (e, t, ...) 前两维
                        v = v.reshape(-1, *shape[2:])  # 先展平成逐帧样本
                if k.startswith('pixels') or k.startswith('goal'):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))   # HWC -> CHW (numpy)
                    else:
                        v = v.permute(0, 3, 1, 2)           # HWC -> CHW (torch)
                # 逐帧套用图像变换(缩放/归一化等)，再堆叠回批量张量
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))   # stack 后已是 tensor，更新标记

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])   # 还原 (e, t) 前两维

            # 普通数值数组转 torch(字符串/结构化 dtype 跳过)
            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            info_dict[k] = v   # 写回字典

        return info_dict

    # 从归一化器的 mean_ 推出它期望的最后一维特征数(没有则返回 None)
    @staticmethod
    def _processor_dim(processor: Transformable | None) -> int | None:
        mean = getattr(processor, 'mean_', None)   # sklearn 风格归一化的均值属性
        if mean is None:
            return None   # 没有 mean_ → 判断不了维度
        mean = np.asarray(mean)   # 统一转成 ndarray 方便取形状
        if mean.ndim == 0:
            return None   # 标量均值 → 不含维度信息
        return int(mean.shape[-1])   # 最后一维就是特征维数

    # 尝试用 qpos+qvel 拼出 target_dim 维向量(兼容旧 checkpoint 的输入格式)
    @staticmethod
    def _merge_qpos_qvel_for_dim(
        info_dict: dict,
        *,
        prefix: str,      # 键前缀: '' 或 'goal_'
        target_dim: int,   # 期望拼出的维度
    ) -> np.ndarray | None:
        qpos = info_dict.get(f'{prefix}qpos')   # 取关节位置
        qvel = info_dict.get(f'{prefix}qvel')   # 取关节速度
        if qpos is None or qvel is None:
            return None   # 缺任何一侧都无法拼
        # 两者必须都是 numpy 才能拼接
        if not isinstance(qpos, (np.ndarray | np.generic)) or not isinstance(
            qvel,
            (np.ndarray | np.generic),
        ):
            return None

        # 两种候选拼法: 全量 qpos+qvel；或丢掉 qpos 前两维(平面位姿)再拼
        candidates = (
            np.concatenate([qpos, qvel], axis=-1),
            np.concatenate([qpos[..., 2:], qvel], axis=-1),
        )
        for candidate in candidates:
            if candidate.shape[-1] == target_dim:   # 维度对得上就采用
                return candidate
        return None   # 两种都不匹配，放弃

    # 让 proprio/goal_proprio 的维度与归一化器一致(必要时重拼/截断)
    def _align_proprio_to_processor(self, info_dict: dict) -> None:
        if not hasattr(self, 'process'):
            return   # 没配置归一化器，无需对齐

        # 依次处理 proprio(无前缀) 和 goal_proprio(前缀 'goal_')
        for key, prefix in (('proprio', ''), ('goal_proprio', 'goal_')):
            if key not in info_dict or key not in self.process:
                continue   # 缺观测列或缺归一化配置就跳过
            value = info_dict[key]
            if not isinstance(value, (np.ndarray | np.generic)):
                continue   # 非数组(如字符串)跳过

            # 归一化器期望的维度(None=不关心维度)
            target_dim = self._processor_dim(self.process.get(key))
            if target_dim is None or value.shape[-1] == target_dim:
                continue   # 维度已经匹配，无需修改

            # 维度不匹配 → 先试着用 qpos+qvel 重新拼出来
            merged = self._merge_qpos_qvel_for_dim(
                info_dict,
                prefix=prefix,
                target_dim=target_dim,
            )
            if merged is not None:
                info_dict[key] = merged   # 用拼好的结果替换原值
                continue

            # 特例: 只是多了前两维(平面 x,y) → 直接截掉前两维
            if value.shape[-1] - 2 == target_dim:
                info_dict[key] = value[..., 2:]


# ---- 段落：随机策略——直接从动作空间均匀采样(数据收集/基线用) ----
class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Optional random seed for the action space.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)   # 先走基类 __init__ 挂载 kwargs 属性
        self.type = 'random'         # 标记策略类型
        self.seed = seed             # 记录随机种子

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()   # 忽略 obs，直接从动作空间随机采样

    def set_seed(self, seed: int) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
        if self.env is not None:
            self.env.action_space.seed(seed)   # 给动作空间设定种子，保证可复现


# ---- 段落：世界模型策略——用 solver(MPC) 在世界模型里规划出动作 ----
class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,                                # 规划求解器(封装世界模型)
        config: PlanConfig | HierarchicalPlanConfig,   # 单层或分层的 MPC 规划配置
        process: dict[str, Transformable] | None = None,   # {列名: 数值归一化器}
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,                                 # {列名: 图像变换函数}
        **kwargs: Any,                                 # 其余参数由基类挂成属性
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'          # 策略类型标签
        self.cfg = config                   # PlanConfig 或 HierarchicalPlanConfig
        self.solver = solver                # 规划求解器(世界模型前向 + 代价搜索)
        self.process = process or {}        # 数值归一化器字典 {列名: scaler}
        self.transform = transform or {}    # 图像变换字典 {列名: 变换函数}
        # 待执行动作队列：一次规划产出的动作先入队，逐步弹出执行
        self._action_buffer: deque[torch.Tensor] | None = None
        # 评估步数上限(可由 kwargs 注入)，None=不限
        self.eval_budget: int | None = getattr(self, 'eval_budget', None)
        self._steps_taken = 0          # 本次 rollout 已执行的环境步数
        self._policy_call_count = 0    # 触发规划(get_action 且队列空)的次数

    @property
    def execution_plan_config(self) -> PlanConfig:
        """Return the low-level plan config used to execute environment actions."""
        # 单层配置：它本身就是执行用配置
        if isinstance(self.cfg, PlanConfig):
            return self.cfg

        # 分层配置必须提供 level1 —— 真正下发动作的是最低层
        if self.cfg.level1 is None:
            raise ValueError(
                'HierarchicalPlanConfig.level1 must be provided for action execution.'
            )

        return self.cfg.level1   # 用 level1 的 action_block/horizon 执行

    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        plan_cfg = self.execution_plan_config
        # 规划步数 × 每个动作重复次数 = 环境的原始步数
        return self.cfg.receding_horizon * plan_cfg.action_block

    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)   # 并行环境个数，缺省按 1
        # 把动作空间/并行数/规划配置喂给 solver 完成配置
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self.reset()   # 清空 buffer 和计数器，准备新一轮 rollout

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def reset(self) -> None:
        """Clear action buffers and step counters for a new rollout."""
        # 队列容量 = 一次规划最多产出的环境步动作数
        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)
        self._steps_taken = 0            # 步数清零
        self._policy_call_count = 0      # 规划次数清零

    def _remaining_eval_steps(self) -> int | None:
        """Return the remaining environment-step budget, if one is set."""
        if self.eval_budget is None:
            return None   # 没设预算 → 一直可执行
        return max(0, self.eval_budget - self._steps_taken)   # 剩余可走步数(不小于 0)

    # ---- 段落：核心流程——缓冲空则重新规划，否则按序吐出缓存动作 ----
    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        # 前置检查：环境已绑定、必须带像素观测和目标
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pixels' in info_dict, "'pixels' must be provided in info_dict"
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        # 归一化 + 图像变换，得到模型可吃的输入张量
        info_dict = self._prepare_info(info_dict)

        # need to replan if action buffer is empty
        if len(self._action_buffer) == 0:   # 缓存里没有待执行动作才需要规划
            # 规划前先检查评估步数预算是否用尽
            remaining_steps = self._remaining_eval_steps()
            if remaining_steps is not None and remaining_steps <= 0:
                raise RuntimeError('Evaluation step budget exhausted.')

            self._policy_call_count += 1   # 记录一次规划调用
            print(f'policy call number: {self._policy_call_count}')
            # 调 solver 做 MPC 规划(内部跑世界模型前向 + 代价搜索)
            outputs = self.solver(
                info_dict,
                steps_taken=self._steps_taken,   # 供 horizon 递减调度参考
                eval_budget=self.eval_budget,
            )

            actions = outputs['actions']  # (num_envs, horizon, action_dim)
            # 只保留前 receding_horizon 步(滚动执行，其余丢弃)
            keep_horizon = min(self.cfg.receding_horizon, actions.shape[1])
            plan = actions[:, :keep_horizon]   # 切出要执行的规划段

            # frameskip back to timestep
            # 每个规划动作展开成 action_block 个重复的环境步动作:
            # (env, 步, 维) -> (env, 步*block, 维)
            plan = plan.reshape(
                self.env.num_envs,
                keep_horizon * self.execution_plan_config.action_block,
                -1,   # 动作维自动推断
            )

            # 转置成 (步, env)，按时间顺序逐个入队
            self._action_buffer.extend(plan.transpose(0, 1))

        # 从队首弹出一个待执行动作(每个并行环境各一个)
        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)   # 整形成动作空间要求的形状
        action = action.numpy()   # torch -> numpy，交给环境执行

        # post-process action
        if 'action' in self.process:
            # 动作若做过归一化，这里逆变换还原到真实量纲
            action = self.process['action'].inverse_transform(action)

        self._steps_taken += 1   # 累计已执行的环境步数
        return action  # (num_envs, action_dim)


# ---- 段落：通用加载工具——从 checkpoint 里找出带指定方法的子模块 ----
def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    """Helper function to load a model checkpoint and find a module with the specified attribute.

    Args:
        run_name: Path or name of the model run
        attribute_name: Name of the attribute to look for in the module (e.g., 'get_action', 'get_cost')
        cache_dir: Optional cache directory path

    Returns:
        The module with the specified attribute

    Raises:
        RuntimeError: If no module with the specified attribute is found
    """
    # 展开 ~ 等，得到 run 名或路径
    run_path = Path(run_name).expanduser()
    if run_path.suffix == '.ckpt':
        # 直接给了 .ckpt 文件路径
        path = run_path
        if not path.exists() and not path.is_absolute():
            # 相对路径且不存在 → 再到缓存目录下找一次
            path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

        # 路径必须真实存在，否则提示先跑预训练
        assert path.exists(), (
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )
    else:
        # 给的是 run 名或目录名(没有 .ckpt 后缀)
        if not run_path.exists():
            # 本地不存在 → 当作缓存目录下的相对名
            run_path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

        if run_path.is_dir():
            # 目录里找所有 *_object.ckpt，按创建时间倒序取最新一个
            ckpt_files = list(run_path.glob('*_object.ckpt'))
            ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            path = ckpt_files[0]   # 最新的 checkpoint
            logging.info(f'Loading model from checkpoint: {path}')
        else:
            # 约定: run 名 + '_object.ckpt' 就是模型文件
            path = Path(f'{run_path}_object.ckpt')
            assert path.exists(), (
                f'Checkpoint path does not exist: {path}. Launch pretraining first.'
            )

    # 反序列化整棵模块(weights_only=False 允许自定义类)，放到 CPU 上
    spt_module = torch.load(path, weights_only=False, map_location='cpu')

    # 深度优先在模块树里找第一个带 attribute_name 的子模块
    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
                module = module.eval()   # 命中后切评估模式(关 dropout、停 BN 更新)
            return module
        for child in module.children():   # 遍历直接子模块
            result = scan_module(child)   # 递归下钻
            if result is not None:
                return result   # 子树里找到了，逐层返回
        return None   # 这棵子树里没有

    result = scan_module(spt_module)   # 从根模块开始扫描
    if result is not None:
        return result   # 找到目标模块，返回它

    # 整个模型都没有该方法 → 报错
    raise RuntimeError(
        f"No module with '{attribute_name}' found in the loaded world model."
    )


def AutoCostModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_cost` method.

    Automatically scans the checkpoint for a module implementing a cost function
    (i.e., has a `get_cost` method) for use with planning solvers.

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to HJEPA_HOME.

    Returns:
        The module with a `get_cost` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_cost` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)   # 找实现 get_cost 的模块


# Alias for backward compatibility and type hinting
Policy = BasePolicy   # Policy 只是 BasePolicy 的别名，兼容旧代码里的 Policy 名称
