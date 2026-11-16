"""World environment manager for vectorized Gymnasium environments.

本模块同时承担「数据采集落盘」职责，写出的 HDF5 契约如下（新数据集必须
与之严格一致，才能被 stable_worldmodel/data/dataset.py 的 HDF5Dataset 读取）：

- 所有数据列按 episode 顺序连续写入同一条全局轴（不按 episode 分组存）；
- 元数据三列：
    ep_offset[i]：第 i 个 episode 在全局轴上的起始下标（int64）；
    ep_len[i]    ：第 i 个 episode 的帧数（int32）；
    ep_idx[t]    ：每个 step 属于第几个 episode（int32）；
- action 列做「左移对齐」：落盘后第 t 行表示 (o_t, a_t)，即观测在前、
  作用于它的动作在后；episode 末尾动作补 NaN；
- ndim>=2 的列用 Blosc(lz4, clevel=5, shuffle) 压缩、chunk=(100,)+shape；
  1D 列不压缩、chunk=(1000,)；
- 文件以 SWMR 模式打开 + 每 episode flush，进程被杀后可续采
  （record_dataset 里按 ep_offset[-1]+ep_len[-1] 推进写指针）。
"""

# ---- 段落标题：标准库与第三方库导入 ----
# hashlib：对评测配置字符串做哈希，生成「断点续评」临时文件名
import hashlib
# json：把评测配置序列化成可哈希的字符串
import json
# os：删除评测临时文件（os.remove）
import os
# time：评测计时（perf_counter 统计到达目标的耗时）
import time
# defaultdict：每个并行环境的 episode 缓冲（列名 -> step 列表），缺列自动建空表
from collections import defaultdict
# Callable / Sequence：类型标注，分别表示「可调用对象」和「序列」
from collections.abc import Callable, Sequence
# deepcopy：评测时深拷贝初始状态/目标，避免原地改动数据集返回的数据
from copy import deepcopy
# partial：把 wrapper 参数预先固化，制造可复用的单环境工厂函数
from functools import partial
# Path：面向对象路径操作（拼 .h5 路径、判断文件是否存在）
from pathlib import Path
# Any：宽松类型标注（任意类型）
from typing import Any

# gymnasium：新版 Gym 接口，提供 env / 向量化环境 / AutoresetMode
import gymnasium as gym
# h5py：读写 HDF5 文件（数据集录制与读取的核心依赖）
import h5py
# hdf5plugin：提供 Blosc(lz4, shuffle) 压缩插件，多维列压缩用
import hdf5plugin
# numpy：所有落盘数据最终都转换成 ndarray 处理
import numpy as np
# torch：评测轨迹 payload 的 torch.save / torch.load 读写
import torch
# VectorEnv：向量化环境基类，仅用于类型标注
from gymnasium.vector import VectorEnv
# loguru 的 logger 起别名 logging，全模块统一用 logging.info 打日志
from loguru import logger as logging
# rich 的 print：美化打印动作空间/观测空间字符串
from rich import print
# tqdm：录制数据集时的进度条
from tqdm import tqdm

# get_cache_dir：数据集默认保存的缓存目录
from stable_worldmodel.data.utils import get_cache_dir
# Policy：策略抽象类型，World.step 调用它来产生动作
from stable_worldmodel.policy import Policy

# MegaWrapper：图像缩放/帧堆叠等预处理包装器；
# SyncWorld：多进程同步向量化环境；VariationWrapper：场景变体随机化包装器
from .wrapper import MegaWrapper, SyncWorld, VariationWrapper


# 评测初始状态的两种回放模式（_validate_eval_state_mode 只放行这两种）：
# dataset_full = 直接用数据集存下的完整状态；xy_neutral = 只取 xy 坐标，
# 由环境自己「结算」出一份静止/中性状态
EVAL_STATE_MODES = ('dataset_full', 'xy_neutral')
# 轨迹 pad 之前的原始帧数，存进 dict 的内部键（下划线开头，不参与落盘）
ORIGINAL_LENGTH_KEY = '_original_length'
# pad 前每个数据列各自的长度字典所用的键名
ORIGINAL_LENGTHS_KEY = '_original_lengths'
# pad 后「有效最后一帧」的下标 = 原始长度-1，评测只认可这一帧之前的结果
VALID_LAST_INDEX_KEY = '_valid_last_index'


# ---- 段落标题：_make_env 单环境工厂（被 SyncWorld 为每个并行槽位调用一次） ----
def _make_env(env_name, max_episode_steps, wrappers, **kwargs):
    """Create a gymnasium environment with specified wrappers.

    Factory function for creating environments within a vectorized setup.
    Creates the base environment and applies wrappers in order.

    Args:
        env_name: Name of the gymnasium environment to create.
        max_episode_steps: Maximum steps per episode before truncation.
        wrappers: List of wrapper functions/classes to apply. Each wrapper
            should accept an environment and return a wrapped environment.
        **kwargs: Additional keyword arguments passed to gym.make.

    Returns:
        The wrapped gymnasium environment.

    Example:
        >>> from functools import partial
        >>> wrappers = [partial(MegaWrapper, image_shape=(64, 64))]
        >>> env = _make_env("CartPole-v1", max_episode_steps=500, wrappers=wrappers)
    """
    # gym.make 创建基础环境，同时注入单集最大步数（超了就截断 truncated）
    env = gym.make(env_name, max_episode_steps=max_episode_steps, **kwargs)
    # 依次套 wrapper：每层包在外面一层（图像缩放 -> 帧堆叠 -> ...）
    for wrapper in wrappers:
        env = wrapper(env)
    # 返回包装完成的单个环境，供向量化环境管理
    return env



# ---- 段落标题：_find_env_method 沿 wrapper 链查找环境的可选方法 ----
def _find_env_method(env: gym.Env, method_name: str) -> Callable | None:
    # 从最外层 wrapper 开始当作起点
    method_owner = env
    # 沿着 .env 属性（wrapper -> 内层 env）一路向下找目标方法
    while method_owner is not None and not hasattr(method_owner, method_name):
        # 取当前对象的内层 env（没有则为 None）
        next_env = getattr(method_owner, 'env', None)
        # 防御：内层指向自己（循环引用）就跳出，避免死循环
        if next_env is method_owner:
            break
        # 指针下移一层，继续找
        method_owner = next_env

    # 链上某层找到了，就返回绑定好的方法（method_owner.method_name）
    if method_owner is not None and hasattr(method_owner, method_name):
        return getattr(method_owner, method_name)

    # 链上没找到，再看最底层的 unwrapped 真实环境对象
    env_unwrapped = env.unwrapped
    # 找到则返回真实环境上的方法
    if hasattr(env_unwrapped, method_name):
        return getattr(env_unwrapped, method_name)

    # 全链都没有该方法，返回 None（调用方按「不支持」处理）
    return None


# ---- 段落标题：_validate_eval_state_mode 校验状态回放模式合法性 ----
def _validate_eval_state_mode(mode: str, mode_name: str) -> None:
    # 模式必须是 'dataset_full' 或 'xy_neutral' 之一
    if mode not in EVAL_STATE_MODES:
        # 否则抛 ValueError，报错里带上参数名和合法取值
        raise ValueError(
            f'{mode_name} must be one of {EVAL_STATE_MODES}, got {mode!r}.'
        )


# ---- 段落标题：_episode_sequence_length 推断一条轨迹的时间长度 T ----
def _episode_sequence_length(
    episode: dict[str, Any],
    columns: Sequence[str],
) -> int | None:
    # 检查顺序：pixels 优先（最稳定的 T 来源），其余列按传入顺序兜底
    keys = ['pixels', *[key for key in columns if key != 'pixels']]
    # 逐列尝试，找到第一个「按时间堆叠的序列值」就用它的第 0 维当 T
    for key in keys:
        # 取该列（没有则 None，跳过）
        value = episode.get(key)
        # torch 张量且至少 1 维：shape[0] 就是时间步数 T
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        # numpy 数组同理，shape[0] 即 T
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
    # 所有列都推断不出（全是标量），返回 None 让调用方报错
    return None


# ---- 段落标题：_is_sequence_value 判断值是否为「时间序列」（0 维以上） ----
def _is_sequence_value(value: Any) -> bool:
    # torch 张量：维数 > 0 才算带时间轴的序列
    if torch.is_tensor(value):
        return value.ndim > 0
    # numpy 数组同理；纯 Python 标量一律不算序列
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return False


# ---- 段落标题：_pad_time_sequence 把某列沿时间轴补到 target_length ----
def _pad_time_sequence(value: Any, target_length: int, *, zero_pad: bool) -> Any:
    # 当前时间长度 T（第 0 维）
    current_length = int(value.shape[0])
    # 需要补的帧数 = 目标 - 当前
    pad_length = target_length - current_length
    # 已经够长或超长：原样返回，不做裁剪（pad 只加不减）
    if pad_length <= 0:
        return value

    # 以下按类型分别处理：torch 张量分支
    if torch.is_tensor(value):
        # zero_pad=True（如 action 列）：补全 0
        if zero_pad:
            # 造 (pad_length, *其余维度) 的全 0 张量，dtype/device 跟原值一致
            padding = torch.zeros(
                (pad_length, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
        # 否则（如 pixels/状态列）：复制最后一帧并 expand 出 pad_length 份
        else:
            # 取最后一帧 [1, ...] 再沿第 0 维扩展、clone 成独立内存
            padding = value[-1:].expand(pad_length, *value.shape[1:]).clone()
        # 沿第 0 维（时间轴）把原序列和补的部分首尾拼接
        return torch.cat([value, padding], dim=0)

    # numpy 数组分支：同样逻辑的 numpy 版
    if zero_pad:
        # 全 0 补齐，形状 (pad_length, *其余维度)
        padding = np.zeros((pad_length, *value.shape[1:]), dtype=value.dtype)
    else:
        # 复制最后一帧 [1, ...] 重复 pad_length 次（np.repeat 沿 axis=0）
        padding = np.repeat(value[-1:], pad_length, axis=0)
    # 沿 axis=0 拼接成等长序列
    return np.concatenate([value, padding], axis=0)


# ---- 段落标题：_pad_loaded_eval_trajectories 把多条加载轨迹对齐成等长 ----
# 评测时每个并行环境吃一条轨迹，形状必须一致才能 np.stack / broadcast；
# 短轨迹沿时间轴补齐（action 补 0、其余补最后一帧），并记录 pad 前的真实长度
def _pad_loaded_eval_trajectories(
    data: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    # 逐条轨迹推断它的时间长度 T
    lengths = [_episode_sequence_length(episode, columns) for episode in data]
    # 任何一条推断不出长度 -> 直接报错（数据不符合契约）
    if any(length is None for length in lengths):
        raise ValueError('Could not infer lengths for all loaded eval trajectories.')

    # 只保留非 None 的长度并转 int 列表
    original_episode_lengths = [int(length) for length in lengths if length is not None]
    # 空数据：原样返回，无需 pad
    if not original_episode_lengths:
        return list(data)

    # 所有轨迹中最长的那条长度，作为统一目标长度
    max_episode_length = max(original_episode_lengths)
    # 长度必须为正（空轨迹没有评测意义）
    if max_episode_length <= 0:
        raise ValueError('Loaded eval trajectories must have positive length.')

    # 统计「每个列」各自的最大时间长度（不同列可能长度不同，比如 goal 列更短）
    target_lengths: dict[str, int] = {}
    for episode in data:
        # 遍历该轨迹的每一列
        for key, value in episode.items():
            # 跳过内部元数据键（_ 开头）和非序列值
            if key.startswith('_') or not _is_sequence_value(value):
                continue
            # 取「历史最大」与「本条该列长度」的较大者，逐步求出每列目标 T
            target_lengths[key] = max(
                target_lengths.get(key, 0),
                int(value.shape[0]),
            )

    # 逐条轨迹做 pad，产出新的 dict（不改动原数据）
    padded = []
    for episode, original_length in zip(data, original_episode_lengths):
        # 浅拷贝一条轨迹 dict
        padded_episode = dict(episode)
        # 记录 pad 前每列各自的长度（后面恢复真实长度用）
        key_lengths = {
            key: int(value.shape[0])
            for key, value in episode.items()
            if not key.startswith('_') and _is_sequence_value(value)
        }
        # 存入三个内部元数据键：原始总长 / 各列原始长 / 有效最后一帧下标
        padded_episode[ORIGINAL_LENGTH_KEY] = original_length
        padded_episode[ORIGINAL_LENGTHS_KEY] = key_lengths
        padded_episode[VALID_LAST_INDEX_KEY] = original_length - 1

        # 逐列检查是否需要补齐到该列的目标长度
        for key, value in list(padded_episode.items()):
            # 该列的目标 T（内部键没有目标，跳过）
            target_length = target_lengths.get(key)
            if key.startswith('_') or target_length is None:
                continue
            # 非序列值、或已经 >= 目标长度的列不用处理
            if not _is_sequence_value(value) or int(value.shape[0]) >= target_length:
                continue
            # 补齐该列：只有 action 用 0 填充，其余列复制最后一帧填充
            padded_episode[key] = _pad_time_sequence(
                value,
                target_length,
                zero_pad=key == 'action',
            )
        padded.append(padded_episode)

    # 长度不一致时打一条 info 日志，说明 pad 到了多长、原始长度集合是什么
    if len(set(original_episode_lengths)) > 1:
        logging.info(
            'Padded loaded eval trajectories to length {} from original lengths {}.',
            max_episode_length,
            sorted(set(original_episode_lengths)),
        )

    # 返回全部对齐后的轨迹列表
    return padded


class World:
    """High-level manager for vectorized Gymnasium environments.

    Manages a set of synchronized vectorized environments with automatic
    preprocessing (resizing, frame stacking, goal conditioning).

    Args:
        env_name: Name of the Gymnasium environment to create.
        num_envs: Number of parallel environments.
        image_shape: Target shape for image observations (H, W).
        seed: Random seed for reproducibility.
        history_size: Number of frames to stack.
        frame_skip: Number of frames to skip per step.
        max_episode_steps: Maximum steps per episode before truncation.
        verbose: Verbosity level (0: silent, >0: info).
        **kwargs: Additional keyword arguments passed to `gym.make_vec`.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple[int, int],
        seed: int = 2349867,
        history_size: int = 1,
        frame_skip: int = 1,
        max_episode_steps: int = 100,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        # ---- 段落标题：构造向量化环境 ----
        # wrapper 链：目前只套 MegaWrapper（负责图像 resize 到 image_shape、
        # history_size 帧堆叠、frame_skip 跳帧）
        wrappers = [
            partial(
                MegaWrapper,
                image_shape=image_shape,
                history_size=history_size,
                frame_skip=frame_skip,
            ),
        ]

        # 把「环境名 + 步数上限 + wrapper 链 + 其余参数」固化成无参 env 工厂
        env_fn = partial(
            _make_env, env_name, max_episode_steps, wrappers, **kwargs
        )
        # 复制 num_envs 份工厂：一个环境一个子进程/实例
        env_fns = [env_fn for _ in range(num_envs)]
        # SyncWorld 创建同步向量化环境，再套 VariationWrapper 支持场景变体
        self.envs: VectorEnv = VariationWrapper(SyncWorld(env_fns))
        # 关闭 gym 的自动 reset：本模块自己在 episode 结束后手动 reset 单个
        # 环境（否则 done 那一步的观测会被 reset 后的新观测顶掉，没法落盘）
        self.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

        # 记录帧堆叠历史长度（record_dataset 里会校验是否 > 1）
        self._history_size = history_size
        # 当前挂载的策略，step() 前必须先 set_policy
        self.policy: Policy | None = None
        # 最近一次 step/reset 得到的观测（含 per-env info）
        self.states: dict | None = None
        # 最近一次返回的 infos 字典（policy 从这里取输入、录制从这里取数据列）
        self.infos: dict = {}
        # 最近一步各环境的奖励数组（shape = [num_envs]）
        self.rewards: np.ndarray | None = None
        # 各环境是否终止（任务成功/失败 terminated）
        self.terminateds: np.ndarray | None = None
        # 各环境是否截断（步数上限等 truncated）
        self.truncateds: np.ndarray | None = None

        # verbose > 0 时打印环境信息
        if verbose > 0:
            # 初始化完成日志
            logging.info(f'🌍🌍🌍 World {env_name} initialized 🌍🌍🌍')

            # 打印动作空间（离散/连续、维度）
            logging.info('🕹️ 🕹️ 🕹️ Action space 🕹️ 🕹️ 🕹️')
            logging.info(f'{self.envs.action_space}')

            # 打印观测空间（像素形状等）
            logging.info('👁️ 👁️ 👁️ Observation space 👁️ 👁️ 👁️')
            logging.info(f'{str(self.envs.observation_space)}')

            # 有变体空间则打印单环境变体描述（如房间布局选项）
            if self.envs.variation_space is not None:
                logging.info('⚗️ ⚗️ ⚗️ Variation space ⚗️ ⚗️ ⚗️')
                print(self.envs.single_variation_space.to_str())
            # 没有变体空间只警告一次
            else:
                logging.warning('No variation space provided!')

        # 保存默认随机种子（reset(seed=None) 时上层可能用它）
        self.seed = seed

    @property
    def num_envs(self) -> int:
        """Number of parallel environment instances."""
        # 并行环境个数直接问底层 VectorEnv 要
        return self.envs.num_envs

    def close(self, **kwargs: Any) -> None:
        """Close all environments and clean up resources."""
        # 关闭全部子环境、释放进程/渲染资源
        return self.envs.close(**kwargs)

    def step(self) -> None:
        """Advance all environments by one step using the current policy."""
        # note: reset happens before because of auto-reset, should fix that
        # 没挂策略就直接报错（step 必须由 policy 提供动作）
        if self.policy is None:
            raise RuntimeError('No policy set. Call set_policy() first.')

        # 策略根据当前 infos（观测）批量给出 num_envs 个动作
        actions = self.policy.get_action(self.infos)
        # 所有环境同步走一步，解包返回五元组
        (
            # 新观测（每环境的观测 + 附加信息）
            self.states,
            # 奖励数组 [num_envs]
            self.rewards,
            # 是否终止 [num_envs]（任务级成功/失败）
            self.terminateds,
            # 是否截断 [num_envs]（如达到步数上限）
            self.truncateds,
            # 新的 infos 字典，供下一步 policy/录制使用
            self.infos,
        ) = self.envs.step(actions)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset all environments to initial states.

        Args:
            seed: Random seed(s) for the environments.
            options: Additional options passed to the environment reset.
        """
        if self.policy is not None:
            # 有策略时先清策略内部状态（如 RNN hidden state / 历史缓存）
            self.policy.reset()
        # 所有并行环境一起 reset，拿回初始观测和 infos
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy: Policy) -> None:
        """Attach a policy to the world.

        Args:
            policy: The policy instance to use for determining actions.
        """
        # 挂载策略，之后 step() 才能跑
        self.policy = policy
        # 把环境交给策略（策略可能需要动作空间/环境引用）
        self.policy.set_env(self.envs)

        # 策略自带 seed 时，把它同步给策略（保证可复现）
        if hasattr(self.policy, 'seed') and self.policy.seed is not None:
            self.policy.set_seed(self.policy.seed)

    # ---- 段落标题：record_dataset 主录制循环（HDF5 落盘 + 断点续采） ----
    def record_dataset(
        self,
        dataset_name: str,
        episodes: int = 10,
        seed: int | None = None,
        cache_dir: os.PathLike | str | None = None,
        options: dict | None = None,
    ) -> None:
        """Records episodes from the environment into an HDF5 dataset.

        Args:
            dataset_name: Name of the dataset file (without extension).
            episodes: Total number of episodes to record.
            seed: Initial random seed.
            cache_dir: Directory to save the dataset. Defaults to standard cache.
            options: Reset options passed to environments.

        Raises:
            NotImplementedError: If history_size > 1.
        """
        # 禁止录制帧堆叠历史：HDF5 契约按单帧写，history>1 无法对齐
        if self._history_size > 1:
            raise NotImplementedError(
                'Frame history > 1 not supported for dataset recording.'
            )

        # 拼出数据集文件路径：缓存目录/<dataset_name>.h5
        path = Path(cache_dir or get_cache_dir()) / f'{dataset_name}.h5'
        # 递归创建父目录（已存在也不报错）
        path.parent.mkdir(parents=True, exist_ok=True)

        # terminated/truncated 标志先全部清零，作为本轮录制的初始状态
        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)

        # 每个并行环境一个 episode 缓冲：{列名: [step_0, step_1, ...]}，
        # episode 结束时整体搬进 HDF5（见 _handle_done_ep / _write_episode）
        episode_buffers = [defaultdict(list) for _ in range(self.num_envs)]

        # mode='a'：文件已存在则追加（断点续采），否则新建。
        # fs_strategy='page' 仅新建时生效：按 4MB 页扩展文件，
        # 避免可 resize 数据集在 SSD 上产生大段稀疏空洞
        h5_kwargs = {
            # 文件名（转成字符串给 h5py）
            'name': str(path),
            # 已存在 -> 'a' 追加续采；不存在 -> 'w' 新建
            'mode': 'a' if path.exists() else 'w',
            # 用最新 HDF5 格式特性（SWMMR 等需要）
            'libver': 'latest',
        }

        if not path.exists():  # creation only args
            # 新建时按 4MB 页预分配，避免稀疏空洞
            h5_kwargs.update(
                {'fs_strategy': 'page', 'fs_page_size': 4 * 1024 * 1024}
            )

        # 以上述参数打开（或创建）HDF5 文件，f 是文件句柄
        with h5py.File(**h5_kwargs) as f:
            # 打开单写多读模式
            f.swmr_mode = True  # avoid issue when killed
            # SWMR 允许写入期间被外部（读端/崩溃恢复）读取，
            # 配合每 episode flush，进程被杀也不会损坏已写数据

            # 续采逻辑：磁盘上已有 ep_len 说明之前采过，
            # 写指针从「最后一集末尾」继续，seed 顺延避免重复采样
            if 'ep_len' in f:
                # 已落盘的 episode 数（进度条起点）
                n_ep_recorded = f['ep_len'].shape[0]
                # 写指针 = 最后一集的全局起点 + 该集长度 = 全局轴下一个空位
                global_step_ptr = (
                    f['ep_offset'][-1] + f['ep_len'][-1]
                    if n_ep_recorded > 0
                    else 0
                )
                # 数据集列已经建好，不用再初始化
                initialized = True
                # seed 顺延 n_ep_recorded，避免续采重复采到相同轨迹
                seed = None if seed is None else (seed + n_ep_recorded)
                # 告知用户从第几集续采
                logging.info(
                    f'Resuming: {n_ep_recorded} episodes already on disk.'
                )
            else:
                # 全新文件：还没录过任何一集
                n_ep_recorded = 0
                # 全局写指针从 0 开始
                global_step_ptr = 0
                # 列还没建，等第一集结束再「懒初始化」
                initialized = False

            # 重置全部并行环境，拿初始观测（seed 已按续采偏移）
            self.reset(seed, options=options)
            # 推进 seed：本次 reset 消耗了 num_envs 个种子，下次单环境
            # reset 从 seed+num_envs 开始，保证每个环境种子不重复
            seed = None if seed is None else (seed + self.num_envs)
            # 把 t=0 的初始观测先存进各环境缓冲（episode 第一帧来自 reset）
            self._dump_step_data(episode_buffers)  # record initial state

            # 进度条：总数 episodes，已录 n_ep_recorded 起步
            with tqdm(
                total=episodes, initial=n_ep_recorded, desc='Recording'
            ) as pbar:
                # 录满 episodes 之前一直循环
                while n_ep_recorded < episodes:
                    # 所有并行环境同步走一步（策略产生动作）
                    self.step()
                    # 把这一步各环境的数据追加进各自 episode 缓冲
                    self._dump_step_data(episode_buffers)

                    # 逐个环境检查是否本步结束（terminated 或 truncated）
                    for i in range(self.num_envs):
                        # 第 i 个环境结束了这一集
                        if self.terminateds[i] or self.truncateds[i]:
                            # 取出该环境的完整 episode 数据并清空缓冲，
                            # 同时做 action 左移对齐、盖上 ep_idx 标记
                            finished_ep = self._handle_done_ep(
                                episode_buffers, i, n_ep_recorded
                            )

                            # lazy dataset initialization
                            # 第一集结束才用真实 shape/dtype 建 HDF5 列
                            if not initialized:
                                self._init_h5_datasets(f, finished_ep)
                                initialized = True

                            # contiguous writing
                            # 整集沿全局连续轴追加写入，返回本集帧数
                            steps_written = self._write_episode(
                                f, finished_ep, global_step_ptr
                            )
                            # 写指针前移：下一批数据接在本集末尾之后
                            global_step_ptr += steps_written
                            # 已完成集数 +1
                            n_ep_recorded += 1
                            # 进度条前进 1
                            pbar.update(1)

                            # 每集 flush 元数据，进程被杀也不丢已录数据
                            f.flush()  # flush metadata to avoid corruption

                            # 达到目标集数：跳出环境循环，退出 while
                            if n_ep_recorded >= episodes:
                                break

                            # reset terminated env and record initial state
                            # 只 reset 刚结束的第 i 个环境（autoreset 已禁用），
                            # seed 按已录集数推进避免与历史种子冲突
                            n_seed = (
                                None
                                if seed is None
                                else (seed + n_ep_recorded)
                            )
                            self._reset_single_env(i, n_seed, options)
                            # 新环境的 t=0 初始观测入缓冲，开新一集
                            self._dump_step_data(episode_buffers, env_idx=i)

        # 全部录完：打印累计帧数（= 全局轴总长度）
        logging.info(f'Recording complete. Total frames: {global_step_ptr}')

    def _init_h5_datasets(
        self, f: h5py.File, sample_episode: dict[str, list[Any]]
    ) -> None:
        """Initialize resizable HDF5 datasets based on the first episode.

        Args:
            f: The open HDF5 file handle.
            sample_episode: A dictionary containing data from a single episode,
                used to determine shapes and dtypes.
        """
        # ---- 段落标题：用第一集的真实数据懒创建 HDF5 可扩展列 ----
        # 遍历 episode 缓冲的每一列（列名 -> step 值列表）
        for key, data_list in sample_episode.items():
            # 元数据列 ep_len/ep_idx/policy 不在这里建（下面单独建 ep_*）
            if key in ['ep_len', 'ep_idx', 'policy']:
                continue

            # 列名里的 '/' 换成 '_'（HDF5 里 '/' 是层级分隔符）
            key = key.replace('/', '_')  # sanitize keys for h5

            # determine array shape and dtype from sample data
            # 取该列第一帧转 ndarray，用它推断单帧 shape 和 dtype
            sample_data = np.array(data_list[0])
            # 初始长度为 0，后面按集追加；帧维度 = sample_data.shape
            shape = (0,) + sample_data.shape
            maxshape = (None,) + sample_data.shape  # 第 0 维可无限 resize 追加

            # determine chunk size and compression
            # 压缩契约：多维列（pixels/qpos 等）按 100 行一个 chunk 走
            # Blosc(lz4)+shuffle；1D 标量列（step_idx 等）按 1000 行一个
            # chunk、不压缩。HDF5Dataset 按列读取，两种 chunk 大小都够用
            if sample_data.ndim >= 2:
                # 多维列：100 行一个 chunk（帧图像压缩收益高）
                chunks = (100,) + sample_data.shape
                # Blosc(lz4) 压缩 + shuffle（字节重排提升压缩率）
                compression = hdf5plugin.Blosc(
                    cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                )

            else:
                # 1D 标量列：1000 行一个 chunk，读整列更快
                chunks = (1000,) + sample_data.shape
                # 不压缩（标量本来就小）
                compression = None

            # 用首帧的 dtype 作为该列 dtype
            dtype = sample_data.dtype
            # 字符串类型（str_/bytes_）换成 HDF5 专用变长字符串 dtype
            if np.issubdtype(dtype, np.str_) or np.issubdtype(
                dtype, np.bytes_
            ):
                dtype = h5py.string_dtype()

            # 创建可 resize 的数据集（shape 从 0 起步）
            f.create_dataset(
                key,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )

        # index metadata
        # ep_offset/ep_len：episode 级索引，是 HDF5Dataset 定位切片的唯一
        # 依据（g_start = ep_offset[ep] + start），缺一不可读
        f.create_dataset(
            'ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64
        )
        f.create_dataset(
            'ep_len', shape=(0,), maxshape=(None,), dtype=np.int32
        )

        # per-step episode index
        # ep_idx：逐 step 记录所属 episode，供按 episode 聚合/评测使用
        f.create_dataset(
            'ep_idx',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1000,),
        )

    def _reset_single_env(
        self,
        env_idx: int,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset a single environment and update infos dict.

        Args:
            env_idx: Index of the environment to reset.
            seed: Random seed for this specific environment.
            options: Reset options.
        """
        # ---- 段落标题：_reset_single_env 单独 reset 某个并行环境 ----
        # 清空底层 autoreset 标记，防止 gym 又替我们偷偷 reset 别的环境
        self.envs.unwrapped._autoreset_envs = np.zeros(self.num_envs)
        # 直接对第 env_idx 个子环境 reset，用给定 seed/options 拿新初始 infos
        _, infos = self.envs.envs[env_idx].reset(seed=seed, options=options)

        # 把该环境的新 infos 合并回整体 infos（只覆盖已存在的键）
        for k, v in infos.items():
            # 键已在 self.infos 里 -> 就地替换第 env_idx 行的数据
            if k in self.infos:
                self.infos[k][env_idx] = v

        # 策略若支持按环境重置（如每个环境独立的 RNN 隐状态），通知它
        if self.policy is not None and hasattr(self.policy, 'reset_env'):
            self.policy.reset_env(env_idx)

    # ---- 段落标题：_apply_xy_neutral_eval_state 用 xy 坐标反推「中性状态」 ----
    # dataset_full 模式直接用数据集状态；xy_neutral 模式只给位置 xy，
    # 让环境方法结算出对应的 qpos/qvel/proprio/pixels 写回 init/goal step
    def _apply_xy_neutral_eval_state(
        self,
        init_step: dict[str, np.ndarray],
        goal_step: dict[str, np.ndarray],
        start_state_mode: str,
        goal_state_mode: str,
    ) -> None:
        # 先校验两个模式名是否合法（只能是 dataset_full / xy_neutral）
        _validate_eval_state_mode(start_state_mode, 'start_state_mode')
        _validate_eval_state_mode(goal_state_mode, 'goal_state_mode')

        # 两端都是 dataset_full：状态直接用数据集原值，无需任何处理
        if (
            start_state_mode == 'dataset_full'
            and goal_state_mode == 'dataset_full'
        ):
            return

        # 起点要 xy_neutral 但数据集没有 xy 列 -> 报错
        if start_state_mode == 'xy_neutral' and 'xy' not in init_step:
            raise ValueError(
                "start_state_mode='xy_neutral' requires dataset column 'xy'."
            )
        # 终点同理：必须有 goal_xy 列
        if goal_state_mode == 'xy_neutral' and 'goal_xy' not in goal_step:
            raise ValueError(
                "goal_state_mode='xy_neutral' requires dataset column 'xy'."
            )

        # 逐个并行环境处理（每个环境的初始状态各不相同）
        for env_idx, env in enumerate(self.envs.unwrapped.envs):
            # 优先找 settled_info_from_xy（结算后的静止状态）
            xy_info_fn = _find_env_method(env, 'settled_info_from_xy')
            # 没有则退回 neutral_info_from_xy
            if xy_info_fn is None:
                xy_info_fn = _find_env_method(env, 'neutral_info_from_xy')
            # 两个方法都没有 -> 该环境不支持 xy_neutral 模式
            if xy_info_fn is None:
                raise ValueError(
                    'xy_neutral eval state mode requires an environment with '
                    'settled_info_from_xy() or neutral_info_from_xy().'
                )

            # 起点需要中性化：拿数据集里的 xy 让环境算状态
            if start_state_mode == 'xy_neutral':
                # 用该环境起点的 xy 计算 info（需要像素时顺带渲染）
                info = xy_info_fn(
                    init_step['xy'][env_idx],
                    render_pixels='pixels' in init_step,
                )
                # 把算出的各字段写回 init_step 的对应列（空前缀 = 非 goal）
                self._write_neutral_step_info(init_step, env_idx, info, '')

            # 终点需要中性化：拿 goal_xy 计算目标处的状态
            if goal_state_mode == 'xy_neutral':
                # 目标点 xy -> 目标状态 info
                info = xy_info_fn(
                    goal_step['goal_xy'][env_idx],
                    render_pixels='goal' in goal_step,
                )
                # 写回 goal_step 的 goal_* 列（带 'goal_' 前缀）
                self._write_neutral_step_info(goal_step, env_idx, info, 'goal_')

    @staticmethod
    def _proprio_from_qpos_qvel(
        info: dict[str, np.ndarray],
        target_dim: int,
    ) -> np.ndarray | None:
        # 从 info 里取关节位置 qpos 与关节速度 qvel
        qpos = info.get('qpos')
        qvel = info.get('qvel')
        # 缺任意一个就无法拼 proprio，返回 None
        if qpos is None or qvel is None:
            return None

        # 两种候选拼法：完整 [qpos, qvel]；去掉 qpos 前两维（x,y 位姿）
        # 再拼 qvel（某些任务 proprio 不含全局位姿）
        candidates = (
            np.concatenate([qpos, qvel], axis=-1),
            np.concatenate([qpos[..., 2:], qvel], axis=-1),
        )
        # 逐个候选检查最后一维是否正好等于目标维度
        for candidate in candidates:
            if candidate.shape[-1] == target_dim:
                return candidate
        # 都不匹配 -> 返回 None，由调用方决定回退方案
        return None

    @classmethod
    def _match_proprio_dim(
        cls,
        value: np.ndarray,
        target_dim: int,
        info: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        # 已经是目标维度：原样返回
        if value.shape[-1] == target_dim:
            return value

        # 有 info 时，尝试用 qpos+qvel 现拼一个目标维度的 proprio
        if info is not None:
            candidate = cls._proprio_from_qpos_qvel(info, target_dim)
            if candidate is not None:
                # 候选可能多一个时间维：把前面的维都收缩到只剩最后一帧
                while candidate.ndim > value.ndim:
                    candidate = candidate[:, -1]
                # 返回维度对齐后的 proprio
                return candidate

        # 兜底1：若差的正好是前两维（全局 x,y），砍掉前 2 维
        if value.shape[-1] - 2 == target_dim:
            return value[..., 2:]

        # 兜底2：无法对齐，原样返回（保持现状，不报错）
        return value

    @classmethod
    def _write_neutral_step_info(
        cls,
        step: dict[str, np.ndarray],
        env_idx: int,
        neutral_info: dict[str, np.ndarray],
        prefix: str,
    ) -> None:
        # ---- 段落标题：把中性化算出的 info 写回 step dict 的对应列 ----
        # 映射表：step 列名 -> neutral_info 里的字段名
        # （prev_qpos/prev_qvel 复用 qpos/qvel 的值）
        key_pairs = (
            ('xy', 'xy'),
            ('qpos', 'qpos'),
            ('qvel', 'qvel'),
            ('prev_qpos', 'qpos'),
            ('prev_qvel', 'qvel'),
            ('proprio', 'proprio'),
            ('pixels', 'pixels'),
        )

        # 逐对检查哪些列需要写
        for step_key, info_key in key_pairs:
            # 目标键名：goal 前缀 + 列名；只有 pixels 特殊（goal 图像叫 'goal'）
            target_key = (
                'goal' if prefix == 'goal_' and step_key == 'pixels'
                else f'{prefix}{step_key}'
            )
            # step 里有该列 且 info 里有对应字段，才写入
            if target_key in step and info_key in neutral_info:
                # 取中性化得到的值
                value = neutral_info[info_key]
                # proprio 的维度可能和数据集不一致，先做维度对齐
                if step_key == 'proprio':
                    value = cls._match_proprio_dim(
                        value,
                        step[target_key].shape[-1],
                        neutral_info,
                    )
                # 只替换第 env_idx 个环境那一行，其他环境保持原值
                step[target_key][env_idx] = value

    # ---- 段落标题：_handle_done_ep 收尾一集：action 左移对齐 + 打 ep_idx ----
    def _handle_done_ep(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int,
        n_ep_recorded: int,
    ) -> dict[str, list[Any]]:
        """Prepare the episode buffer for writing.

        Args:
            tmp_buffer: List of dictionaries accumulating step data per env.
            env_idx: Index of the environment that finished an episode.
            n_ep_recorded: Number of episodes recorded so far.

        Returns:
            A dictionary containing the complete episode data.
        """
        # 取出第 env_idx 个环境正在积累的缓冲
        ep_buffer = tmp_buffer[env_idx]
        # 本集帧数 = step_idx 列的长度（第 0 帧来自 reset）
        ep_len = len(ep_buffer['step_idx'])

        # left-shift actions to align with observations i.e. (o_t, a_t).
        # Some policies expose action-side metadata, such as AntMaze explore
        # directions, that should be shifted with actions.
        # 【中文】采集时每个 step 先存「上一步动作」再存观测，所以 buffer 里
        # action 比观测晚一拍；这里把 action 整体左移一位，使落盘后第 t 行
        # 同时是 o_t 和作用于它的 a_t（(o_t, a_t) 配对），末位补 NaN。
        # 新采集器写 bruce 数据集时必须复刻同样的对齐，否则 H-JEPA 训练
        # 会把动作错位一帧
        # 需要左移的键：至少是 action；策略还可以声明随动作一起移的元数据键
        action_aligned_keys = ['action']
        # 有策略时把策略声明的 action_aligned_info_keys 并进来（如探索方向）
        if self.policy is not None:
            action_aligned_keys.extend(
                getattr(self.policy, 'action_aligned_info_keys', ())
            )
        # 逐个键做左移对齐
        for key in action_aligned_keys:
            # 缓冲里没有这一列（比如策略没产出该元数据）就跳过
            if key in ep_buffer:
                # 取该列的 step 值列表（原地修改）
                values = ep_buffer[key]
                # 长度与帧数一致：首帧是「上一集残留/初始动作」，弹出后
                # 末尾补同一个值占位（本质是把动作整体往左挪一位）
                if len(values) == ep_len:
                    nan = values.pop(0)
                    values.append(nan)
                # 长度比帧数少 1：说明首帧本来就没动作，直接补一帧 NaN
                elif len(values) == ep_len - 1 and values:
                    values.append(np.full_like(values[-1], np.nan))

        # Extract a copy and clear the temporary buffer
        # 深一层拷贝出本集数据（各列都是新 list）
        out = {k: list(v) for k, v in ep_buffer.items()}
        # 清空该环境缓冲，准备积累下一集
        ep_buffer.clear()
        # 复位该环境的结束标志，下一轮继续判断
        self.terminateds[env_idx] = False
        self.truncateds[env_idx] = False

        # Add episode index to all steps
        # 给本集每一帧都盖上同一集号（ep_idx 列，供按集聚合查询）
        out['ep_idx'] = [n_ep_recorded] * ep_len

        # 返回完整的一集数据
        return out

    # ---- 段落标题：_write_episode 把一整集沿全局连续轴追加写入 HDF5 ----
    def _write_episode(
        self, f: h5py.File, ep_data: dict[str, list[Any]], global_ptr: int
    ) -> int:
        """Write a single contiguous episode to the HDF5 file.

        Args:
            f: The open HDF5 file handle.
            ep_data: The episode data dictionary.
            global_ptr: The global step index where this episode starts.

        Returns:
            The length of the episode written.
        """
        # 本集帧数（各列对齐后长度一致）
        ep_len = len(ep_data['step_idx'])

        # append data to each dataset
        # 每列沿第 0 维 resize 后整段追加，保证各列共享同一条全局连续轴
        for key in ep_data:
            # 列名清洗：'/' -> '_'（与建列时的规则保持一致）
            h5_key = key.replace('/', '_')  # sanitize keys for h5
            # ep_len/policy 不作为数据列写（ep_len 是元数据、policy 是对象）
            if h5_key in ['ep_len', 'policy']:
                continue

            # 拿到该列的 HDF5 数据集对象
            ds = f[h5_key]
            # 当前行数 = 本集数据要插入的起始下标
            curr_size = ds.shape[0]
            # 把第 0 维扩长 ep_len 行（新行先为空占位）
            ds.resize(curr_size + ep_len, axis=0)
            # 把整集数据一次性写进新扩出来的区间
            ds[curr_size:] = np.array(ep_data[key])

        # update metadata
        # 记录本集的全局起点与长度；调用方负责累加 global_ptr，
        # 从而 ep_offset 永远等于前面所有 ep_len 之和
        meta_idx = f['ep_offset'].shape[0]
        # ep_offset / ep_len 各追加一个元素
        f['ep_offset'].resize(meta_idx + 1, axis=0)
        f['ep_len'].resize(meta_idx + 1, axis=0)

        # 本集在全局轴上的起始下标
        f['ep_offset'][meta_idx] = global_ptr
        # 本集帧数
        f['ep_len'][meta_idx] = ep_len

        # 返回帧数，供调用方推进写指针
        return ep_len

    def _dump_step_data(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int | None = None,
    ) -> None:
        """Append current step data to temporary episode buffers.

        Args:
            tmp_buffer: List of dictionaries accumulating step data.
            env_idx: Optional index to dump data for a single environment.
                If None, dumps for all environments.
        """
        # ---- 段落标题：_dump_step_data 把当前一步的 infos 追加进各环境缓冲 ----
        # env_idx=None -> 所有环境；否则只处理指定的那个环境
        env_indices = range(self.num_envs) if env_idx is None else [env_idx]

        # infos 的每一列（像素、qpos、action、step_idx...）
        for col, data in self.infos.items():
            # 下划线开头的内部元数据列不落盘（上面已有说明）
            if col.startswith('_'):
                continue  # 下划线前缀 = 内部元数据列，不落盘（如 _original_length）

            # normalize data shape and type
            # ndarray 且时间维被多包了一层 [num_envs, 1, ...] -> 挤掉第 1 维
            if isinstance(data, np.ndarray):
                data = (
                    np.squeeze(data, axis=1)
                    if data.ndim > 1 and data.shape[1] == 1
                    else data
                )
                # object 数组（每元素是变长数组）-> 拼成普通 list
                if data.dtype == object:
                    data = np.concatenate(data).tolist()

            # append to buffers
            # 把每个环境那一帧的值分别塞进它的缓冲
            for i in env_indices:
                # 是 ndarray 就 copy 一份（防止 infos 原地被下一步覆盖）
                env_data = (
                    data[i].copy()
                    if isinstance(data[i], np.ndarray)
                    else data[i]
                )
                # 第 i 个环境的该列追加这一帧
                tmp_buffer[i][col].append(env_data)

    # ---- 段落标题：evaluate 随机 rollout 评测策略成功率（支持断点续评） ----
    def evaluate(
        self,
        episodes: int = 10,
        eval_keys: list[str] | None = None,
        seed: int | None = None,
        options: dict | None = None,
        dump_every: int = -1,
    ) -> dict:
        """Evaluate the current policy over multiple episodes.

        Args:
            episodes: Number of episodes to evaluate.
            eval_keys: List of keys in `infos` to collect and return.
            seed: Random seed for evaluation.
            options: Reset options.
            dump_every: Interval to save intermediate results (for long evals).

        Returns:
            Dictionary containing success rates, seeds, and collected keys.
        """
        # options 缺省时用空 dict
        options = options or {}

        # 结果字典：先初始化计数、成功率、逐集成功标志、逐集种子
        results: dict = {
            'episode_count': 0,  # 已完成的评测集数（断点续评的进度）
            'success_rate': 0.0,  # 最后统一算的百分比成功率
            'episode_successes': np.zeros(episodes),  # 每集是否成功(0/1)
            'seeds': np.zeros(episodes, dtype=np.int32),  # 每集实际用的种子
        }

        # 调用方额外要求收集的 infos 键（如到达目标步数等），逐个开数组
        if eval_keys:
            for key in eval_keys:
                results[key] = np.zeros(episodes)

        # 本地维护 terminated/truncated 标志数组（shape=[num_envs]）
        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        # 每个并行环境当前负责第几集（初始 = 0..num_envs-1）
        episode_idx = np.arange(self.num_envs)
        # 全体环境 reset：seed 会按环境自动分流成 num_envs 个子种子
        self.reset(seed=seed, options=options)
        # root_seed：第二批及以后 reset 的种子基准（+num_envs 跳过首批）
        root_seed = seed + self.num_envs if seed is not None else None

        # 评测是否全部完成的总开关
        eval_done = False

        # determine "unique" hash for this eval run
        # 把评测配置打包（tuple 化保证 json 可序列化、顺序稳定）
        config = {
            'episodes': episodes,
            'eval_keys': tuple(sorted(eval_keys)) if eval_keys else None,
            'seed': seed,
            'options': tuple(sorted(options.items())) if options else None,
            'dump_every': dump_every,
        }

        # 配置 -> 排序 JSON 字符串（同配置必得同字符串）
        config_str = json.dumps(config, sort_keys=True)
        # SHA256 后取前 8 位十六进制作 run id：同配置共享同一个临时文件
        run_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        # 中间结果临时文件路径（评测可中断重跑）
        run_tmp_path = Path(f'eval_tmp_{run_hash}.npy')

        # load back intermediate results if file exists
        # 存在临时文件说明这次评测之前中断过 -> 加载续评
        if run_tmp_path.exists():
            # 读回 dict（allow_pickle 因为 np.save 存的是对象）
            tmp_results = np.load(run_tmp_path, allow_pickle=True).item()
            # 用已保存的进度覆盖本地 results
            results.update(tmp_results)

            # 已完成集数，从下一集号继续
            ep_count = results['episode_count']
            # 各环境的新集号 = ep_count .. ep_count+num_envs-1
            episode_idx = np.arange(ep_count, ep_count + self.num_envs)

            # reset seed where we left off
            # 种子也从断点处顺延，保证和上次跑的序列接上
            last_seed = seed + ep_count if seed is not None else None
            self.reset(seed=last_seed, options=options)

            # 打印续评信息
            logging.success(
                f'Found existing eval tmp file {run_tmp_path}, resuming from episode {ep_count}/{episodes}'
            )

        # 评测主循环：直到 eval_done 被置真
        while True:
            # 所有并行环境同步走一步（策略决策）
            self.step()

            # start new episode for done envs
            # 扫描每个环境，处理本步结束的 episode
            for i in range(self.num_envs):
                # 第 i 个环境 terminated（成功/失败）或 truncated（超步数）
                if self.terminateds[i] or self.truncateds[i]:
                    # record eval info
                    # 该环境当前负责的全局集号
                    ep_idx = episode_idx[i]
                    # 终止视为成功（成功定义 = terminated=True）
                    results['episode_successes'][ep_idx] = self.terminateds[i]
                    # 记录该环境实际使用的随机种子（从底层环境读）
                    results['seeds'][ep_idx] = self.envs.envs[
                        i
                    ].unwrapped.np_random_seed

                    # 收集调用方指定的附加指标（如 infos 里的到达步数）
                    if eval_keys:
                        for key in eval_keys:
                            # infos 里必须有这个键，否则直接断言失败
                            assert key in self.infos, (
                                f'key {key} not found in infos'
                            )
                            # 取第 i 个环境的该指标存进结果
                            results[key][ep_idx] = self.infos[key][i]

                    # determine new episode idx
                    # re-reset env with seed and options (no supported by auto-reset)
                    # 新一集的种子 = 根种子 + 已完成集数（不重复）
                    new_seed = (
                        root_seed + results['episode_count']
                        if seed is not None
                        else None
                    )
                    # 给该环境分配下一个全局集号（当前最大集号+1）
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    # 已完成集数 +1
                    results['episode_count'] += 1

                    # break if enough episodes evaluated
                    # 集数够了：置完成标志
                    if results['episode_count'] >= episodes:
                        eval_done = True
                        # 删除断点续评用的临时文件（不再需要）
                        if run_tmp_path.exists():
                            logging.info(
                                f'Eval done, deleting tmp file {run_tmp_path}'
                            )
                            os.remove(run_tmp_path)
                        # 跳出环境扫描 for（外层 while 见 eval_done 退出）
                        break

                    # dump temporary results in a file
                    # 每 dump_every 集把中间结果落盘一次（长评测可中断）
                    if dump_every > 0 and (
                        results['episode_count'] % dump_every == 0
                    ):
                        # 保存 results dict 到临时 npy
                        np.save(run_tmp_path, results)
                        # 打印保存进度
                        logging.success(
                            f'Dumped intermediate eval results to {run_tmp_path} ({results["episode_count"]}/{episodes})'
                        )
                    # 清 autoreset 标记，确保我们手动 reset 不被干扰
                    self.envs.unwrapped._autoreset_envs = np.zeros(
                        (self.num_envs,)
                    )
                    # 手动 reset 第 i 个环境（autoreset 禁用，需带种子/options）
                    _, infos = self.envs.envs[i].reset(
                        seed=new_seed, options=options
                    )

                    # 把新初始 infos 合并回整体 infos（只更新已有的键）
                    for k, v in infos.items():
                        # 新键不在跟踪列表里就跳过
                        if k not in self.infos:
                            continue
                        # Convert to array and extract scalar to preserve dtype
                        # 转成 ndarray 写回第 i 行，保持 dtype 一致
                        self.infos[k][i] = np.asarray(v)

            # 本轮评测已全部完成 -> 退出 while
            if eval_done:
                break

        # compute success rate
        # 成功率 = 成功集数 / 总集数 * 100（百分比）
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / episodes * 100.0
        )

        # 断言：计数必须正好等于目标集数（防漏算/重复）
        assert results['episode_count'] == episodes, (
            f'episode_count {results["episode_count"]} != episodes {episodes}'
        )

        # 断言：种子必须全部互不相同（防 seed 冲突导致轨迹重复）
        assert np.unique(results['seeds']).shape[0] == episodes, (
            'Some episode seeds are identical!'
        )

        # 返回完整结果 dict
        return results

    # ---- 段落标题：evaluate_from_dataset 从数据集状态起步评测 ----
    # 流程：加载/采样轨迹 chunk -> 提取每环境起点与目标 -> reset 后把
    # 数据集状态回写进 infos -> 预算内 rollout 统计成功率与到达耗时
    def evaluate_from_dataset(
        self,
        dataset: Any | None,
        episodes_idx: Sequence[int] | None,
        start_steps: Sequence[int] | None,
        goal_offset_steps: int,
        eval_budget: int,
        callables: list[dict] | None = None,
        dump_eval_trajs_path: str | Path | None = None,
        load_eval_trajs_path: str | Path | dict | None = None,
        process: dict[str, Any] | None = None,
        start_state_mode: str = 'dataset_full',
        goal_state_mode: str = 'dataset_full',
    ) -> dict:
        """Evaluate the policy starting from states sampled from a dataset.

        Args:
            dataset: The source dataset to sample initial states/goals from.
            episodes_idx: Indices of episodes to sample from.
            start_steps: Step indices within those episodes to start from.
            goal_offset_steps: Number of steps ahead to look for the goal.
            eval_budget: Maximum steps allowed for the agent to reach the goal.
            callables: Optional list of method calls to setup the env.

        Returns:
            Dictionary containing success rates and other metrics.

        Raises:
            ValueError: If input sequence lengths mismatch or don't match num_envs.
        """
        # 断言：环境步数上限必须 >= 目标前瞻步数（否则 goal 永远够不到）
        assert (
            self.envs.envs[0].spec.max_episode_steps is None
            or self.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
        ), 'env max_episode_steps must be greater than eval_budget'

        # ==== 分支1：从磁盘/调用方预载的轨迹文件加载评测初始状态 ====
        if load_eval_trajs_path is not None:
            # a dict is a payload preloaded by the caller (chunked eval)
            # 传进 dict 视为已预载 payload；否则按路径用 torch.load 读文件
            payload = (
                load_eval_trajs_path
                if isinstance(load_eval_trajs_path, dict)
                else torch.load(
                    Path(load_eval_trajs_path), map_location='cpu', weights_only=False
                )
            )
            # 每环境一条轨迹的 dict 列表（record 时 dump 的）
            data = payload['data']
            # 这批数据包含哪些列
            columns = payload['columns']
            # 各轨迹来自数据集的哪一集（没有则留空数组）
            ep_idx_arr = np.array(payload.get('episodes_idx', []), dtype=np.int64)
            # 没记录集号就按 0..N-1 补全
            if ep_idx_arr.size == 0:
                ep_idx_arr = np.arange(len(data), dtype=np.int64)

            # 轨迹条数少于并行环境数：没法一条环境喂一条 -> 报错
            if len(data) < self.num_envs:
                raise ValueError(
                    f'Loaded eval trajectories contain {len(data)} episodes, '
                    f'fewer than num_envs={self.num_envs}. Decrease eval.num_eval.'
                )

            # 轨迹条数多于环境数：只取前 num_envs 条
            if len(data) > self.num_envs:
                logging.info(
                    f'Loaded {len(data)} episodes; using the first {self.num_envs} '
                    'to match num_envs.'
                )
                # 截断数据与对应集号到 num_envs 条
                data = data[:self.num_envs]
                ep_idx_arr = ep_idx_arr[:self.num_envs]

            # 各轨迹长度不一 -> 统一 pad 到等长（并记录原始长度）
            data = _pad_loaded_eval_trajectories(data, columns)
        # ==== 分支2：直接从数据集按 (集号, 起始步) 现场切 chunk ====
        else:
            # 没传数据集直接报错
            if dataset is None:
                raise ValueError('dataset must be provided when not loading eval trajectories')
            # 起评的集号与起始步必须给全
            if episodes_idx is None or start_steps is None:
                raise ValueError('episodes_idx and start_steps must be provided when not loading eval trajectories')

            # 集号/起始步转成 ndarray 方便向量化运算
            ep_idx_arr = np.array(episodes_idx)
            start_steps_arr = np.array(start_steps)
            # 终止步 = 起始步 + 目标前瞻步数（chunk 里最后一步即目标）
            end_steps = start_steps_arr + goal_offset_steps

            # 集号与起始步必须一一对应（等长）
            if not (len(ep_idx_arr) == len(start_steps_arr)):
                raise ValueError(
                    'episodes_idx and start_steps must have the same length'
                )

            # 任务数必须正好等于并行环境数（一环境一条）
            if len(ep_idx_arr) != self.num_envs:
                raise ValueError(
                    'Number of episodes to evaluate must match number of envs'
                )

            # 从数据集批量切出每个环境的轨迹 chunk（含起点到目标帧）
            data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
            # 该数据集的全部列名
            columns = dataset.column_names

            # 若指定了 dump 路径：只把 chunk 存成文件就返回（分块评测模式）
            if dump_eval_trajs_path is not None:
                # 目标路径
                dump_path = Path(dump_eval_trajs_path)
                # 递归创建父目录
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                # 组装可复现的 payload（数据 + 元信息）
                payload = {
                    'data': data,  # 轨迹 chunk 本体
                    'columns': list(columns),  # 列名
                    'episodes_idx': ep_idx_arr.tolist(),  # 来源集号
                    'start_steps': start_steps_arr.tolist(),  # 起始步
                    'goal_offset_steps': int(goal_offset_steps),  # 目标前瞻
                    'eval_budget': int(eval_budget),  # 评测步数预算
                    'process': process,  # 进程/分块信息（由调用方填）
                }
                # torch.save 序列化落盘（后面 load 分支原样读回）
                torch.save(payload, dump_path)
                # 打印 dump 成功
                logging.success(f'Dumped eval trajectories to {dump_path}')
                # 提前返回：这个调用只负责导出，不执行评测
                return {
                    'dumped_eval_trajs_path': str(dump_path),
                    'num_episodes': int(len(ep_idx_arr)),
                }

        # ---- 段落标题：从 chunk 中抽出每个环境的「起点帧」与「目标帧」 ----
        # keep relevant part of the chunk
        # 每列一个 list，按环境顺序收集该列的第 0 帧（起点）
        init_step_per_env: dict[str, list[Any]] = defaultdict(list)
        # 同上，收集每列的最后一帧（目标/终点）
        goal_step_per_env: dict[str, list[Any]] = defaultdict(list)

        # 遍历每条轨迹（第 i 条将喂给第 i 个并行环境）
        for i, ep in enumerate(data):
            # 遍历数据集的每一列
            for col in columns:
                # goal* 列在下面单独归入 goal_step，这里只处理普通列
                if col.startswith('goal'):
                    continue
                # 像素列目前是 (T, C, H, W)：把通道维挪到最后 (T, H, W, C)
                if col.startswith('pixels'):
                    # permute channel to be last
                    ep[col] = ep[col].permute(0, 2, 3, 1)

                # 非张量/数组的列（字符串等）暂不参与回放
                if not isinstance(ep[col], (torch.Tensor | np.ndarray)):
                    continue

                # 取该列时间轴第 0 帧 = 起点状态
                init_data = ep[col][0]
                # 最后一帧 = 目标状态
                goal_data = ep[col][-1]

                # TODO handle that better
                # 帧不是数组类型（如 python 标量）就跳过并警告
                if not isinstance(init_data, (np.ndarray | torch.Tensor)):
                    logging.warning(
                        f'Data type {type(init_data)} for column {col} not supported, yet skipping conversion'
                    )
                    continue

                # torch 张量转 numpy（环境回放吃 ndarray）
                init_data = (
                    init_data.numpy()
                    if isinstance(init_data, torch.Tensor)
                    else init_data
                )
                # 目标帧同理转 numpy
                goal_data = (
                    goal_data.numpy()
                    if isinstance(goal_data, torch.Tensor)
                    else goal_data
                )

                # 起点帧按环境序 append
                init_step_per_env[col].append(init_data)
                # 目标帧按环境序 append
                goal_step_per_env[col].append(goal_data)

        # 每列把 [env0, env1, ...] stack 成 (num_envs, ...) 的数组
        init_step = {
            k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()
        }

        # 起点的 step_idx 保留为 dataset_step_idx（数据集内原始步号），
        # step_idx 本身重置为 0（评测从第 0 步重新计数）
        if 'step_idx' in init_step:
            init_step['dataset_step_idx'] = init_step['step_idx'].copy()
            init_step['step_idx'] = np.zeros_like(init_step['step_idx'])

        # 目标列统一改名：pixels -> goal，其余 -> goal_<列名>
        goal_step = {}
        for key, value in goal_step_per_env.items():
            # 像素的目标帧键就叫 'goal'
            key = 'goal' if key == 'pixels' else f'goal_{key}'
            # 同样 stack 成 (num_envs, ...)
            goal_step[key] = np.stack(value)

        # get dataset info
        # 数据集里每集的 reset 种子（有就用它 reset，复现采集时的初始条件）
        seeds = init_step.get('seed')
        # get dataset variation
        # 数据集里记录的场景变体（variation.xxx 列）
        vkey = 'variation.'
        # 把 'variation.foo' 形式的键剥成 'foo'，值仍是各环境数组
        variations_dict = {
            k.removeprefix(vkey): v
            for k, v in init_step.items()
            if k.startswith(vkey)
        }

        # 每个环境一个 options dict 的列表
        options = [{} for _ in range(self.num_envs)]

        # 有变体信息：写进各环境的 reset options 以还原场景
        if len(variations_dict) > 0:
            for i in range(self.num_envs):
                # 该环境要加载哪些变体名
                options[i]['variation'] = list(variations_dict.keys())
                # 该环境每个变体的具体取值
                options[i]['variation_values'] = {
                    k: v[i] for k, v in variations_dict.items()
                }

        # 用数据集种子 + 变体选项 reset 全部环境，建立「数据集原初始条件」
        self.reset(seed=seeds, options=options)  # set seeds for all envs
        # 若选了 xy_neutral 模式：就地把 init/goal 状态换成中性化状态
        self._apply_xy_neutral_eval_state(
            init_step,
            goal_step,
            start_state_mode,
            goal_state_mode,
        )
        # 把 goal_* 列也并进 init_step（方便统一 broadcast 与回写 infos）
        init_step.update(deepcopy(goal_step))

        # ---- 段落标题：用 callables / goal 信息把环境精确设成数据集状态 ----
        # apply callable list (e.g used for set initial position if not access to seed)
        # callables 为空时用空列表（调用方可省略）
        callables = callables or []
        # 逐个并行环境执行状态设置
        for i, env in enumerate(self.envs.unwrapped.envs):
            # 拿解包后的真实环境对象（报错信息里显示用）
            env_unwrapped = env.unwrapped

            # 特判：若环境有「设置关门索引」的私有方法且数据集里有该列
            set_closed_door = _find_env_method(env, '_set_closed_door_idx')
            if set_closed_door is not None and 'closed_door_idx' in init_step:
                # 用数据集记录的关门位置复原场景（深拷贝防改动原数据）
                set_closed_door(deepcopy(init_step['closed_door_idx'][i]))

            # 逐条执行调用方声明的方法调用规格
            for spec in callables:
                # 规格里声明要调用的环境方法名
                method_name = spec['method']
                # 沿 wrapper 链查找该方法
                method = _find_env_method(env, method_name)
                # 环境没这个方法 -> 警告并跳过该条
                if method is None:
                    logging.warning(
                        f'Env {env} has no method {method_name}, skipping callable'
                    )
                    continue

                # 参数表：优先取 spec['args']，否则把整个 spec 当参数表
                args = spec.get('args', spec)

                # prepare args
                # 组装实参字典
                prepared_args = {}
                for args_name, args_data in args.items():
                    # 参数值（可能是个数据集列名占位）
                    value = args_data.get('value', None)
                    # 该参数是否表示「去数据集取这个列」（默认 True）
                    is_in_datset = args_data.get('in_dataset', True)

                    if is_in_datset:
                        # 是列名但数据集没有这一列 -> 警告并跳过本条调用
                        if value not in init_step:
                            logging.warning(
                                f'Col {value} not found in dataset, skipping callable for env {env_unwrapped}'
                            )
                            continue
                        # 从 init_step 取该环境这一帧的值作实参（深拷贝）
                        prepared_args[args_name] = deepcopy(
                            init_step[value][i]
                        )
                    else:
                        # 不依赖数据集：直接用规格里写的字面值
                        prepared_args[args_name] = args_data.get('value')

                # call method with prepared args
                # 以关键字参数调用环境方法（如 set_qpos(...)）
                method(**prepared_args)

            # 把目标状态喂给环境（供奖励/成功判断/可视化用）
            set_goal_info = _find_env_method(env, 'set_current_goal_info')
            # 环境支持设置目标信息
            if set_goal_info is not None:
                # 收集数据集中存在的 goal 相关列
                goal_info = {}
                for key in (
                    'goal',
                    'goal_xy',
                    'goal_qpos',
                    'goal_qvel',
                    'goal_proprio',
                ):
                    # 该 goal 列存在就取第 i 环境那份（深拷贝）
                    if key in init_step:
                        goal_info[key] = deepcopy(init_step[key][i])
                # 至少有一个 goal 列才调用
                if goal_info:
                    set_goal_info(**goal_info)

        # 结果容器：先放成功率、逐集成功标志（长度=任务数）、种子
        results: dict = {
            'success_rate': 0.0,  # 最后算的百分比成功率
            'episode_successes': np.zeros(len(ep_idx_arr)),  # 每任务是否成功
            'seeds': seeds,  # 数据集里的种子（或 None）
        }

        # expend all data to the right shape (x, y, (original_shape))
        # 当前 infos 里像素的前两维（num_envs, 历史帧数）作为广播目标前缀
        shape_prefix = self.infos['pixels'].shape[:2]

        # TODO get the data from the previous step in the dataset for history
        # 起点：在第 1 维插一个长度 1 的历史帧轴，再广播成 (num_envs, H, ...)
        # （H>1 时同一帧被复制满历史槽，等将来用数据集前帧替换）
        init_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in init_step.items()
        }
        # 目标做同样的广播，形状与 infos 对齐
        goal_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in goal_step.items()
        }

        # update the reset with our new init and goal infos
        # 把数据集起点状态盖进 self.infos（policy 第一步看到的就是它）
        self.infos.update(deepcopy(init_step))
        # 目标状态也盖进去（goal/goal_xy... 供策略读取）
        self.infos.update(deepcopy(goal_step))

        # 一致性自检：goal 图像必须与刚写入的一致
        if 'goal' in goal_step and 'goal' in self.infos:
            assert np.allclose(self.infos['goal'], goal_step['goal']), (
                'Goal info does not match'
            )

        # 每条轨迹 pad 前的真实长度（加载分支记的 _original_length）
        original_lengths = np.array(
            [
                # 没有该键就用 pixels 帧数兜底
                int(ep.get(ORIGINAL_LENGTH_KEY, ep['pixels'].shape[0]))
                for ep in data
            ],
            dtype=np.int64,
        )
        # 每环境「首次成功的墙钟耗时」初值 NaN（没成功就一直是 NaN）
        wall_clock_to_success = np.full(self.num_envs, np.nan, dtype=np.float64)
        # 每环境「首次成功用了几步」初值 NaN
        steps_to_success = np.full(self.num_envs, np.nan, dtype=np.float64)

        # 评测计时起点
        eval_start_time = time.perf_counter()
        # 评测主循环：最多走 eval_budget 步
        for i in range(eval_budget):
            # 每步都刷新 goal 相关 infos（防止环境 step 内部覆盖它们）
            self.infos.update(deepcopy(goal_step))
            # 所有并行环境同步走一步
            self.step()
            # 新成功 = 本步 terminated 且此前还没成功过（只记第一次）
            newly_successful = np.logical_and(
                self.terminateds, np.isnan(wall_clock_to_success)
            )
            # 记录首次成功的墙钟耗时（从评测开始算）
            wall_clock_to_success[newly_successful] = (
                time.perf_counter() - eval_start_time
            )
            # 记录首次成功时的步数（i 从 0 起，故 +1）
            steps_to_success[newly_successful] = i + 1

            # 成功标志做或累计：成功过就保持 True（不被后续覆盖）
            results['episode_successes'] = np.logical_or(
                results['episode_successes'], self.terminateds
            )
            # for auto-reset
            # 禁用 autoreset：不让 gym 在 terminated 后自动重置，
            # 评测只要统计成功，不做新一轮 rollout
            self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))

        # 实际评测的任务数
        n_episodes = len(ep_idx_arr)

        # compute success rate
        # 成功率 = 成功任务数 / 任务总数 * 100
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / n_episodes * 100.0
        )
        # 附上首次成功耗时（秒）与步数，供上层分析
        results['wall_clock_to_success'] = wall_clock_to_success
        results['steps_to_success'] = steps_to_success
        # 附上加载轨迹的原始长度（用于对齐/过滤 pad 帧）
        results['loaded_eval_original_lengths'] = original_lengths
        # 只取成功任务的步数（排除 NaN）
        successful_steps = steps_to_success[np.isfinite(steps_to_success)]
        # 成功任务的平均到达步数；一个都没成功则 NaN
        results['steps_to_success_success_only'] = (
            float(np.mean(successful_steps))
            if successful_steps.size > 0
            else float('nan')
        )

        # 有种子时断言互不相同（防重复采样）
        if results['seeds'] is not None:
            assert np.unique(results['seeds']).shape[0] == n_episodes, (
                'Some episode seeds are identical!'
            )

        # 返回全部评测指标
        return results
