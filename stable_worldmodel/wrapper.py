# 从标准库 copy 导入 deepcopy：深拷贝，用于复制观测数组，防止外部修改污染内部缓存
from copy import deepcopy
# 导入 re 正则模块：用来按"模式"校验 info 字典里是否包含必需的键
import re
# 导入 time 模块：用于记录渲染像素等操作的耗时（墙钟时间）
import time
# 从 collections 导入 deque 双端队列：frame stack 的环形缓冲区（自动丢弃最旧元素）
from collections import deque
# 从 collections.abc 导入类型标注 Callable（可调用对象）与 Iterable（可迭代对象）
from collections.abc import Callable, Iterable
# 从 typing 导入类型标注 Any（任意类型）与 Sequence（序列类型）
from typing import Any, Sequence

# 导入 Gymnasium 强化学习库：本文件所有 Wrapper 都基于它的 gym.Wrapper 接口
import gymnasium as gym
# 导入 NumPy：图像数组、掩码、np.nan 等数值操作都用它
import numpy as np
# 导入 PyTorch：部分观测/堆叠数据是 Tensor，需要与 ndarray 分开处理
import torch
# 导入 is_space_dtype_shape_equiv：判断两个空间的 dtype 与 shape 是否等价（VariationWrapper 校验用）
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
# 导入 VectorWrapper：向量化环境的包装器基类，VariationWrapper 继承自它
from gymnasium.vector import VectorWrapper
# 从 gymnasium.vector.utils 导入两个批量空间工具：
#   batch_space —— 把单个子环境的空间复制扩展成 (num_envs, ...) 的批量空间
#   batch_differing_spaces —— 当每个子环境空间不同时，把它们组合成一个带掩码的批量空间
from gymnasium.vector.utils import (
    batch_differing_spaces,
    batch_space,
)

# 从本仓库 utils 导入 get_in：按路径（如 'a.b' 拆开的键列表）深入嵌套空间/字典取值
from stable_worldmodel.utils import get_in


# ---------------------------------------------------------------------------
# 段落标题：校验类 Wrapper —— 只做检查、不改变数据，位于包装链中层用于尽早暴露错误
# ---------------------------------------------------------------------------
class EnsureInfoKeysWrapper(gym.Wrapper):
    """Validates that required keys are present in the info dict."""

    def __init__(self, env: gym.Env, required_keys: Iterable[str]):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            required_keys: Iterable of regex patterns that must match keys in info.
        """
        # 调用父类 gym.Wrapper 的构造函数：把 env 存到 self.env，建立转发关系
        super().__init__(env)
        # 初始化一个空列表，用来存放编译好的正则模式对象
        self._patterns: list[re.Pattern] = []
        # 遍历调用方要求的每一个"必需键"（其实是正则表达式字符串）
        for k in required_keys:
            # 把每个正则字符串编译成 Pattern 对象存入列表，编译一次、多次复用
            self._patterns.append(re.compile(k))

    def _check(self, info: dict, where: str) -> None:
        """Check if all required patterns have at least one match in info.

        Args:
            info: The info dictionary to check.
            where: String indicating where the check is performed (e.g., "reset").

        Raises:
            RuntimeError: If any required pattern is missing from info.
        """
        # 取出 info 字典的全部键名，转成普通 list 方便后续多次遍历
        keys = list(info.keys())
        # 列表推导：收集所有"没有任何键能 fullmatch 它"的模式（即缺失的必需键）
        missing = [
            # 记录缺失模式的原始字符串（p.pattern），便于报错时人类可读
            p.pattern
            # 对每个编译好的模式 p 进行判断
            for p in self._patterns
            # 只要存在任意一个键 k 与模式 p 完全匹配，就不算缺失；一个都没有才算缺失
            if not any(p.fullmatch(k) for k in keys)
        ]
        # 如果存在缺失的必需键
        if missing:
            # 抛出运行时错误，同时报告：在哪里出错、缺了哪些模式、实际有哪些键
            raise RuntimeError(
                f'{where}: required info keys missing (patterns with no match): {missing}. Present keys: {keys}'
            )

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform environment step and validate info keys.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        # 调用被包装环境的 step，得到标准五元组：观测、奖励、终止、截断、info
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 校验 info 中包含全部必需键，否则直接抛异常（where 标记来源是 step）
        self._check(info, 'step()')
        # 原样返回五元组，本 Wrapper 不修改任何数据
        return obs, reward, terminated, truncated, info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and validate info keys.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        # 调用被包装环境的 reset，得到 (初始观测, info 字典)
        obs, info = self.env.reset(*args, **kwargs)
        # 校验 reset 返回的 info 同样包含必需键（where 标记来源是 reset）
        self._check(info, 'reset()')
        # 原样返回观测与 info
        return obs, info


class EnsureImageShape(gym.Wrapper):
    """Validates that an image in the info dict has the expected spatial dimensions."""

    def __init__(
        self, env: gym.Env, image_key: str, image_shape: tuple[int, int]
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            image_key: Key in info dict containing the image.
            image_shape: Expected (height, width) of the image.
        """
        # 调用父类构造，建立对内层 env 的引用
        super().__init__(env)
        # 记录 info 中存放图像数据的键名，例如 'pixels' 或 'goal'
        self.image_key = image_key
        # 记录期望的空间尺寸 (height, width)，后续每次 step/reset 都会比对
        self.image_shape = image_shape  # (height, width)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and validate image shape.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If image shape does not match expected shape.
        """
        # 向内层环境推进一步，拿到标准五元组
        obs, reward, terminated, truncated, info = self.env.step(action)
        # shape[:-1] 去掉最后一维（通常是通道维 C），只比较 (H, W) 是否符合期望
        if info[self.image_key].shape[:-1] != self.image_shape:
            # 不匹配就报错，提示实际形状与期望形状
            raise RuntimeError(
                f'Image shape {info[self.image_key].shape} should be {self.image_shape}'
            )
        # 形状正确，原样返回
        return obs, reward, terminated, truncated, info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset and validate image shape.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.

        Raises:
            RuntimeError: If image shape does not match expected shape.
        """
        # 重置内层环境，拿到初始观测与 info
        obs, info = self.env.reset(*args, **kwargs)
        # 同样比对图像的空间维度 (H, W)
        if info[self.image_key].shape[:-1] != self.image_shape:
            # 不一致则抛出运行时错误
            raise RuntimeError(
                f'Image shape {info[self.image_key].shape} should be {self.image_shape}'
            )
        # 一致则原样返回
        return obs, info


class EnsureGoalInfoWrapper(gym.Wrapper):
    """Validates that 'goal' key is present in info dict."""

    def __init__(
        self, env: gym.Env, check_reset: bool, check_step: bool = False
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            check_reset: Whether to check 'goal' presence on reset.
            check_step: Whether to check 'goal' presence on each step.
        """
        # 调用父类构造，保持 Wrapper 链的连接
        super().__init__(env)
        # 记录是否在 reset 时检查 info 中存在 'goal' 键
        self.check_reset = check_reset
        # 记录是否在每次 step 时也检查 'goal' 键（默认不检查，因为 goal 通常只在 reset 给出）
        self.check_step = check_step

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset and validate goal presence.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.

        Raises:
            RuntimeError: If 'goal' is missing and check_reset is True.
        """
        # 重置内层环境
        obs, info = self.env.reset(*args, **kwargs)
        # 仅当开关打开且 info 里没有 'goal' 时才报错
        if self.check_reset and 'goal' not in info:
            # 提示调用方：reset() 返回的 info 必须带 'goal'
            raise RuntimeError(
                "The info dict returned by reset() must contain the key 'goal'."
            )
        # 检查通过，原样返回
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and validate goal presence.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If 'goal' is missing and check_step is True.
        """
        # 向内层环境推进一步
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 仅当 check_step 打开且 info 缺少 'goal' 时才报错
        if self.check_step and 'goal' not in info:
            # 提示 step() 的 info 也必须带 'goal'
            raise RuntimeError(
                "The info dict returned by step() must contain the key 'goal'."
            )
        # 检查通过，原样返回五元组
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 段落标题：EverythingToInfoWrapper —— 把观测/奖励/终止/动作等全部"搬进" info，
#           统一数据出口：下游只需读 info 就能拿到一次转移的全部信息
# ---------------------------------------------------------------------------
class EverythingToInfoWrapper(gym.Wrapper):
    """Moves all transition information into the info dict."""

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
        """
        # 调用父类构造，连接内层环境
        super().__init__(env)
        # 需要被监控的 variation（域随机化参数）键名列表，reset 时可由 options 指定
        self._variations_watch: Sequence[str] = []
        # 当前 episode 内的步数计数器，每次 step 加一、每次 reset 清零
        self._step_counter = 0
        # 当前 episode 的唯一 ID，reset 时重新生成
        self._id = 0

    def _gen_id(self) -> int:
        """Generate a random unique identifier for the current episode.

        Returns:
            A random 64-bit integer.
        """
        # 取 int64 的最大值作为随机 ID 的上界（不含）
        max_int = np.iinfo(np.int64).max
        # 取被包装环境自带的随机数生成器（Gymnasium 保证可复现性）
        rng = self.env.unwrapped.np_random
        # 新版 numpy Generator 用 integers()，旧版 RandomState 用 randint()，兼容两者
        return int(
            rng.integers(0, max_int)
            if hasattr(rng, 'integers')
            else rng.randint(0, max_int)
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and move all data to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        # 新 episode 开始，步数计数器归零
        self._step_counter = 0
        # 重置内层环境，拿到原始观测 obs 和内层 info
        obs, info = self.env.reset(*args, **kwargs)
        # 如果 obs 不是字典（裸数组），把它包装成 {'observation': obs} 统一接口
        if not isinstance(obs, dict):
            _obs = {'observation': obs}
        # 本来就是字典（多模态观测，如 image + state），直接使用
        else:
            _obs = obs

        # 把观测字典的每一个键值对搬进 info（断言保证键不与内层已有键冲突）
        for key, val in _obs.items():
            # 若键已存在于 info 中说明数据流混乱，直接断言失败
            assert key not in info
            # 把观测内容复制到 info，下游统一从 info 读取
            info[key] = val

        # ---- 段落：为 reset 填充"伪转移"字段，使 info 结构与 step 时完全一致 ----
        # 断言 info 尚无 reward 键（reset 时并没有奖励）
        assert 'reward' not in info
        # 用 NaN 表示"本次没有奖励"，保持键存在、类型为浮点
        info['reward'] = np.nan
        # 断言 info 尚无 terminated 键
        assert 'terminated' not in info
        # reset 视为既未终止也未截断，填 False 占位
        info['terminated'] = False
        # 断言 info 尚无 truncated 键
        assert 'truncated' not in info
        # 同样填 False 占位
        info['truncated'] = False
        # 断言 info 尚无 action 键
        assert 'action' not in info
        # 先按动作空间采样一个"形状正确"的占位动作（稍后会被替换成 NaN）
        info['action'] = self.env.action_space.sample()
        # 断言 info 尚无 step_idx 键
        assert 'step_idx' not in info
        # 记录当前步数（reset 后为 0）
        info['step_idx'] = self._step_counter
        # 断言 info 尚无 id 键
        assert 'id' not in info
        # 为本 episode 生成一个随机唯一 ID
        self._id = self._gen_id()
        # 把该 ID 写入 info，之后每一步都复用同一个 ID 以标识"同一局"
        info['id'] = self._id

        # add all variations to info if needed
        # 读取 reset 的 options 参数（可能没传，取空字典兜底）
        options = kwargs.get('options') or {}

        # 如果 options 中指定了要监控的 variation 列表
        if 'variation' in options:
            # 取出 variation 选项，应为键名列表，如 ['color', 'size']
            var_opt = options['variation']
            # 断言它必须是 list 或 tuple，否则给出可读的报错信息
            assert isinstance(var_opt, list | tuple), (
                'variation option must be a list or tuple containing variation names to sample, found: '
                f'{type(var_opt)}'
            )
            # 特殊写法 ['all']：表示监控该环境的全部 variation
            if len(var_opt) == 1 and var_opt[0] == 'all':
                # 从底层环境的 variation_space 取出所有参数名
                self._variations_watch = (
                    self.env.unwrapped.variation_space.names()
                )
            # 否则按调用方给定的键名列表来监控
            else:
                self._variations_watch = var_opt

        # 把被监控的每个 variation 的当前取值写进 info
        for key in self._variations_watch:
            # info 中的键统一加 'variation.' 前缀，避免与其他键冲突
            var_key = f'variation.{key}'
            # 断言键不重复
            assert var_key not in info
            # 按 '.' 拆分键名，用 get_in 深入嵌套的 variation_space 找到对应子空间
            subvar_space = get_in(
                self.env.unwrapped.variation_space, key.split('.')
            )
            # 取该子空间"当前采样到的值"写入 info，供下游记录域随机化配置
            info[var_key] = subvar_space.value

        # 如果占位动作是 dict（分层动作空间），尚未实现
        if isinstance(info['action'], dict):
            raise NotImplementedError
        # 普通数组动作：把占位动作全部替换成 NaN，表示"reset 时没有真实动作"
        else:
            info['action'] = np.full_like(info['action'], np.nan)
        # 返回原始观测（本 Wrapper 不改 obs）与填充完整的 info
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and move all data to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        # 向内层环境执行一步，拿到标准五元组
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 步数计数器加一
        self._step_counter += 1
        # 与 reset 相同：把裸数组观测包装成字典，统一接口
        if not isinstance(obs, dict):
            _obs = {'observation': obs}
        else:
            _obs = obs
        # 把观测各字段搬进 info
        for key, val in _obs.items():
            # 断言键不与内层 info 冲突
            assert key not in info
            info[key] = val
        # 写入本步真实奖励
        assert 'reward' not in info
        info['reward'] = reward
        # 写入是否终止（转成原生 bool，去掉 numpy 类型）
        assert 'terminated' not in info
        info['terminated'] = bool(terminated)
        # 写入是否因超时截断
        assert 'truncated' not in info
        info['truncated'] = bool(truncated)
        # 写入本步实际执行的动作
        assert 'action' not in info
        info['action'] = action
        # 写入当前步数
        assert 'step_idx' not in info
        info['step_idx'] = self._step_counter
        # 写入本 episode 的 ID（与 reset 时一致）
        assert 'id' not in info
        info['id'] = self._id

        # 与 reset 相同：把监控中的 variation 当前值写入 info，记录本步的域随机化配置
        for key in self._variations_watch:
            # 加前缀生成 info 键
            var_key = f'variation.{key}'
            assert var_key not in info
            # 按路径深入 variation_space 找到子空间
            subvar_space = get_in(
                self.env.unwrapped.variation_space, key.split('.')
            )
            # 写入子空间当前值
            info[var_key] = subvar_space.value

        # 五元组原样返回：reward/terminated/truncated 仍按 Gym 标准接口返回，info 只是"多一份副本"
        return obs, reward, terminated, truncated, info


class AddPixelsWrapper(gym.Wrapper):
    """Adds rendered environment pixels to info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for rendered pixels.
            torchvision_transform: Optional transform to apply to the pixels.
        """
        # 调用父类构造，连接内层环境
        super().__init__(env)
        # 记录渲染像素的目标尺寸 (H, W)
        self.pixels_shape = pixels_shape
        # 记录可选的 torchvision 变换（如 ToTensor、颜色归一化等）
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        # 局部导入 PIL：torchvision 的变换要求输入是 PIL 图像
        from PIL import Image

        # 把 Image 模块存成成员，供后面 _get_pixels/_format 复用（避免重复导入）
        self.Image = Image

    def _get_pixels(self) -> tuple[dict[str, np.ndarray], float]:
        """Render environment and process pixels.

        Returns:
            A tuple of (pixels dictionary, render time).
        """
        # Render the environment as an RGB array
        # 优先找底层环境是否提供 render_multiview（一次渲染多视角）
        render = getattr(self.env.unwrapped, 'render_multiview', None)
        # 若存在且可调用就用它，否则退回标准的 gym render()
        render_fn = render if callable(render) else self.env.render

        # 记录渲染开始时刻
        t0 = time.time()
        # 执行渲染，返回 ndarray / dict / list（取决于环境实现）
        img = render_fn()
        # 记录渲染结束时刻，用于统计渲染耗时
        t1 = time.time()

        # 内部函数：把单张原始图像统一处理成目标尺寸的数组或张量
        def _process_img(img_array: np.ndarray) -> np.ndarray:
            # Convert to PIL Image for resizing
            # 把 numpy 数组转成 PIL 图像，PIL 提供 resize 接口
            pil_img = self.Image.fromarray(img_array)
            # 解出目标高、宽
            height, width = self.pixels_shape
            # 缩放到目标尺寸：PIL.resize 要求 (width, height)，用双线性插值
            pil_img = pil_img.resize((width, height), self.Image.BILINEAR)
            # Optionally apply torchvision transform
            # 若配置了 torchvision 变换（如转 Tensor、归一化），在此应用
            if self.torchvision_transform is not None:
                pixels = self.torchvision_transform(pil_img)
            # 否则直接转回 numpy 数组
            else:
                pixels = np.array(pil_img)
            # 返回处理后的单帧像素
            return pixels

        # 若渲染结果是 dict（多视角，如 front/side），为每个视角键加 'pixels.' 前缀并逐个处理
        if isinstance(img, dict):
            pixels = {f'pixels.{k}': _process_img(v) for k, v in img.items()}
        # 若渲染结果是 list/tuple（多相机序列），用下标作为视角名
        elif isinstance(img, (list | tuple)):
            pixels = {
                f'pixels.{i}': _process_img(v) for i, v in enumerate(img)
            }
        # 单张图像：直接放在 'pixels' 键下
        else:
            pixels = {'pixels': _process_img(img)}

        # 返回 {键: 像素数组} 字典与渲染耗时（秒）
        return pixels, t1 - t0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and add pixels to info.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        # 重置内层环境
        obs, info = self.env.reset(*args, **kwargs)
        # 渲染当前画面：左侧接收像素字典，右侧把耗时写入 info['render_time']
        pixels, info['render_time'] = self._get_pixels()
        # 把 pixels.* 各键合并进 info
        info.update(pixels)
        # 返回观测与补充了像素的 info
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and add pixels to info.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        # 向内层环境推进一步
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 渲染执行动作后的新画面并记录耗时
        pixels, info['render_time'] = self._get_pixels()
        # 把新像素合并进 info
        info.update(pixels)
        # 返回五元组，info 中已包含最新一帧图像
        return obs, reward, terminated, truncated, info


class ResizeGoalWrapper(gym.Wrapper):
    """Resizes goal images in info dict."""

    def __init__(
        self,
        env: gym.Env,
        pixels_shape: tuple[int, int] = (84, 84),  # (height, width)
        torchvision_transform: Callable[[Any], Any] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            pixels_shape: Target (height, width) for resizing goal images.
            torchvision_transform: Optional transform to apply to goal images.
        """
        # 调用父类构造，连接内层环境
        super().__init__(env)
        # 记录目标图像要缩放到的尺寸，必须与观测像素尺寸一致，模型才能比对
        self.pixels_shape = pixels_shape
        # 记录可选的 torchvision 变换（与像素变换分开配置）
        self.torchvision_transform = torchvision_transform
        # For resizing, use PIL (required for torchvision transforms)
        # 局部导入 PIL 模块用于图像缩放
        from PIL import Image

        # 缓存为成员变量供 _format 使用
        self.Image = Image

    def _format(self, img: np.ndarray) -> np.ndarray:
        """Resize and transform a goal image.

        Args:
            img: The original goal image as a numpy array.

        Returns:
            The processed goal image.
        """
        # Convert to PIL Image for resizing
        # numpy -> PIL，为缩放做准备
        pil_img = self.Image.fromarray(img)
        # 解出目标高、宽
        height, width = self.pixels_shape
        # PIL 的尺寸参数顺序是 (width, height)，用双线性插值缩放
        pil_img = pil_img.resize((width, height), self.Image.BILINEAR)
        # Optionally apply torchvision transform
        # 配置了变换则应用（保证 goal 与观测经历完全相同的预处理）
        if self.torchvision_transform is not None:
            pixels = self.torchvision_transform(pil_img)
        # 否则转回 numpy
        else:
            pixels = np.array(pil_img)
        # 返回处理后的 goal 图像
        return pixels

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and format goal image.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        # 重置内层环境
        obs, info = self.env.reset(*args, **kwargs)
        # info 中有 goal 图像时，把它缩放/变换到统一规格
        if 'goal' in info:
            info['goal'] = self._format(info['goal'])
        # 返回处理后的 info
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and format goal image.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.
        """
        # 向内层环境推进一步
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 每步都可能刷新 goal，同样保证格式统一
        if 'goal' in info:
            info['goal'] = self._format(info['goal'])
        # 返回五元组
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 段落标题：StackedWrapper —— 对 info 中指定键做"历史帧堆叠"（frame stack）
#           capacity = history_size × frameskip：缓冲区按"环境步"计数，
#           取出时每隔 frameskip 抽一帧，共抽 history_size 帧
# ---------------------------------------------------------------------------
class StackedWrapper(gym.Wrapper):
    """Stacks specified key(s) in the info dict over the last k steps."""

    def __init__(
        self,
        env: gym.Env,
        key: str | list[str],
        history_size: int = 1,
        frameskip: int = 1,
    ):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            key: Key(s) in info dict to stack.
            history_size: Number of steps to stack.
            frameskip: Number of steps to skip between stacked frames.
        """
        # 调用父类构造，连接内层环境
        super().__init__(env)
        # 单个键名转成列表，统一按"键列表"处理（要堆叠的 info 字段）
        self.keys = [key] if isinstance(key, str) else key
        # 历史长度：最终希望堆叠出多少帧
        self.history_size = history_size
        # 抽帧间隔：每隔多少个环境步取一帧（区别于 frame skip 的动作重复！）
        self.frameskip = frameskip
        # 为每个要堆叠的键建立一个定长 deque 缓冲区（maxlen 超限时自动丢最旧）
        self.buffers: dict[str, deque[Any]] = {
            k: deque([], maxlen=self.capacity) for k in self.keys
        }

    @property
    def capacity(self) -> int:
        """Total capacity of the buffers in environment steps."""
        # 缓冲区容量 = 历史帧数 × 抽帧间隔（以环境步为单位）
        return self.history_size * self.frameskip

    def get_buffer_data(self, key: str) -> Any:
        """Retrieve stacked data from the buffer for a given key.

        Args:
            key: Key to retrieve data for.

        Returns:
            Stacked data as a numpy array or torch tensor.

        Raises:
            AssertionError: If buffer length is incorrect.
        """
        # 取出该键对应的缓冲区
        buffer = self.buffers[key]
        # 缓冲区为空（还没写入过）时返回空列表，避免索引错误
        if not buffer:
            return []

        # 关键步骤：list(buffer) 是按时间从旧到新排列的全部帧；
        # [:: -self.frameskip] 从尾部每隔 frameskip 取一个（先拿到较新的帧，顺序还是旧→新方向被反转）；
        # 再 [::-1] 反转回来，得到"从旧到新、间隔 frameskip"的 history_size 个帧
        new_info = list(buffer)[:: -self.frameskip][::-1]
        # 断言取出的帧数正好等于 history_size（缓冲区长度不对说明逻辑有误）
        assert len(new_info) == self.history_size, (
            f'Buffer for key {key} has incorrect length.'
        )

        # 把多帧沿新维度堆叠成一个张量/数组
        return self._stack_elements(new_info)

    def _stack_elements(self, elements: list[Any]) -> Any:
        """Stack elements based on their type.

        Args:
            elements: List of elements to stack.

        Returns:
            Stacked collection of elements.
        """
        # 空列表原样返回
        if not elements:
            return elements

        # 取第一个元素，用它的类型决定堆叠方式
        first_elem = elements[0]

        # 张量用 torch.stack：新增最前面的"帧"维度，形状 (T, ...)
        if torch.is_tensor(first_elem):
            return torch.stack(elements)
        # numpy 数组用 np.stack：同样新增帧维度
        elif isinstance(first_elem, np.ndarray):
            return np.stack(elements)
        # 标量（int/float/bool/np.number）直接转成一维数组，形状 (T,)
        elif isinstance(first_elem, (int, float, bool)) or issubclass(
            type(first_elem), np.number
        ):
            return np.array(elements)
        # 其他类型（如字符串、dict）保持列表，不强求堆叠
        else:
            return elements

    def init_buffer(self, info: dict) -> dict:
        """Initialize buffers with data from the current info dict.

        Args:
            info: Current environment info dict.

        Returns:
            Updated info dict with stacked data.

        Raises:
            AssertionError: If a required key is missing from info.
        """
        # 遍历所有要堆叠的键
        for k in self.keys:
            # 断言 info 中确实有该键
            assert k in info, (
                f'Key {k} not found in info dict during buffer initialization.'
            )
            # 取出该键当前的值（reset 后的首帧）
            data = info[k]
            # 取出对应缓冲区
            buffer = self.buffers[k]
            # 清空旧数据（新 episode 不继承上一局的历史）
            buffer.clear()
            # 用首帧填满整个缓冲区：让 episode 开头的每一"历史位"都是首帧
            buffer.extend([data] * self.capacity)
            # 把 info 中的单帧替换成堆叠后的多帧
            info[k] = self.get_buffer_data(k)
        # 返回被改写过的 info
        return info

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and initialize stacking buffers.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        # 重置内层环境
        obs, info = self.env.reset(*args, **kwargs)
        # 用 reset 得到的首帧初始化（填满）所有堆叠缓冲区
        info = self.init_buffer(info)
        # 返回观测与已含堆叠数据的 info
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform step and update stacking buffers.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            AssertionError: If a required key is missing from info.
        """
        # 向内层环境推进一步
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 对每个被堆叠的键更新历史
        for k in self.keys:
            # 断言本步 info 中含有该键
            assert k in info, f'Key {k} not found in info dict during step.'
            # 把最新一帧追加进缓冲区（超过容量时最旧的一帧被挤出）
            self.buffers[k].append(info[k])
            # 用缓冲区里的历史帧替换 info 中的单帧值
            info[k] = self.get_buffer_data(k)
        # 返回五元组，info 中相关键现在是多帧堆叠数据
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 段落标题：MegaWrapper —— 一条链串起"渲染像素 → 全部入 info → 键校验 → goal 缩放 → 堆叠"
#           观测/动作空间改写发生在何处：内层各 Wrapper 主要改写 info 的内容结构，
#           StackedWrapper 把 info 中每个值变成多帧（等价于在时间维扩展观测），
#           真正的 observation_space 语义由最外层训练代码按 info 键重新声明
# ---------------------------------------------------------------------------
class MegaWrapper(gym.Wrapper):
    """Combines multiple wrappers for comprehensive environment preprocessing."""

    def __init__(
        self,
        env: gym.Env,
        image_shape: tuple[int, int] = (84, 84),
        pixels_transform: Callable[[Any], Any] | None = None,
        goal_transform: Callable[[Any], Any] | None = None,
        required_keys: Iterable[str] | None = None,
        separate_goal: bool = True,
        history_size: int = 1,
        frame_skip: int = 1,
    ):
        """Initialize the mega wrapper pipeline.

        Args:
            env: The environment to wrap.
            image_shape: Target (height, width) for all image processing.
            pixels_transform: Optional transform for rendered pixels.
            goal_transform: Optional transform for goal images.
            required_keys: Keys that must be present in info dict.
            separate_goal: Whether to handle goal separately.
            history_size: Number of frames to stack.
            frame_skip: Number of frames to skip for stacking.
        """
        # 调用 gym.Wrapper 构造；注意后面会手动改写 self.env 来动态插层
        super().__init__(env)

        # 若调用方给了必需键就转成列表，否则为空列表
        req_keys = list(required_keys) if required_keys is not None else []
        # 追加正则 '^pixels(?:\..*)?$'：要求 info 中存在 'pixels' 或 'pixels.xxx' 键
        req_keys.append(r'^pixels(?:\..*)?$')

        # Build pipeline
        # this adds `pixels` key to info with optional transform
        # 第 1 层：渲染环境图像并缩放后写入 info['pixels*']
        env = AddPixelsWrapper(env, image_shape, pixels_transform)
        # this removes the info output, everything is in observation!
        # 第 2 层：把 obs/reward/terminated/action 等统一搬进 info
        env = EverythingToInfoWrapper(env)
        # check that necessary keys are in the observation
        # 第 3 层：校验 info 中包含全部必需键（含 pixels 正则）
        env = EnsureInfoKeysWrapper(env, req_keys)
        # check goal is provided
        # env = EnsureGoalInfoWrapper(env, check_reset=separate_goal, check_step=separate_goal)
        # 第 4 层：把 info['goal'] 缩放到与观测一致的尺寸（goal 校验层被注释掉）
        env = ResizeGoalWrapper(env, image_shape, goal_transform)

        # We will wrap with StackedWrapper dynamically after we know the keys
        # 先把当前链条存到 self.env（覆盖父类的引用）
        self.env = env
        # 记录堆叠的历史帧数
        self._history_size = history_size
        # 记录抽帧间隔（注意：这里是堆叠用的 frame skip，不是动作重复的 frame skip）
        self._frameskip = frame_skip
        # 标记堆叠层是否已挂上（首次 reset 前为 False）
        self._stack_initialized = False

    def _init_stack(self, info: dict) -> None:
        """Attach a StackedWrapper around self.env dynamically.

        Args:
            info: Current environment info dict to determine keys to stack.
        """
        # 取当前 info 的全部键：这些就是要堆叠的历史字段（此时才知道有哪些键）
        keys = list(info.keys())
        # 用这些键在现有链条外面再包一层 StackedWrapper
        self.env = StackedWrapper(
            self.env, keys, self._history_size, self._frameskip
        )
        # 标记已初始化，之后的 reset/step 不会重复包层
        self._stack_initialized = True
        # 用本次 info 初始化（填满）堆叠缓冲区
        self.env.init_buffer(info)

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset environment and initialize stack if needed.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            Standard Gymnasium reset results.
        """
        # 重置内层链条（首次调用时还没有 StackedWrapper）
        obs, info = self.env.reset(*args, **kwargs)

        # 首次 reset 才知道 info 的键集合，此时才动态挂上堆叠层
        if not self._stack_initialized:
            self._init_stack(info)

        # 返回观测与 info（info 中各字段已堆叠历史帧）
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Perform environment step.

        Args:
            action: Action to perform.

        Returns:
            Standard Gymnasium step results.

        Raises:
            RuntimeError: If reset() has not been called yet.
        """
        # 若还没 reset 过、堆叠层未挂上，直接报错（防止 KeyError 混淆）
        if not self._stack_initialized:
            raise RuntimeError(
                'StackedWrapper not yet initialized — call reset() first.'
            )
        # 把动作转给整条包装链执行一步
        return self.env.step(action)


# ---------------------------------------------------------------------------
# 段落标题：SyncWorld —— 同步向量化环境，reset 时允许"每个子环境传不同的 options/seeds"
#           （Gymnasium 原版 SyncVectorEnv 的 reset 只能给一份统一 options）
# ---------------------------------------------------------------------------
class SyncWorld(gym.vector.SyncVectorEnv):
    """Synchronous vectorized environment with per-environment options support.

    Extends SyncVectorEnv to allow passing different options to each
    sub-environment during reset, enabling per-environment variations
    and configurations.

    This is useful for scenarios where each environment in the vector
    needs different initialization parameters, such as different
    variation values or seeds.

    Example:
        >>> env_fns = [lambda: gym.make("MyEnv-v0") for _ in range(3)]
        >>> vec_env = SyncWorld(env_fns)
        >>> # Reset with per-env options
        >>> options = [{"variation": ["color"]}, {"variation": ["size"]}, None]
        >>> obs, infos = vec_env.reset(options=options)
    """

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ):
        """Reset all environments with optional per-environment configuration.

        Args:
            seed: Random seed(s) for reproducibility. Can be:
                - None: No seeding
                - int: Base seed, each env gets seed + env_index
                - list[int | None]: Explicit seed per environment
            options: Reset options. Can be:
                - None: No options for any environment
                - dict: Same options applied to all environments
                - list[dict | None]: Per-environment options (must match num_envs)

        Returns:
            A tuple of (observations, infos) where observations is the
            concatenated observations from all environments and infos
            is a dictionary with batched info from each environment.

        Raises:
            AssertionError: If options list length doesn't match num_envs.
        """

        # ---- 段落：把 seed / options 归一化成"长度 = num_envs 的列表" ----
        # 没给种子：每个子环境都是 None（不固定种子）
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        # 给了一个整数：各子环境用 seed+i，保证并行环境互不相同又可复现
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]

        # 没给 options：每个子环境都传 None
        if options is None:
            options_list = [None for _ in range(self.num_envs)]
        # 给的是列表：必须与子环境数量一致，逐个下发（可为每个环境指定不同 variation）
        elif isinstance(options, list):
            assert len(options) == self.num_envs, (
                f'options list length must match num_envs={self.num_envs}'
            )
            options_list = options
        # 给的是单个字典：广播给所有子环境（同一份配置）
        else:
            options_list = [options for _ in range(self.num_envs)]

        # 重置三组状态数组：是否 terminated / 是否 truncated / 是否待自动重置
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        # 准备收集各子环境 info 的批量字典
        infos = {}
        # 逐个子环境同步 reset：'同步'意味着所有子环境在同一个循环里依次推进、步调一致
        for i, (env, single_seed, single_options) in enumerate(
            zip(self.envs, seed, options_list, strict=True)
        ):
            # 用各自独立的 seed 与 options 重置第 i 个子环境
            self._env_obs[i], env_info = env.reset(
                seed=single_seed, options=single_options
            )
            # 把第 i 个环境的 info 合并进批量 infos（按索引 i 批处理）
            infos = self._add_info(infos, env_info, i)

        # 把各子环境观测沿第 0 维拼接成批量观测，复用传入的 self._observations 缓冲
        self._observations = gym.vector.utils.concatenate(
            self.single_observation_space, self._env_obs, self._observations
        )

        # 若配置 copy 则深拷贝后返回（避免调用方修改内部缓冲），否则直接返回引用
        return (
            deepcopy(self._observations) if self.copy else self._observations
        ), infos


# ---------------------------------------------------------------------------
# 段落标题：VariationWrapper —— 为向量化环境附加"variation 空间"（域随机化参数空间）
#           'same'：所有子环境共享同一套 variation 定义，批量成 (num_envs, ...) 空间
#           'different'：各子环境 variation 定义可不同，用 batch_differing_spaces 组合
# ---------------------------------------------------------------------------
class VariationWrapper(VectorWrapper):
    """Manages variation spaces for vectorized environments."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        variation_mode: str | gym.Space = 'same',
    ):
        """Initialize the variation wrapper.

        Args:
            env: The vectorized environment to wrap.
            variation_mode: Either 'same', 'different', or a custom Gymnasium space.

        Raises:
            ValueError: If variation_mode is invalid or sub-environments are incompatible.
        """
        # 调用 VectorWrapper 构造：保存内层向量环境、计算 num_envs 等
        super().__init__(env)

        # 取第一个子环境解包后的底层环境对象（unwrapped 去掉所有 Wrapper）
        base_env = env.envs[0].unwrapped

        # 底层环境没有 variation_space（不支持域随机化）时，两个空间都记为 None 并提前返回
        if not hasattr(base_env, 'variation_space'):
            self.single_variation_space: gym.Space | None = None
            self.variation_space: gym.Space | None = None
            return

        # 模式 'same'：单环境 variation 空间直接取第一个子环境的
        if variation_mode == 'same':
            self.single_variation_space = base_env.variation_space
            # 批量空间 = 把单环境空间复制扩展到 num_envs 份
            self.variation_space = batch_space(
                self.single_variation_space, self.num_envs
            )

        # 模式 'different'：收集每个子环境各自的 variation 空间
        elif variation_mode == 'different':
            self.single_variation_space = base_env.variation_space
            # 把"各不相同"的空间列表组合成一个带差异掩码的批量空间
            self.variation_space = batch_differing_spaces(
                [sub_env.unwrapped.variation_space for sub_env in env.envs]
            )

        # 其他取值（既不是 'same' 也不是 'different'）视为非法配置
        else:
            raise ValueError(
                f"Invalid `variation_mode`, expected: 'same' or 'different' or tuple of single and batch variation space, actual got {variation_mode}"
            )

        # check sub-environment obs and action spaces
        # 逐一检查各子环境的观测空间是否能与声明的单环境空间对齐
        for sub_env in env.envs:
            # 'same' 模式下所有子环境观测空间必须 dtype/shape 等价
            if variation_mode == 'same':
                # 比较子环境观测空间与 self.single_observation_space
                if not is_space_dtype_shape_equiv(
                    sub_env.unwrapped.observation_space,
                    self.single_observation_space,
                ):
                    # 不等价则无法 batch，直接报错并打印两者信息
                    raise ValueError(
                        f"VariationWrapper(..., variation_mode='same') however the sub-environments observation spaces do not share a common shape and dtype, single_observation_space={self.single_observation_space}, sub-environment observation_space={sub_env.observation_space}"
                    )
            # 'different' 模式下观测空间仍需 dtype/shape 一致（只是 variation 内容允许不同）
            else:
                if not is_space_dtype_shape_equiv(
                    sub_env.unwrapped.observation_space,
                    self.single_observation_space,
                ):
                    # 不一致则报错
                    raise ValueError(
                        f"VariationWrapper(..., variation_mode='different' or custom space) however the sub-environments observation spaces do not share a common shape and dtype, single_observation_space={self.single_observation_space}, sub-environment observation_space={sub_env.observation_space}"
                    )

    @property
    def envs(self) -> list[gym.Env] | None:
        """Access sub-environments if available."""
        # 透传内层向量环境的子环境列表；若内层没有该属性则返回 None
        return getattr(self.env, 'envs', None)
