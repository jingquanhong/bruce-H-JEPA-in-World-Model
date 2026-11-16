"""Dataset classes for episode-based reinforcement learning data."""

# ---- 段落：导入依赖 ----
import logging                          # 标准库日志：打印缓存/合并信息
from pathlib import Path                # 面向对象的路径拼接
import re                               # 正则：校验 levelN 参数名、按正则匹配列名
from typing import Any                  # 类型标注：Any 表示"任意类型"

import h5py                             # 读写 HDF5 数据文件
# hdf5plugin 注册 HDF5 压缩插件(Blosc 等)，需在打开文件前 import 才能解压
import hdf5plugin  # noqa: F401
import numpy as np                      # ndarray：读出数据的基础容器
import torch                            # PyTorch：数值列最终转成 torch 张量
from omegaconf import OmegaConf         # 解析 Hydra/OmegaConf 配置对象

from stable_worldmodel.data.utils import get_cache_dir  # 取默认缓存数据目录


# ---- 段落：HDF5Dataset —— 单个 HDF5 文件的回合(episode)式数据集 ----
class HDF5Dataset:
    """Episode dataset loaded from a single HDF5 file.

    Args:
        name: Name of the dataset (filename without extension).
        keys_to_load: Specific keys to load (defaults to all except metadata).
        keys_to_cache: Keys to load entirely into memory for faster access.
        keys_to_merge: Target column -> source columns to concatenate and cache.
        cache_dir: Directory containing the dataset file.
        level1, level2, ...: Per-level clip configs (frameskip, num_steps, window_size).
    """

    def __init__(
        self,
        name: str,                                               # 数据集名(不含 .h5 后缀)
        keys_to_load: list[str] | None = None,                   # 要读入的列；None=除元数据外全部
        keys_to_cache: list[str] | None = None,                  # 启动时整列读进内存的列
        keys_to_merge: dict[str, list[str] | str] | None = None, # 合并列配置 {目标列: 源列}
        cache_dir: str | Path | None = None,                     # h5 所在目录；None 用默认缓存目录
        level1: dict | None = None,                              # level1 配置(frameskip/num_steps/window_size)
        **level_kwargs,                                          # 其余 level2、level3... 的配置
    ) -> None:
        # 拼出数据文件完整路径: <cache_dir>/<name>.h5
        self.h5_path = Path(cache_dir or get_cache_dir(), f'{name}.h5')
        # h5 文件句柄占位，首次读数据时才真正打开(见 _open)
        self.h5_file: h5py.File | None = None
        # 内存缓存: 列名 -> 整列 ndarray，放高频列和合并出的新列
        self._cache: dict[str, np.ndarray] = {}

        # 临时打开文件读取元数据与预缓存列，读完 with 块自动关闭
        with h5py.File(self.h5_path, 'r') as f:
            # ep_len/ep_offset 是 world.py 写出的契约元数据（必读）：
            # length/offset 描述每个 episode 在全局连续轴上的位置，
            # 后续所有切片读取都用 offsets[ep] + start 定位
            self.lengths, self.offsets = f['ep_len'][:], f['ep_offset'][:]  # ':'=读整列
            # keys_to_load 缺省 = 文件里除元数据外的全部列；
            # 若配置了它（如 ant.yaml），文件里的其余列（例如 bruce 的
            # sim_* 仿真真值列）不会被读入，也就不会进模型/归一化
            self._keys = keys_to_load or [   # 没显式指定就取全部数据列
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')  # 排除两列元数据
            ]

            # 把 keys_to_cache 指定的列整列读进内存，后续读取走缓存更快
            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]  # ':'=该列全部行
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        # 配置多层级(level1/level2...)并推导累计抽帧率与 span
        self._setup_levels(level1, **level_kwargs)

        # ---- 段落：枚举所有可截取的训练窗口(clip)索引 ----
        # 每个元素是 (episode编号, 窗口起始帧)；只考虑长度 >= 最高层 span
        # 的 episode(太短的整个跳过)，起点从 0 逐帧平移到 length-span
        self.clip_indices = [
            (ep, start)                                   # 产出 (ep, start) 元组
            for ep, length in enumerate(self.lengths)     # 遍历每个 episode 及其长度
            if length >= self.last_level['span']          # 装不下最高层跨度的 episode 跳过
            for start in range(length - self.last_level['span'] + 1)  # 窗口起点逐帧滑动
        ]

        self.transform = None  # 变换钩子默认关闭，之后可由外部赋值(如归一化)

        # 配置了合并列(如 qpos+qvel -> proprio)就立即执行合并
        if keys_to_merge:
            for target, source in keys_to_merge.items():  # 逐个目标列处理
                self.merge_col(source, target)            # 拼接并写入 _cache

    # ---- 段落：校验并规范化各层级配置，推导累计抽帧率与 span ----
    def _setup_levels(self, level1: dict | None, **level_kwargs) -> None:
        def _normalize_level(level_cfg: dict | None, level_name: str) -> dict | None:
            # 该层未配置(None) → 返回 None，后面直接跳过
            if level_cfg is None:
                return None

            # Convert to a plain dict so we can attach derived fields without
            # mutating Hydra/OmegaConf struct configs.
            if OmegaConf.is_config(level_cfg):
                # OmegaConf 对象转成普通 dict(resolve=True 展开插值)
                normalized = OmegaConf.to_container(level_cfg, resolve=True)
            else:
                normalized = dict(level_cfg)   # 本就是 dict，浅拷贝一份再改

            # 配置必须是键值映射，否则类型报错
            if not isinstance(normalized, dict):
                raise TypeError(f"{level_name} must be a mapping")

            # 三个关键参数统一强制转 int: frameskip=观测抽帧倍率
            normalized["frameskip"] = int(normalized["frameskip"])
            # num_steps=该层一个样本包含的步数
            normalized["num_steps"] = int(normalized["num_steps"])
            # window_size=窗口两端扩展帧数，缺省为 1
            normalized["window_size"] = int(normalized.get("window_size", 1))
            return normalized

        # 汇总配置入口: level1 走显式参数
        level_inputs = {"level1": level1}
        # level2、level3... 收在 **level_kwargs 里
        for level_name, level_cfg in level_kwargs.items():
            # 名字必须严格是 level+数字(允许 level10 等多位数)
            match = re.fullmatch(r"level([1-9]\d*)", level_name)
            if match is None:
                raise TypeError(f"Unexpected dataset argument: {level_name}")
            level_inputs[level_name] = level_cfg

        # 按层级序号重新组织: {1: 配置1, 2: 配置2, ...}
        levels_by_idx = {}
        for level_name, level_cfg in level_inputs.items():
            if level_cfg is None:
                continue   # 该层没配置，跳过
            level_idx = int(level_name.removeprefix("level"))  # 'level2' -> 2
            levels_by_idx[level_idx] = _normalize_level(level_cfg, level_name)  # 规范化后存入

        # 至少必须配置 level1
        if not levels_by_idx:
            raise ValueError("At least level1 must be configured")

        # 层级编号必须从 1 开始连续，不能缺号(有 level3 就必须有 level2)
        max_level = max(levels_by_idx)   # 最大层级号
        missing_levels = [
            level_idx for level_idx in range(1, max_level + 1)   # 1..max 全部序号
            if level_idx not in levels_by_idx                    # 找出其中没配置的
        ]
        if missing_levels:
            raise ValueError(f"Missing dataset level configs: {missing_levels}")

        # 排成有序列表，levels[0] 就是 level1 的配置
        levels = [levels_by_idx[level_idx] for level_idx in range(1, max_level + 1)]
        self.level1 = None   # 先置 None 占位，下面按 levelN 属性重新挂载
        for level_idx, level in enumerate(levels, start=1):
            setattr(self, f"level{level_idx}", level)   # 动态挂成 self.level1 / self.level2...

        self.levels = len(levels)     # 一共有几层
        self.last_level = levels[-1]  # 最高层配置(决定 span 和 clip 枚举)
        self.level_configs = levels   # 全部层级配置的列表

        # ---- 段落：计算每层相对 level1 的累计抽帧倍率 ----
        for i, level in enumerate(levels):
            if i:
                # 非首层: 本层 frameskip × 上一层的累计值(逐层连乘)
                level['effective_frameskip'] = level['frameskip'] * levels[i-1]['effective_frameskip']
            else:
                # level1 的累计倍率就是它自己的 frameskip
                level['effective_frameskip'] = level['frameskip']

        # ---- 段落：反推取一个第 level_idx 层样本需要多少 level1 原始帧 ----
        def _required_level1_frames(level_idx: int) -> int:
            # 起点 = 目标层一个样本的步数
            required = int(levels[level_idx - 1]['num_steps'])
            # 从目标层向 level1 方向逐层换算回更细的帧
            for upper_idx in range(level_idx - 1, 0, -1):
                upper_level = levels[upper_idx]   # 当前处理层的配置
                # 每降一层: 帧数 = (上一步数-1)×该层frameskip + 该层window_size
                # (步与步之间隔 frameskip 帧，窗口两端还要扩展 window_size)
                required = (
                    (required - 1) * int(upper_level['frameskip'])
                    + int(upper_level['window_size'])
                )
            return required

        # 所有层级里对 level1 帧数需求的最大值(保证任一层都能取样)
        required_level1_frames = max(
            _required_level1_frames(level_idx)            # 每层各自的需求
            for level_idx in range(1, self.levels + 1)    # 遍历第 1..levels 层
        )
        # span：取完最高层级的一个样本所需的 level1 原始帧数。
        # clip 索引只枚举 length >= span 的窗口——即 episode 太短时
        # 会被整体跳过（新数据集若 episode 过短会"一个样本都没有"）
        self.last_level['span'] = (
            # 最后再乘 level1 自己的 frameskip，把 span 统一到 level1 序列的单位
            required_level1_frames * self.level1['frameskip']
        )

    # 对外暴露的列名列表(含合并生成的新列)
    @property
    def column_names(self) -> list[str]:
        return self._keys   # 直接返回内部 _keys

    def __len__(self) -> int:
        return len(self.clip_indices)   # 样本数 = 可截取窗口总数

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]          # 查表得到 (episode, 窗口起点)
        return self._load_slice_with_levels(ep_idx, start)  # 读出该窗口的 level1 序列

    # ---- 段落：按 (ep, start, end) 批量读出若干原始片段(回放/评估用) ----
    def load_chunk(
        self, episodes_idx: np.ndarray,   # 各段所在 episode 编号(数组)
        start: np.ndarray,                # 各段起始帧(episode 内相对下标)
        end: np.ndarray,                  # 各段结束帧(不含)
    ) -> list[dict]:
        chunk = []   # 收集每个片段结果的列表
        for ep, s, e in zip(episodes_idx, start, end):   # 三个数组按位置一一配对
            # 按 level1 的 frameskip 抽帧读取 [s, e) 区间
            steps = self._load_slice(ep, s, e, frameskip=self.level1['frameskip'])
            if 'action' in steps:
                # action 是满帧率读出的，压成 (抽帧后帧数, 动作维)
                steps['action'] = steps['action'].reshape(
                    (e - s) // self.level1['frameskip'], -1   # 行数=帧数，-1 自动算动作维
                )
            chunk.append(steps)   # 收进结果列表
        return chunk

    def _open(self) -> None:
        # swmr=True：以只读共享模式打开，兼容 world.py 写端的 SWMR 文件
        # （进程被杀后正在写的文件依然可读）
        if self.h5_file is None:   # 尚未打开才打开，避免重复 open
            self.h5_file = h5py.File(
                # 只读模式; SWMR 共享读; 256MB 块缓存(减少磁盘 IO)
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    # ---- 段落：读一个 clip → 返回 level1 抽帧序列(高层由模型自行派生) ----
    def _load_slice_with_levels(self, ep_dix: int, start: int) -> dict:
        # Return the full level-1 sequence covering the largest span; the model
        # derives the higher levels from it.
        level1_steps = self._load_slice(
            ep_idx=ep_dix,
            start=start,                            # 窗口起点
            end=start + self.last_level['span'],    # 终点 = 起点 + 最高层 span
            frameskip=self.level1['frameskip'],     # 用 level1 抽帧率抽观测
            apply_transform=False,                  # 先不 transform，reshape 完再统一做
        )

        # 校验: 满帧率读出的 action 长度必须能被 frameskip 整除
        assert level1_steps['action'].shape[0] % self.level1['frameskip'] == 0
        # 抽帧后的帧数 = 满帧 action 数 ÷ frameskip
        level1_frames = level1_steps['action'].shape[0] // self.level1['frameskip']
        # 动作维度 = 最后一维
        action_dim = level1_steps['action'].shape[-1]
        # 把 (满帧动作,) 重排成 (帧, frameskip, 动作维): 每个观测帧配一小段连续动作
        level1_steps['action'] = level1_steps['action'].reshape(
            level1_frames, self.level1['frameskip'], action_dim   # 目标三元形状
        )

        # 若外部注册了 transform(归一化等)，在这里统一应用
        if self.transform:
            level1_steps = self.transform(level1_steps)

        # transform 可能又把 action 展平，这里再压回二维 (帧, frameskip×动作维)
        if 'action' in level1_steps:
            level1_steps['action'] = level1_steps['action'].reshape(
                level1_steps['action'].shape[0], -1   # 行(帧数)不变，列自动合并
            )

        # 所有列统一加后缀 _level1 返回(如 obs -> obs_level1)
        return {f'{col}_level1': steps for col, steps in level1_steps.items()}

    # ---- 段落：读取单个 episode 内 [start, end) 的原始切片(抽帧前) ----
    def _load_slice(
        self,
        ep_idx: int,          # episode 编号
        start: int,           # episode 内起始帧(相对)
        end: int,             # episode 内结束帧(不含)
        frameskip: int,       # 观测抽帧倍率
        apply_transform: bool = True,   # 是否在末尾应用 self.transform
    ) -> dict:
        self._open()   # 确保 h5 文件已打开
        # start/end 是 episode 内相对下标，加 offset 换算成全局连续轴下标
        g_start, g_end = (
            self.offsets[ep_idx] + start,   # 全局起点
            self.offsets[ep_idx] + end,     # 全局终点
        )
        steps = {}   # 收集各列读出的数据 {列名: 值}
        for col in self._keys:   # 只读 _keys 里的列，其余列(如 sim_* 真值)不读入
            # 优先用内存缓存，没缓存就从 h5 文件读
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]   # 按全局行区间切片
            # 【核心契约】抽帧（frameskip）只作用于观测列：
            # 非 action 列按 data[::frameskip] 降采样（100Hz → 20Hz）；
            # action 保持满帧率原样读出，由上层 reshape 成
            # (帧数, frameskip, action_dim) —— 每个观测帧对应一小段
            # 连续动作。所以 action 的落盘帧率必须与观测一致
            if col != 'action':
                data = data[:: frameskip]   # 观测列按步长 frameskip 抽帧

            # 字符串类列(如语言指令)：取第一行并解码成 str
            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''   # 空切片给空字节串兜底
                steps[col] = val.decode() if isinstance(val, bytes) else val  # bytes -> str
            else:
                # 数值列转成 torch 张量
                steps[col] = torch.from_numpy(data)
                # 图像 (T,H,W,C) 且通道为 1/3 → 转成通道在前 (T,C,H,W)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        # 请求了变换且注册了 transform 时才执行，否则原样返回
        if apply_transform and self.transform:
            return self.transform(steps)
        return steps

    # 取整列数据：缓存优先，否则从文件读全部行
    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]   # 命中内存缓存，直接返回
        self._open()
        return self.h5_file[col][:]   # ':'=读该列全部行

    # 公开接口：返回某一列的完整数据
    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    # 按行号(单个下标或下标列表)取出多列对应的值
    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        data = {}   # {列名: 该行(些行)的值}
        for col in self._keys:   # 只取 _keys 里的列
            src = self._cache if col in self._cache else self.h5_file   # 缓存优先
            data[col] = src[col][row_idx]   # 花式索引：取指定行
        return data

    # ---- 段落：解析合并列的 source 描述，返回 [(标签, 整列数据), ...] ----
    def _load_merge_sources(
        self,
        source: list[str | dict[str, Any]] | str | dict[str, Any],  # 正则串 / 切片字典 / 它们的列表
        dim: int,   # 后续切片所沿的维度(-1 = 最后一维)
    ) -> list[tuple[str, np.ndarray]]:
        self._open()
        # OmegaConf 配置对象先转成普通 Python 容器
        if OmegaConf.is_config(source):
            source = OmegaConf.to_container(source, resolve=True)

        # 写法一：字符串 = 列名正则，匹配文件里所有同模式的列
        if isinstance(source, str):
            matches = [k for k in self.h5_file.keys() if re.match(source, k)]   # 正则匹配列名
            if not matches:
                raise KeyError(f"No columns matched merge source {source!r}")
            return [(key, self._get_col(key)) for key in matches]   # 每列各自读出

        # 写法二：字典 {key, start, end} —— 读某一列并切掉一段维度
        if isinstance(source, dict):
            key = source.get("key")   # 必填：源列名
            if not isinstance(key, str) or not key:
                raise ValueError("Merge source dicts must define a non-empty 'key'.")

            data = self._get_col(key)         # 读整列
            start = source.get("start", None) # 切片起(可缺省)
            end = source.get("end", None)     # 切片止(可缺省)
            if start is not None or end is not None:
                # 负 dim 换算成正轴号(如 -1 -> ndim-1)
                axis = dim if dim >= 0 else data.ndim + dim
                if axis < 0 or axis >= data.ndim:
                    raise ValueError(
                        f"Cannot slice merge source {key!r} along dim={dim}; "
                        f"source has {data.ndim} dimensions."
                    )
                # 构造每维的切片列表：其余维全取，目标维取 [start:end]
                slices = [slice(None)] * data.ndim
                slices[axis] = slice(
                    None if start is None else int(start),   # 起点，None=从头
                    None if end is None else int(end),       # 终点，None=到尾
                )
                data = data[tuple(slices)]   # 应用切片得到子块

            # 生成可读标签：带切片时形如 qpos[2:8]，仅用于日志/调试
            label = key
            if start is not None or end is not None:
                start_label = "" if start is None else str(start)
                end_label = "" if end is None else str(end)
                label = f"{key}[{start_label}:{end_label}]"
            return [(label, data)]

        # 写法三：列表 —— 递归展开每一项再汇总
        sources = []
        for item in source:
            sources.extend(self._load_merge_sources(item, dim))   # 递归收集
        return sources

    def merge_col(
        self,
        source: list[str | dict[str, Any]] | str | dict[str, Any],
        target: str,
        dim: int = -1,
    ) -> None:
        """把若干源列沿最后一维拼成一个新列并缓存（如 qpos+qvel → proprio）。

        source 支持三种写法：列名正则字符串、{key,start,end} 切片字典、
        或它们的列表；拼接结果放进 _cache 且追加进 _keys，后续
        get_row_data/__getitem__ 会像普通列一样返回它。
        """
        # 解析 source，拿到 [(标签, 数据), ...]
        sources = self._load_merge_sources(source, dim)

        # 沿 dim 维把所有源数据拼成一整块(如 qpos+qvel 拼成 proprio)
        merged = np.concatenate([data for _, data in sources], axis=dim)
        self._cache[target] = merged   # 放进缓存，之后当普通列一样读
        if target not in self._keys:
            self._keys.append(target)   # 注册进列名列表，__getitem__ 才会返回它
        labels = [label for label, _ in sources]   # 只取标签用于日志
        logging.info(f"Merged columns {labels} into '{target}' and cached it")

    # 该列的特征维度 = 除行维外所有维度之积(标量列视为 1)
    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1   # shape[1:] 求积
