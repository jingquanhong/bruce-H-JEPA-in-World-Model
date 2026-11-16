# 从 pathlib 导入 Path 类，用于拼接与校验文件/目录路径（跨平台）
from pathlib import Path

# 导入 numpy（别名 np）：数组运算、NaN 掩码判断与随机数生成器
import numpy as np
# 导入 OmegaConf：解析 YAML 配置对象，并把它转换成普通 Python 容器
from omegaconf import OmegaConf
# 导入 stable_worldmodel（别名 swm）：复用其中的 HDF5Dataset 等数据集实现
import stable_worldmodel as swm
# 导入 torch：张量转换，以及归一化统计的保存与加载（torch.save / torch.load）
import torch


# 归一化统计文件的固定文件名，保存在 run_dir 下，供规划评测时加载复用
NORMALIZER_ARTIFACT_FILENAME = "normalizer.pt"
# ImageNet 图像统计常量：RGB 三通道的均值与标准差，用于像素观测的标准化
IMAGENET_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ---- 段落标题：单列变换——对样本字典的某一列施加函数并写入目标列 ----
class ColumnTransform:
    """Apply `fn` to `sample[source]` and store the result in `sample[target]`."""

    def __init__(self, fn, source: str, target: str):
        # 一次性保存三元组：变换函数 fn、读取用的源列名 source、写入用的目标列名 target
        self.fn, self.source, self.target = fn, source, target

    def __call__(self, sample: dict) -> dict:
        # 读取 sample[source]，经 fn 变换后写入 sample[target]（同名即原地覆盖该列）
        sample[self.target] = self.fn(sample[self.source])
        # 返回样本字典本身，方便多个变换串联调用
        return sample


# ---- 段落标题：变换组合器——把多个样本变换按顺序串成一条流水线 ----
class Compose:
    """Apply dict-sample transforms in sequence."""

    def __init__(self, *transforms):
        # *transforms 把所有位置参数收集成一个元组，按传入顺序保存各变换
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        # 依次调用每个变换，前一个的输出样本作为后一个的输入
        for t in self.transforms:
            sample = t(sample)
        # 返回处理完毕的最终样本
        return sample


# ---- 段落标题：构建数据集——解析配置、DROID 分支、按转移数预算做子采样 ----
def build_hdf5_dataset(dataset_cfg, cache_dir=None):
    # 若传入的是 OmegaConf 配置对象（YAML 解析得到的 DictConfig）
    if OmegaConf.is_config(dataset_cfg):
        # 解析其中的插值占位符（resolve=True）并转成普通 dict/list 容器
        cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        # 否则按普通 dict 处理，复制一份，避免修改调用者传入的原字典
        cfg = dict(dataset_cfg)
    # 弹出与数据集构造无关的 val_name（验证集名由上层另行管理），键不存在时静默忽略
    cfg.pop("val_name", None)
    # 弹出 type 键；值为 "droid" 时说明要走 DROID 真机数据的专用分支
    if cfg.pop("type", None) == "droid":
        # 在分支内局部导入，避免未安装 droid 依赖时整个模块导入失败
        from droid_data import DROIDDataset

        # 用剩余配置关键字构造 DROID 数据集并直接返回（不走下面的子采样逻辑）
        return DROIDDataset(**cfg)
    # 弹出子采样随机种子 subset_seed（默认 0），保证 episode 子集可复现
    subset_seed = int(cfg.pop("subset_seed", 0))
    # 弹出转移数预算 total_transitions；为 None 表示用全部数据、不子采样
    total_transitions = cfg.pop("total_transitions", None)
    # 用剩余配置加 cache_dir 构造 HDF5 数据集，此时 clip_indices 覆盖全部合法窗口
    dataset = swm.data.HDF5Dataset(**cfg, cache_dir=cache_dir)
    # 只有设置了转移数预算时才执行子采样
    if total_transitions is not None:
        # 用固定种子的随机数生成器挑选若干 episode，使其总转移数逼近预算
        episodes = _select_episodes_for_transition_budget(
            dataset, int(total_transitions), np.random.default_rng(subset_seed)
        )
        # 把选中的 episode 编号转成集合，便于后面 O(1) 判断"是否保留"
        selected = set(episodes.tolist())
        # 重建 clip 索引列表：仅保留 episode 属于选中集合的 (episode, start) 窗口，
        # 把数据集规模裁剪到预算附近（整个列表推导式替换 dataset.clip_indices）
        dataset.clip_indices = [
            (episode, start)
            for episode, start in dataset.clip_indices
            if episode in selected
        ]
    # 返回（可能已被子采样的）数据集，交给上层做归一化与装 DataLoader
    return dataset


# ---- 段落标题：按"总转移数预算"贪心挑选 episode（尽量接近目标且不超可用量） ----
def _select_episodes_for_transition_budget(
    dataset,
    target_transitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # 合法 episode = 长度不小于最末层级窗口跨度 span 的轨迹（太短切不出一个完整样本）
    valid_episodes = np.flatnonzero(dataset.lengths >= dataset.last_level["span"])
    # 一个合法 episode 都没有时
    if valid_episodes.size == 0:
        # 抛出错误并指明是哪个 HDF5 文件，便于排查数据问题
        raise ValueError(f"No valid episodes in {dataset.h5_path}.")

    # 合法 episode 的转移数总和，即当前可用的数据总量
    available_transitions = int(dataset.lengths[valid_episodes].sum())
    # 请求的预算超过可用总量时直接报错（不静默截断，暴露配置错误）
    if target_transitions > available_transitions:
        raise ValueError(
            f"Requested {target_transitions} transitions from {dataset.h5_path}, "
            f"but only {available_transitions} valid episode transitions are available."
        )

    # 用给定的随机数生成器打乱合法 episode 顺序，避免总按编号顺序挑选造成偏差
    shuffled = rng.permutation(valid_episodes)
    # 记录最终选中的 episode 编号列表
    chosen = []
    # 累计已选 episode 的转移数（贪心过程中的当前总量）
    total = 0

    # ---- 段落标题：贪心累加 episode，直到累计转移数达到/超过预算为止 ----
    # 遍历打乱后的每个 episode 编号
    for episode in shuffled:
        # 当前 episode 的长度（含多少个转移步）
        length = int(dataset.lengths[episode])
        # 记录"加入本条之前"的累计值，稍后用于比较加与不加哪个更接近预算
        previous_total = total
        # 先把本条 episode 计入候选选择
        chosen.append(int(episode))
        # 累计转移数加上本条长度
        total += length

        # 累计量一旦达到或超过预算就停止挑选
        if total >= target_transitions:
            # 若已选不止一条，且"不加本条"比"加本条"更接近预算
            if (
                len(chosen) > 1
                and abs(target_transitions - previous_total)
                < abs(target_transitions - total)
            ):
                # 撤回刚加入的本条，保留更贴近预算的组合
                chosen.pop()
            # 结束贪心循环
            break

    # 一条都没选中时（预算极小），至少选打乱后的第一条，保证返回非空
    if not chosen:
        chosen.append(int(shuffled[0]))

    # 排序后转成 int64 数组返回：排序使结果与随机遍历顺序无关，便于复现与比较
    return np.asarray(sorted(chosen), dtype=np.int64)


# ---- 段落标题：基于当前数据集在线统计，生成某列的 z-score 归一化变换 ----
def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    # 数据集自带按列统计接口时优先走该路径（大数据集不必读出整列原始数据）
    if hasattr(dataset, "get_col_stats"):
        # 取该列的均值与标准差（numpy 数组；目前仅 DROIDDataset 实现了该接口）
        mean_np, std_np = dataset.get_col_stats(source)
        # 均值转成 torch float32 张量，便于随后与样本张量直接做广播运算
        mean = torch.from_numpy(np.array(mean_np)).float()
        # 标准差同样转成 torch float32 张量
        std = torch.from_numpy(np.array(std_np)).float()
    else:
        # 回退路径：读出该列的全部原始数据（HDF5Dataset 走这里）
        col_data = dataset.get_col_data(source)
        # numpy 数据转成 torch 张量
        data = torch.from_numpy(np.array(col_data))
        # 逐行检查 NaN：只要某行含一个 NaN 就整行剔除，保证统计不受缺失值污染
        data = data[~torch.isnan(data).any(dim=1)]
        # 沿第 0 维（样本维）求均值，keepdim 保持 (1, D) 形状便于广播；clone 避免共享内存
        mean = data.mean(0, keepdim=True).clone()
        # 同样求标准差（torch 默认为样本标准差，即 ddof=1）
        std = data.std(0, keepdim=True).clone()

    # 定义该列的归一化函数（z-score 标准化）
    def norm_fn(x):
        # 减均值、除标准差，把该列变换到 0 均值、单位方差
        x = (x - mean) / std
        # 把 NaN 与 ±inf（如 std=0 造成的除零）都替换为 0，再统一转成 float32
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()

    # 把函数封装成列变换对象：作用于 source 列，结果写入 target 列
    normalizer = ColumnTransform(norm_fn, source=source, target=target)

    # 返回变换对象，供 Compose 组装进数据流水线
    return normalizer


# ---- 段落标题：用已保存的训练集统计 artifact 构造归一化变换（评测/探针用） ----
def get_column_normalizer_from_artifact(
    artifact: dict, source: str, target: str, fill_nan: bool = True
):
    """Get a column normalizer from saved training-set stats.

    With fill_nan=False, NaN rows are preserved so callers can mask them (used
    for probe targets whose padded/inactive slots are stored as NaN).
    """
    # 取出 artifact 中的 stats 字典（结构：列名 -> {"mean", "std", ...}）
    stats = artifact.get("stats", {})
    # 请求的列不在统计里，说明该列当初没被选入归一化集合
    if source not in stats:
        # 把实际存在的列名排序拼成字符串，作为报错提示的一部分
        available = ", ".join(sorted(stats.keys()))
        # 抛 KeyError，附带缺失列与可用列，方便定位配置问题
        raise KeyError(
            f"Normalizer artifact is missing column {source!r}. "
            f"Available columns: {available}"
        )

    # 取出该列对应的统计条目
    col_stats = stats[source]
    # 训练集均值转成 torch float32 张量（必须复用训练统计，保证 train/eval 尺度一致）
    mean = torch.from_numpy(np.asarray(col_stats["mean"])).float()
    # 训练集标准差同样转成 torch float32 张量
    std = torch.from_numpy(np.asarray(col_stats["std"])).float()

    # 定义与训练期完全一致的 z-score 归一化函数
    def norm_fn(x):
        # 用训练集的 mean/std 做标准化
        x = (x - mean) / std
        # fill_nan=True 时把 NaN/inf 填 0；False 时保留 NaN，交由上层自行做掩码
        if fill_nan:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # 统一转成 float32 返回
        return x.float()

    # 封装成 source -> target 的列变换对象返回
    return ColumnTransform(norm_fn, source=source, target=target)


def normalizer_columns_from_dataset_cfg(dataset_cfg) -> list[str]:
    """Return non-pixel dataset columns that need z-score normalization.

    【选列规则】归一化/入模的列集合 = keys_to_load ∪ keys_to_cache ∪
    keys_to_merge 的键，并跳过 pixels*。凡是不在这个集合里的文件列
    （例如 bruce 数据集里的 sim_* 仿真真值列）既不会被 HDF5Dataset
    读入，也不会出现在归一化 artifact 中——即"文件里存了但训练用不到"，
    这是 sim2real 数据隔离的实现点。
    """
    # 初始化有序列表，收集需要归一化的非像素列（边遍历边去重、保持发现顺序）
    columns = []
    # 依次遍历 keys_to_load（按需读取的列）与 keys_to_cache（整列缓存进内存的列）
    for key in ("keys_to_load", "keys_to_cache"):
        # 取出该配置键的列名列表；键缺失或为 None 时用空列表兜底（or []）
        for col in dataset_cfg.get(key, []) or []:
            # pixels* 是图像观测列，不做 z-score 归一化，跳过
            if col.startswith("pixels"):
                continue
            # 尚未收集过才追加，保证列名不重复
            if col not in columns:
                columns.append(col)

    # 取出 keys_to_merge 配置（dict：目标列名 -> 源列，源列沿最后一维拼接成目标列并缓存）
    keys_to_merge = dataset_cfg.get("keys_to_merge", None)
    # 配置了合并键时，把参与合并的目标列也纳入归一化集合
    if keys_to_merge is not None:
        # 遍历 keys_to_merge 的键，即拼接生成的新目标列名（如 qpos+qvel 合成的 proprio）
        for col in keys_to_merge.keys():
            # 图像列同样跳过
            if col.startswith("pixels"):
                continue
            # 去重后追加进集合
            if col not in columns:
                columns.append(col)

    # 返回最终需要做 z-score 归一化的列名列表
    return columns


# ---- 段落标题：单列统计——优先用数据集自带接口，否则读全列自行计算 mean/std ----
def _column_mean_std(dataset, col: str) -> tuple[np.ndarray, np.ndarray, int]:
    # 数据集实现了 get_col_stats（如 DROIDDataset）时走快速路径
    if hasattr(dataset, "get_col_stats"):
        # 直接取该列的均值与标准差
        mean, std = dataset.get_col_stats(col)
        # 样本计数取所有轨迹长度之和（总转移数），表示统计覆盖的数据量
        count = int(dataset.lengths.sum())
        # 三者统一转成 float64 数组返回，即 (mean, std, count)
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(std, dtype=np.float64),
            count,
        )

    # 回退路径：读出该列的完整原始数据（HDF5Dataset 走这里）
    data = np.asarray(dataset.get_col_data(col))
    # 展平成 (样本数, 每样本元素数) 的二维数组，便于按行统一判断 NaN
    flat = data.reshape(data.shape[0], -1)
    # 行级有效掩码：该行不含任何 NaN 才为 True（沿特征维 any 检查后取反）
    valid_mask = ~np.isnan(flat).any(axis=1)
    # 只保留有效行并转成 float64 提高统计精度（copy=False 尽量不复制数据）
    valid = data[valid_mask].astype(np.float64, copy=False)
    # 有效样本数
    count = int(valid.shape[0])
    # 该列一行有效数据都没有时无法统计
    if count == 0:
        raise ValueError(f"Column {col!r} has no finite rows.")

    # 沿样本轴求均值，keepdims 保持 (1, ...) 形状方便后续广播
    mean = valid.mean(axis=0, keepdims=True)
    # 求标准差：样本数大于 1 用无偏估计 ddof=1，仅 1 个样本时退化为 ddof=0
    std = valid.std(axis=0, ddof=1 if count > 1 else 0, keepdims=True)
    # 返回均值、标准差、有效样本数
    return mean, std, count


# ---- 段落标题：构建归一化 artifact——在训练集上逐列统计并打包成可序列化 dict ----
def build_normalizer_artifact(cfg, train_dataset) -> dict:
    """Build serializable training normalizer stats for planning eval.

    对训练集里每个需要归一化的列统计 mean/std（跳过含 NaN 的行），
    序列化成 normalizer.pt；规划器评测时用同一份统计做归一化，
    保证 train/eval 的尺度一致。
    """
    # 从总配置中取出数据集子配置（用于选列以及记录数据集名）
    dataset_cfg = cfg.data.dataset
    # 初始化"列名 -> 统计量"字典
    stats = {}
    # 按选列规则遍历所有需要归一化的非像素列
    for col in normalizer_columns_from_dataset_cfg(dataset_cfg):
        # 在训练集上计算该列的 mean/std 与有效样本数（只用训练集，避免信息泄漏）
        mean, std, count = _column_mean_std(train_dataset, col)
        # 以列名为键记录统计量；mean/std 转 float32 减小文件体积，count 留作溯源信息
        stats[col] = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "count": count,
        }

    # 组装最终 artifact 字典：format 为格式版本号（加载时校验），
    # stats 为上面统计好的各列 mean/std/count，metadata 记录数据集名、
    # 列清单与标准差 ddof（1 表示无偏估计，与 _column_mean_std 一致）
    return {
        "format": "lejepa_training_normalizer_v1",
        "stats": stats,
        "metadata": {
            "dataset_name": str(dataset_cfg.get("name", "")),
            "columns": list(stats.keys()),
            "std_ddof": 1,
        },
    }


# ---- 段落标题：归一化 artifact 的落盘与读取（含格式校验） ----
def save_normalizer_artifact(artifact: dict, run_dir: str | Path) -> Path:
    # 拼出保存路径 run_dir/normalizer.pt
    path = Path(run_dir) / NORMALIZER_ARTIFACT_FILENAME
    # 用 torch.save 把整个 artifact 字典序列化写入 .pt 文件
    torch.save(artifact, path)
    # 返回写入的文件路径，供调用方打印或登记到训练日志
    return path


def load_normalizer_artifact(path: str | Path) -> dict:
    # 加载 .pt 文件到 CPU；weights_only=False 允许其中包含普通 Python 对象
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    # artifact 必须是字典结构，否则抛 TypeError
    if not isinstance(artifact, dict):
        raise TypeError(f"Normalizer artifact must be a dict, got {type(artifact)}.")
    # 校验格式版本号，不符则抛 ValueError，防止误用旧版或不兼容文件
    if artifact.get("format") != "lejepa_training_normalizer_v1":
        raise ValueError(
            f"Unsupported normalizer artifact format: {artifact.get('format')!r}."
        )
    # stats 必须是"列名 -> 统计"的字典，否则抛 ValueError
    if not isinstance(artifact.get("stats"), dict):
        raise ValueError("Normalizer artifact is missing a 'stats' dictionary.")
    # 全部校验通过，返回 artifact 供 get_column_normalizer_from_artifact 使用
    return artifact
