# ==== 模块总览：规划评测 —— 把学到的世界模型接进环境做闭环 rollout ====
# 评测主流程 run_planning_eval：
#   1) 加载评测配置，构建图像预处理 transform 与观测归一化 process；
#   2) _build_policy：加载世界模型 checkpoint -> 构建 solver（分层或平坦）-> 打包成 WorldModelPolicy；
#   3) 建立 swm.World（若干并行环境），调用 world.evaluate_from_dataset：
#      从数据集取"起点状态 + 真值 goal"，环境复位到起点，policy 每步用世界模型规划动作，
#      环境执行后产生新观测，再进入下一轮规划 —— 形成 观测->规划->动作->新观测 的闭环，
#      直到到达 goal 或耗尽 eval_budget；
#   4) 聚合各 episode 的成功率 / 到达步数等指标，写入 results_dir/metrics.yaml。
# 两条支线：
#   - dump_eval_trajs_path：只从数据集采样并导出固定评测轨迹（保证多次评测起点/goal 一致）；
#   - chunked eval：按 chunk_size 分块断点续跑，每块重播种子，最后拼接聚合。

# ---- 标准库导入 ----
# os：os.replace 实现块结果文件的原子替换（防中断留下半截文件）
import os
# shlex：把命令行参数列表安全拼成可复制执行的字符串（溯源信息用）
import shlex
# subprocess：调用外部 git 命令，记录评测时的代码版本
import subprocess
# sys：读取本次运行的命令行参数 sys.argv，写入溯源信息
import sys
# time：记录各块/总评测耗时与生成时间戳
import time
# Path：面向对象的路径操作
from pathlib import Path

# ---- 第三方库导入 ----
# hydra：配置实例化工具，hydra.utils.instantiate 按 _target_ 构建 solver
import hydra
# numpy：数组计算，用于采样评测起点与聚合指标
import numpy as np
# stable_worldmodel（swm）：本仓库核心库，提供 World、policy、HDF5Dataset 等
import stable_worldmodel as swm
# torch：张量计算；torch.load 读取 checkpoint 与评测轨迹文件
import torch
# SDPBackend/sdpa_kernel：指定注意力算子后端；评测强制 MATH 后端，以支持带梯度的规划计算
from torch.nn.attention import SDPBackend, sdpa_kernel
# OmegaConf：读写/解析 YAML 配置；DictConfig 是配置对象类型
from omegaconf import DictConfig, OmegaConf
# sklearn.preprocessing：用训练时的均值方差构造 StandardScaler 做观测归一化
from sklearn import preprocessing
# torchvision.transforms v2：图像预处理链（转张量、归一化、缩放）
from torchvision.transforms import v2 as transforms

# ---- 本仓库内部导入 ----
# data 模块：ImageNet 归一化常量、normalizer 工件文件名与加载函数
from data import (
    IMAGENET_STATS,
    NORMALIZER_ARTIFACT_FILENAME,
    load_normalizer_artifact,
)
# eval_config_utils：从 yaml 构建规划配置、判断当前配置是否为分层 solver
from eval_config_utils import _build_policy_plan_config, _is_hierarchical_solver
# 分层求解器工厂函数（本文件用它组装分层规划器）
from hierarchical_solver import build_hierarchical_solver
# 解析 policy checkpoint 文件路径的工具
from utils import resolve_model_checkpoint_path


# ==== 图像预处理链：把环境原始像素变成模型可用的输入 ====
def img_transform(cfg):
    """按训练时同款流程转换单张图像：转张量 -> [0,1] -> ImageNet 标准化 -> 缩放到 img_size。"""
    return transforms.Compose(
        [
            # PIL/ndarray 转 torch 图像张量（CHW）
            transforms.ToImage(),
            # 转 float32 并按 1/255 缩放到 [0,1]（scale=True 表示自动除以 dtype 最大值）
            transforms.ToDtype(torch.float32, scale=True),
            # 用 ImageNet 各通道均值/方差标准化（与训练保持一致，否则分布偏移）
            transforms.Normalize(**IMAGENET_STATS),
            # 缩放到评测配置指定的图像尺寸
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


# ==== 生成"结果溯源信息"：命令行、工作目录、git 版本、时间 ====
def generation_provenance():
    # 本文件所在目录，作为执行 git 命令的仓库目录
    repo = Path(__file__).resolve().parent
    # lambda 封装 git 子命令调用：在 repo 下执行并返回去空白后的标准输出
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    return {
        # 用于复现的完整命令行（python + 各参数，shlex 安全拼接）
        "command": shlex.join(["python", *sys.argv]),
        # 运行时的工作目录
        "cwd": str(Path.cwd()),
        # 当前 git commit 哈希
        "git_commit": git("rev-parse", "HEAD"),
        # 当前有改动的文件列表（非空说明结果不能精确复现）
        "git_dirty_files": git("status", "--porcelain").splitlines(),
        # 生成时间戳
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


# ==== 加载评测用的 YAML 配置 ====
def load_eval_config(config_name: str | None = None, config_path: str | None = None):
    # 显式给了配置文件路径：直接从该路径读
    if config_path is not None:
        return OmegaConf.load(config_path)

    # 既没给路径也没给名字：参数不合法
    if config_name is None:
        raise ValueError("Either config_name or config_path must be provided.")

    # 只给了名字：解释成本文件目录下 config/eval/<名字>.yaml
    eval_dir = Path(__file__).parent / "config" / "eval"
    return OmegaConf.load(eval_dir / f"{config_name}.yaml")


# ==== 把相对路径解析成真实存在的路径：依次尝试 当前目录、本文件目录 ====
def _resolve_existing_path(path: str | Path) -> Path:
    # 展开 ~ 并转成 Path 对象
    raw_path = Path(path).expanduser()
    for candidate in (Path.cwd() / raw_path, Path(__file__).parent / raw_path):
        # 哪个候选位置存在就返回哪个
        if candidate.exists():
            return candidate
    # 都不存在：原样返回（相对路径），由后续打开时报错
    return raw_path




# ==== 计算数据集中每个 episode 的长度（步数） ====
def get_episodes_length(dataset, episodes):
    """对每条 episode 返回 max(step_idx)+1，即该轨迹的总步数。"""
    # episode 列名在不同数据集里可能是 episode_idx 或 ep_idx，先探测
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    # 读取"每一行属于哪条 episode"的整列数据
    episode_idx = dataset.get_col_data(col_name)
    # 读取"每一行是该 episode 内第几步"的整列数据
    step_idx = dataset.get_col_data("step_idx")
    # 存放逐条 episode 的长度
    lengths = []
    for ep_id in episodes:
        # 长度 = 该 episode 的最大 step_idx + 1（step_idx 从 0 开始计数）
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


# ==== 构建评测用的 HDF5 数据集对象 ====
def get_dataset(cfg, dataset_name):
    # 缓存目录：配置指定优先，否则用 swm 的默认缓存目录
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        # 需要常驻内存缓存的列（如 pixels）
        keys_to_cache=cfg.dataset.keys_to_cache,
        # 需要跨列拼接合并的列（可为空）
        keys_to_merge=cfg.dataset.get("keys_to_merge", None),
        cache_dir=dataset_path,
        # 是否按 level1 粒度读取数据
        level1=cfg.dataset["level1"],
    )


# ==== 把本次评测的（已解析）配置快照写进结果目录，方便溯源复现 ====
def save_eval_config(results_dir: Path, cfg: DictConfig) -> Path:
    # 目标文件名固定为 eval_config.yaml
    config_path = results_dir / "eval_config.yaml"
    # to_yaml(resolve=True) 会把 ${...} 插值全部展开成最终值再写盘
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return config_path


# ==== 决定评测结果写到哪个目录 ====
def resolve_results_dir(
    cfg: DictConfig,
    results_dir: str | Path | None = None,
) -> Path:
    # 显式指定了结果目录：直接使用
    if results_dir is not None:
        return Path(results_dir)

    # 配置里可选的 output.dir
    output_dir = cfg.output.get("dir", None)
    # 无策略或 random 基线：放在本文件目录（或 output.dir）
    if cfg.policy in (None, "random"):
        return Path(__file__).parent if output_dir is None else Path(output_dir)
    # 有模型策略：放在 缓存目录/策略名 所在的目录，再拼接 output.dir
    return Path(swm.data.utils.get_cache_dir(), cfg.policy).parent / (output_dir or "")


# ==== 列出需要做归一化处理的观测列（排除图像 pixels） ====
def _required_process_columns(cfg: DictConfig) -> list[str]:
    # 合并 keys_to_cache 与 keys_to_merge 里出现的所有列名
    columns = [*cfg.dataset.keys_to_cache, *(cfg.dataset.get("keys_to_merge", None) or {})]
    # dict.fromkeys 去重且保持原有顺序；排除 pixels（图像单独走 transform 归一化）
    return [col for col in dict.fromkeys(columns) if col != "pixels"]


# ==== 把"训练时 normalizer 的统计量"变成可直接调用的 sklearn StandardScaler 字典 ====
def _normalizer_artifact_to_process(artifact: dict, required_cols: list[str]) -> dict:
    # artifact["stats"]：{列名: {mean, std, count}} 的训练期统计
    stats = artifact["stats"]
    # 找出缺少统计量的必需列
    missing = [col for col in required_cols if col not in stats]
    if missing:
        # 缺列直接报错：漏归一化会让观测分布与训练时不一致
        available = ", ".join(sorted(stats.keys()))
        raise KeyError(
            "Training normalizer artifact is missing required columns "
            f"{missing}. Available columns: {available}"
        )

    # 结果字典：{列名: StandardScaler}
    process = {}
    for col in required_cols:
        # 该列的统计量
        col_stats = stats[col]
        # 均值拉平成一维 float64 数组
        mean = np.asarray(col_stats["mean"], dtype=np.float64).reshape(-1)
        # 标准差拉平，作为 scaler 的 scale
        scale = np.asarray(col_stats["std"], dtype=np.float64).reshape(-1)

        # StandardScaler 通常由 fit 产生；这里手工注入已知统计量，跳过 fit
        processor = preprocessing.StandardScaler()
        # 写入训练时的均值（归一化公式 (x - mean) / scale 的 mean）
        processor.mean_ = mean
        # 写入训练时的标准差（scale）
        processor.scale_ = scale
        # 方差 = 标准差平方（StandardScaler 的内部字段，需保持自洽）
        processor.var_ = np.square(scale)
        # 特征维度（该列向量的长度）
        processor.n_features_in_ = int(mean.shape[0])
        # 参与统计的样本数（缺省用 0）
        processor.n_samples_seen_ = int(col_stats.get("count", 0))
        # 挂到该列名下
        process[col] = processor

        # 非动作列还要为 'goal_<列>' 存一份同样的归一化器（目标观测用同一套统计量）
        if col != "action":
            process[f"goal_{col}"] = processor

    return process


# ==== 从 policy checkpoint 旁边加载训练期 normalizer 工件 ====
def _load_policy_normalizer_artifact(cfg: DictConfig) -> dict:
    # 定位 checkpoint 文件路径
    ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.get("cache_dir"))
    # normalizer 文件约定与 checkpoint 同目录
    normalizer_path = ckpt_path.parent / NORMALIZER_ARTIFACT_FILENAME
    if not normalizer_path.exists():
        # 找不到就报错：归一化必须与训练一致，否则指标失真
        raise FileNotFoundError(
            "Training normalizer artifact not found for policy checkpoint. "
            f"Expected: {normalizer_path}."
        )
    return load_normalizer_artifact(normalizer_path)


# ==== 生成本次评测使用的归一化 process 字典 ====
def _build_eval_process_from_policy_normalizer(
    cfg: DictConfig,
    model=None,
) -> dict:
    # 若传入了内存中的模型，优先使用它自带的 normalizer 工件
    artifact = getattr(model, "normalizer_artifact", None) if model is not None else None
    if artifact is None:
        # random 基线不做归一化，返回空字典
        if cfg.get("policy", "random") == "random":
            return {}
        # 否则从磁盘上 checkpoint 旁的文件加载
        artifact = _load_policy_normalizer_artifact(cfg)

    # 转成 {列名: StandardScaler}，只保留本次评测需要的列
    return _normalizer_artifact_to_process(artifact, _required_process_columns(cfg))


# ==== 裁掉配置中"超过模型实际层数"的层级配置 ====
def _drop_levels_above_model(cfg, num_levels):
    """Let one planning config (e.g. the lN_project configs) serve models of any depth."""
    # 遍历 solver 配置的键的快照（循环体内要删除，不能边遍历边删）
    for key in list(cfg.solver.solvers):
        # 'level3' -> 3；层号超出模型层数
        if int(key[len("level"):]) > num_levels:
            # 删除多余的该层 solver 配置
            del cfg.solver.solvers[key]
            # 同步删除该层的分层规划配置，保持两处一致
            del cfg.hierarchical_plan_config[key]


# ==== 加载用于规划的世界模型（搬到 cuda、切 eval 模式） ====
def load_model(cfg: DictConfig, model=None):
    """Load cfg.policy (or pick the cost module of an in-memory flat model) for planning on cuda."""
    if model is None:
        if _is_hierarchical_solver(cfg):
            # 分层模型：checkpoint 里存的就是完整模型对象，直接反序列化（先放 cpu）
            ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.get("cache_dir"))
            model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        else:
            # 平坦模型：用 AutoCostModel 包装出能计算 get_cost 的模块
            model = swm.policy.AutoCostModel(cfg.policy)

        # 搬到 GPU 上做规划
        model = model.to("cuda")
    elif not _is_hierarchical_solver(cfg):
        # 已有内存模型且非分层：挑出第一个带 get_cost 的子模块（即代价头）
        selected_model = next((m for m in model.modules() if hasattr(m, "get_cost")), None)
        if selected_model is None:
            # 找不到代价模块说明模型结构不符合预期
            raise RuntimeError(
                "No module with 'get_cost' found in the provided in-memory model."
            )
        model = selected_model

    # 切到评估模式：关闭 dropout、把 BatchNorm 固定为推理统计
    model = model.eval()
    # 允许位置编码按当前序列长度插值（推理分辨率可与训练不同）
    model.interpolate_pos_encoding = True
    return model


# ==== 构建求解器：分层走分层工厂，平坦则用 hydra 实例化 cfg.solver ====
def build_solver(cfg: DictConfig, model):
    if _is_hierarchical_solver(cfg):
        # 分层：每层一个子 solver，由 build_hierarchical_solver 统一组装
        return build_hierarchical_solver(cfg, model)
    # 平坦：按 cfg.solver 的 _target_ 实例化，并把 model 注入进去
    return hydra.utils.instantiate(cfg.solver, model=model)


# ==== 组装最终交给环境的 policy（solver + 归一化 + 图像变换） ====
def _build_policy(
    cfg: DictConfig,
    process: dict,
    transform: dict,
    model=None,
):
    # 没给模型且策略为 random：返回随机动作基线（对照组）
    if model is None and cfg.get("policy", "random") == "random":
        return swm.policy.RandomPolicy()

    # 加载（或从内存选取）世界模型
    model = load_model(cfg, model)
    # 分层时删掉配置里模型不存在的更高层，避免配置与模型层数不匹配
    if _is_hierarchical_solver(cfg):
        _drop_levels_above_model(cfg, model.num_levels)
    # 从 yaml 的 plan_config / hierarchical_plan_config 构建规划配置对象
    policy_config = _build_policy_plan_config(cfg)
    # 构建求解器（分层或平坦）
    solver = build_solver(cfg, model)

    # 打包成"世界模型策略"：policy.get_action 内部会调用 solver.solve 做规划
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=policy_config,
        # 观测归一化器（StandardScaler 字典）
        process=process,
        # 图像预处理链
        transform=transform,
        # 本次评测允许的规划/交互步数预算
        eval_budget=int(cfg.eval.eval_budget),
    )


# ==== 分层配额采样评测起点：先在轨迹间均衡分配，再在每条轨迹内均匀布点 ====
def _sample_stratified_eval_starts(cfg: DictConfig, dataset):
    """Balance eval starts across trajectories, then space them out within each.

    Trajectories get near-equal quotas (capped by how many valid starts each has),
    and within a trajectory the valid-start range is split into contiguous bins with
    one start drawn per bin. Removes the length bias of uniform step sampling and
    avoids near-duplicate segments from the same trajectory.
    """
    # 探测 episode 列名（episode_idx 或 ep_idx）
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    # 所有出现过的 episode 编号（去重升序）
    ep_indices = np.unique(dataset.get_col_data(col_name))
    # 每条 episode 的长度
    episode_len = get_episodes_length(dataset, ep_indices)

    # valid starts per episode: step_idx in [0, length - offset - 1]
    # 中文：每条 episode 的合法起点集合 = step_idx ∈ [0, 长度-goal_offset-1]，
    # 保证起点之后还留得出到 goal 的偏移量；长度不足则为空数组。
    allowed = [np.arange(max(int(n) - int(cfg.eval.goal_offset_steps), 0)) for n in episode_len]

    # 长度不够、一个起点都留不出的 episode 标为 False
    eligible = np.array([len(a) > 0 for a in allowed])
    # 只保留合格的 episode 编号
    ep_indices = ep_indices[eligible]
    allowed = [a for a, keep in zip(allowed, eligible) if keep]
    # 每条 episode 的"容量" = 可提供的合法起点个数
    capacity = np.array([len(a) for a in allowed], dtype=np.int64)
    # 打印全部合法起点总数
    print(int(capacity.sum()), "valid starting points found for evaluation.")

    # 本次评测需要采样的起点总数
    num_eval = int(cfg.eval.num_eval)
    if capacity.sum() < num_eval:
        # 合法起点不够就报错，避免悄悄少评
        raise ValueError(
            f"Not enough valid starts for evaluation: requested {num_eval}, "
            f"found {int(capacity.sum())}."
        )

    # 用配置里的 seed 建立可复现的随机数生成器
    g = np.random.default_rng(cfg.seed)

    # capacity-constrained round-robin: hand out one slot at a time in a shuffled
    # episode order until num_eval slots are placed, skipping saturated episodes.
    # 中文：容量受限的轮转分配 —— 按打乱后的 episode 顺序每次发 1 个名额，
    # 已达容量上限的 episode 跳过，直到发完 num_eval 个；这样各条轨迹的起点数尽量均衡，
    # 消除"均匀按步采样"对长轨迹的偏置。
    order = g.permutation(len(ep_indices))
    quotas = np.zeros(len(ep_indices), dtype=np.int64)
    remaining = num_eval
    while remaining > 0:
        for i in order:
            # 名额已发完，退出本轮
            if remaining == 0:
                break
            # 该 episode 还没到容量上限：给它多分配一个名额
            if quotas[i] < capacity[i]:
                quotas[i] += 1
                remaining -= 1

    # 输出 1：每个起点所属的 episode 编号
    eval_episodes = []
    # 输出 2：每个起点在该 episode 内的步号
    eval_start_idx = []
    for i in range(len(ep_indices)):
        # 该 episode 分到的起点个数
        k = int(quotas[i])
        if k == 0:
            # 没分到名额就跳过
            continue
        # 该 episode 的合法起点总数（容量）
        v = int(capacity[i])
        # 把 [0, v) 均匀切成 k 个连续区间（分箱），让 k 个点在轨迹内散开
        edges = np.linspace(0, v, k + 1).astype(np.int64)
        for j in range(k):
            # 第 j 个箱的左右边界
            low, high = edges[j], edges[j + 1]
            # 记录该起点所属的 episode
            eval_episodes.append(int(ep_indices[i]))
            # 在第 j 个箱内随机抽一个合法起点，保证同轨迹内不扎堆
            eval_start_idx.append(int(allowed[i][g.integers(low, high)]))

    return np.array(eval_episodes), np.array(eval_start_idx)


# ==== 把指标转换成可安全 json/yaml 序列化的纯 Python 类型 ====
def _serialize_metrics(metrics: dict):
    # 输出字典
    metrics_to_save = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            # numpy 数组 -> list
            metrics_to_save[key] = value.tolist()
        elif isinstance(value, np.generic):
            # numpy 标量 -> Python 原生标量
            metrics_to_save[key] = value.item()
        elif torch.is_tensor(value):
            # torch 张量 -> 先去图、搬到 cpu，再转 list
            metrics_to_save[key] = value.detach().cpu().tolist()
        else:
            # 其余类型（int/float/str/list/dict）原样保留
            metrics_to_save[key] = value
    return metrics_to_save


# ==== 收集并清空各层 solver 的 solve_records（每次 GD 求解的计算量元数据） ====
def pop_solve_records(solver) -> list:
    """从分层 solver 取全部子 solver 的记录；平坦 solver 视作 {1: solver}；None 返回空。"""
    # level_solvers 存在则是分层；否则若是 None 用空字典，否则把自身当作第 1 层
    solvers = getattr(solver, "level_solvers", {} if solver is None else {1: solver})
    # 按层号升序收集所有 solve_records（每条含层、地平线、采样数、迭代数、耗时）
    records = [r for level in sorted(solvers) for r in solvers[level].solve_records]
    # 清空各层记录，避免下一次评测重复累计
    for s in solvers.values():
        s.solve_records = []
    return records


# ==== 分块（可断点续跑）评测：每次评 chunk_size 个任务，结果落盘可跳过已完成块 ====
def _run_chunked_eval(
    cfg: DictConfig,
    chunk_size: int,
    world_cfg: DictConfig,
    image_shape: tuple,
    load_eval_trajs_path: str,
    process: dict,
    transform: dict,
    model,
    results_dir: Path,
) -> dict:
    """Evaluate the first eval.num_eval loaded tasks chunk_size at a time (one World of chunk_size
    envs per chunk), so that a preempted eval only redoes the chunk it was in.

    Each chunk writes results_dir/chunks/tasks_<start>-<end>.json (per-task results + GD solve
    records) and is skipped when that file exists. Every chunk reseeds the planner generators with
    base seed + its first task index: its result does not depend on which chunks ran before it in
    the process, and a single chunk (chunk_size >= num_eval) plans with the unchunked eval's seed.
    metrics.yaml aggregates the chunks with the keys of the unchunked eval.
    """
    # json 用于读写每块的结果文件（延迟导入）
    import json

    # 载入之前 dump 好的固定评测轨迹（含每个任务的起点/goal 列数据）
    payload = torch.load(load_eval_trajs_path, map_location="cpu", weights_only=False)
    # 期望的任务总数
    n = int(cfg.eval.num_eval)
    if len(payload["data"]) < n:
        # 轨迹数少于配置要求：直接报错
        raise ValueError(f"{load_eval_trajs_path}: {len(payload['data'])} tasks < eval.num_eval={n}")
    # 每个任务来自哪条 episode（可能缺失）
    ep_idx = payload.get("episodes_idx")
    # 缺失就退化为 0..n-1 的默认编号
    ep_idx = list(ep_idx) if ep_idx is not None and len(ep_idx) else list(range(len(payload["data"])))
    # 取评测期需要额外调用的环境设置回调（callables），解析成普通容器
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    # 存放各块结果的子目录
    chunk_dir = results_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # 预先算好每块"起始任务号 -> 结果 json 路径"（文件已存在 = 该块已完成）
    chunk_paths = {
        s: chunk_dir / f"tasks_{s:03d}-{min(s + chunk_size, n):03d}.json" for s in range(0, n, chunk_size)
    }

    # policy 惰性构建（只在遇到第一个未完成块时构建一次）；gens 记录各层 solver 的随机生成器
    policy, gens = None, []
    for start, path in chunk_paths.items():
        if path.exists():
            # 该块已有结果文件，跳过 —— 这就是断点续跑
            continue
        # 本块的任务区间 [start, end)，最后一块可能不足 chunk_size
        end = min(start + chunk_size, n)
        if policy is None:
            # 首次真正需要评测时才构建 policy（加载模型、构建 solver）
            policy = _build_policy(cfg, process, transform, model=model)
            solver = getattr(policy, "solver", None)
            # 记录每个子 solver 的 torch 随机生成器及其初始种子
            gens = [
                (s.torch_gen, s.torch_gen.initial_seed())
                for s in getattr(solver, "level_solvers", {1: solver}).values()
                if hasattr(s, "torch_gen")
            ]
        # 每块把生成器重播为 基础种子 + 本块首个任务号，
        # 使本块结果与"之前跑过哪些块"无关，单独跑某块也能复现
        for gen, seed in gens:
            gen.manual_seed(seed + start)
        # shallow per-episode copies: evaluate_from_dataset reassigns their columns in place
        # 中文：浅拷贝每个 episode 的字典（evaluate_from_dataset 会原地改写这些列），
        # 并只切出本块 [start, end) 的任务，避免污染原始 payload。
        chunk = {**payload, "data": [dict(ep) for ep in payload["data"][start:end]], "episodes_idx": ep_idx[start:end]}
        # 为本块创建一个含 (end-start) 个并行环境的 World（一个任务占一个 env）
        world = swm.World(**{**world_cfg, "num_envs": end - start}, image_shape=image_shape)
        try:
            # 把策略挂到 World 上（内部会 configure solver 并 reset 状态）
            world.set_policy(policy)
            t0 = time.time()
            # enable_grad：规划的梯度下降需要梯度；MATH 后端的 sdpa 支持带梯度的注意力
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                # 闭环评测主入口：加载本块固定轨迹的起点与真值 goal，让 policy 在环境里
                # 逐步 规划->执行->观测，直到到达 goal 或耗尽 eval_budget，返回指标字典
                metrics = world.evaluate_from_dataset(
                    None,
                    episodes_idx=None,
                    start_steps=None,
                    goal_offset_steps=cfg.eval.goal_offset_steps,
                    eval_budget=cfg.eval.eval_budget,
                    callables=callables,
                    load_eval_trajs_path=chunk,
                    process=process,
                    start_state_mode=cfg.eval.get("start_state_mode", "dataset_full"),
                    goal_state_mode=cfg.eval.get("goal_state_mode", "dataset_full"),
                )
            # 记录本块墙钟耗时
            metrics["evaluation_time"] = time.time() - t0
        finally:
            # 无论成败都关闭环境，释放并行进程/资源
            world.close()
        # 取走本块各层 solver 累计的 GD 求解记录并清空
        metrics["solve_records"] = pop_solve_records(getattr(policy, "solver", None))
        # 先写 .tmp 文件
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_serialize_metrics(metrics)))
        os.replace(tmp, path)  # atomic: a preemption never leaves a partial chunk file
        # 打印本块进度与成功率
        print(f"tasks {start}-{end}: success_rate {metrics['success_rate']:.1f}")

    # ---- 段落：聚合所有块的指标 ----
    # 逐块把 json 结果读回（按块顺序）
    chunks = [json.loads(p.read_text()) for p in chunk_paths.values()]
    # 拼接各块的"逐任务是否成功"布尔数组 -> 总长度 n
    successes = np.concatenate([np.asarray(c["episode_successes"], dtype=bool) for c in chunks])
    # 拼接"到达成功所用步数"（未成功者为 inf/NaN）
    steps = np.concatenate([np.asarray(c["steps_to_success"], dtype=np.float64) for c in chunks])
    # 拼接各块的种子列表；全为 None 时保持 None
    seeds = None if chunks[0]["seeds"] is None else [s for c in chunks for s in c["seeds"]]
    if seeds is not None:
        # 断言所有 episode 种子互不相同，保证各任务独立、结果可比
        assert np.unique(np.asarray(seeds)).shape[0] == n, "Some episode seeds are identical!"
    # 组装最终要保存的指标（键与不分块评测保持一致）
    metrics_to_save = {
        # 总成功率 = 成功任务数 / 任务总数 × 100
        "success_rate": float(successes.sum()) / n * 100.0,
        # 逐任务成功标志
        "episode_successes": successes.tolist(),
        # 逐任务种子
        "seeds": seeds,
        # 逐任务到达成功的墙钟时间
        "wall_clock_to_success": [x for c in chunks for x in c["wall_clock_to_success"]],
        # 逐任务到达成功的步数
        "steps_to_success": steps.tolist(),
        # 数据集中各原始轨迹长度（供分析起点/轨迹关系）
        "loaded_eval_original_lengths": [x for c in chunks for x in c["loaded_eval_original_lengths"]],
        # 只统计成功 episode 的平均到达步数（一个都没成功则为 NaN）
        "steps_to_success_success_only": (
            float(np.mean(steps[np.isfinite(steps)])) if np.isfinite(steps).any() else float("nan")
        ),
        # 所有块耗时之和
        "evaluation_time": sum(c["evaluation_time"] for c in chunks),
        # 本次使用的块大小
        "chunk_size": chunk_size,
    }
    # 写出最终指标文件
    (results_dir / "metrics.yaml").write_text(OmegaConf.to_yaml(metrics_to_save))
    print(metrics_to_save)
    # 汇总所有块的 GD 求解计算量记录
    solve_records = [r for c in chunks for r in c["solve_records"]]
    if solve_records:
        # 写出规划计算量明细（供 planner-FLOPs 分析）
        (results_dir / "planning_compute.json").write_text(json.dumps(solve_records))
    return metrics_to_save


# ==== 评测总入口：配置 -> policy -> World 闭环评测 -> 指标落盘 ====
def run_planning_eval(
    cfg: DictConfig,
    model=None,
    results_dir: str | Path | None = None,
):
    # 深拷贝配置且暂不解析插值（配置稍后可能被原地修改，如 _drop_levels_above_model）
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    # 决定结果写入目录
    resolved_results_dir = resolve_results_dir(cfg, results_dir)
    # 创建结果目录（含父目录）
    resolved_results_dir.mkdir(parents=True, exist_ok=True)
    # 保存本次评测的配置快照
    save_eval_config(resolved_results_dir, cfg)

    # 环境最大步数设为 2 × 评测预算，为"到达 goal 后的收尾"留余量
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    # 解析插值后再拷出一份干净的 World 构造配置
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))

    # 像素与 goal 各挂一条完全相同的图像预处理链
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    # 可选：加载已 dump 的固定评测轨迹（让起点/goal 与之前某次评测完全一致）
    load_eval_trajs_path = cfg.get("load_eval_trajs_path", None)
    if load_eval_trajs_path:
        # 解析成真实存在的路径（尝试 cwd 与本文件目录）
        load_eval_trajs_path = str(_resolve_existing_path(load_eval_trajs_path))
    # 可选：把本次采样出的评测轨迹导出，供以后复用
    dump_eval_trajs_path = cfg.get("dump_eval_trajs_path", None)
    # 只导出轨迹、不做评测的模式
    dump_eval_only = dump_eval_trajs_path is not None

    # 评测图像的空间尺寸 (H, W)
    image_shape = (int(cfg.eval.img_size),) * 2

    # 以下三项只在"直接从数据集采样起点"时才需要
    dataset = None
    eval_episodes = None
    eval_start_idx = None
    # 观测归一化 process，默认为空字典
    process = {}

    if dump_eval_only:
        # 导出模式：打开数据集并分层采样评测起点
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        eval_episodes, eval_start_idx = _sample_stratified_eval_starts(cfg, dataset)
    else:
        # 评测模式：构建与训练一致的观测归一化器
        process = _build_eval_process_from_policy_normalizer(cfg, model=model)

    # 每块任务数（可为 null）；配置了 chunk_size 就走"可断点续跑"的分块评测
    chunk_size = cfg.eval.get("chunk_size", 1)  # resumable per chunk of N tasks; null: one World of all tasks
    if chunk_size and not dump_eval_only:
        if not load_eval_trajs_path:
            # 分块评测必须建立在"已 dump 的固定轨迹"之上
            raise ValueError("eval.chunk_size needs load_eval_trajs_path (+eval.chunk_size=null to run unchunked)")
        return _run_chunked_eval(
            cfg, int(chunk_size), world_cfg, image_shape, load_eval_trajs_path,
            process, transform, model, resolved_results_dir,
        )

    # 创建 World：cfg.world 提供环境类型/数量等，这里再补图像形状
    world = swm.World(**world_cfg, image_shape=image_shape)

    try:
        # 构建 policy（内部完成模型加载 + solver 构建）
        policy = _build_policy(cfg, process, transform, model=model)
        # 挂到 World：会调用 solver.configure 并 reset 各自状态
        world.set_policy(policy)

        start_time = time.time()
        # 规划的梯度下降需要梯度；MATH 注意力后端支持带梯度的 sdpa 计算
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            # 解析评测期环境设置回调（callables）
            callables = OmegaConf.to_container(
                cfg.eval.get("callables"), resolve=True
            )
            # 起点步号（分层采样得到）转 list；None 表示由 World 自行决定
            start_steps = (
                eval_start_idx.tolist() if eval_start_idx is not None else None
            )
            # 起点所属的 episode 编号
            episodes_idx = (
                eval_episodes.tolist() if eval_episodes is not None else None
            )
            # 核心：闭环评测。World 内并行跑 num_envs 个环境，每步把观测交给 policy
            # （其内部调 solver.solve 用世界模型规划动作），环境执行动作产生新观测，
            # 如此闭环推进直到各 episode 到达真值 goal 或耗尽 eval_budget，
            # 最后聚合出 success_rate、steps_to_success 等指标
            metrics = world.evaluate_from_dataset(
                dataset,
                start_steps=start_steps,
                goal_offset_steps=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=episodes_idx,
                callables=callables,
                dump_eval_trajs_path=dump_eval_trajs_path,
                load_eval_trajs_path=load_eval_trajs_path,
                process=process,
                start_state_mode=cfg.eval.get(
                    "start_state_mode", "dataset_full"
                ),
                goal_state_mode=cfg.eval.get(
                    "goal_state_mode", "dataset_full"
                ),
            )
        end_time = time.time()
    finally:
        # 无论评测是否异常都关闭环境，避免残留子进程
        world.close()

    if dump_eval_only:
        # 读回刚导出的轨迹文件，补写元信息
        payload = torch.load(dump_eval_trajs_path, weights_only=False)
        payload["task_info"] = {
            # 来源数据集名
            "dataset_name": str(cfg.eval.dataset_name),
            # 采样方式标记
            "sampling_mode": "stratified",
            # 采样种子
            "seed": int(cfg.seed),
            # 采样的评测任务数
            "num_episodes": int(cfg.eval.num_eval),
            # 完整评测配置
            "eval_config": OmegaConf.to_container(cfg, resolve=False),
            # 命令行 / git 版本 / 时间等复现溯源信息
            **generation_provenance(),
        }
        # 覆盖保存轨迹文件
        torch.save(payload, dump_eval_trajs_path)
        print(metrics)
        return metrics

    # 序列化指标（numpy/torch -> 纯 Python 类型）
    metrics_to_save = _serialize_metrics(metrics)
    # 补上总评测耗时
    metrics_to_save["evaluation_time"] = end_time - start_time

    metrics_path = resolved_results_dir / "metrics.yaml"
    # 写出最终指标文件
    metrics_path.write_text(OmegaConf.to_yaml(metrics_to_save))
    print(metrics)

    # Per-solve GD compute metadata (level, horizon, num_samples, per-env n_iters,
    # wall_clock) for the planner-FLOPs Pareto analysis.
    # 中文：收集每次 GD 求解的计算量元数据（层、地平线、采样数、每环境迭代数、
    # 墙钟时间），用于 planner-FLOPs 的帕累托分析。
    solve_records = pop_solve_records(getattr(policy, "solver", None))
    if solve_records:
        import json
        # 写出规划计算量明细文件
        (resolved_results_dir / "planning_compute.json").write_text(
            json.dumps(solve_records)
        )

    return metrics_to_save
