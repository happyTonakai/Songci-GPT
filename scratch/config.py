"""配置文件管理模块

使用 dataclass 定义配置结构，支持从 YAML 文件加载配置。
"""

from dataclasses import dataclass

import yaml

@dataclass
class DPOParams:
    """DPO 专用超参数"""
    ref_ckpt_path: str
    data_path: str
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"


@dataclass
class TrainConfig:
    """训练配置"""

    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: float
    device: str
    save_interval: int
    ckpt_path: str
    num_workers: int
    warmup_steps: int = 0
    aux_loss_target_ratio: float = 0.01
    dpo: DPOParams | None = None


@dataclass
class ModelConfig:
    """模型配置"""

    vocab_size: int
    max_seq_len: int
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    n_experts: int
    topk: int
    dropout: float
    use_mla: bool = False
    latent_dim: int | None = None
    rope_head_dim: int | None = None
    use_attn_res: bool | str = False
    n_shared_experts: int = 0  # 共享专家数（always-on，不参与路由和负载均衡）


@dataclass
class DataConfig:
    """数据配置"""

    data_path: str
    tokenizer_path: str
    max_seq_len: int
    num_workers: int


@dataclass
class InferenceConfig:
    """推理配置"""

    temperature: float
    top_k: int
    top_p: float
    max_len: int


@dataclass
class ElasticConfig:
    """Elastic Training 配置 (ERNIE 5.0)

    在训练时随机缩减模型的深度、宽度、稀疏度，
    使得同一套参数在各种配置下都能正常工作。
    """

    depth_prob: float = 0.0  # 缩减深度的概率
    width_prob: float = 0.0  # 缩减专家宽度的概率
    sparsity_prob: float = 0.0  # 缩减 top-k 的概率
    # 预定义配置库：各维度可选的值
    depth_levels: list[int] | None = None  # 可选深度列表，如 [6, 4, 2]
    width_levels: list[int] | None = None  # 可选专家数列表，如 [4, 2]
    sparsity_levels: list[int] | None = None  # 可选 top-k 列表，如 [2, 1]
    # 课程学习：前 N 步使用全量网络，之后才开启弹性
    warmup_steps: int = 0
    # 核心专家：这些专家永不被宽度缩减屏蔽（类似 DeepSeek 的 Shared Expert）
    core_experts: list[int] | None = None


@dataclass
class Config:
    """总配置类"""

    train: TrainConfig
    model: ModelConfig
    data: DataConfig
    inference: InferenceConfig
    elastic: ElasticConfig | None = None


def load_config(config_path: str) -> Config:
    """从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config: 配置对象
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    train_dict = config_dict["train"]
    if "dpo" in train_dict and train_dict["dpo"] is not None:
        train_dict["dpo"] = DPOParams(**train_dict["dpo"])
    train_cfg = TrainConfig(**train_dict)
    model_cfg = ModelConfig(**config_dict["model"])
    data_cfg = DataConfig(**config_dict["data"])
    inference_cfg = InferenceConfig(**config_dict["inference"])

    elastic_cfg = None
    if "elastic" in config_dict and config_dict["elastic"] is not None:
        elastic_cfg = ElasticConfig(**config_dict["elastic"])

    return Config(
        train=train_cfg,
        model=model_cfg,
        data=data_cfg,
        inference=inference_cfg,
        elastic=elastic_cfg,
    )
