# SongCi GPT

基于深度学习的宋词生成项目，包含两种实现方式：

1. **从零实现**：基于 PyTorch 从零实现 GPT 模型，支持 MoE（共享专家 + 路由专家）、MLA、弹性训练（ERNIE 5.0 Elastic Training），以及 DPO 对齐训练
2. **Unsloth 微调**：基于 Unsloth 框架微调 Qwen3-0.6B

## 项目结构

```
├── scratch/              # 从零实现
│   ├── model.py          # GPT 模型（MoE/MLA/共享专家/弹性训练）
│   ├── attention.py      # 注意力机制（MHA/MLA）
│   ├── tokenizer.py      # BPE 分词器
│   ├── train_sft.py      # SFT 训练脚本（含弹性训练）
│   ├── train_dpo.py      # DPO 对齐训练脚本
│   ├── generate_dpo_pairs.py  # DPO 偏好对生成（SongEval 作为奖励模型）
│   ├── inference.py      # 推理脚本
│   ├── configs/          # 配置文件
│   │   ├── mha.yaml      # 6层 MHA 配置
│   │   ├── mha_24l.yaml  # 24层全 MoE + 弹性训练配置
│   │   └── mla.yaml      # MLA 配置
│   ├── ckpt/             # 模型检查点
│   └── songeval/         # 格律评估系统 ⭐
│
├── unsloth/              # Unsloth 微调
│   └── train_qwen_songci.py
│
├── dataset/              # 宋词数据集
│   └── dpo/              # DPO 偏好数据
└── pyproject.toml        # 项目依赖
```

## 格律评估系统

基于逆向工程从训练语料提取格律标准，自动评估生成宋词的格律符合度。

### 评估指标

| 指标 | 说明 |
|------|------|
| 结构匹配 | 句数、每句字数是否符合词牌 |
| 平仄准确度 | 与标准平仄模板的符合度 |
| 押韵一致性 | 韵脚组内韵部的一致性（支持换韵） |
| 综合格律分 | 加权得分（结构40% + 平仄40% + 押韵20%） |

### 最新评估结果（50词牌）

| 模型 | 结构匹配 | 平仄准确度 | 押韵一致性 | 综合得分 |
|------|---------|-----------|-----------|---------|
| MHA | 78.00% | 85.88% | 66.24% | 75.06 |
| MLA | **82.00%** | 85.37% | 55.90% | **75.34** |
| full_attn_res | 74.00% | **84.72%** | **69.22%** | 72.88 |

详细评估方法和结果见 [`scratch/songeval/README.md`](scratch/songeval/README.md)

## DPO 对齐训练

使用 SongEval 格律评估作为奖励模型，自动标注偏好对，通过 DPO 提升生成质量。

```bash
# 1. 生成偏好对（需要先有 SFT 模型）
uv run python scratch/generate_dpo_pairs.py \
  --config_path=./scratch/configs/mha.yaml \
  --num_candidates=8 \
  --temperatures="0.7,0.9,1.0,1.2"

# 2. DPO 训练
uv run python scratch/train_dpo.py \
  --config_path=./scratch/configs/mha.yaml \
  --dpo_data_path=./dataset/dpo \
  --ref_ckpt_path=./scratch/ckpt/mha.pt
```

**流程**: SFT 预训练 → SongEval 评估候选 → 标注 chosen/rejected → DPO 对齐

## 弹性训练 (Elastic Training)

受 ERNIE 5.0 启发，一次预训练同时优化整个模型家族。训练时随机缩减模型的深度、宽度、稀疏度，使同一套参数在各种配置下都能正常工作，部署时可按需提取不同大小的子网络。

```bash
# 24层全 MoE 模型（含弹性训练）
uv run python scratch/train_sft.py --config_path=./scratch/configs/mha_24l.yaml
```

**三个维度：**
- **弹性深度**：25% 概率 Bypassing 跳层（Stochastic Depth，残差直连）
- **弹性宽度**：20% 概率缩减路由专家（共享专家不受影响）
- **弹性稀疏度**：20% 概率缩减路由 top-k

**模型对比：**

| 配置 | 总参数 | 激活参数 | MoE 策略 |
|------|--------|---------|---------|
| 6层 mha.yaml | 41.8M | 29.1M | 交错 MoE（4专家, top-1） |
| 24层 mha_24l.yaml | 489.2M | 186.5M | 全 MoE（1共享+8路由, top-2） |

## 快速开始

```bash
# 安装依赖
uv sync

# 训练分词器
uv run python scratch/tokenizer.py

# SFT 训练（6层基础模型）
uv run python scratch/train_sft.py --config_path=./scratch/configs/mha.yaml

# SFT 训练（24层全 MoE + 弹性训练，推荐）
uv run python scratch/train_sft.py --config_path=./scratch/configs/mha_24l.yaml

# DPO 对齐（可选）
uv run python scratch/generate_dpo_pairs.py
uv run python scratch/train_dpo.py --ref_ckpt_path=./scratch/ckpt/mha.pt

# 交互式推理
uv run python scratch/inference.py

# 评估模型
cd scratch/songeval
uv run python evaluate_model.py --model_type scratch --config_path ../configs/mha.yaml --num_titles 50
```

## 两种实现对比

| 特性 | Scratch | Unsloth |
|------|---------|---------|
| 基础模型 | 从零实现 | Qwen3-0.6B |
| 分词器 | 自定义 BPE | Qwen3 Tokenizer |
| 训练方式 | SFT + DPO 对齐 | LoRA (1-10%) |
| 显存优化 | KV Cache + MLA | 4-bit 量化 |
| 架构特性 | MoE（共享专家+路由专家）、弹性训练 | 标准 Transformer |
| 适用场景 | 学习原理、架构实验、质量对齐 | 生产部署 |

## 详细文档

- Scratch 实现：[`scratch/README.md`](scratch/README.md)
- Unsloth 微调：[`unsloth/README.md`](unsloth/README.md)
- 格律评估：[`scratch/songeval/README.md`](scratch/songeval/README.md)

## 数据集

宋词数据集来源：https://github.com/chinese-poetry/chinese-poetry

## 许可证

MIT License
