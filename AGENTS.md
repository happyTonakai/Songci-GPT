# AGENTS.md - Songci-GPT 项目指南

## 项目概述

**Songci-GPT** 是一个基于深度学习的宋词生成项目，提供两种实现方式：

1. **从零实现 (scratch/)**：基于 PyTorch 从零实现 GPT 模型，包含自定义 BPE 分词器、MoE (Mixture of Experts)、MLA (Multi-head Latent Attention)、KV Cache 推理优化、ERNIE 5.0 弹性训练 (Elastic Training)，以及基于 DPO 的对齐训练
2. **Unsloth 微调 (unsloth/)**：基于 Unsloth 框架微调 Qwen3-0.6B，使用 LoRA 高效微调技术

## 项目结构

```
/home/hanzerui/joyspace/gpt/
├── README.md                 # 项目主文档
├── pyproject.toml           # 项目依赖配置 (uv 管理)
├── uv.lock                  # 依赖锁定文件
├── AGENTS.md               # 本文件
│
├── scratch/                # 从零实现的 GPT 模型
│   ├── README.md           # scratch 详细文档
│   ├── model.py            # GPT 模型定义（支持 MoE 和 MLA）
│   ├── attention.py        # 注意力机制（MHA 和 MLA 实现）
│   ├── config.py           # YAML 配置管理
│   ├── tokenizer.py        # BPE 分词器实现
│   ├── dataset.py          # 数据集加载和预处理
│   ├── train_sft.py        # SFT 训练脚本
│   ├── train_dpo.py        # DPO 对齐训练脚本
│   ├── generate_dpo_pairs.py # DPO 偏好对生成脚本（SongEval 作为奖励模型）
│   ├── inference.py        # 交互式推理脚本
│   ├── tokenizer.json      # 预训练分词器
│   ├── ckpt/              # 模型检查点目录
│   ├── configs/           # 配置文件目录
│   │   ├── mha.yaml       # 6层 MHA 配置（交错 MoE）
│   │   ├── mha_24l.yaml   # 24层全 MoE 配置（共享专家 + 弹性训练）
│   │   └── mla.yaml       # Multi-head Latent Attention 配置
│   └── songeval/          # 宋词格律评估系统
│
├── unsloth/               # Unsloth 微调实现
│   ├── README.md          # unsloth 详细文档
│   └── train_qwen_songci.py  # 训练和推理脚本
│
├── dataset/               # 数据集
│   ├── generate_songci_qa.py      # OpenAI API 问答对生成脚本
│   ├── README_songci_qa.md        # 问答生成文档
│   ├── train-en.txt               # 英文训练语料
│   ├── train-zh.txt               # 中文训练语料
│   ├── dpo/                       # DPO 偏好数据
│   │   └── dpo_pairs.json         # 生成的偏好对
│   └── 宋词/                     # 宋词 JSON 数据
│       ├── ci.song.*.json        # 宋词数据文件（多个分片）
│       ├── 宋词三百首.json        # 宋词三百首精选
│       ├── main.py               # 数据处理脚本
│       ├── UpdateCi.py           # 数据更新脚本
│       └── songci_qa/            # 生成的问答对数据
│
├── outputs/               # 训练输出（检查点）
│   └── checkpoint-*/     # 各阶段检查点
│
└── qwen3-0.6b-songci-lora/  # Unsloth 微调保存的 LoRA 权重
```

## 环境配置

### 依赖管理

项目使用 **uv** 作为包管理器：

```bash
# 安装依赖
uv sync

# 运行脚本
uv run python <script.py>
```

### Python 版本

- 主项目: Python >= 3.13
- 实际运行环境: Python 3.8.10 (Linux)

### 关键依赖

- `torch>=2.9.1` - 深度学习框架
- `transformers>=4.57.1` - 模型加载和推理
- `unsloth>=2026.1.2` - 高效微调框架
- `peft==0.18.1` - LoRA 等参数高效微调
- `jieba>=0.42.1` - 中文分词
- `orjson>=3.11.5` - 高性能 JSON 处理
- `pyyaml>=6.0.3` - YAML 配置解析
- `einops>=0.8.2` - 张量操作
- `fire>=0.7.1` - 命令行参数解析

## 三种训练方式

| 阶段 | 脚本 | 说明 |
|------|------|------|
| SFT | `train.py` / `train_sft.py` | 监督微调，学习宋词格式 |
| DPO 偏好对生成 | `generate_dpo_pairs.py` | 用 SongEval 评估，自动标注 chosen/rejected |
| DPO 对齐训练 | `train_dpo.py` | 从偏好数据学习，提升格律符合度 |

## 两种实现方式对比

| 特性 | Scratch 实现 | Unsloth 微调 |
|------|-------------|--------------|
| 基础模型 | 从零实现 | Qwen3-0.6B-Base |
| 分词器 | 自定义 BPE | Qwen3 Tokenizer |
| 训练方式 | SFT + DPO 对齐 | LoRA (1-10% 参数) |
| 显存优化 | KV Cache + MLA | 4-bit 量化 |
| 架构特性 | MoE、MLA、共享专家、弹性训练、RoPE | 标准 Transformer |
| 对齐方法 | SongEval 格律评估 → DPO | 无 |
| 适用场景 | 学习原理、架构实验、质量对齐 | 生产部署 |

## 使用方法

### 方式一：Scratch 实现

#### 1. 训练分词器

```bash
uv run python scratch/tokenizer.py
```

- 读取 `dataset/宋词/` 目录下的所有 JSON 文件
- 训练词表大小为 10000 的 BPE 分词器
- 保存到 `scratch/ckpt/songci_tokenizer.json`

#### 2. 训练模型

**6层 MHA 配置（基础模型）：**

```bash
uv run python scratch/train_sft.py --config_path=./scratch/configs/mha.yaml
```

**24层全 MoE 配置（推荐，含弹性训练）：**

```bash
uv run python scratch/train_sft.py --config_path=./scratch/configs/mha_24l.yaml
```

**MLA 配置（更低内存占用）：**

```bash
uv run python scratch/train_sft.py --config_path=./scratch/configs/mla.yaml
```

**配置参数说明：**

| 参数 | 6层模型 (mha.yaml) | 24层模型 (mha_24l.yaml) |
|------|-------------------|------------------------|
| num_layers | 6 | 24 |
| embedding_dim | 512 | 512 |
| hidden_dim | 2048 | 2048 |
| n_experts | 4 (交错 MoE) | 8 (全 MoE) |
| n_shared_experts | 0 | 1 (always-on) |
| topk | 1 | 2 |
| 总参数 | 41.8M | 489.2M |
| 激活参数 | 29.1M | 186.5M |
| 弹性训练 | 关闭 | 开启 |

#### 3. 推理生成

```bash
# 默认配置
uv run python scratch/inference.py

# MLA 配置
uv run python scratch/inference.py --config_path=./scratch/configs/mla.yaml
```

交互式输入词牌名即可生成宋词，输入 `q` 退出。

#### 4. DPO 对齐训练（可选）

DPO (Direct Preference Optimization) 通过偏好数据对模型进行对齐，使生成的宋词更符合格律规范。

**第一步：生成偏好对**

使用 SongEval 格律评估作为奖励模型，自动标注偏好对：

```bash
# 生成偏好对（需要先有训练好的 SFT 模型）
uv run python scratch/generate_dpo_pairs.py \
  --config_path=./scratch/configs/mha.yaml \
  --num_pairs=1000 \
  --num_candidates=8 \
  --temperatures="0.7,0.9,1.0,1.2" \
  --output_path=./dataset/dpo/dpo_pairs.json
```

**参数说明：**

- `num_pairs`: 目标偏好对数量，循环遍历 75 个词牌直到达到该数量（默认 1000）
- `num_candidates`: 每个词牌每轮生成的候选数量（越多偏好对质量越高）
- `temperatures`: 逗号分隔的温度列表，循环使用以增加多样性
- 奖励模型：SongEval 综合评分（结构 40% + 平仄 40% + 押韵 20%）

**第二步：DPO 训练**

```bash
uv run python scratch/train_dpo.py \
  --config_path=./scratch/configs/mha.yaml \
  --dpo_data_path=./dataset/dpo \
  --ref_ckpt_path=./scratch/ckpt/mha.pt
```

**参数说明：**

- `dpo_data_path`: 偏好数据目录（包含 dpo_pairs.json）
- `ref_ckpt_path`: reference model 的 checkpoint（通常是 SFT 训练好的模型）

**DPO 核心组件：**

- **Policy Model**: 可训练的模型，学习偏好
- **Reference Model**: 冻结的 SFT 模型，作为基线
- **DPO Loss**: `-log(sigmoid(beta * (π_chosen - π_rejected)))`
  - `π = log_prob(policy) - log_prob(reference)`
  - `beta`: 温度参数，控制偏离 reference model 的程度

### 方式二：Unsloth 微调

#### 1. 训练模型

```bash
# 从头训练
uv run python unsloth/train_qwen_songci.py --mode train

# 从检查点继续
uv run python unsloth/train_qwen_songci.py --mode train --ckpt qwen3-0.6b-songci-lora
```

#### 2. 推理生成

```bash
# 流式输出（默认）
uv run python unsloth/train_qwen_songci.py --mode infer

# 禁用流式输出
uv run python unsloth/train_qwen_songci.py --mode infer --no-stream
```

### 问答对生成（可选）

使用 OpenAI API 生成宋词问答对：

```bash
# 设置环境变量
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='gpt-3.5-turbo'

# 运行生成脚本
cd dataset
uv run python generate_songci_qa.py \
  --input-dir 宋词 \
  --output-dir songci_qa \
  --max-files 5 \
  --delay 1.0
```

## 核心技术特点

### 1. BPE 分词器 (scratch)

- 支持中英文混合分词
- 使用 `jieba` 进行中文预分词
- 特殊标记: `<bos>`, `<eos>`, `<unknown>`, `</w>`, `<mask>`, `<sep>`, `<pad>`
- `</w>` 标记表示词边界

### 2. MoE (Mixture of Experts)

- **6层模型**：交错式架构，奇数层 MoE（4专家，top-1），偶数层 Dense
- **24层模型**：全 MoE 架构，每层包含 1 个共享专家（always-on）+ 8 个路由专家（top-2）
- **共享专家**：每个 token 必定经过，提取通用特征，不参与路由和负载均衡
- **路由专家**：通过 Router 动态选择 top-k 个，负载均衡损失确保均匀使用
- 每个专家是独立的 FFN（Linear → GELU → Linear）

### 3. MLA (Multi-head Latent Attention)

- DeepSeek-V2/V3 提出的注意力机制
- 压缩 KV 到低维潜在空间 (latent_dim=64)
- 解耦 RoPE：Content 部分压缩，RoPE 部分携带位置信息
- 大幅降低 KV Cache 内存占用

### 4. KV Cache 优化

- 缓存历史 token 的 K 和 V
- 每次只计算新 token 的 K 和 V
- 推理速度提升约 10-100 倍
- 通过 `offset` 参数正确递增位置索引

### 5. RoPE (旋转位置编码)

- 替代传统位置编码
- 更好的长序列外推能力
- 支持 KV Cache 场景

### 6. YAML 配置驱动

- 训练和推理参数通过 YAML 管理
- 便于实验复现和超参调优
- 支持 MHA 和 MLA 两种配置

### 7. DPO 对齐训练

- **Direct Preference Optimization**: 无需训练独立的奖励模型，直接从偏好数据学习
- **SongEval 作为奖励模型**: 使用格律评估系统（结构 + 平仄 + 押韵）自动标注偏好对
- **多样性采样**: 对同一词牌使用不同 temperature 生成多个候选，评估后选最优/最差作为偏好对
- **Reference Model**: 冻结的 SFT 模型作为基线，防止模型遗忘已学到的知识
- **训练流程**: SFT 预训练 → 偏好对生成 → DPO 对齐 → 生成质量提升
- **DPO 实验结果** (375样本): 结构匹配率 80.80%→84.80% (+4.00%)，平仄准确度 87.45%→91.32% (+3.87%)，押韵一致性 67.42%→70.12% (+2.70%)，综合格律得分 81.99%→85.74% (+3.75%)
- **DPO 训练配置**: 1000偏好对，beta=0.5，lr=5e-5，5 epochs，batch_size=32
- **偏好对生成参数**: num_candidates=8，temperatures=[0.7, 0.9, 1.0, 1.2]，min_chosen_score=0.90，max_rejected_score=0.50

### 8. 弹性训练 (Elastic Training)

受 ERNIE 5.0 论文启发，在预训练阶段同时优化一个完整的模型家族。核心思想：每次训练迭代以一定概率随机缩减模型的深度、宽度、稀疏度，使得同一套参数在各种配置下都能正常工作。

**三个正交维度：**

| 维度 | 机制 | 概率 | 说明 |
|------|------|------|------|
| **弹性深度** | Bypassing 跳层 | 25% | 被跳过的层执行恒等映射（Stochastic Depth） |
| **弹性宽度** | Masking 路由专家 | 20% | 将非活跃专家 logits 设为 -inf，共享专家不受影响 |
| **弹性稀疏度** | 缩减 top-k | 20% | 从预定义范围中随机选更小的 k |

**工程细节：**

- **Bypassing vs 截断**：采用跳层（Bypassing）而非截断前缀。残差连接使得被跳过的层成为恒等映射 `X_{l+1} = X_l`，梯度无损穿过，而截断会强迫浅层学深层特征
- **课程学习**：`warmup_steps` 之前使用全量网络稳定训练，之后才开启弹性
- **核心专家**：`core_experts` 指定的路由专家永不被弹性宽度屏蔽（类似 DeepSeek Shared Expert）
- **共享专家**：always-on，不参与弹性宽度和负载均衡

**配置示例：**

```yaml
elastic:
  depth_prob: 0.25
  width_prob: 0.20
  sparsity_prob: 0.20
  depth_levels: [24, 18, 12, 6]    # 预定义配置库
  width_levels: [8, 4, 2]
  sparsity_levels: [2, 1]
  warmup_steps: 1000               # 课程学习
  core_experts: [0]                # 核心专家永不屏蔽
```

**训练日志**：进度条显示当前弹性配置，如 `Elastic [depth=12, width=4, topk=1]`

## 宋词格律评估系统 (SongEval)

位于 `scratch/songeval/`，用于自动评估生成宋词的格律符合度。

### 功能特性

- **格律库构建**：从训练语料逆向工程提取 75 个常用词牌的格律标准
- **结构匹配**：评估句数、每句字数是否符合标准
- **平仄准确度**：使用 `pypinyin` 判定每个字的平仄，对比标准模板
- **押韵一致性**：分析押韵位置韵部的一致性
- **综合打分**：加权计算格律得分 (结构40% + 平仄40% + 押韵20%)

### 格律库 (standard.json)

已提交到 Git，包含 75 个词牌：

- **结构标准**：每句字数（如浣溪沙 `[7,7,7,7,7,7]`）
- **平仄模板**：`P`=平声, `Z`=仄声, `*`=平仄不拘
- **押韵位置**：强制押韵的句索引

### 模型评估结果（750样本）

| 指标 | MHA | MLA | 差异 |
|------|-----|-----|------|
| 结构匹配率 | **82.4%** | 76.8% | MHA +5.6% |
| 平均平仄准确度 | **87.2%** | 85.9% | MHA +1.3% |
| 综合格律得分 | **83.9/100** | 79.8/100 | **MHA +4.1分** |

**结论**：MHA 模型生成质量更好，MLA 虽然节省内存但牺牲了一定精度。

### 使用方法

```bash
cd scratch/songeval

# 评估 MHA 模型
uv run python evaluate_model.py \
  --model_type scratch \
  --config_path ../configs/mha.yaml \
  --num_titles 50 \
  --output eval_mha.json

# 评估 MLA 模型
uv run python evaluate_model.py \
  --model_type scratch \
  --config_path ../configs/mla.yaml \
  --num_titles 50 \
  --output eval_mla.json

# 对比结果
python3 << 'EOF'
import json
with open('eval_mha.json') as f: mha = json.load(f)
with open('eval_mla.json') as f: mla = json.load(f)
print(f"MHA: {mha['aggregate_scores']['avg_form_score']*100:.1f}/100")
print(f"MLA: {mla['aggregate_scores']['avg_form_score']*100:.1f}/100")
EOF
```

### 训练集质量分析

评估系统还包含训练集格律符合度统计：

- **高质量词牌** (>=90%)：31个，如浣溪沙、鹧鸪天、玉楼春
- **中质量词牌** (70-90%)：21个，如水调歌头、满江红
- **低质量词牌** (<70%)：23个，如沁园春、水龙吟（变体过多）

详细文档见 `scratch/songeval/README.md`

## 数据格式

### 宋词 JSON 格式

```json
{
  "rhythmic": "水调歌头",
  "paragraphs": ["明月几时有？把酒问青天。", "不知天上宫阙，今夕是何年。"]
}
```

### 训练格式

**Scratch 实现（SFT）：**

```
<bos>{词牌名}<sep>{正文}<eos>
```

**Scratch 实现（DPO 偏好对）：**

```json
{
  "prompt": "浣溪沙<sep>",
  "chosen": "一曲新词酒一杯，去年天气旧亭台...",
  "rejected": "无可奈何花落去，似曾相识燕归来...",
  "chosen_score": 0.85,
  "rejected_score": 0.32
}
```

DPO 训练时，模型学习区分 chosen（高分）和 rejected（低分）的生成结果，隐式地学习格律规范。

**Unsloth 实现：**

```
<|im_start|>user
请按照词牌名《水调歌头》写一首宋词：<|im_end|>
<|im_start|>assistant
明月几时有？把酒问青天。<|im_end|>
```

## 配置文件详解

### MHA 配置 (configs/mha.yaml) — 6层基础模型

```yaml
model:
  vocab_size: 10000
  max_seq_len: 256
  embedding_dim: 512
  hidden_dim: 2048
  num_heads: 8
  num_layers: 6
  n_experts: 4
  topk: 1
  use_mla: false

train:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 100
  ckpt_path: "./scratch/ckpt/mha.pt"
```

### MHA 24层配置 (configs/mha_24l.yaml) — 全 MoE + 弹性训练

```yaml
model:
  vocab_size: 10000
  max_seq_len: 256
  embedding_dim: 512
  hidden_dim: 2048
  num_heads: 8
  num_layers: 24
  n_experts: 8           # 路由专家数
  topk: 2                # 每 token 激活 2 个路由专家
  n_shared_experts: 1    # 共享专家 (always-on)
  dropout: 0.1
  use_mla: false

train:
  batch_size: 32
  ckpt_path: "./scratch/ckpt/mha_24l.pt"

elastic:
  depth_prob: 0.25
  width_prob: 0.20
  sparsity_prob: 0.20
  depth_levels: [24, 18, 12, 6]
  width_levels: [8, 4, 2]
  sparsity_levels: [2, 1]
  warmup_steps: 1000
  core_experts: [0]
```

### MLA 配置 (configs/mla.yaml)

```yaml
model:
  use_mla: true
  latent_dim: 64
  rope_head_dim: 16
  # ... 其他同 MHA

train:
  ckpt_path: "./scratch/ckpt/mla.pt"
```

## 开发规范

### 代码风格

- 使用类型提示 (Python 3.13+ 特性)
- 模块清晰分离：model.py, attention.py, tokenizer.py, dataset.py
- 配置与代码分离：YAML 配置驱动

### Git 提交规范

根据 git log，项目使用以下提交规范：

- `feat:` 新功能
- `fix:` 修复问题
- `docs:` 文档更新
- `refactor:` 代码重构

### 训练监控

- 使用 `tqdm` 显示训练进度
- 每 20 轮自动保存检查点
- 支持从检查点恢复训练（Unsloth）

## 常见问题

### 1. 显存不足

- Scratch: 使用 MLA 配置降低 KV Cache 内存
- Unsloth: 已使用 4-bit 量化，可降低 batch_size

### 2. 分词器未找到

确保先运行 `uv run python scratch/tokenizer.py` 训练分词器

### 3. 模型检查点加载失败

检查 `ckpt_path` 配置与实际文件路径一致

### 4. Unsloth EOS token 错误

确保 `unsloth` 在 `trl` 之前导入

## 参考资料

- 宋词数据集：<https://github.com/chinese-poetry/chinese-poetry>
- Unsloth 框架：<https://unsloth.ai/>
- Qwen3 模型：<https://huggingface.co/Qwen/Qwen3-0.6B-Base>
- DeepSeek MLA 论文

## 许可证

MIT License
