# AGENTS.md - Songci-GPT 项目指南

## 项目概述

**Songci-GPT** 是一个基于深度学习的宋词生成项目，提供两种实现方式：

1. **从零实现 (scratch/)**：基于 PyTorch 从零实现 GPT 模型，包含自定义 BPE 分词器、MoE (Mixture of Experts)、MLA (Multi-head Latent Attention) 和 KV Cache 推理优化
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
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 交互式推理脚本
│   ├── tokenizer.json      # 预训练分词器
│   ├── ckpt/              # 模型检查点目录
│   └── configs/           # 配置文件目录
│       ├── mha.yaml       # 标准 Multi-Head Attention 配置
│       └── mla.yaml       # Multi-head Latent Attention 配置
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

## 两种实现方式对比

| 特性 | Scratch 实现 | Unsloth 微调 |
|------|-------------|--------------|
| 基础模型 | 从零实现 | Qwen3-0.6B-Base |
| 分词器 | 自定义 BPE | Qwen3 Tokenizer |
| 训练方式 | 全参数微调 | LoRA (1-10% 参数) |
| 显存优化 | KV Cache + MLA | 4-bit 量化 |
| 架构特性 | MoE、RoPE、YAML 配置 | 标准 Transformer |
| 适用场景 | 学习原理、架构实验 | 生产部署 |

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

**标准 MHA 配置：**
```bash
uv run python scratch/train.py
```

**MLA 配置（更低内存占用）：**
```bash
uv run python scratch/train.py --config_path=./scratch/configs/mla.yaml
```

**配置参数说明：**
- `batch_size`: 32
- `learning_rate`: 1e-4
- `epochs`: 100
- `save_interval`: 每 20 轮保存
- 模型: vocab_size=10000, max_seq_len=256, embedding_dim=512, hidden_dim=2048, num_heads=8, num_layers=6
- MoE: n_experts=4, topk=1（奇数层使用 MoE，偶数层使用 Dense）

#### 3. 推理生成

```bash
# 默认配置
uv run python scratch/inference.py

# MLA 配置
uv run python scratch/inference.py --config_path=./scratch/configs/mla.yaml
```

交互式输入词牌名即可生成宋词，输入 `q` 退出。

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

- 交错式 MoE 架构：奇数层使用 MoE，偶数层和最后一层使用 Dense
- Top-k 路由机制（默认 topk=1）
- 负载均衡损失确保专家均匀使用
- 每个专家是独立的 FFN

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

**Scratch 实现：**
```
<bos>{词牌名}<sep>{正文}<eos>
```

**Unsloth 实现：**
```
<|im_start|>user
请按照词牌名《水调歌头》写一首宋词：<|im_end|>
<|im_start|>assistant
明月几时有？把酒问青天。<|im_end|>
```

## 配置文件详解

### MHA 配置 (configs/mha.yaml)

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

- 宋词数据集：https://github.com/chinese-poetry/chinese-poetry
- Unsloth 框架：https://unsloth.ai/
- Qwen3 模型：https://huggingface.co/Qwen/Qwen3-0.6B-Base
- DeepSeek MLA 论文

## 许可证

MIT License
