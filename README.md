# SongCi GPT

一个基于深度学习的宋词生成项目，包含两种实现方式：

1. **从零实现**：基于 PyTorch 从零实现 GPT 模型，包含自定义 BPE 分词器和 KV Cache 推理优化
2. **Unsloth 微调**：基于 Unsloth 框架微调 Qwen3-0.6B，使用 LoRA 高效微调技术

## 项目结构

```
Songci-GPT/
├── scratch/              # 从零实现的 GPT 模型
│   ├── README.md         # scratch 实现文档
│   ├── model.py          # GPT 模型定义
│   ├── tokenizer.py      # BPE 分词器实现
│   ├── dataset.py        # 数据集加载和预处理
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 交互式推理脚本
│   └── ckpt/             # 模型检查点
│
├── unsloth/              # Unsloth 微调实现
│   ├── README.md         # unsloth 实现文档
│   └── train_qwen_songci.py  # 训练和推理脚本
│
├── dataset/              # 宋词数据集
│   ├── train-en.txt
│   ├── train-zh.txt
│   └── 宋词/            # 宋词 JSON 数据
│
└── pyproject.toml       # 项目依赖
```

## 两种实现对比

| 特性 | Scratch 实现 | Unsloth 微调 |
|------|-------------|--------------|
| 基础模型 | 从零实现 | Qwen3-0.6B-Base |
| 分词器 | 自定义 BPE | Qwen3 Tokenizer |
| 训练方式 | 全参数微调 | LoRA (1-10% 参数) |
| 显存优化 | KV Cache | 4-bit 量化 |
| 适用场景 | 学习原理 | 生产部署 |

## 快速开始

### 环境安装

```bash
uv sync
```

### 方式一：Scratch 实现

详细文档见 [`scratch/README.md`](scratch/README.md)

```bash
# 训练分词器
uv run python scratch/tokenizer.py

# 训练模型
uv run python scratch/train.py

# 交互式推理
uv run python scratch/inference.py
```

### 方式二：Unsloth 微调

详细文档见 [`unsloth/README.md`](unsloth/README.md)

```bash
# 训练模型
python unsloth/train_qwen_songci.py --mode train

# 推理生成
python unsloth/train_qwen_songci.py --mode infer
```

## 数据集

宋词数据集使用 JSON 格式，每条数据包含：

```json
{
  "rhythmic": "水调歌头",
  "paragraphs": ["明月几时有？把酒问青天。", "不知天上宫阙，今夕是何年。"]
}
```

数据集来源：https://github.com/chinese-poetry/chinese-poetry

## 示例输出

使用 Unsloth 微调的模型生成：

```
请输入词牌名：水调歌头

=== 水调歌头 ===
明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。

转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。
```

## 技术特点

### Scratch 实现
- 完整的 BPE 分词器实现（支持中英文混合分词）
- Decoder-only Transformer 架构
- 自定义 KV Cache 推理优化（推理速度提升 10-100 倍）
- 可学习的位置编码

### Unsloth 微调
- 4-bit 量化训练，大幅降低显存占用
- LoRA 高效微调，仅更新 1-10% 参数
- 使用 Qwen3 的 chat template 进行对话格式训练

## 依赖项

- `torch`：深度学习框架
- `transformers`：模型加载和推理
- `unsloth`：高效微调框架
- `trl`：SFT 训练
- `jieba`：中文分词（仅 Scratch 实现）
- `orjson`：高性能 JSON 处理

## 许可证

MIT License

## 致谢

- 宋词数据集：https://github.com/chinese-poetry/chinese-poetry
- Unsloth 框架：https://unsloth.ai/
- Qwen3 模型：https://huggingface.co/Qwen/Qwen3-0.6B-Base