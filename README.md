# SongCi GPT

一个基于 Transformer 的 GPT 模型，专门用于生成宋词。项目包含完整的 BPE 分词器训练、模型训练和文本生成功能。

## 项目简介

本项目实现了一个从零开始的 GPT 模型，用于宋词生成。主要特点：

- **自定义 BPE 分词器**：支持中英文混合分词，使用 `jieba` 进行中文分词
- **Decoder-only Transformer 架构**：基于 PyTorch 实现，包含可学习的位置编码
- **宋词数据集**：使用宋词 JSON 数据进行训练，支持根据词牌名生成宋词
- **灵活的采样策略**：支持 temperature、top-k、top-p 等采样方法

## 项目结构

```
gpt/
├── model.py           # GPT 模型定义
├── tokenizer.py       # BPE 分词器实现
├── dataset.py         # 数据集加载和预处理
├── train.py           # 训练脚本
├── inference.py       # 交互式推理脚本
├── dataset/           # 训练数据
│   ├── train-en.txt   # 英文训练语料
│   ├── train-zh.txt   # 中文训练语料
│   └── 宋词/          # 宋词 JSON 数据
└── ckpt/              # 模型检查点目录
```

## 环境要求

- Python >= 3.13
- PyTorch >= 2.9.1
- 其他依赖见 `pyproject.toml`

## 安装

使用 uv 安装依赖：

```bash
uv sync
```

## 使用方法

### 1. 训练分词器

首先训练 BPE 分词器：

```bash
uv run python tokenizer.py
```

这将：
- 读取 `dataset/宋词/` 目录下的所有 JSON 文件
- 训练一个词表大小为 10000 的 BPE 分词器
- 保存到 `ckpt/songci_tokenizer.json`

### 2. 训练模型

训练 GPT 模型：

```bash
uv run python train.py
```

训练参数：
- 模型参数：vocab_size=10000, max_seq_len=256, embedding_dim=512, hidden_dim=2048, num_head=8, num_layers=6
- 训练轮数：100 epochs
- Batch size：32
- 学习率：1e-4
- 每 20 轮保存一次检查点

模型将保存到 `ckpt/model.pt`

### 3. 生成宋词

#### 方式一：交互式推理

使用交互式推理脚本：

```bash
uv run python inference.py
```

输入词牌名即可生成对应的宋词，输入 `q` 退出。

#### 方式二：代码调用

```python
from model import SongCiGPT
from tokenizer import BPETokenizer

# 加载模型和分词器
model = SongCiGPT()
model.load_state_dict(torch.load("ckpt/model.pt"))
model.to("cuda")

tokenizer = BPETokenizer()
tokenizer.load("ckpt/songci_tokenizer.json")

# 生成宋词
response = model.generate(
    tokenizer,
    prompt_text="水调歌头",
    max_len=256,
    temperature=1.0,
    top_k=100,
    top_p=0.9
)
print(response)
```


## 模型架构

### SongCiGPT

- **Embedding 层**：词嵌入 + 可学习的位置编码
- **Transformer 层**：6 层 TransformerEncoder，每层包含：
  - 8 头自注意力机制
  - 前馈网络（hidden_dim=2048）
  - GELU 激活函数
  - Dropout (0.1)
- **输出层**：线性映射到词表大小

### BPE 分词器

- 支持中英文混合分词
- 使用 `jieba` 进行中文分词
- 特殊标记：`<bos>`, `<eos>`, `<unknown>`, `</w>`, `<mask>`, `<sep>`, `<pad>`
- `</w>` 标记用于表示词边界，防止跨词合并

## 数据格式

宋词数据集使用 JSON 格式，每条数据包含：

```json
{
  "rhythmic": "水调歌头",
  "paragraphs": ["明月几时有？把酒问青天。", "不知天上宫阙，今夕是何年。"]
}
```

训练时将词牌名和正文用 `<sep>` 分隔，格式为：`rhythmic<sep>paragraphs`

## 采样策略

模型支持多种采样策略：

- **Temperature**：控制输出的随机性，值越大越随机
- **Top-k**：只从概率最高的 k 个 token 中采样
- **Top-p (Nucleus Sampling)**：从累积概率达到 p 的最小 token 集合中采样

## 训练细节

### 损失计算

- 使用交叉熵损失
- 词牌名部分（`<bos>` 到 `<sep>`）不参与损失计算
- Padding token 不参与损失计算
- 标签右移一位，预测下一个 token

### 注意力掩码

- 使用因果掩码（causal mask）确保自回归特性
- Padding mask 用于屏蔽 padding token

## 示例输出

使用交互式推理：

```bash
$ uv run python inference.py
请输入(输入q退出)：水调歌头
明月几时有？把酒问青天。不知天上宫阙，今夕是何年。
请输入(输入q退出)：江城子
十年生死两茫茫，不思量，自难忘。
请输入(输入q退出)：q
```

或在代码中调用：

```python
# 生成水调歌头
model.generate(tokenizer, "水调歌头", max_len=256, top_k=100, top_p=0.9)

# 生成江城子
model.generate(tokenizer, "江城子", max_len=256, top_k=100, top_p=0.9)
```

## 技术特点

1. **完整的 BPE 实现**：从零实现的 BPE 分词器，支持中英文混合
2. **Decoder-only 架构**：使用 `nn.TransformerEncoder` 实现，更符合 GPT 的设计
3. **可学习的位置编码**：相比固定的正弦位置编码，更灵活
4. **高效的数据加载**：使用 `DataLoader` 的 `persistent_workers` 加速训练
5. **灵活的生成策略**：支持多种采样方法，可调节生成质量

## 依赖项

- `torch`：深度学习框架
- `jieba`：中文分词
- `orjson`：高性能 JSON 处理
- `tqdm`：进度条显示
- `numpy`：数值计算

## 许可证

MIT License

## 致谢

- 宋词数据集来自开源数据 https://github.com/chinese-poetry/chinese-poetry
- 感谢 https://www.bilibili.com/video/BV1SZ42177SH

