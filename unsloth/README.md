# Qwen3-0.6B 宋词生成器

基于 Unsloth 框架微调 Qwen3-0.6B-Base 模型，使其能够根据词牌名生成宋词。

## 特点

- **高效微调**：使用 LoRA 技术，仅更新 1-10% 的模型参数
- **低显存占用**：4-bit 量化训练，可在消费级 GPU 上运行
- **对话格式**：使用 Qwen3 的 chat template 进行训练

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- Unsloth
- 其他依赖见 `pyproject.toml`

## 安装依赖

```bash
uv sync
```

## 使用方法

### 训练模型

```bash
uv run python unsloth/train_qwen_songci.py --mode train
```

从检查点继续训练：

```bash
uv run python unsloth/train_qwen_songci.py --mode train --ckpt qwen3-0.6b-songci-lora
```

### 推理生成

```bash
uv run python unsloth/train_qwen_songci.py --mode infer
```

禁用流式输出：

```bash
uv run python unsloth/train_qwen_songci.py --mode infer --no-stream
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_seq_length | 512 | 最大序列长度 |
| num_train_epochs | 20 | 训练轮数 |
| per_device_train_batch_size | 32 | 每设备 batch 大小 |
| learning_rate | 2e-4 | 学习率 |
| lora_r | 16 | LoRA rank |
| lora_alpha | 16 | LoRA 缩放系数 |

## 生成示例

```
请输入词牌名：水调歌头

=== 水调歌头 ===
明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。

转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。
```

## 文件说明

- `train_qwen_songci.py`：训练和推理脚本

## 技术细节

### 数据格式

训练数据使用对话格式：

```
<|im_start|>user
请按照词牌名《水调歌头》写一首宋词：<|im_end|>
<|im_start|>assistant
明月几时有？把酒问青天。<|im_end|>
```

### LoRA 配置

- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- r: 16
- alpha: 16
- dropout: 0
- bias: none

### 生成参数

- temperature: 1.0
- top_p: 0.9
- top_k: 100
- repetition_penalty: 1.1
- max_new_tokens: 256

## 参考资料

- [Unsloth 文档](https://unsloth.ai/docs/)
- [Qwen3 模型](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [宋词数据集](https://github.com/chinese-poetry/chinese-poetry)
