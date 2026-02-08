"""
基于 Unsloth 微调 Qwen3-0.6B 写宋词

使用 Unsloth 框架对 Qwen3-0.6B-Base 模型进行 LoRA 微调，使其能够生成宋词。
主要特点：
- 4-bit 量化训练，大幅降低显存占用
- LoRA 高效微调，仅更新 1-10% 的参数
- 使用 Qwen3 的 chat template 进行对话格式训练
"""

import glob
import json
from datasets import Dataset
from unsloth import FastLanguageModel

# 注意：unsloth 必须在 trl 之前导入，否则可能出现 EOS token 相关的错误
# 参考：https://stackoverflow.com/questions/79663362/sfttrainer-the-specified-eos-token-eos-token-is-not-found-in-the-vocabu


# Qwen3 的 chat template (完整版，包含 add_generation_prompt)
# 格式说明：
# - system 消息：<|im_start|>system\n内容<|im_end|>
# - user 消息：<|im_start|>user\n内容<|im_end|>
# - assistant 消息：<|im_start|>assistant\n内容<|im_end|>
# - generation prompt：<|im_start|>assistant\n (用于触发模型生成)
QWEN3_CHAT_TEMPLATE = """{% for message in messages %}
{% if loop.first and message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% endif %}
{% if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}"""


def load_songci_dataset(tokenizer, max_samples: int = None):
    """
    加载宋词数据集并转换为对话格式

    Args:
        tokenizer: 分词器对象
        max_samples: 最大样本数，None 表示加载全部数据

    Returns:
        HuggingFace Dataset，包含 'text' 字段
    """
    files = glob.glob("./dataset/宋词/*.json")
    data = []

    # 为 base 模型设置 chat template（如果是首次使用）
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                rhythmic = item["rhythmic"]  # 词牌名
                # 合并所有段落，去除换行符
                content = "".join(item["paragraphs"]).replace("\n", "").strip()
                # 添加 EOS token，表示生成结束
                # ! 注意：这里使用 eos_token 而不是 <|im_end|>，因为后者在训练时会被 mask
                content += tokenizer.eos_token

                # 构建对话样本：用户给定词牌名，模型生成宋词
                conversation = [
                    {"role": "user", "content": f"请按照词牌名《{rhythmic}》写一首宋词："},
                    {"role": "assistant", "content": content},
                ]

                # 使用 chat template 转换为训练文本
                # add_generation_prompt=False 表示不添加 assistant 的 generation prompt
                # ! we cannot use im_end as eos token, because it seems to be masked during training
                text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                ).strip()

                data.append({"text": text})

                if max_samples and len(data) >= max_samples:
                    break
        if max_samples and len(data) >= max_samples:
            break

    return Dataset.from_list(data)


def train(resume_from: str = None):
    """
    训练宋词生成模型

    Args:
        resume_from: 从指定检查点继续训练，为 None 则从头训练
    """
    from trl import SFTConfig, SFTTrainer

    # ============ 1. 加载模型和分词器 ============
    max_seq_length = 512  # 序列长度，可扩展到 2048
    dtype = None  # 自动选择数据类型 (FP16 for A100, BF16 for H100)
    load_in_4bit = True  # 4-bit 量化，大幅减少显存占用

    if resume_from:
        print(f"从检查点继续训练: {resume_from}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=resume_from,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    else:
        # 加载 Qwen3-0.6B-Base 基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen3-0.6B-Base",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

    # 确保 chat template 已设置（base 模型默认不带）
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    # ============ 2. 配置 LoRA 高效微调 ============
    # LoRA 只更新少量参数，大幅减少显存和计算量
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank，越大微调能力越强，但参数量也越多
        target_modules=[
            "q_proj",  # Query 投影
            "k_proj",  # Key 投影
            "v_proj",  # Value 投影
            "o_proj",  # Output 投影
            "gate_proj",  # FFN 门控
            "up_proj",  # FFN 上投影
            "down_proj",  # FFN 下投影
        ],
        lora_alpha=16,  # LoRA 缩放系数，通常设为 rank 或 rank*2
        lora_dropout=0,  # Dropout 概率，0 表示不使用
        bias="none",  # 不更新 bias
        use_gradient_checkpointing="unsloth",  # 使用 Unsloth 的梯度检查点，节省显存
    )

    # ============ 3. 准备数据 ============
    print("加载数据集...")
    dataset = load_songci_dataset(tokenizer, max_samples=None)  # 加载全部数据
    print(f"数据集样本数: {len(dataset)}")

    # ============ 4. 配置训练参数 ============
    args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=32,  # 每设备 batch 大小
        gradient_accumulation_steps=1,  # 梯度累积步数，用于模拟更大 batch
        warmup_steps=5,  # 学习率 warmup 步数
        num_train_epochs=20,  # 训练轮数
        learning_rate=2e-4,  # 学习率
        logging_steps=1,  # 日志打印间隔
        optim="adamw_8bit",  # 8-bit AdamW，节省显存
        weight_decay=0.001,  # 权重衰减
        lr_scheduler_type="linear",  # 学习率调度器
        seed=42,  # 随机种子
        report_to="none",  # 不上报训练指标
        output_dir="outputs",  # 输出目录
        max_length=max_seq_length,  # 最大序列长度
    )

    # ============ 5. 创建 SFTTrainer ============
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=args,
    )

    # ============ 6. 开始训练 ============
    print("开始训练...")
    trainer.train()

    # ============ 7. 保存模型 ============
    model.save_pretrained("qwen3-0.6b-songci-lora")
    tokenizer.save_pretrained("qwen3-0.6b-songci-lora")
    print("模型已保存到 qwen3-0.6b-songci-lora")


def infer(ckpt: str | None = None, stream: bool = True):
    """
    宋词生成推理

    Args:
        ckpt: 模型检查点路径
        stream: 是否使用流式输出
    """
    import torch
    from transformers import TextStreamer

    if ckpt is None:
        ckpt = "qwen3-0.6b-songci-lora"

    # 加载微调后的模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ckpt,
        max_seq_length=512,
        load_in_4bit=True,
    )
    # 切换到推理模式，优化内存和推理速度
    FastLanguageModel.for_inference(model)

    # 为 base 模型设置 chat template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = QWEN3_CHAT_TEMPLATE

    print("宋词生成器已启动，输入词牌名开始生成，输入 q 退出")
    while True:
        rhythmic = input("请输入词牌名：").strip()
        if rhythmic.lower() == "q":
            break

        # 构造提示词（只包含 user 消息）
        conversation = [{"role": "user", "content": f"请按照词牌名《{rhythmic}》写一首宋词："}]
        # add_generation_prompt=True 在结尾添加 <|im_start|>assistant\n，触发模型生成
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 终止符：EOS token 或 <|im_end|>
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]

        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        print(f"\n=== {rhythmic} ===")
        if stream:
            # 流式输出：实时显示生成的 token
            _ = model.generate(
                **inputs,
                max_new_tokens=256,  # 最大生成 token 数
                do_sample=True,  # 使用采样而非贪心
                temperature=1.0,  # 温度，控制随机性
                top_p=0.9,  # top-p 采样
                top_k=100,  # top-k 采样
                repetition_penalty=1.1,  # 重复惩罚，避免生成重复内容
                eos_token_id=terminators,
                streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
            )
            print()
        else:
            # 非流式输出：一次性生成完整结果
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=100,
                    eos_token_id=terminators,
                    repetition_penalty=1.1,
                )

            # 解码生成的文本（跳过输入部分）
            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            print(output_text)

            # 调试信息：显示每个 token 的文本和 ID
            debug_text = ""
            for token_id in generated_ids:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                debug_text += f"{token_text}({token_id}) "
            print(debug_text)
            print(f"序列长度: {len(generated_ids)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="宋词生成模型训练和推理")
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="运行模式：train（训练）或 infer（推理）",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="检查点路径：训练时用于继续训练，推理时用于加载模型",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出（默认启用流式输出）",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(resume_from=args.ckpt)
    else:
        infer(ckpt=args.ckpt, stream=not args.no_stream)
