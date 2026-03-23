#!/usr/bin/env python3
"""
evaluate_model.py - 对训练好的模型进行格律评估

支持两种模型：
1. scratch 实现的 PyTorch 模型 (mha.pt / mla.pt)
2. unsloth 微调的 LoRA 模型 (qwen3-0.6b-songci-lora/)

用法：
    # 评估 scratch MHA 模型
    python evaluate_model.py --model_type scratch --config_path ../configs/mha.yaml
    
    # 评估 scratch MLA 模型  
    python evaluate_model.py --model_type scratch --config_path ../configs/mla.yaml
    
    # 评估 unsloth LoRA 模型
    python evaluate_model.py --model_type unsloth --model_path ../../qwen3-0.6b-songci-lora
    
    # 指定输出和样本数
    python evaluate_model.py --model_type scratch --config_path ../configs/mha.yaml \
        --num_samples 50 --output eval_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator import Evaluator


def load_scratch_model(config_path: str, device: str = "cuda"):
    """加载 scratch 实现的模型"""
    try:
        from config import load_config
        from model import SongCiGPT
        from tokenizer import BPETokenizer
    except ImportError as e:
        print(f"错误: 无法导入 scratch 模块: {e}")
        sys.exit(1)

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent

    config = load_config(config_path)

    # 转换相对路径为绝对路径
    ckpt_path = Path(config.train.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    tokenizer_path = Path(config.data.tokenizer_path)
    if not tokenizer_path.is_absolute():
        tokenizer_path = project_root / tokenizer_path

    # 加载模型
    print(f"加载模型: {ckpt_path}")
    model = SongCiGPT(config.model)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✓ 成功加载检查点: {ckpt_path}")
    else:
        print(f"⚠️ 警告: 检查点不存在，使用随机初始化: {ckpt_path}")

    model.to(device)
    model.eval()

    # 加载分词器
    print(f"加载分词器: {tokenizer_path}")
    tokenizer = BPETokenizer()
    if tokenizer_path.exists():
        tokenizer.load(str(tokenizer_path))
        print("✓ 成功加载分词器")
    else:
        print(f"⚠️ 警告: 分词器不存在: {tokenizer_path}")
        print("请先运行: python scratch/tokenizer.py")
        sys.exit(1)

    return model, tokenizer, config


def load_unsloth_model(model_path: str, device: str = "cuda"):
    """加载 unsloth LoRA 模型"""
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        print(f"错误: 无法导入 unsloth: {e}")
        sys.exit(1)

    print(f"加载 Unsloth 模型: {model_path}")

    # 加载基础模型和 LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)
    model.eval()

    print("✓ 成功加载 LoRA 模型")

    return model, tokenizer


def generate_scratch(model, tokenizer, title: str, config, device: str = "cuda") -> str:
    """使用 scratch 模型生成"""
    with torch.no_grad():
        generated = model.generate(
            tokenizer=tokenizer,
            prompt_text=title,
            max_len=config.inference.max_len,
            temperature=config.inference.temperature,
            top_k=config.inference.top_k,
            top_p=config.inference.top_p,
        )

    # 提取内容（去除特殊标记）
    content = generated.replace(f"<bos>{title}<sep>", "").replace("<eos>", "").strip()
    return content


def generate_unsloth(model, tokenizer, title: str, device: str = "cuda") -> str:
    """使用 unsloth 模型生成"""
    # 构建提示
    prompt = f"请按照词牌名《{title}》写一首宋词："

    # 使用 chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 编码
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.9,
            top_k=100,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取 assistant 的回复
    if "<|im_start|>assistant" in generated:
        content = generated.split("<|im_start|>assistant")[-1]
        content = content.replace("<|im_end|>", "").strip()
    else:
        content = generated[len(text) :].strip()

    return content


def evaluate_model(
    model_type: str,
    model,
    tokenizer,
    config_or_path,
    evaluator: Evaluator,
    titles: List[str],
    num_titles: int = None,
    samples_per_title: int = 1,
    device: str = "cuda",
) -> Dict:
    """
    评估模型在多个词牌上的表现

    Args:
        num_titles: 评估的词牌数量（从前N个词牌中选取）
        samples_per_title: 每个词牌生成的样本数量

    Returns:
        评估结果字典
    """
    if num_titles:
        titles = titles[:num_titles]

    results = {
        "model_type": model_type,
        "total_titles": len(titles),
        "generated_samples": [],
        "aggregate_scores": {
            "structure_match_count": 0,
            "total_samples": 0,
            "avg_tonal_accuracy": 0.0,
            "avg_rhyme_consistency": 0.0,
            "avg_form_score": 0.0,
        },
    }

    tonal_accuracies = []
    rhyme_consistencies = []
    form_scores = []

    print(f"\n开始评估 {len(titles)} 个词牌...")
    print("=" * 80)

    total_generated = 0
    for title in tqdm(titles, desc="生成与评估"):
        for sample_idx in range(samples_per_title):
            try:
                # 生成
                if model_type == "scratch":
                    generated = generate_scratch(
                        model, tokenizer, title, config_or_path, device
                    )
                else:  # unsloth
                    generated = generate_unsloth(model, tokenizer, title, device)

                # 评估
                report = evaluator.evaluate(title, generated)

                # 记录结果
                sample_result = {
                    "title": title,
                    "sample_idx": sample_idx,
                    "generated": generated,
                    "structure_match": report.get("structure", {}).get("match", False),
                    "tonal_accuracy": report.get("tonal", {}).get("accuracy", 0.0),
                    "rhyme_consistency": report.get("rhyme", {}).get(
                        "consistency", 0.0
                    ),
                    "form_score": report.get("form_score", 0.0),
                }
                results["generated_samples"].append(sample_result)

                # 累加统计
                if sample_result["structure_match"]:
                    results["aggregate_scores"]["structure_match_count"] += 1
                tonal_accuracies.append(sample_result["tonal_accuracy"])
                rhyme_consistencies.append(sample_result["rhyme_consistency"])
                form_scores.append(sample_result["form_score"])
                total_generated += 1

            except Exception as e:
                print(f"\n错误: 评估 {title} (样本 {sample_idx}) 时出错: {e}")
                continue

    # 计算平均值
    n = len(results["generated_samples"])
    if n > 0:
        results["aggregate_scores"]["total_samples"] = n
        results["aggregate_scores"]["avg_tonal_accuracy"] = sum(tonal_accuracies) / n
        results["aggregate_scores"]["avg_rhyme_consistency"] = (
            sum(rhyme_consistencies) / n
        )
        results["aggregate_scores"]["avg_form_score"] = sum(form_scores) / n
        results["aggregate_scores"]["structure_match_rate"] = (
            results["aggregate_scores"]["structure_match_count"] / n
        )
    else:
        results["aggregate_scores"]["total_samples"] = 0
        results["aggregate_scores"]["avg_tonal_accuracy"] = 0.0
        results["aggregate_scores"]["avg_rhyme_consistency"] = 0.0
        results["aggregate_scores"]["avg_form_score"] = 0.0
        results["aggregate_scores"]["structure_match_rate"] = 0.0

    return results


def print_results(results: Dict):
    """打印评估结果"""
    scores = results["aggregate_scores"]

    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print(f"模型类型: {results['model_type']}")
    print(f"评估样本数: {scores['total_samples']}")
    print(
        f"\n结构匹配率: {scores['structure_match_rate'] * 100:.2f}% ({scores['structure_match_count']}/{scores['total_samples']})"
    )
    print(f"平均平仄准确度: {scores['avg_tonal_accuracy'] * 100:.2f}%")
    print(f"平均押韵一致性: {scores['avg_rhyme_consistency'] * 100:.2f}%")
    print(f"综合格律得分: {scores['avg_form_score'] * 100:.2f}/100")
    print("=" * 80)

    # 打印一些示例
    print("\n生成示例 (前5个):")
    print("-" * 80)
    for i, sample in enumerate(results["generated_samples"][:5], 1):
        print(f"\n{i}. 【{sample['title']}】")
        print(f"   生成: {sample['generated'][:50]}...")
        print(f"   结构匹配: {'✓' if sample['structure_match'] else '✗'}")
        print(f"   平仄准确度: {sample['tonal_accuracy'] * 100:.1f}%")
        print(f"   格律分: {sample['form_score'] * 100:.1f}/100")


def main():
    parser = argparse.ArgumentParser(
        description="评估宋词生成模型的格律符合度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估 scratch MHA 模型
    python evaluate_model.py --model_type scratch --config_path ../configs/mha.yaml
    
    # 评估 scratch MLA 模型
    python evaluate_model.py --model_type scratch --config_path ../configs/mla.yaml
    
    # 评估 unsloth 模型
    python evaluate_model.py --model_type unsloth --model_path ../../qwen3-0.6b-songci-lora
    
    # 只评估前20个词牌
    python evaluate_model.py --model_type scratch --config_path ../configs/mha.yaml --num_samples 20
        """,
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["scratch", "unsloth"],
        required=True,
        help="模型类型: scratch (PyTorch) 或 unsloth (LoRA)",
    )

    parser.add_argument(
        "--config_path", type=str, help="模型配置文件路径 (scratch 模型必需)"
    )

    parser.add_argument("--model_path", type=str, help="模型路径 (unsloth 模型必需)")

    parser.add_argument(
        "--registry",
        type=str,
        default="standard.json",
        help="格律库文件路径 (默认: standard.json)",
    )

    parser.add_argument(
        "--num_titles",
        type=int,
        default=None,
        help="评估的词牌数量，默认评估所有词牌 (从格律库中选取前 N 个)",
    )

    parser.add_argument(
        "--samples_per_title",
        type=int,
        default=1,
        help="每个词牌生成的样本数量 (默认: 1)",
    )

    parser.add_argument("--output", type=str, help="结果保存路径 (JSON 格式)")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备 (默认: cuda)",
    )

    args = parser.parse_args()

    # 检查参数
    if args.model_type == "scratch" and not args.config_path:
        print("错误: scratch 模型需要指定 --config_path")
        sys.exit(1)

    if args.model_type == "unsloth" and not args.model_path:
        print("错误: unsloth 模型需要指定 --model_path")
        sys.exit(1)

    # 检查格律库
    if not os.path.exists(args.registry):
        print(f"错误: 格律库不存在: {args.registry}")
        print("请先运行: python analyzer.py")
        sys.exit(1)

    # 加载评估器
    print("=" * 80)
    print("宋词模型格律评估")
    print("=" * 80)
    print(f"格律库: {args.registry}")
    evaluator = Evaluator(registry_path=args.registry)

    # 获取所有词牌
    all_titles = list(evaluator.registry.keys())
    print(f"词牌总数: {len(all_titles)}")

    # 加载模型
    if args.model_type == "scratch":
        model, tokenizer, config = load_scratch_model(args.config_path, args.device)
        config_or_path = config
    else:  # unsloth
        model, tokenizer = load_unsloth_model(args.model_path, args.device)
        config_or_path = args.model_path

    # 进行评估
    results = evaluate_model(
        model_type=args.model_type,
        model=model,
        tokenizer=tokenizer,
        config_or_path=config_or_path,
        evaluator=evaluator,
        titles=all_titles,
        num_titles=args.num_titles,
        samples_per_title=args.samples_per_title,
        device=args.device,
    )

    # 打印结果
    print_results(results)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
