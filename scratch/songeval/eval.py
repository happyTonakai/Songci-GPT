#!/usr/bin/env python3
"""
eval.py - 宋词生成模型评估脚本

功能:
1. 对生成的宋词进行格律符合度打分
2. 支持单条评估和批量评估
3. 与 scratch 模型集成，支持生成时实时评估

用法:
    # 评估单条生成
    python eval.py --title "水调歌头" --text "明月几时有..."

    # 批量评估 (JSON 文件)
    python eval.py --batch_file samples.json

    # 与模型集成，生成并评估
    python eval.py --mode generate --config_path ../configs/mha.yaml --num_samples 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加 scratch 目录

from evaluator import Evaluator


def load_samples_from_json(file_path: str) -> List[Dict]:
    """从 JSON 文件加载样本"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持两种格式:
    # 1. [{"title": "...", "text": "..."}, ...]
    # 2. {"title": "...", "paragraphs": [...]}

    samples = []
    if isinstance(data, list):
        for item in data:
            if "title" in item and "text" in item:
                samples.append(item)
            elif "rhythmic" in item and "paragraphs" in item:
                samples.append(
                    {"title": item["rhythmic"], "text": "".join(item["paragraphs"])}
                )
    elif isinstance(data, dict):
        if "title" in data and "text" in data:
            samples.append(data)
        elif "rhythmic" in data and "paragraphs" in data:
            samples.append(
                {"title": data["rhythmic"], "text": "".join(data["paragraphs"])}
            )

    return samples


def print_report(report: Dict, detailed: bool = False):
    """打印评估报告"""
    if "error" in report:
        print(f"\n❌ 错误: {report['error']}")
        return

    print("\n" + "=" * 60)
    print(f"【{report['title']}】评估报告")
    print("=" * 60)

    # 结构匹配
    struct = report["structure"]
    match_icon = "✓" if struct["match"] else "✗"
    print(f"\n{match_icon} 结构匹配:")
    print(f"   标准: {struct['expected']}")
    print(f"   实际: {struct['actual']}")

    # 平仄准确度
    tonal = report["tonal"]
    print(f"\n📊 平仄准确度: {tonal['accuracy'] * 100:.2f}%")
    if detailed:
        print(f"   标准模板: {tonal['expected']}")
        print(f"   实际模式: {tonal['actual']}")

    # 押韵一致性
    rhyme = report["rhyme"]
    print(f"\n🎵 押韵一致性: {rhyme['consistency'] * 100:.2f}%")
    if detailed and "details" in rhyme:
        details = rhyme["details"]
        if "rhyme_distribution" in details:
            print(f"   韵部分布: {details['rhyme_distribution']}")

    # 综合分数
    print(f"\n⭐ 综合格律分: {report['form_score'] * 100:.2f}/100")
    print("=" * 60)


def generate_and_evaluate(args):
    """使用模型生成并评估"""
    try:
        import torch
        from config import load_config
        from model import SongCiGPT
        from tokenizer import BPETokenizer
    except ImportError as e:
        print(f"错误: 无法导入模型相关模块: {e}")
        print("请确保在 scratch 目录下运行，且已安装依赖")
        return

    # 加载配置
    config = load_config(args.config_path)
    device = config.train.device

    # 加载模型
    print(f"加载模型: {config.train.ckpt_path}")
    model = SongCiGPT(config.model)

    if os.path.exists(config.train.ckpt_path):
        model.load_state_dict(torch.load(config.train.ckpt_path, map_location=device))
    else:
        print(f"警告: 模型文件不存在: {config.train.ckpt_path}")
        print("将使用随机初始化的模型进行评估（结果无意义）")

    model.to(device)
    model.eval()

    # 加载分词器
    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    # 加载评估器
    evaluator = Evaluator(registry_path=args.registry)

    # 生成并评估
    titles = list(evaluator.registry.keys())[: args.num_samples]
    samples = []

    print(f"\n生成并评估 {len(titles)} 首宋词...")

    for title in titles:
        print(f"\n生成【{title}】...")

        # 生成
        with torch.no_grad():
            generated = model.generate(
                tokenizer=tokenizer,
                prompt_text=title,
                max_len=config.inference.max_len,
                temperature=config.inference.temperature,
                top_k=config.inference.top_k,
                top_p=config.inference.top_p,
                device=device,
            )

        # 提取内容部分
        content = (
            generated.replace(f"<bos>{title}<sep>", "").replace("<eos>", "").strip()
        )
        print(f"生成内容: {content[:50]}...")

        samples.append({"title": title, "text": content})

    # 批量评估
    print("\n" + "=" * 60)
    print("批量评估结果")
    print("=" * 60)

    stats = evaluator.evaluate_batch(samples)

    print(f"\n总样本数: {stats['total_samples']}")
    print(f"结构匹配率: {stats['structure_match_rate'] * 100:.2f}%")
    print(f"平均平仄准确度: {stats['avg_tonal_accuracy'] * 100:.2f}%")
    print(f"平均押韵一致性: {stats['avg_rhyme_consistency'] * 100:.2f}%")
    print(f"平均格律分: {stats['avg_form_score'] * 100:.2f}/100")

    # 保存详细结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {args.output}")


def evaluate_single(args):
    """评估单条生成"""
    evaluator = Evaluator(registry_path=args.registry)

    report = evaluator.evaluate(args.title, args.text)
    print_report(report, detailed=args.detailed)


def evaluate_batch_file(args):
    """批量评估文件"""
    evaluator = Evaluator(registry_path=args.registry)

    # 加载样本
    samples = load_samples_from_json(args.batch_file)
    print(f"加载了 {len(samples)} 个样本")

    # 批量评估
    stats = evaluator.evaluate_batch(samples)

    print("\n" + "=" * 60)
    print("批量评估统计")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"结构匹配率: {stats['structure_match_rate'] * 100:.2f}%")
    print(f"平均平仄准确度: {stats['avg_tonal_accuracy'] * 100:.2f}%")
    print(f"平均押韵一致性: {stats['avg_rhyme_consistency'] * 100:.2f}%")
    print(f"平均格律分: {stats['avg_form_score'] * 100:.2f}/100")

    # 打印前几个详细结果
    if args.detailed:
        print("\n" + "=" * 60)
        print("详细评估结果 (前 5 条)")
        print("=" * 60)
        for report in stats["details"][:5]:
            print_report(report, detailed=False)

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="宋词生成模型评估工具 - 格律符合度评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估单条生成
    python eval.py --title "水调歌头" --text "明月几时有把酒问青天..."
    
    # 批量评估
    python eval.py --batch_file samples.json --detailed
    
    # 生成并评估
    python eval.py --mode generate --config_path ../configs/mha.yaml --num_samples 10
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch", "generate"],
        default="single",
        help="评估模式 (默认: single)",
    )

    parser.add_argument(
        "--registry",
        type=str,
        default="standard.json",
        help="格律库文件路径 (默认: standard.json)",
    )

    # 单条评估参数
    parser.add_argument("--title", type=str, help="词牌名 (单条评估模式)")

    parser.add_argument("--text", type=str, help="生成文本 (单条评估模式)")

    # 批量评估参数
    parser.add_argument("--batch_file", type=str, help="批量评估的 JSON 文件路径")

    # 生成评估参数
    parser.add_argument(
        "--config_path",
        type=str,
        default="../configs/mha.yaml",
        help="模型配置文件路径 (生成模式)",
    )

    parser.add_argument(
        "--num_samples", type=int, default=10, help="生成样本数量 (生成模式)"
    )

    # 通用参数
    parser.add_argument("--output", type=str, help="输出结果保存路径")

    parser.add_argument("--detailed", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 检查格律库文件
    if not os.path.exists(args.registry):
        print(f"错误: 格律库文件不存在: {args.registry}")
        print("请先运行 analyzer.py 生成格律库")
        sys.exit(1)

    # 根据模式执行
    if args.mode == "single":
        if not args.title or not args.text:
            print("错误: 单条评估模式需要提供 --title 和 --text")
            parser.print_help()
            sys.exit(1)
        evaluate_single(args)

    elif args.mode == "batch":
        if not args.batch_file:
            print("错误: 批量评估模式需要提供 --batch_file")
            parser.print_help()
            sys.exit(1)
        evaluate_batch_file(args)

    elif args.mode == "generate":
        generate_and_evaluate(args)


if __name__ == "__main__":
    main()
