#!/usr/bin/env python3
"""
analyzer.py - 宋词格律库构建主脚本

用法:
    python analyzer.py --data_path ../../dataset/宋词/*.json --output standard.json --top_n 50 --confidence 0.8

参数:
    --data_path: 宋词数据文件路径 (支持通配符)
    --output: 输出文件路径 (默认: standard.json)
    --top_n: 保留的词牌数量 (默认: 50)
    --confidence: 平仄判定阈值 (默认: 0.8)
"""

import argparse
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from registry_builder import RegistryBuilder


def main():
    parser = argparse.ArgumentParser(
        description="宋词格律库构建工具 - 从训练语料中提取格律标准",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法 (样本量 >= 50 的词牌，约75个)
    python analyzer.py
    
    # 使用 Top N 策略 (覆盖前50个词牌)
    python analyzer.py --top_n 50
    
    # 自定义样本量阈值 (更严格)
    python analyzer.py --min_samples 100
    
    # 自定义所有参数
    python analyzer.py --data_path "../../dataset/宋词/*.json" --output standard.json --min_samples 50 --confidence 0.8
        """,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="../../dataset/宋词/*.json",
        help="宋词数据文件路径 (支持通配符，默认: ../../dataset/宋词/*.json)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="standard.json",
        help="输出文件路径 (默认: standard.json)",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="保留 Top N 个词牌 (与 --min_samples 二选一，默认 None)",
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        default=50,
        help="最小样本量阈值，只保留样本数 >= 此值的词牌 (默认: 50)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="平仄判定阈值，超过此比例才判定为 P 或 Z (默认: 0.8)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("宋词格律库构建工具 (SongEval v2.0)")
    print("=" * 70)
    print(f"数据路径: {args.data_path}")
    print(f"输出文件: {args.output}")
    if args.top_n is not None:
        print(f"筛选策略: Top {args.top_n}")
    else:
        print(f"筛选策略: 样本量 >= {args.min_samples}")
    print(f"置信度阈值: {args.confidence}")
    print("=" * 70)

    # 构建格律库
    builder = RegistryBuilder(
        data_path=args.data_path,
        top_n=args.top_n,
        min_samples=args.min_samples,
        confidence=args.confidence,
    )

    registry = builder.build()
    builder.save(args.output)

    print("\n构建完成！")
    print(f"格律库已保存到: {args.output}")
    print(f"共包含 {len(registry)} 个词牌的格律标准")


if __name__ == "__main__":
    main()
