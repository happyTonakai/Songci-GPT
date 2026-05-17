"""Loop 弹性训练评估脚本

测试所有弹性降级级别 (1x/2x/3x/4x) 的生成质量：
- Block Loop: block_loop_count=1/2/3/4
- Layer Loop: layer_loop_counts=[1/2/3/4]*6
"""
from __future__ import annotations

from pathlib import Path

import torch
from config import load_config
from fire import Fire
from model import SongCiGPT
from tokenizer import BPETokenizer
from tqdm import tqdm


def score_model(
    model: SongCiGPT,
    tokenizer: BPETokenizer,
    evaluator,
    titles: list[str],
    config,
    label: str,
    block_loop_count: int = 1,
    layer_loop_counts: list[int] | None = None,
    num_samples: int = 5,
) -> dict:
    """评估模型在指定 loop 配置下的格律得分"""
    device = config.train.device
    model.eval()

    form_scores = []
    tonal_accs = []
    rhyme_cons = []
    struct_hits = 0
    total = 0

    for title in tqdm(titles, desc=f"评估 {label}"):
        for _ in range(num_samples):
            with torch.no_grad():
                text = model.generate(
                    tokenizer,
                    title,
                    temperature=config.inference.temperature,
                    top_k=config.inference.top_k,
                    top_p=config.inference.top_p,
                    max_len=config.inference.max_len,
                    block_loop_count=block_loop_count,
                    layer_loop_counts=layer_loop_counts,
                )
            text = text.replace("<bos>", "").replace("<eos>", "").replace("<sep>", "").strip()
            if title in text:
                text = text[len(title):]
            report = evaluator.evaluate(title, text)
            if "error" in report:
                continue
            form_scores.append(report["form_score"])
            tonal_accs.append(report["tonal"]["accuracy"])
            rhyme_cons.append(report["rhyme"]["consistency"])
            if report["structure"]["match"]:
                struct_hits += 1
            total += 1

    n = max(total, 1)
    return {
        "total_samples": total,
        "structure_match_rate": struct_hits / n,
        "avg_tonal_accuracy": sum(tonal_accs) / len(tonal_accs) if tonal_accs else 0,
        "avg_rhyme_consistency": sum(rhyme_cons) / len(rhyme_cons) if rhyme_cons else 0,
        "avg_form_score": sum(form_scores) / len(form_scores) if form_scores else 0,
    }


def print_multi_comparison(title: str, results: dict[int, dict]):
    """打印多级别对比表"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    header = f"  {'指标':<20}"
    for level in sorted(results.keys()):
        header += f" {level}x".rjust(10)
    print(header)
    print(f"{'-' * 70}")

    metrics = [
        ("结构匹配率", "structure_match_rate"),
        ("平均平仄准确度", "avg_tonal_accuracy"),
        ("平均押韵一致性", "avg_rhyme_consistency"),
        ("综合格律得分", "avg_form_score"),
    ]
    for name, key in metrics:
        line = f"  {name:<20}"
        for level in sorted(results.keys()):
            line += f" {results[level][key]*100:>9.2f}%"
        print(line)

    print(f"\n  评估样本数: {results[sorted(results.keys())[0]]['total_samples']} (每组)")
    print(f"{'=' * 70}")


_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# 切换到项目根目录，确保配置文件中的相对路径正确
import os
os.chdir(_PROJECT_ROOT)


def evaluate_block_loop(
    config_path: str = None,
    num_titles: int = 75,
    num_samples: int = 5,
):
    """评估 Block Loop 模型: 所有弹性级别 (1x/2x/3x/4x)"""
    if config_path is None:
        config_path = str(_SCRIPT_DIR / "configs" / "loop_block.yaml")
    config = load_config(config_path)
    device = config.train.device

    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    from songeval import Evaluator
    registry_path = str(Path(__file__).parent / "songeval" / "standard.json")
    evaluator = Evaluator(registry_path=registry_path)
    titles = list(evaluator.registry.keys())[:num_titles]

    model = SongCiGPT(config.model)
    model.load_state_dict(torch.load(config.train.ckpt_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    results = {}
    for level in [1, 2, 3, 4]:
        results[level] = score_model(
            model, tokenizer, evaluator, titles, config,
            label=f"Block Loop {level}x",
            block_loop_count=level,
            num_samples=num_samples,
        )

    print_multi_comparison("Block Loop: 弹性级别对比", results)
    return results


def evaluate_layer_loop(
    config_path: str = None,
    num_titles: int = 75,
    num_samples: int = 5,
):
    """评估 Layer Loop 模型: 所有弹性级别 (1x/2x/3x/4x)"""
    if config_path is None:
        config_path = str(_SCRIPT_DIR / "configs" / "loop_layer.yaml")
    config = load_config(config_path)
    device = config.train.device

    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    from songeval import Evaluator
    registry_path = str(Path(__file__).parent / "songeval" / "standard.json")
    evaluator = Evaluator(registry_path=registry_path)
    titles = list(evaluator.registry.keys())[:num_titles]

    model = SongCiGPT(config.model)
    model.load_state_dict(torch.load(config.train.ckpt_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    num_layers = config.model.num_layers

    results = {}
    for level in [1, 2, 3, 4]:
        results[level] = score_model(
            model, tokenizer, evaluator, titles, config,
            label=f"Layer Loop {level}x",
            layer_loop_counts=[level] * num_layers,
            num_samples=num_samples,
        )

    print_multi_comparison("Layer Loop: 弹性级别对比", results)
    return results


def evaluate_all(
    num_titles: int = 75,
    num_samples: int = 5,
):
    """评估所有 loop 模型"""
    print("=" * 70)
    print("  Loop 弹性训练评估: 所有弹性级别 (1x/2x/3x/4x)")
    print("=" * 70)

    print("\n>>> Block Loop 评估")
    block_results = evaluate_block_loop(
        num_titles=num_titles, num_samples=num_samples,
    )

    print("\n>>> Layer Loop 评估")
    layer_results = evaluate_layer_loop(
        num_titles=num_titles, num_samples=num_samples,
    )

    # 汇总对比
    print(f"\n{'=' * 70}")
    print(f"  汇总对比 (综合格律得分)")
    print(f"{'=' * 70}")
    header = f"  {'模型':<20}"
    for level in [1, 2, 3, 4]:
        header += f" {level}x".rjust(10)
    print(header)
    print(f"{'-' * 70}")
    for name, results in [("Block Loop", block_results), ("Layer Loop", layer_results)]:
        line = f"  {name:<20}"
        for level in [1, 2, 3, 4]:
            line += f" {results[level]['avg_form_score']*100:>9.2f}%"
        print(line)
    print(f"{'=' * 70}")


if __name__ == "__main__":
    Fire({
        "block": evaluate_block_loop,
        "layer": evaluate_layer_loop,
        "all": evaluate_all,
    })
