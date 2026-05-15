#!/usr/bin/env python3
"""评估弹性训练子网络性能

对每个深度搜索多个随机组合，找出最佳/最差/平均表现，
分析哪些层是关键层。
"""

import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from model import SongCiGPT
from tokenizer import BPETokenizer

sys.path.insert(0, str(Path(__file__).parent / "songeval"))
from evaluator import Evaluator

EXCLUDE_TITLES = {"失调名"}


def generate_with_elastic(model, tokenizer, title, config, **elastic_params):
    with torch.no_grad():
        generated = model.generate(
            tokenizer=tokenizer,
            prompt_text=title,
            max_len=config.inference.max_len,
            temperature=config.inference.temperature,
            top_k=config.inference.top_k,
            top_p=config.inference.top_p,
            **elastic_params,
        )
    content = generated.replace(f"<bos>{title}<sep>", "").replace("<eos>", "").strip()
    return content


def evaluate_config(model, tokenizer, config, evaluator, eval_items, label, **elastic_params):
    tonal_accs = []
    rhyme_conss = []
    form_scores = []
    struct_match = 0
    total = 0

    for title, _ in tqdm(eval_items, desc=f"[{label}]", leave=False):
        try:
            generated = generate_with_elastic(
                model, tokenizer, title, config, **elastic_params
            )
            report = evaluator.evaluate(title, generated)
            if report.get("structure", {}).get("match", False):
                struct_match += 1
            tonal_accs.append(report.get("tonal", {}).get("accuracy", 0.0))
            rhyme_conss.append(report.get("rhyme", {}).get("consistency", 0.0))
            form_scores.append(report.get("form_score", 0.0))
            total += 1
        except Exception:
            continue

    if total == 0:
        return None
    return {
        "label": label,
        "struct_match": struct_match / total,
        "tonal_acc": sum(tonal_accs) / total,
        "rhyme_cons": sum(rhyme_conss) / total,
        "form_score": sum(form_scores) / total,
        "total": total,
    }


def main():
    config_path = "./configs/mha_24l.yaml"
    project_root = Path(__file__).parent.parent
    total_samples = 500
    num_combos = 20  # 每个深度搜索 20 个随机组合

    config = load_config(config_path)
    device = "cuda"

    print("加载模型...")
    model = SongCiGPT(config.model)
    ckpt_path = project_root / config.train.ckpt_path
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer()
    tokenizer.load(str(project_root / config.data.tokenizer_path))

    evaluator = Evaluator(registry_path=str(Path(__file__).parent / "songeval" / "standard.json"))
    all_titles = [t for t in evaluator.registry.keys() if t not in EXCLUDE_TITLES]
    print(f"可用词牌: {len(all_titles)} 个（排除: {EXCLUDE_TITLES}）")

    # 构建评估项
    eval_items = []
    sample_idx = 0
    while len(eval_items) < total_samples:
        for title in all_titles:
            eval_items.append((title, sample_idx))
            if len(eval_items) >= total_samples:
                break
        sample_idx += 1
    print(f"评估项: {len(eval_items)} 条（每个词牌约 {sample_idx} 首）")

    num_layers = config.model.num_layers  # 24
    num_experts = config.model.n_experts  # 8

    # 全量基线（之前已跑过，用已知值）
    full_score = 0.805  # 之前 500 条评估的结果

    # ── 深度搜索 ──
    print("\n" + "=" * 60)
    print(f"深度搜索（每个深度 {num_combos} 个随机组合）")
    print("=" * 60)

    # 统计每层被选中的次数和对应的平均得分
    layer_scores = {n: {} for n in [6, 12, 18]}  # {depth: {layer_idx: [scores]}}
    depth_results = {}  # {depth: [list of (combo, score)]}

    for n_active in [6, 12, 18]:
        print(f"\n── Depth={n_active}（从 24 层中选 {n_active} 层）──")
        combos = []
        for seed in range(num_combos):
            random.seed(seed)
            active_layers = sorted(random.sample(range(num_layers), n_active))
            combos.append(active_layers)

        combo_scores = []
        for i, active_layers in enumerate(combos):
            label = f"D{n_active} #{i:02d} {active_layers}"
            result = evaluate_config(
                model, tokenizer, config, evaluator, eval_items, label,
                active_layer_indices=active_layers,
            )
            if result:
                combo_scores.append((active_layers, result["form_score"]))
                print(f"  [{i+1}/{num_combos}] {active_layers} → {result['form_score']*100:.1f}")
                # 记录每层的贡献
                for layer in active_layers:
                    if layer not in layer_scores[n_active]:
                        layer_scores[n_active][layer] = []
                    layer_scores[n_active][layer].append(result["form_score"])

        depth_results[n_active] = combo_scores

    # ── 输出结果 ──
    print("\n" + "=" * 80)
    print("弹性深度搜索结果汇总")
    print("=" * 80)

    # 基线
    print(f"\n全量模型基线: {full_score*100:.1f}")

    # 深度搜索
    for n_active in [6, 12, 18]:
        if n_active not in depth_results or not depth_results[n_active]:
            continue
        scores = [s for _, s in depth_results[n_active]]
        best_combo, best_score = max(depth_results[n_active], key=lambda x: x[1])
        worst_combo, worst_score = min(depth_results[n_active], key=lambda x: x[1])
        avg_score = sum(scores) / len(scores)

        print(f"\n{'─' * 60}")
        print(f"Depth={n_active}（{num_combos} 个随机组合）")
        print(f"  最佳: {best_score*100:.1f}  层: {best_combo}")
        print(f"  平均: {avg_score*100:.1f}")
        print(f"  最差: {worst_score*100:.1f}  层: {worst_combo}")
        print(f"  标准差: {(sum((s-avg_score)**2 for s in scores)/len(scores))**0.5*100:.1f}")
        print(f"  最佳较全量: {(best_score - full_score)*100:+.1f}")
        print(f"  最差较全量: {(worst_score - full_score)*100:+.1f}")

    # 分析每层重要性
    print(f"\n{'─' * 60}")
    print("每层重要性分析（该层被选中时的平均得分）")
    print(f"{'─' * 60}")

    for n_active in [6, 12, 18]:
        if n_active not in layer_scores or not layer_scores[n_active]:
            continue
        print(f"\nDepth={n_active}:")
        avg_when_present = {}
        for layer in range(num_layers):
            if layer in layer_scores[n_active]:
                scores = layer_scores[n_active][layer]
                avg_when_present[layer] = sum(scores) / len(scores)

        # 按平均得分排序，显示哪些层"最值得选"
        sorted_layers = sorted(avg_when_present.items(), key=lambda x: x[1], reverse=True)
        print(f"  {'层':>4} {'平均得分':>8} {'柱状图'}")
        for layer, avg in sorted_layers:
            bar = "█" * int((avg - 0.3) * 100)  # 简单柱状图
            print(f"  {layer:>4} {avg*100:>7.1f}  {bar}")


if __name__ == "__main__":
    main()
