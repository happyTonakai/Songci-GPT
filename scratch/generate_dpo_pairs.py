"""使用 SongEval 格律评估作为奖励模型，生成 DPO 偏好对

流程：
1. 加载训练好的 SFT 模型
2. 对每个词牌名生成多个候选宋词（不同 temperature）
3. 用 SongEval Evaluator 对每个候选打分
4. 取最高分作为 chosen，最低分作为 rejected，组成偏好对
5. 保存为 DPODataset 可读取的 JSON 格式
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from fire import Fire
from tqdm import tqdm

from config import load_config
from model import SongCiGPT
from songeval import Evaluator
from tokenizer import BPETokenizer


def generate_candidates(
    model: SongCiGPT,
    tokenizer: BPETokenizer,
    title: str,
    num_candidates: int,
    temperatures: list[float],
    top_k: int,
    top_p: float,
    max_len: int,
    device: str,
) -> list[str]:
    """对一个词牌名生成多个候选宋词

    使用不同的 temperature 采样，增加候选多样性。
    """
    candidates = []
    with torch.no_grad():
        for i in range(num_candidates):
            temp = temperatures[i % len(temperatures)]
            text = model.generate(
                tokenizer,
                title,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                max_len=max_len,
            )
            candidates.append(text)
    return candidates


def build_dpo_pairs(
    config_path: str = "./scratch/configs/mha.yaml",
    num_pairs: int = 1000,
    num_candidates: int = 8,
    temperatures: str = "0.7,0.9,1.0,1.2",
    output_path: str = "./dataset/dpo/dpo_pairs.json",
    seed: int = 42,
):
    """生成 DPO 偏好对

    Args:
        config_path: 模型配置文件路径
        num_pairs: 目标偏好对数量，循环遍历词牌直到达到该数量
        num_candidates: 每个词牌每轮生成的候选数量
        temperatures: 逗号分隔的 temperature 列表，循环使用
        output_path: 输出 JSON 路径
        seed: 随机种子
    """
    random.seed(seed)

    # 解析 temperatures
    temp_list = [float(t) for t in temperatures.split(",")]

    # 加载配置和模型
    config = load_config(config_path)
    device = config.train.device

    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    model = SongCiGPT(config.model)
    model.load_state_dict(torch.load(config.train.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"模型加载完成: {config.train.ckpt_path}")

    # 加载格律库
    registry_path = str(Path(__file__).parent / "songeval" / "standard.json")
    evaluator = Evaluator(registry_path=registry_path)
    titles = list(evaluator.registry.keys())
    print(f"格律库加载完成，共 {len(titles)} 个词牌")

    # 循环遍历词牌，直到生成足够数量的偏好对
    dpo_pairs = []
    skipped = 0
    round_num = 0

    pbar = tqdm(total=num_pairs, desc="生成偏好对")
    while len(dpo_pairs) < num_pairs:
        round_num += 1
        random.shuffle(titles)  # 每轮打乱顺序，增加多样性

        for title in titles:
            if len(dpo_pairs) >= num_pairs:
                break

            # 生成候选
            candidates = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                title=title,
                num_candidates=num_candidates,
                temperatures=temp_list,
                top_k=config.inference.top_k,
                top_p=config.inference.top_p,
                max_len=config.inference.max_len,
                device=device,
            )

            # 评估每个候选
            scored = []
            for text in candidates:
                report = evaluator.evaluate(title, text)
                if "error" in report:
                    continue
                scored.append((text, report["form_score"]))

            # 至少需要 2 个有效候选才能组成偏好对
            if len(scored) < 2:
                skipped += 1
                continue

            # 按分数排序
            scored.sort(key=lambda x: x[1], reverse=True)

            # 取最高分作为 chosen，最低分作为 rejected
            chosen_text, chosen_score = scored[0]
            rejected_text, rejected_score = scored[-1]

            # 跳过分数相同的（没有区分度）
            if chosen_score == rejected_score:
                skipped += 1
                continue

            # 清理文本：去掉 <bos>、<eos>、<sep>，再去掉标题前缀
            chosen_text = (
                chosen_text.replace("<bos>", "")
                .replace("<eos>", "")
                .replace("<sep>", "")
                .strip()
            )
            rejected_text = (
                rejected_text.replace("<bos>", "")
                .replace("<eos>", "")
                .replace("<sep>", "")
                .strip()
            )
            if chosen_text.startswith(title):
                chosen_text = chosen_text[len(title) :]
            if rejected_text.startswith(title):
                rejected_text = rejected_text[len(title) :]

            dpo_pairs.append(
                {
                    "prompt": title,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                }
            )
            pbar.update(1)

    pbar.close()

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n生成完成！")
    print(f"  有效偏好对: {len(dpo_pairs)}")
    print(f"  跳过: {skipped}")
    print(f"  轮次: {round_num}")
    print(f"  保存到: {output_path}")

    # 打印几个示例
    if dpo_pairs:
        print(f"\n示例偏好对:")
        for pair in dpo_pairs[:3]:
            print(f"  词牌: {pair['prompt']}")
            print(
                f"  chosen (score={pair['chosen_score']:.2f}): {pair['chosen'][:50]}..."
            )
            print(
                f"  rejected (score={pair['rejected_score']:.2f}): {pair['rejected'][:50]}..."
            )
            print()


if __name__ == "__main__":
    Fire(build_dpo_pairs)
