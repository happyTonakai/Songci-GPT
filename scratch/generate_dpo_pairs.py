"""使用 SongEval 格律评估作为奖励模型，生成 DPO 偏好对

流程：
1. 加载训练好的 SFT 模型
2. 对每个词牌名生成多个候选宋词（不同 temperature）
3. 用 SongEval Evaluator 对每个候选打分
4. 取最高分作为 chosen，最低分作为 rejected，组成偏好对
5. 保存为 DPODataset 可读取的 JSON 格式

支持多进程并行：利用 GPU 推理低显存的特点，开多个进程共享一张卡。
使用共享队列动态分配任务，快的 worker 多干，慢的少干，整体一起结束。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import torch.multiprocessing as mp
from fire import Fire
from tqdm import tqdm

from config import load_config
from model import SongCiGPT
from songeval import Evaluator
from tokenizer import BPETokenizer

_SENTINEL = None  # 结束信号


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
    """对一个词牌名生成多个候选宋词"""
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


def _refill_queue(queue: mp.Queue, titles: list[str]):
    """将打乱顺序的词牌放入队列"""
    shuffled = titles[:]
    random.shuffle(shuffled)
    for t in shuffled:
        queue.put(t)


def _worker(
    rank: int,
    config_path: str,
    queue: mp.Queue,
    counter,  # mp.Value('i')
    num_pairs: int,
    num_candidates: int,
    temp_list: list[float],
    min_score_diff: float,
    min_chosen_score: float,
    max_rejected_score: float,
    output_path: str,
    seed: int,
):
    """工作进程：从队列取词牌，生成候选，评估，写结果"""
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    config = load_config(config_path)
    device = config.train.device

    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    model = SongCiGPT(config.model)
    model.load_state_dict(torch.load(config.train.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    registry_path = str(Path(__file__).parent / "songeval" / "standard.json")
    evaluator = Evaluator(registry_path=registry_path)

    dpo_pairs = []
    skipped = 0
    worker_output = f"{output_path}.worker{rank}"
    pbar = tqdm(total=num_pairs, desc=f"Worker {rank}", position=rank, leave=True)

    while True:
        # 检查全局计数是否已达标
        with counter.get_lock():
            if counter.value >= num_pairs:
                break

        try:
            title = queue.get(timeout=5)
        except Exception:
            # 队列空了，等待其他 worker 填充
            continue

        if title is _SENTINEL:
            break

        if title == "失调名":
            continue

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

        # 评估
        scored = []
        for text in candidates:
            # 清理特殊标记和标题（model.generate 的输出包含 prompt 部分）
            clean_text = (
                text.replace("<bos>", "")
                .replace("<eos>", "")
                .replace("<sep>", "")
                .strip()
            )
            if clean_text.startswith(title):
                clean_text = clean_text[len(title) :]
            report = evaluator.evaluate(title, clean_text)
            if "error" in report:
                continue
            scored.append((text, report["form_score"]))

        if len(scored) < 2:
            skipped += 1
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_text, chosen_score = scored[0]
        rejected_text, rejected_score = scored[-1]

        if chosen_score - rejected_score < min_score_diff:
            skipped += 1
            continue

        # 过滤绝对分数：chosen 太低说明最好的也烂，rejected 太高说明最差的也不够差
        if chosen_score < min_chosen_score or rejected_score > max_rejected_score:
            skipped += 1
            continue

        # 清理文本
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

        pair = {
            "prompt": title,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        }
        dpo_pairs.append(pair)

        # 实时追加写入（JSONL 格式，一行一条）
        with open(worker_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # 更新全局计数
        with counter.get_lock():
            counter.value += 1
        pbar.update(1)

    pbar.close()


def build_dpo_pairs(
    config_path: str = "./scratch/configs/mha.yaml",
    num_pairs: int = 1000,
    num_candidates: int = 16,
    temperatures: str = "0.8,1.0,1.3,1.5",
    min_score_diff: float = 0.06,
    min_chosen_score: float = 0.90,
    max_rejected_score: float = 0.50,
    output_path: str = "./dataset/dpo/dpo_pairs.json",
    num_workers: int = 3,
    seed: int = 42,
):
    """生成 DPO 偏好对

    Args:
        config_path: 模型配置文件路径
        num_pairs: 目标偏好对数量
        num_candidates: 每个词牌每轮生成的候选数量
        temperatures: 逗号分隔的 temperature 列表
        min_score_diff: chosen 和 rejected 的最小分数差
        min_chosen_score: chosen 的最低分数，低于此值说明"最好的也烂"
        max_rejected_score: rejected 的最高分数，高于此值说明"最差的也不够差"
        output_path: 输出 JSON 路径
        num_workers: 并行进程数
        seed: 随机种子
    """
    temp_list = [float(t) for t in temperatures.split(",")]

    # 加载格律库获取词牌列表
    registry_path = str(Path(__file__).parent / "songeval" / "standard.json")
    evaluator = Evaluator(registry_path=registry_path)
    all_titles = [t for t in evaluator.registry.keys() if t != "失调名"]
    print(f"格律库加载完成，共 {len(all_titles)} 个词牌（已排除失调名）")
    print(f"启动 {num_workers} 个并行进程...")

    # 必须先设置 start method，再创建共享对象
    mp.set_start_method("spawn", force=True)

    # 共享队列和计数器
    queue = mp.Queue()
    counter = mp.Value("i", 0)

    # 预填队列
    for _ in range(20):
        _refill_queue(queue, all_titles)

    # 启动 worker 进程（用 mp.Process 替代 mp.spawn 以支持 join=False）
    workers = []
    for rank in range(num_workers):
        p = mp.Process(
            target=_worker,
            args=(
                rank,
                config_path,
                queue,
                counter,
                num_pairs,
                num_candidates,
                temp_list,
                min_score_diff,
                min_chosen_score,
                max_rejected_score,
                output_path,
                seed,
            ),
        )
        p.start()
        workers.append(p)

    # 主进程持续补充队列，直到达到目标数量
    while any(p.is_alive() for p in workers):
        with counter.get_lock():
            if counter.value >= num_pairs:
                break
        # 队列快空了就补充
        if queue.qsize() < len(all_titles):
            for _ in range(10):
                _refill_queue(queue, all_titles)
        import time
        time.sleep(1)

    # 发送结束信号
    for _ in range(num_workers):
        queue.put(_SENTINEL)

    for p in workers:
        p.join()

    # 合并结果（JSONL 格式：每行一个 JSON 对象）
    dpo_pairs = []
    for rank in range(num_workers):
        worker_file = f"{output_path}.worker{rank}"
        if os.path.exists(worker_file):
            with open(worker_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dpo_pairs.append(json.loads(line))
            os.remove(worker_file)

    # 按分数差排序，取前 num_pairs 条
    dpo_pairs.sort(key=lambda d: d["chosen_score"] - d["rejected_score"], reverse=True)
    dpo_pairs = dpo_pairs[:num_pairs]

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n生成完成！")
    print(f"  有效偏好对: {len(dpo_pairs)}")
    print(f"  保存到: {output_path}")

    if dpo_pairs:
        diffs = [d["chosen_score"] - d["rejected_score"] for d in dpo_pairs]
        print(f"  平均分数差: {sum(diffs) / len(diffs):.4f}")
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
