from __future__ import annotations

from pathlib import Path

import torch
from config import Config, load_config
from fire import Fire
from model import SongCiGPT
from tokenizer import BPETokenizer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# DPO Dataset
# ============================================================
class DPODataset(Dataset):
    """DPO 偏好数据集

    数据格式 (JSON):
    [
        {
            "prompt": "浣溪沙",
            "chosen": "一曲新词酒一杯...",
            "rejected": "无可奈何花落去..."
        },
        ...
    ]

    每条样本包含:
    - prompt: 条件输入（词牌名）
    - chosen: 人工偏好的回答
    - rejected: 不偏好的回答
    """

    def __init__(self, config, dpo_data_path: str):
        self.config = config
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(config.tokenizer_path)

        from glob import glob

        import orjson

        files = glob(f"{dpo_data_path}/*.json")
        raw_data = []
        for file in files:
            with open(file, "rb") as f:
                raw_data += orjson.loads(f.read())

        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        sep_id = self.tokenizer.sep_id
        pad_id = self.tokenizer.pad_id
        max_seq_len = config.max_seq_len

        self.data = []
        for item in raw_data:
            prompt_tokens = self.tokenizer.encode(item["prompt"])
            chosen_tokens = self.tokenizer.encode(item["chosen"])
            rejected_tokens = self.tokenizer.encode(item["rejected"])

            # 构造完整序列: <bos> + prompt + <sep> + chosen/rejected + <eos> + <pad>
            chosen_full = [bos_id] + prompt_tokens + [sep_id] + chosen_tokens + [eos_id]
            rejected_full = (
                [bos_id] + prompt_tokens + [sep_id] + rejected_tokens + [eos_id]
            )

            if len(chosen_full) > max_seq_len or len(rejected_full) > max_seq_len:
                continue

            # padding
            chosen_full += [pad_id] * (max_seq_len - len(chosen_full))
            rejected_full += [pad_id] * (max_seq_len - len(rejected_full))

            # prompt 长度（用于计算 loss 时屏蔽 prompt 部分）
            prompt_len = 2 + len(prompt_tokens)  # <bos> + prompt

            self.data.append(
                {
                    "chosen_input_ids": torch.tensor(chosen_full, dtype=torch.long),
                    "rejected_input_ids": torch.tensor(rejected_full, dtype=torch.long),
                    "prompt_len": prompt_len,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen_ids = item["chosen_input_ids"]
        rejected_ids = item["rejected_input_ids"]
        pad_id = self.tokenizer.pad_id

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_ids == pad_id,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_ids == pad_id,
            "prompt_len": item["prompt_len"],
        }


# ============================================================
# DPO Trainer
# ============================================================
class DPOTrainer:
    def __init__(
        self,
        model: SongCiGPT,
        ref_model: SongCiGPT,
        dataset: DPODataset,
        config: Config,
    ):
        self.config = config
        self.device = torch.device(config.train.device)

        # policy model（可训练）
        self.model = model.to(self.device)

        # reference model（冻结，用于计算 DPO loss 中的隐式奖励）
        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            persistent_workers=config.train.num_workers > 0,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.tokenizer = dataset.tokenizer
        self.global_step = 0

    def _sample_elastic_config(self) -> dict:
        """采样弹性训练配置（与 SFT Trainer 相同逻辑）"""
        import random

        elastic_cfg = self.config.elastic
        if elastic_cfg is None:
            return {}
        if self.global_step < elastic_cfg.warmup_steps:
            return {}

        num_layers = self.config.model.num_layers
        n_experts = self.config.model.n_experts
        topk = self.config.model.topk
        result = {}

        # Elastic Depth
        if random.random() < elastic_cfg.depth_prob:
            if elastic_cfg.depth_levels:
                levels = [d for d in elastic_cfg.depth_levels if d < num_layers]
                if levels:
                    n_active = random.choice(levels)
                else:
                    n_active = num_layers
            else:
                n_active = random.randint(2, num_layers)
            result["active_layer_indices"] = sorted(random.sample(range(num_layers), n_active))

        # Elastic Width
        if n_experts > 1 and random.random() < elastic_cfg.width_prob:
            core = {i for i in (elastic_cfg.core_experts or []) if 0 <= i < n_experts}
            remaining = [i for i in range(n_experts) if i not in core]
            if elastic_cfg.width_levels:
                levels = [w for w in elastic_cfg.width_levels if w < n_experts]
                if levels:
                    n_active = random.choice(levels)
                else:
                    n_active = n_experts
            else:
                n_active = random.randint(1, n_experts)
            n_active = max(n_active, len(core))
            n_from_remaining = min(n_active - len(core), len(remaining))
            result["active_experts"] = sorted(core | set(random.sample(remaining, n_from_remaining)))

        # Elastic Sparsity
        if topk > 1 and random.random() < elastic_cfg.sparsity_prob:
            if elastic_cfg.sparsity_levels:
                levels = [k for k in elastic_cfg.sparsity_levels if 1 <= k < topk]
                if levels:
                    result["elastic_topk"] = random.choice(levels)
            else:
                result["elastic_topk"] = random.randint(1, topk)

        # Per-Block Loop: 默认循环 block_loop_count 次，小概率降级（含 1=不循环）
        if elastic_cfg.block_loop_count >= 2:
            result["block_loop_count"] = elastic_cfg.block_loop_count
            if random.random() < elastic_cfg.block_loop_drop_prob:
                levels = [t for t in (elastic_cfg.block_loop_levels or []) if 1 <= t < elastic_cfg.block_loop_count]
                if levels:
                    result["block_loop_count"] = random.choice(levels)

        # Per-Layer Loop: 默认循环 layer_loop_count 次，小概率降级（含 1=不循环）
        if elastic_cfg.layer_loop_count >= 2:
            result["layer_loop_counts"] = [elastic_cfg.layer_loop_count] * num_layers
            if random.random() < elastic_cfg.layer_loop_drop_prob:
                levels = [t for t in (elastic_cfg.layer_loop_levels or []) if 1 <= t < elastic_cfg.layer_loop_count]
                if levels:
                    if random.random() < 0.5:
                        result["layer_loop_counts"] = [random.choice(levels)] * num_layers
                    else:
                        result["layer_loop_counts"] = [random.choice(levels) for _ in range(num_layers)]

        return result

    def compute_log_probs(
        self,
        model: SongCiGPT,
        input_ids: Tensor,
        attention_mask: Tensor,
        prompt_len: int,
        **elastic_params,
    ) -> Tensor:
        """计算模型对 response 部分的 log 概率

        Args:
            model: 模型
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            prompt_len: prompt 的 token 长度（从 prompt_len 开始计算 loss）
            **elastic_params: 弹性训练参数

        Returns:
            每个样本的 response 部分 log 概率之和 (batch,)
        """
        # 1. 前向传播得到 logits（只用最终步 logits，丢弃 loop_logits）
        logits, _, _, _ = model(
            input_ids.to(self.device), attention_mask.to(self.device),
            **elastic_params,
        )
        # 2. 错位对齐（偏移 1 位）
        # 预测 logits[t] -> 目标 target_ids[t]
        # 形状：logits 变为 (B, L-1, V), target_ids 变为 (B, L-1)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        # 3. 抽奖：从全家桶里抠出正确答案的概率
        # 这一步执行完，per_token_logps 的形状是 (B, L-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        # 3. 只取 response 部分 (从 prompt_len 到序列末尾) 的 log_probs
        _, max_len_minus_1 = target_ids.shape

        # 构造一个 [[0,1,2...], [0,1,2...]] 的索引矩阵
        # 形状 (L-1,) -> (1, L-1)
        step_indices = torch.arange(max_len_minus_1).unsqueeze(0)

        # 利用广播机制：(1, L-1) 与 (B, 1) 比较
        # prompt_len 形状 (B,) -> (B, 1)
        prompt_mask = step_indices >= (prompt_len.unsqueeze(1) - 1)
        prompt_mask = prompt_mask.to(input_ids.device)

        pad_mask = ~attention_mask[:, 1:]

        final_mask = prompt_mask & pad_mask

        logp = (final_mask * per_token_logps).sum(dim=-1)
        return logp

    def compute_dpo_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
        beta: float,
    ) -> Tensor:
        """计算 DPO loss

        DPO loss = -log(sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x)
                                    - log pi(y_l|x) + log pi_ref(y_l|x))))

        Args:
            chosen_logps: policy model 对 chosen 的 log 概率
            rejected_logps: policy model 对 rejected 的 log 概率
            ref_chosen_logps: reference model 对 chosen 的 log 概率
            ref_rejected_logps: reference model 对 rejected 的 log 概率
            beta: 温度参数，控制偏离 reference model 的程度

        Returns:
            DPO loss (标量)
        """
        policy_logp_diff = chosen_logps - rejected_logps
        ref_logp_diff = ref_chosen_logps - ref_rejected_logps
        logits = policy_logp_diff - ref_logp_diff
        loss = -F.logsigmoid(beta * logits).mean()
        return loss

    def train(self, num_epochs: int):
        beta = self.config.train.dpo.beta

        for epoch in range(num_epochs):
            self.model.train()
            pbar = tqdm(self.dataloader, total=len(self.dataloader))
            pbar.set_description(f"Epoch {epoch}")
            for batch_dict in pbar:
                chosen_ids = batch_dict["chosen_input_ids"].to(self.device)
                chosen_mask = batch_dict["chosen_attention_mask"].to(self.device)
                rejected_ids = batch_dict["rejected_input_ids"].to(self.device)
                rejected_mask = batch_dict["rejected_attention_mask"].to(self.device)
                prompt_len = batch_dict["prompt_len"]

                # Elastic Training: 采样弹性配置（policy 和 ref 用同一份）
                elastic_params = self._sample_elastic_config()

                # 1. 用 policy model 计算 chosen/rejected 的 log_probs
                chosen_logps = self.compute_log_probs(
                    self.model, chosen_ids, chosen_mask, prompt_len,
                    **elastic_params,
                )
                rejected_logps = self.compute_log_probs(
                    self.model, rejected_ids, rejected_mask, prompt_len,
                    **elastic_params,
                )
                # 2. 用 ref model 计算 chosen/rejected 的 log_probs (no_grad)
                with torch.no_grad():
                    ref_chosen_logps = self.compute_log_probs(
                        self.ref_model, chosen_ids, chosen_mask, prompt_len,
                        **elastic_params,
                    )
                    ref_rejected_logps = self.compute_log_probs(
                        self.ref_model, rejected_ids, rejected_mask, prompt_len,
                        **elastic_params,
                    )
                self.global_step += 1
                # 3. 计算 DPO loss
                loss = self.compute_dpo_loss(
                    chosen_logps,
                    rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta,
                )
                # 4. 反向传播 + 更新参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if epoch % self.config.train.save_interval == 0:
                self.save(f"./scratch/ckpt/dpo_model_{epoch}.pt")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")


# ============================================================
# Evaluation
# ============================================================
def evaluate(
    config_path: str = "./scratch/configs/dpo_mha.yaml",
    ref_ckpt_path: str = None,
    dpo_ckpt_path: str = None,
    num_titles: int = 75,
    num_samples: int = 5,
):
    """对比评估 ref 模型和 DPO 模型的格律得分

    Args:
        config_path: 配置文件路径
        ref_ckpt_path: SFT 模型 checkpoint 路径（默认从配置读取）
        dpo_ckpt_path: DPO 模型 checkpoint 路径（默认从配置推导）
        num_titles: 评估的词牌数量
        num_samples: 每个词牌生成的样本数
    """
    config = load_config(config_path)
    if ref_ckpt_path is None:
        ref_ckpt_path = config.train.dpo.ref_ckpt_path
    if dpo_ckpt_path is None:
        dpo_ckpt_path = config.train.ckpt_path.replace(".pt", "_dpo.pt")
    from songeval import Evaluator

    device = config.train.device
    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    registry_path = str(Path(__file__).parent / "songeval" / "standard.json")
    evaluator = Evaluator(registry_path=registry_path)
    titles = list(evaluator.registry.keys())[:num_titles]

    def load_model(ckpt_path: str) -> SongCiGPT:
        model = SongCiGPT(config.model)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        model.to(device)
        model.eval()
        return model

    def generate(model: SongCiGPT, title: str) -> str:
        with torch.no_grad():
            text = model.generate(
                tokenizer,
                title,
                temperature=config.inference.temperature,
                top_k=config.inference.top_k,
                top_p=config.inference.top_p,
                max_len=config.inference.max_len,
            )
        return text.replace("<bos>", "").replace("<eos>", "").replace("<sep>", "").strip()

    def score_model(model: SongCiGPT, label: str) -> dict:
        form_scores = []
        tonal_accs = []
        rhyme_cons = []
        struct_hits = 0
        total = 0

        for title in tqdm(titles, desc=f"评估 {label}"):
            for _ in range(num_samples):
                text = generate(model, title)
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

    # 加载两个模型
    ref_model = load_model(ref_ckpt_path)
    dpo_model = load_model(dpo_ckpt_path)

    # 评估
    ref_scores = score_model(ref_model, "Ref (SFT)")
    dpo_scores = score_model(dpo_model, "DPO")

    # 打印对比表
    print("\n" + "=" * 70)
    print("  DPO 对齐效果评估")
    print("=" * 70)
    print(f"  {'指标':<20} {'Ref (SFT)':>12} {'DPO':>12} {'Delta':>10}")
    print("-" * 70)

    metrics = [
        ("结构匹配率", "structure_match_rate", True),
        ("平均平仄准确度", "avg_tonal_accuracy", True),
        ("平均押韵一致性", "avg_rhyme_consistency", True),
        ("综合格律得分", "avg_form_score", True),
    ]
    for name, key, is_pct in metrics:
        ref_v = ref_scores[key]
        dpo_v = dpo_scores[key]
        delta = dpo_v - ref_v
        if is_pct:
            print(f"  {name:<20} {ref_v*100:>11.2f}% {dpo_v*100:>11.2f}% {delta*100:>+9.2f}%")
        else:
            print(f"  {name:<20} {ref_v:>12.4f} {dpo_v:>12.4f} {delta:>+10.4f}")

    print(f"\n  评估样本数: {ref_scores['total_samples']} (每个模型)")
    print("=" * 70)

    return ref_scores, dpo_scores


# ============================================================
# Entry Point
# ============================================================
def train(
    config_path: str = "./scratch/configs/dpo_mha.yaml",
):
    """DPO 训练入口

    Args:
        config_path: 模型配置文件路径
    """
    config = load_config(config_path)
    ref_ckpt_path = config.train.dpo.ref_ckpt_path

    # policy model
    model = SongCiGPT(config.model)

    # reference model（冻结的旧策略）
    ref_model = SongCiGPT(config.model)

    model.load_state_dict(torch.load(ref_ckpt_path, weights_only=True))
    ref_model.load_state_dict(torch.load(ref_ckpt_path, weights_only=True))

    dataset = DPODataset(config.data, config.train.dpo.data_path)
    trainer = DPOTrainer(model, ref_model, dataset, config)

    dpo_ckpt_path = config.train.ckpt_path.replace(".pt", "_dpo.pt")
    try:
        trainer.train(num_epochs=config.train.epochs)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        trainer.save(dpo_ckpt_path)

    # 训练完成后自动评估
    print("\n训练完成，开始评估...")
    evaluate(config_path, ref_ckpt_path, dpo_ckpt_path)


if __name__ == "__main__":
    Fire({"train": train, "evaluate": evaluate})
