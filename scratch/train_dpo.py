from __future__ import annotations

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
            persistent_workers=True,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.tokenizer = dataset.tokenizer

    def compute_log_probs(
        self,
        model: SongCiGPT,
        input_ids: Tensor,
        attention_mask: Tensor,
        prompt_len: int,
    ) -> Tensor:
        """计算模型对 response 部分的 log 概率

        Args:
            model: 模型
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            prompt_len: prompt 的 token 长度（从 prompt_len 开始计算 loss）

        Returns:
            每个样本的 response 部分 log 概率之和 (batch,)
        """
        # 1. 前向传播得到 logits
        logits, _, _ = model(input_ids.to(self.device), attention_mask.to(self.device))
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

                # TODO: 实现训练步骤
                # 1. 用 policy model 计算 chosen/rejected 的 log_probs
                chosen_logps = self.compute_log_probs(
                    self.model, chosen_ids, chosen_mask, prompt_len
                )
                rejected_logps = self.compute_log_probs(
                    self.model, rejected_ids, rejected_mask, prompt_len
                )
                # 2. 用 ref model 计算 chosen/rejected 的 log_probs (no_grad)
                with torch.no_grad():
                    ref_chosen_logps = self.compute_log_probs(
                        self.ref_model, chosen_ids, chosen_mask, prompt_len
                    )
                    ref_rejected_logps = self.compute_log_probs(
                        self.ref_model, rejected_ids, rejected_mask, prompt_len
                    )
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

            if epoch % self.config.train.save_interval == 0:
                self.save(f"./scratch/ckpt/dpo_model_{epoch}.pt")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")


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

    try:
        trainer.train(num_epochs=config.train.epochs)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        trainer.save(config.train.ckpt_path.replace(".pt", "_dpo.pt"))


if __name__ == "__main__":
    Fire(train)
