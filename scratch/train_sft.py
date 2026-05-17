from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from config import Config, load_config
from einops import rearrange
from fire import Fire
from model import SongCiGPT
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SongCiDataset


class Trainer:
    def __init__(
        self,
        model: SongCiGPT,
        dataset: SongCiDataset,
        config: Config,
    ):
        self.config = config
        self.model = model
        self.device = torch.device(config.train.device)
        self.model.to(self.device)
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
        self.max_seq_len = config.model.max_seq_len
        self.global_step = 0  # 全局步数，用于课程学习

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            pbar = tqdm(self.dataloader, total=len(self.dataloader))
            for batch_dict in pbar:
                input_ids = batch_dict["input_ids"].to(self.device)
                labels = batch_dict["labels"].to(self.device)
                attention_mask = batch_dict["attention_mask"].to(self.device)

                # Elastic Training: 每次迭代随机采样弹性配置
                elastic_params = self._sample_elastic_config()
                logits, _, aux_loss, loop_logits = self.model(
                    input_ids, attention_mask, **elastic_params
                )

                if loop_logits is not None:
                    # Per-Block Loop: 只用最后一个 loop step 的 logits 计算 loss
                    # 推理时也只用最后一个 step，保持训练/推理一致
                    loss = F.cross_entropy(
                        rearrange(loop_logits[-1], "b s d -> (b s) d"),
                        rearrange(labels, "b s -> (b s)"),
                    )
                else:
                    loss = F.cross_entropy(
                        rearrange(logits, "b s d -> (b s) d"),
                        rearrange(labels, "b s -> (b s)"),
                    )
                # total_loss = loss + aux_loss
                total_loss = self._compute_total_loss(loss, aux_loss)
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # 构建弹性训练的描述信息
                elastic_desc = ""
                if elastic_params:
                    parts = []
                    if "active_layer_indices" in elastic_params:
                        parts.append(f"depth={len(elastic_params['active_layer_indices'])}")
                    if "active_experts" in elastic_params:
                        parts.append(f"width={len(elastic_params['active_experts'])}")
                    if "elastic_topk" in elastic_params:
                        parts.append(f"topk={elastic_params['elastic_topk']}")
                    if "block_loop_count" in elastic_params:
                        parts.append(f"block_loop={elastic_params['block_loop_count']}")
                    if "layer_loop_counts" in elastic_params:
                        parts.append(f"layer_loop={elastic_params['layer_loop_counts'][0]}")
                    if parts:
                        elastic_desc = f", Elastic [{', '.join(parts)}]"

                pbar.set_description(
                    f"Epoch {epoch}, Loss {loss.item():.4f}, Aux Loss {aux_loss.item():.4f}{elastic_desc}"
                )

            if epoch % self.config.train.save_interval == 0:
                self.save(f"./scratch/ckpt/model_{epoch}.pt")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")

    def _compute_total_loss(self, ce_loss: Tensor, aux_loss: Tensor) -> Tensor:
        """
        Args:
            ce_loss: 主任务交叉熵损失 (Tensor)
            aux_loss: MoE 负载均衡损失 (Tensor)
        """
        # 1. 使用 detach() 提取数值，避免梯度回传到这个比例计算中
        ce_val = ce_loss.detach().item()
        aux_val = aux_loss.detach().item()

        # 2. 防止除以 0 (刚开始训练时 aux_val 可能是 0)
        target_ratio = self.config.train.aux_loss_target_ratio
        if aux_val > 1e-8:
            # 计算当前的动态系数
            # 目标是：lambda * aux_val = target_ratio * ce_val
            dynamic_lambda = target_ratio * (ce_val / aux_val)
        else:
            dynamic_lambda = 0.0

        # 3. 这里的 dynamic_lambda 只是一个标量系数
        total_loss = ce_loss + dynamic_lambda * aux_loss

        return total_loss

    def _sample_elastic_config(self) -> dict:
        """采样弹性训练配置（ERNIE 5.0 Elastic Training）

        三个维度独立采样：
        - Elastic Depth: Bypassing 跳层，被跳过的层执行恒等映射（残差直连）
        - Elastic Width: Masking 屏蔽非活跃专家，核心专家永不屏蔽
        - Elastic Sparsity: 使用更小的 top-k

        课程学习：warmup_steps 之前使用全量网络，之后才开启弹性。

        Returns:
            包含弹性参数的 dict，可直接 ** 解包传给 model.forward()
        """
        elastic_cfg = self.config.elastic
        if elastic_cfg is None:
            return {}

        # 课程学习：warmup 阶段使用全量网络
        if self.global_step < elastic_cfg.warmup_steps:
            return {}

        num_layers = self.config.model.num_layers
        n_experts = self.config.model.n_experts
        topk = self.config.model.topk
        result = {}

        # ── Elastic Depth: Bypassing 跳层 ──
        # 被跳过的层执行恒等映射 X_{l+1} = X_l（残差连接的本意）
        # 梯度直接穿过被跳过的层，不更新其参数
        if random.random() < elastic_cfg.depth_prob:
            # 从预定义配置库中随机选一个深度
            if elastic_cfg.depth_levels:
                levels = [d for d in elastic_cfg.depth_levels if d < num_layers]
                if levels:
                    n_active = random.choice(levels)
                else:
                    n_active = num_layers
            else:
                n_active = random.randint(2, num_layers)
            # 随机均匀选取哪些层保持活跃（Bypassing），而非截断前缀
            active_layers = sorted(random.sample(range(num_layers), n_active))
            result["active_layer_indices"] = active_layers

        # ── Elastic Width: 屏蔽部分专家 ──
        # 核心专家（core_experts）永不被屏蔽
        if n_experts > 1 and random.random() < elastic_cfg.width_prob:
            core = set(elastic_cfg.core_experts or [])
            # 校验 core_experts 不越界
            core = {i for i in core if 0 <= i < n_experts}
            remaining = [i for i in range(n_experts) if i not in core]
            # 从预定义配置库中随机选一个宽度（严格小于，全量由 width_prob 控制）
            if elastic_cfg.width_levels:
                levels = [w for w in elastic_cfg.width_levels if w < n_experts]
                if levels:
                    n_active = random.choice(levels)
                else:
                    n_active = n_experts
            else:
                n_active = random.randint(1, n_experts)
            # 确保 n_active 至少包含所有核心专家
            n_active = max(n_active, len(core))
            n_from_remaining = min(n_active - len(core), len(remaining))
            selected = sorted(core | set(random.sample(remaining, n_from_remaining)))
            result["active_experts"] = selected

        # ── Elastic Sparsity: 缩减 top-k ──
        if topk > 1 and random.random() < elastic_cfg.sparsity_prob:
            if elastic_cfg.sparsity_levels:
                levels = [k for k in elastic_cfg.sparsity_levels if 1 <= k < topk]
                if levels:
                    result["elastic_topk"] = random.choice(levels)
            else:
                result["elastic_topk"] = random.randint(1, topk)

        # ── Per-Block Loop: LoopLM/Ouro ──
        # 默认循环 block_loop_count 次，小概率降到更低的循环次数（含 1=不循环）
        if elastic_cfg.block_loop_count >= 2:
            result["block_loop_count"] = elastic_cfg.block_loop_count
            if random.random() < elastic_cfg.block_loop_drop_prob:
                levels = [t for t in (elastic_cfg.block_loop_levels or []) if 1 <= t < elastic_cfg.block_loop_count]
                if levels:
                    result["block_loop_count"] = random.choice(levels)

        # ── Per-Layer Loop ──
        # 默认循环 layer_loop_count 次，小概率降到更低的循环次数（含 1=不循环）
        if elastic_cfg.layer_loop_count >= 2:
            result["layer_loop_counts"] = [elastic_cfg.layer_loop_count] * num_layers
            if random.random() < elastic_cfg.layer_loop_drop_prob:
                levels = [t for t in (elastic_cfg.layer_loop_levels or []) if 1 <= t < elastic_cfg.layer_loop_count]
                if levels:
                    if random.random() < 0.5:
                        # 50%: 所有层统一降级到同一个 loop count
                        result["layer_loop_counts"] = [random.choice(levels)] * num_layers
                    else:
                        # 50%: 每层独立采样 loop count
                        result["layer_loop_counts"] = [random.choice(levels) for _ in range(num_layers)]

        return result


def train(config_path: str = "./scratch/configs/mha.yaml"):
    config = load_config(config_path)
    model = SongCiGPT(config.model)
    dataset = SongCiDataset(config.data)
    trainer = Trainer(model, dataset, config)
    try:
        trainer.train(num_epochs=config.train.epochs)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        trainer.save(config.train.ckpt_path)


if __name__ == "__main__":
    Fire(train)
