import torch
import torch.nn.functional as F
from einops import rearrange
from model import SongCiGPT
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SongCiDataset


class Trainer:
    def __init__(
        self,
        model: SongCiGPT,
        dataset: SongCiDataset,
        device: torch.device,
        lr: float = 1e-4,
        batch_size: int = 32,
        max_seq_len: int = 256,
    ):
        self.model = model
        self.model.to(device)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.max_seq_len = max_seq_len

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            pbar = tqdm(self.dataloader, total=len(self.dataloader))
            for batch_dict in pbar:
                input_ids = batch_dict["input_ids"].to(self.device)
                labels = batch_dict["labels"].to(self.device)
                attention_mask = batch_dict["attention_mask"].to(self.device)

                logits, _, aux_loss = self.model(input_ids, attention_mask)

                loss = F.cross_entropy(
                    rearrange(logits, "b s d -> (b s) d"),
                    rearrange(labels, "b s -> (b s)"),
                )
                # total_loss = loss + aux_loss
                total_loss = self._compute_total_loss(loss, aux_loss)
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_description(
                    f"Epoch {epoch}, Loss {loss.item():.4f}, Aux Loss {aux_loss.item():.4f}"
                )

            if epoch % 20 == 0:
                self.save(f"./scratch/ckpt/model_{epoch}.pt")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")

    @staticmethod
    def _compute_total_loss(
        ce_loss: torch.Tensor, aux_loss: torch.Tensor, target_ratio: float = 0.01
    ) -> torch.Tensor:
        """
        Args:
            ce_loss: 主任务交叉熵损失 (Tensor)
            aux_loss: MoE 负载均衡损失 (Tensor)
            target_ratio: 你希望 aux_loss 占据总梯度的期望比例 (比如 10%)
        """
        # 1. 使用 detach() 提取数值，避免梯度回传到这个比例计算中
        ce_val = ce_loss.detach().item()
        aux_val = aux_loss.detach().item()

        # 2. 防止除以 0 (刚开始训练时 aux_val 可能是 0)
        if aux_val > 1e-8:
            # 计算当前的动态系数
            # 目标是：lambda * aux_val = target_ratio * ce_val
            dynamic_lambda = target_ratio * (ce_val / aux_val)
        else:
            dynamic_lambda = 0.0

        # 3. 这里的 dynamic_lambda 只是一个标量系数
        total_loss = ce_loss + dynamic_lambda * aux_loss

        return total_loss


def train():
    model = SongCiGPT()
    dataset = SongCiDataset()
    trainer = Trainer(
        model, dataset, device=torch.device("cuda"), batch_size=32, max_seq_len=256
    )
    try:
        trainer.train(num_epochs=100)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        trainer.save("./scratch/ckpt/model.pt")


if __name__ == "__main__":
    train()
