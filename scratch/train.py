import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SongCiDataset
from model import SongCiGPT


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
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True
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

                outputs, _ = self.model(input_ids, attention_mask)

                loss = F.cross_entropy(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_description(f"Epoch {epoch}, Loss {loss.item():.4f}")

            if epoch % 20 == 0:
                self.save(f"./ckpt/model_{epoch}.pt")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model loaded from {path}")


def train():
    model = SongCiGPT()
    dataset = SongCiDataset()
    trainer = Trainer(model, dataset, device=torch.device("cuda"), batch_size=32, max_seq_len=256)
    try:
        trainer.train(num_epochs=100)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        trainer.save("./ckpt/model.pt")


if __name__ == "__main__":
    train()
