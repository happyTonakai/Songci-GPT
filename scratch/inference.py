from __future__ import annotations

import torch
from config import load_config
from fire import Fire
from model import SongCiGPT
from tokenizer import BPETokenizer


def inference(config_path: str = "./scratch/configs/mha.yaml"):
    config = load_config(config_path)

    tokenizer = BPETokenizer()
    tokenizer.load(config.data.tokenizer_path)

    model = SongCiGPT(config.model)
    model.load_state_dict(torch.load(config.train.ckpt_path, weights_only=True))
    model.to(config.train.device)
    model.eval()

    print(f"模型加载完成，从 {config.train.ckpt_path}")
    print(
        f"生成参数: temperature={config.inference.temperature}, top_k={config.inference.top_k}, top_p={config.inference.top_p}"
    )
    print("请输入词牌名(输入q退出)：")

    while True:
        prompt = input("> ")
        if prompt == "q":
            break
        with torch.no_grad():
            output = model.generate(
                tokenizer,
                prompt,
                temperature=config.inference.temperature,
                top_k=config.inference.top_k,
                top_p=config.inference.top_p,
                max_len=config.inference.max_len,
            )
        print(output)
        print()


if __name__ == "__main__":
    Fire(inference)
