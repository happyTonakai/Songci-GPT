import torch
from fire import Fire

from model import SongCiGPT
from tokenizer import BPETokenizer


def inference(ckpt: str = "ckpt/model.pt"):
    tokenizer = BPETokenizer()
    tokenizer.load("ckpt/songci_tokenizer.json")
    model = SongCiGPT()
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.to("cuda")
    model.eval()
    while True:
        prompt = input("请输入(输入q退出)：")
        if prompt == "q":
            break
        with torch.no_grad():
            output = model.generate(tokenizer, prompt, top_k=100, top_p=0.9)
        print(output)


if __name__ == "__main__":
    Fire(inference)
