from glob import glob

import orjson
import torch
from torch.utils.data import Dataset

from tokenizer import BPETokenizer


class SongCiDataset(Dataset):
    def __init__(self, max_seq_len=256):

        files = glob("./dataset/宋词/*.json")
        raw_text = []
        for file in files:
            with open(file, "rb") as f:
                data = orjson.loads(f.read())
                raw_text += [
                    item["rhythmic"] + "<sep>" + "".join(item["paragraphs"]) for item in data
                ]
        self.tokenizer = BPETokenizer()
        self.tokenizer.load("./ckpt/songci_tokenizer.json")
        bos_id = [self.tokenizer.bos_id]
        eos_id = [self.tokenizer.eos_id]
        pad_id = [self.tokenizer.pad_id]

        data = []
        for text in raw_text:
            tokens = self.tokenizer.encode(text)
            if len(tokens) + 2 <= max_seq_len:
                tokens = bos_id + tokens + eos_id + pad_id * (max_seq_len - len(tokens) - 2)
                data.append(torch.Tensor(tokens).long())
        self.data = torch.stack(data)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        input_ids = self.data[index]
        attention_mask = input_ids == self.tokenizer.pad_id
        # Attention mask 用于计算 self attention score 的时候，让模型无法看到特定的 token
        # 比如因果 mask 和 padding mask
        # 而我们当前的任务类似于 SFT，让模型根据词牌名续写宋词，我们需要让词牌名本身不参与计算 loss
        # 但是模型应该能够看到词牌名的内容，因此不能使用 mask 来屏蔽词牌名
        # 这里我们返回 labels
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        # 从 <bos> 到 <sep> 之间的 token 都不需要参与计算，<eos> 之后的 token 也不需要参与计算
        sep_idx = (input_ids == self.tokenizer.sep_id).nonzero(as_tuple=True)[0]
        if len(sep_idx) > 0:
            sep_idx = sep_idx[0].item()
            labels[:sep_idx] = -100

        labels[labels == self.tokenizer.pad_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = SongCiDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(len(dataset))
    for i, batch_dict in enumerate(tqdm(dataloader)):
        # print(batch)
        input_ids, attention_mask, labels = (
            batch_dict["input_ids"],
            batch_dict["attention_mask"],
            batch_dict["labels"],
        )
        print(input_ids.shape, attention_mask.shape, labels.shape)
        if i == 10:
            break

    print(input_ids[0], labels[0], attention_mask[0])
    print(sum(input_ids[0] == dataset.tokenizer.pad_id))
    print(sum(labels[0] == -100))
