import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import BPETokenizer


class PositionalEmbedding(nn.Module):
    """
    Learnable Positional Embedding
    """

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, embedding_dim]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.embedding(positions.unsqueeze(0))
        return x + pos_emb


class SongCiGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_len: int = 256,
        embedding_dim: int = 512,
        hidden_dim: int = 2048,  # usually 4 times of embedding_dim
        num_head: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = PositionalEmbedding(max_seq_len, embedding_dim)
        # 使用 nn.TransformerEncoderLayer 更符合 Decoder-only 结构
        # 它只包含自注意力和前馈网络，没有交叉注意力
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_head,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ffn = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]
        batch_size, seq_len = input_ids.size()
        # 1. Get token embeddings
        x = self.emb(input_ids)
        # 2. Add positional embeddings
        x = self.pos_emb(x)
        # 3. Generate causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == True, float("-inf"))
        # 4. Apply transformer
        x = self.transformer(src=x, mask=causal_mask, src_key_padding_mask=attention_mask)
        # 5. Apply linear layer
        logits = self.ffn(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        tokenizer: BPETokenizer,
        prompt_text: str,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        max_len: int = 256,
    ) -> str:
        self.eval()
        device = self.ffn.weight.device
        # 1. Encode prompt text
        prompt_text = "<bos>" + prompt_text + "<sep>"
        input_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        # 2. Generate tokens
        for _ in range(self.max_seq_len - len(input_ids)):
            logits = self(input_ids)  # [batch_size, seq_len, vocab_size]
            logits = logits[:, -1, :]  # select the last generated logits
            # 3. Apply temperature
            logits = logits / temperature
            # 4. Apply top-k/top-p sampling
            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)
            if top_p is not None:
                logits = self._top_p_logits(logits, top_p)
            # 5. Sample a token
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # 6. Append the sampled token to the input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # 7. Check if the generated text is too long or finished
            if input_ids.size(1) >= max_len or next_token.item() == tokenizer.eos_id:
                break
        # 8. Decode the generated tokens
        generated_text = tokenizer.decode(input_ids[0].tolist())
        return generated_text.replace("</w>", "")

    def _top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if k == 0 or k > logits.size(-1):
            return logits
        # 将不要的 logits 设置为 float('-inf')
        value, idx = torch.topk(logits, k, dim=-1)
        probs = torch.full_like(logits, float("-inf"))
        probs.scatter_(-1, idx, value)
        return probs

    def _top_p_logits(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        if p == 0.0 or p == 1.0:
            return logits

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # 排除累积和超过 top_p 的所有 token，但要保留第一个超过 top_p 的token (如果它本身使得 cumulative_probs 超过 top_p)
        # 也就是把第一个 `True` 后面的都设为 `True`
        # shift the indices to the right to keep the first token above top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits


if __name__ == "__main__":
    # test positional embedding
    emb = PositionalEmbedding(256, 512)

    x = torch.rand(32, 256, 512)
    print(emb(x).shape)

    # test SongCiGPT
    model = SongCiGPT()
    input_ids = torch.randint(0, 1000, (4, 256))
    # attention_mask = torch.ones(4, 256)
    attention_mask = None
    print(model(input_ids, attention_mask).shape)

    # test generate
    model.load_state_dict(torch.load("ckpt/model.pt"))
    model.to("cuda")
    tokenizer = BPETokenizer()
    tokenizer.load("ckpt/songci_tokenizer.json")
    response = model.generate(tokenizer, "水调歌头", max_len=256, top_k=100, top_p=0.9)
    print(response)
    response = model.generate(tokenizer, "江城子", max_len=256, top_k=100, top_p=0.9)
    print(response)
