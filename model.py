import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from tokenizer import BPETokenizer


class PositionalEmbedding(nn.Module):
    """
    Learnable Positional Embedding

    支持 KV Cache 的位置编码：
    - 在推理时使用 KV Cache，我们只生成一个新的 token
    - 需要正确计算这个新 token 的位置索引（offset）
    """

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            offset: 位置偏移量，用于 KV Cache 场景
                   例如：之前已生成 10 个 token，新生成的 1 个 token 位置应该是 10-19
        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        # 生成位置索引：从 offset 开始，避免与历史位置重复
        positions = torch.arange(offset, offset + seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.embedding(positions.unsqueeze(0))
        return x + pos_emb


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention with KV Cache Support

    KV Cache 的核心思想：
    - 在自回归生成中，每个新 token 都需要与之前所有 token 计算注意力
    - 如果不使用 KV Cache，每次都需要重新计算所有历史 token 的 K 和 V
    - 使用 KV Cache 可以缓存历史 K 和 V，只需计算新 token 的 K 和 V，然后拼接

    为什么只缓存 K 和 V，不缓存 Q？
    - 对于自回归生成，每个位置只关心"之前"的 token
    - 新位置的 Q 只与当前位置有关，不需要缓存
    - 但 K 和 V 需要与之前的拼接，所以需要缓存
    """

    def __init__(self, embedding_dim: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.head_dim = embedding_dim // num_head
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.scale = self.head_dim**-0.5
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
               如果 past_kv 不为 None，则 seq_len 通常为 1（只输入最后一个 token）
            mask: 注意力掩码
            past_kv: 缓存的历史 K 和 V，格式为 (past_k, past_v)
                    - past_k: [batch_size, num_head, past_seq_len, head_dim]
                    - past_v: [batch_size, num_head, past_seq_len, head_dim]

        Returns:
            output: 注意力输出 [batch_size, seq_len, embedding_dim]
            present_kv: 当前的 K 和 V（含历史缓存），用于下一轮生成
        """
        # x: [batch_size, seq_len, embedding_dim]
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        # q, k, v: [batch_size, seq_len, embedding_dim]
        # -> [batch_size, num_head, seq_len, embedding_dim // num_head]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_head)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_head)

        # KV Cache: 将新计算的 K、V与历史缓存拼接
        # 这样注意力计算可以一次性处理所有历史 token
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # 在序列维度拼接
            v = torch.cat([past_v, v], dim=2)
        # 保存当前 K、V供下一轮使用
        present_kv = (k, v)

        # 计算注意力分数: Q @ K^T / sqrt(d)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn_score: [batch_size, num_head, seq_len, seq_len]
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))

        # Softmax + Dropout
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 注意力加权求和
        output = attn_weights @ v
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)
        output = self.out_dropout(output)
        return output, present_kv


class TransformerLayer(nn.Module):
    """
    Single Transformer Layer (Decoder Block)

    包含：
    1. Multi-head Self-attention（带 KV Cache）
    2. Feed-forward Network (FFN)
    3. Layer Normalization
    4. Skip connections
    """

    def __init__(self, embedding_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入张量
            mask: 注意力掩码
            past_kv: 来自上一轮的 KV 缓存

        Returns:
            x: 处理后的输出
            present_kv: 当前轮的 KV（含缓存），用于下一轮
        """
        # Self-attention + 残差连接
        x_norm = self.norm1(x)
        attn_out, present_kv = self.self_attn(x_norm, mask, past_kv)
        x = x + attn_out

        # FFN + 残差连接
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x, present_kv


class TransformerEncoder(nn.Module):
    """
    Multi-layer Transformer Encoder

    管理多个 TransformerLayer，并处理 KV Cache 在各层之间的传递
    """

    def __init__(self, layer: TransformerLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv_list: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: 输入张量
            mask: 注意力掩码
            past_kv_list: 各层的 KV 缓存列表
                         如果为 None，表示不使用缓存（首次前向传播）

        Returns:
            x: 处理后的输出
            present_kv_list: 各层当前轮的 KV 缓存，供下一轮使用
        """
        present_kv_list = []
        for i, layer in enumerate(self.layers):
            # 获取当前层的历史缓存（如果有）
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            # 前向传播，获取当前层的输出和新的 KV
            x, present_kv = layer(x, mask, past_kv)
            present_kv_list.append(present_kv)
        return x, present_kv_list


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
        # decoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embedding_dim,
        #     nhead=num_head,
        #     dim_feedforward=hidden_dim,
        #     dropout=dropout,
        #     batch_first=True,
        #     activation="gelu",
        # )
        # self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        decoder_layer = TransformerLayer(embedding_dim, num_head, hidden_dim, dropout)
        self.transformer = TransformerEncoder(decoder_layer, num_layers)

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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv_list: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            input_ids: token ID 序列 [batch_size, seq_len]
            attention_mask: padding 掩码 [batch_size, seq_len]
            past_kv_list: 历史 KV 缓存列表，用于 KV Cache 推理

        Returns:
            logits: 预测的 token 概率分布 [batch_size, seq_len, vocab_size]
            present_kv_list: 当前轮的 KV 缓存，供下一轮生成使用
        """
        batch_size, seq_len = input_ids.size()

        # 1. Token Embedding
        x = self.emb(input_ids)

        # 2. Positional Embedding with KV Cache support
        # 计算已缓存的序列长度，用于正确生成新 token 的位置编码
        past_length = 0
        if past_kv_list is not None:
            # past_kv_list[0][0] 是第 0 层缓存的 K，shape 为 [batch, num_head, past_seq_len, head_dim]
            past_length = past_kv_list[0][0].size(2)
        x = self.pos_emb(x, offset=past_length)

        # 3. 生成因果掩码（causal mask）
        # 确保每个位置只能看到之前的位置，实现自回归特性
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        if attention_mask is not None:
            # 合并 padding mask 和 causal mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            mask = attention_mask | causal_mask
        else:
            mask = causal_mask

        # 4. Transformer Encoder (带 KV Cache)
        x, present_kv_list = self.transformer(x, mask, past_kv_list)

        # 5. 输出层：映射到词表大小
        logits = self.ffn(x)
        return logits, present_kv_list

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
        """
        自回归生成文本（使用 KV Cache 优化）

        KV Cache 优化的工作流程：
        1. 首次前向：使用完整的 prompt，计算所有 KV，并缓存
        2. 后续前向：只输入最后一个 token，利用缓存的 KV 进行计算
        3. 每次前向后，更新缓存（拼接新的 K 和 V）

        这样可以大幅减少计算量（从 O(n²) 降到 O(n)），显著提升生成速度。

        Args:
            tokenizer: BPE 分词器
            prompt_text: 提示词（词牌名）
            temperature: 温度参数，控制随机性
            top_k: Top-k 采样
            top_p: Top-p (Nucleus) 采样
            max_len: 最大生成长度

        Returns:
            生成的宋词文本
        """
        self.eval()
        device = self.ffn.weight.device

        # 初始化 KV Cache
        past_kv_list = None

        # 1. 编码 prompt
        prompt_text = "<bos>" + prompt_text + "<sep>"
        input_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        # 2. 自回归生成
        for _ in range(self.max_seq_len - len(input_ids)):
            # KV Cache 优化：
            # - 首次：使用完整的 input_ids 进行前向
            # - 后续：只使用最后一个 token，大幅减少计算量
            if past_kv_list is None:
                x = input_ids
            else:
                x = input_ids[:, -1:]  # 只取最后一个 token

            logits, past_kv_list = self(x, past_kv_list=past_kv_list)
            logits = logits[:, -1, :]  # 只取最后一个位置的预测

            # 3. 温度调节
            logits = logits / temperature

            # 4. Top-k 采样
            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)

            # 5. Top-p 采样
            if top_p is not None:
                logits = self._top_p_logits(logits, top_p)

            # 6. 从概率分布中采样
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 7. 将采样的 token 追加到序列末尾
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 8. 检查是否达到终止条件
            if input_ids.size(1) >= max_len or next_token.item() == tokenizer.eos_id:
                break

        # 9. 解码生成文本
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
    # input_ids = torch.randint(0, 1000, (4, 256))
    # attention_mask = torch.ones(4, 256).bool()
    # attention_mask = None
    # print(model(input_ids, attention_mask).shape)

    # test generate
    model.load_state_dict(torch.load("ckpt/model.pt"))
    model.to("cuda")
    tokenizer = BPETokenizer()
    tokenizer.load("ckpt/songci_tokenizer.json")
    response = model.generate(tokenizer, "水调歌头", max_len=256, top_k=100, top_p=0.9)
    print(response)
    response = model.generate(tokenizer, "江城子", max_len=256, top_k=100, top_p=0.9)
    print(response)
