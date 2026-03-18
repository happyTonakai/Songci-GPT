import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


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
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_flash_attn: bool = False,
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

        if use_flash_attn:
            output = self._flash_attention_simulated(q, k, v, mask)
        else:
            output = self._scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)
        output = self.out_dropout(output)
        return output, present_kv

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch_size, num_head, seq_len, head_dim]
            k: [batch_size, num_head, seq_len, head_dim]
            v: [batch_size, num_head, seq_len, head_dim]
            mask: [batch_size, 1, 1, seq_len]

        Returns:
            attn_output: [batch_size, num_head, seq_len, head_dim]
        """
        # 计算注意力分数: Q @ K^T / sqrt(d)
        # attn_score: [batch_size, num_head, seq_len, seq_len]
        attn_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))

        # safe Softmax + Dropout, 为了防止 exp(x) 溢出，先减去最大值
        max_value = attn_score.max(dim=-1, keepdim=True).values
        attn_score = attn_score - max_value
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 注意力加权求和
        attn_output = attn_weights @ v
        return attn_output

    def _flash_attention_simulated(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, N, d = q.shape
        # 模拟 Flash Attention 的实现 https://www.bilibili.com/video/BV1UT421k7rA
        # Matrices QKV shape N*d, On-chip SRAM size M, typically M=64k~128k, 65536 or 131072
        # Set block sizes B_c = ceil(M/4d), B_r = min(ceil(M/4d), d)
        # Divide Q into T_r = ceil(N/B_r) blocks, K, V into T_c = ceil(N/B_c) blocks
        M = 65536
        B_c = math.ceil(M / (4 * d * 2))  # * 2 bytes per bf16
        B_r = min(math.ceil(M / (4 * d * 2)), d)
        # 1. KV在外循环，Q在内循环
        raise NotImplementedError


if __name__ == "__main__":
    # test standard self attention and flash attention
    mha = MultiHeadSelfAttention(512, 8)
    x = torch.randn(2, 1024, 512)
    output1, present_kv = mha(x, use_flash_attn=False)
    # output2, _ = mha(x, use_flash_attn=True)
    # print(torch.allclose(output1, output2, atol=1e-6))
