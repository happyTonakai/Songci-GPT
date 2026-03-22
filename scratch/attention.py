from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    # dim 是 head_dim，RoPE 是成对旋转的，所以是 dim // 2
    # 10000 ^ (-2 * i / dim)
    freqs = theta ** (-torch.arange(0, head_dim, 2) / head_dim)
    t = torch.arange(end)  # 位置索引 [0, 1, ..., end]
    freqs = torch.outer(t, freqs)  # 外积，得到 [end, dim // 2] 的矩阵
    # 变成复数形式 e^{it\theta}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # 形状: [max_seq_len, head_dim // 2]


def apply_rotary_embedding(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [batch, num_head, seq_len, head_dim]
    # freqs_cis: [seq_len, head_dim // 2]  (预计算好的复数旋转因子)
    # 核心操作为将x两两分组，并与旋转因子相乘

    # 1. 将 x 拆分为偶数和奇数部分
    x = rearrange(x, "b h s (d two) -> b h s d two", two=2)
    # 2. 转换为复数形式
    x = torch.view_as_complex(x.contiguous())
    # 3. 应用旋转
    freqs_cis = rearrange(freqs_cis, "s d -> 1 1 s d")
    x = x * freqs_cis
    # 4. 转换回实数形式
    x = torch.view_as_real(x)
    # 5. 拼接偶数和奇数部分
    x = rearrange(x, "b h s d two -> b h s (d two)")

    return x


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_dropout: nn.Dropout,
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
    attn_score = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        attn_score = attn_score.masked_fill(mask, float("-inf"))

    # safe Softmax + Dropout, 为了防止 exp(x) 溢出，先减去最大值，detach 防止梯度回传
    max_value = attn_score.max(dim=-1, keepdim=True).values.detach()
    attn_score = attn_score - max_value
    attn_weights = F.softmax(attn_score, dim=-1)
    attn_weights = attn_dropout(attn_weights)

    # 注意力加权求和
    attn_output = attn_weights @ v
    return attn_output


# def flash_attention_simulated(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     mask: torch.Tensor | None = None,
# ) -> torch.Tensor:
#     B, H, N, d = q.shape
#     # 模拟 Flash Attention 的实现 https://www.bilibili.com/video/BV1UT421k7rA
#     # Matrices QKV shape N*d, On-chip SRAM size M, typically M=64k~128k, 65536 or 131072
#     # Set block sizes B_c = ceil(M/4d), B_r = min(ceil(M/4d), d)
#     # Divide Q into T_r = ceil(N/B_r) blocks, K, V into T_c = ceil(N/B_c) blocks
#     M = 65536
#     B_c = math.ceil(M / (4 * d * 2))  # * 2 bytes per bf16
#     B_r = min(math.ceil(M / (4 * d * 2)), d)
#     # 1. KV在外循环，Q在内循环
#     raise NotImplementedError


class MultiHeadAttention(nn.Module):
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
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
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

        # Apply rotary embedding
        s = q.size(2)
        offset = past_kv[0].size(2) if past_kv is not None else 0
        current_freqs_cis = freqs_cis[offset : offset + s]
        q = apply_rotary_embedding(q, current_freqs_cis)
        k = apply_rotary_embedding(k, current_freqs_cis)

        # KV Cache: 将新计算的 K、V与历史缓存拼接
        # 这样注意力计算可以一次性处理所有历史 token
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # 在序列维度拼接
            v = torch.cat([past_v, v], dim=2)
        # 保存当前 K、V供下一轮使用
        present_kv = (k, v)

        output = scaled_dot_product_attention(
            q, k, v, self.scale, self.attn_dropout, mask
        )
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.out_proj(output)
        output = self.out_dropout(output)
        return output, present_kv


class MultiheadLatentAttention(nn.Module):
    """
    MLA 将注意力机制拆解为两个关键部分：压缩的 KV 和 解耦的 RoPE。
    RoPE（旋转位置编码）通常是乘在 Q 和 K 上的。但如果 K 被压缩了，旋转位置信息就没法直接做了。
    MLA 的解决方法是：把 Q 和 K 都拆成两部分。Content 部分：走上面的压缩路径，负责语义匹配。RoPE 部分：专门多出几个维度，只用来带上位置信息，不被压缩。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        latent_dim: int,
        rope_head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.head_dim = embedding_dim // num_head
        self.q_down_proj = nn.Linear(embedding_dim, latent_dim)
        self.q_content_proj = nn.Linear(latent_dim, embedding_dim)
        self.q_rope_proj = nn.Linear(latent_dim, rope_head_dim * num_head)

        self.kv_down_proj = nn.Linear(embedding_dim, latent_dim)
        self.k_up_proj = nn.Linear(latent_dim, embedding_dim)
        self.v_up_proj = nn.Linear(latent_dim, embedding_dim)
        # ! shared k rope, but q rope is different
        # ! k rope 的作用是提供"我是第几个位置"的信息。
        # ! 对于同一个 Token，无论哪个 Attention Head 来读取它，它的位置信息（第n个位置）是客观一致的。
        # ! 但是q rope 是主动发起询问的一方，不可以在所有头之间共享位置，会丧失多头注意力的多样性
        self.k_rope_proj = nn.Linear(latent_dim, rope_head_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        # ! scale 应该包括 content 的 head_dim 和 rope_head_dim
        self.scale = (self.head_dim + rope_head_dim) ** -0.5
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # 1. compress q and decouple content and position
        q_latent = self.q_down_proj(x)  # [B, S, d_c]
        q_content = self.q_content_proj(q_latent)  # [B, S, (h d_h)]
        q_rope = self.q_rope_proj(q_latent)  # [B, S, (h d_r)]

        # 2. compress kv
        kv_latent = self.kv_down_proj(x)  # [B, S, d_c]
        k_rope = self.k_rope_proj(kv_latent)  # [B, S, d_r]

        # 3. apply rope to q and k
        s = x.size(1)
        offset = past_kv[0].size(2) if past_kv is not None else 0
        current_freqs_cis = freqs_cis[offset : offset + s]
        q_rope = rearrange(q_rope, "b s (h d_r) -> b h s d_r", h=self.num_head)
        k_rope = rearrange(k_rope, "b s d_r -> b 1 s d_r")  # broadcast
        q_rope = apply_rotary_embedding(q_rope, current_freqs_cis)
        k_rope = apply_rotary_embedding(k_rope, current_freqs_cis)

        # 4. kv cache
        if past_kv is not None:
            past_kv_latent, past_k_rope = past_kv
            # kv_latent: [B, S, d_c], k_rope: [B, 1, S, d_r]
            kv_latent = torch.cat([past_kv_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)

        present_kv = (kv_latent, k_rope)

        # 5. uncompress kv
        k_content = self.k_up_proj(kv_latent)  # [B, S, (h d_h)]
        v_content = self.v_up_proj(kv_latent)  # [B, S, (h d_h)]

        # 6. split head
        q_content = rearrange(q_content, "b s (h d_h) -> b h s d_h", h=self.num_head)
        k_content = rearrange(k_content, "b s (h d_h) -> b h s d_h", h=self.num_head)
        v_content = rearrange(v_content, "b s (h d_h) -> b h s d_h", h=self.num_head)

        # 7. concat content and rope
        q = torch.cat([q_content, q_rope], dim=-1)
        # ! k is shared across head dim (dim 1), so we need to broadcast before concat
        k_rope = k_rope.expand(-1, self.num_head, -1, -1)
        k = torch.cat([k_content, k_rope], dim=-1)

        # 8. attention
        output = scaled_dot_product_attention(
            q, k, v_content, self.scale, self.attn_dropout, mask
        )

        output = rearrange(output, "b h s d_h -> b s (h d_h)")
        output = self.out_proj(output)
        output = self.out_dropout(output)

        return output, present_kv


if __name__ == "__main__":
    d_model = 512
    num_head = 8
    max_seq_len = 1024
    batch_size = 4
    x_prefill = torch.randn(batch_size, max_seq_len // 2, d_model)
    x_decode = torch.randn(batch_size, 1, d_model)

    freqs_cis = precompute_freqs_cis(d_model // num_head, max_seq_len)
    mha = MultiHeadAttention(d_model, num_head)
    output, present_kv = mha(x_prefill, freqs_cis)
    print(output.shape)
    output, present_kv = mha(x_decode, freqs_cis, past_kv=present_kv)
    print(output.shape)

    latent_dim = 64
    rope_head_dim = 16
    freqs_cis = precompute_freqs_cis(rope_head_dim, max_seq_len)
    mla = MultiheadLatentAttention(d_model, num_head, latent_dim, rope_head_dim)
    output, present_kv = mla(x_prefill, freqs_cis)
    print(output.shape)
    output, present_kv = mla(x_decode, freqs_cis, past_kv=present_kv)
    print(output.shape)
