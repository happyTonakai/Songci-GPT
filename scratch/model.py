from __future__ import annotations

import torch
import torch.nn.functional as F
from attention import MultiHeadAttention, MultiheadLatentAttention, precompute_freqs_cis
from config import ModelConfig
from einops import rearrange
from tokenizer import BPETokenizer
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.float()


class MoELayer(nn.Module):
    def __init__(
        self,
        n_experts: int,
        topk: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        n_shared_experts: int = 0,
    ):
        super().__init__()
        assert topk <= n_experts
        self.topk = topk
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts

        # 共享专家（always-on，不参与路由和负载均衡）
        # 每个 token 都经过共享专家，提取通用特征
        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(n_shared_experts)
            ]
        )

        # 路由专家（通过 Router 动态选择 top-k 个）
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(n_experts)
            ]
        )
        self.router = nn.Linear(embedding_dim, n_experts)

    def forward(
        self,
        x: Tensor,
        active_experts: list[int] | None = None,
        elastic_topk: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        # x shape: (batch, seq_len, d_model)
        B, T, D = x.shape
        x_flat = rearrange(x, "b t d -> (b t) d")

        # ── 共享专家: always-on，每个 token 都经过 ──
        shared_output = torch.zeros_like(x_flat)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(x_flat)

        # ── 路由专家: 通过 Router 动态选择 ──
        logits = self.router(x_flat)
        # Noisy Top-k Gating 在训练过程中强制添加噪声
        if self.training:
            noise = torch.randn_like(logits) * (1.0 / self.n_experts)  # 注入噪声
            logits = logits + noise

        # Elastic Width: 将非活跃路由专家的 logits 设为 -inf
        # 注意：共享专家不受弹性宽度影响，始终激活
        if active_experts is not None:
            mask_tensor = torch.full_like(logits, float("-inf"))
            for idx in active_experts:
                mask_tensor[:, idx] = 0.0
            logits = logits + mask_tensor

        # Elastic Sparsity: 使用缩减的 top-k
        effective_topk = elastic_topk if elastic_topk is not None else self.topk

        # 计算专家路由负载均衡loss（只针对路由专家）
        # 每个token选择每个专家的概率 (n_tokens, n_experts)
        probs = F.softmax(logits, dim=-1)
        P = probs.mean(dim=0)  # 每个专家平均被选择的概率 (n_experts, )
        # 负载均衡loss: 希望每个专家被均匀使用，最小化方差
        # soft constrain: minimize the L2 norm of the expert usage, 训练结果是主loss下降但是aux loss没有下降
        # load_balance_loss = (P * P).sum()
        # KL constrain: encourage the expert usage to be uniform
        # load_balance_loss = (P * (P.log() - torch.log(1 / self.n_experts))).sum()
        # 软约束都存在一个问题，就是：概率接近均匀 ≠ 选择接近均匀
        # 如果大部分token的专家选择都是 [0.26, 0.25, 0.25, 0.24]，哪怕概率非常接近均匀分布，但专家0依然被疯狂选中，造成collapse
        # 因此switch transformer中提出的双重约束，即同时约束概率和频次

        # 计算每个专家实际被选中为topk的频率
        topk_logits, topk_indices = torch.topk(logits, effective_topk, dim=-1)
        f = F.one_hot(topk_indices, num_classes=self.n_experts).float().mean(dim=0)
        # Elastic Width: 负载均衡只考虑活跃专家
        if active_experts is not None:
            active_mask = torch.zeros(self.n_experts, device=logits.device)
            for idx in active_experts:
                active_mask[idx] = 1.0
            P = P * active_mask
            f = f * active_mask
            # 对活跃专家的概率重新归一化，确保负载均衡信号有效
            P_sum = P.sum()
            if P_sum > 0:
                P = P / P_sum
        # 使用固定的 self.n_experts 作为系数，保持正则化强度一致
        # 不随活跃专家数量变化，避免弹性宽度时梯度信号减弱
        load_balance_loss = self.n_experts * (f * P).sum()

        # 进入专家路由
        weights = F.softmax(topk_logits, dim=-1)
        # 在很多主流实现（比如 Switch Transformer）中，人们倾向于对所有专家的 logits 先做 Softmax，
        # 然后再取 Top-K。如果先取 Top-K 再 Softmax，会放大这 $K$ 个专家之间的差距

        # weighted sum of routed experts outputs
        routed_output = torch.zeros_like(x_flat, device=x_flat.device)

        # 当前的实现是遍历所有专家，不涉及到单个专家的容量问题
        # 若修改为并行处理，需要引入单个专家的capacity，防止单个专家爆炸
        # 添加负载均衡loss能够缓解此问题，但是无法根治
        for i, expert in enumerate(self.experts):
            # 遍历专家，找到哪些token用到了这个专家，每行代表每一个token top1/top2是否选中专家i
            mask = topk_indices == i  # (batch * seq_len, topk)
            if not mask.any():
                continue
            # 有哪些token选中了这个专家，以及选中的顺位如何
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            # 专家i的输出
            out = expert(x_flat[token_idx])
            # 该token对于这个专家的权重如何
            w = weights[token_idx, k_idx].unsqueeze(-1)
            # 专家i的加权输出
            routed_output[token_idx] += w * out

        # 最终输出 = 共享专家 + 路由专家
        output = shared_output + routed_output
        output = rearrange(output, "(b t) d -> b t d", b=B, t=T)
        return output, load_balance_loss


class TransformerLayer(nn.Module):
    """
    Single Transformer Layer (Decoder Block)

    包含：
    1. Multi-head Self-attention（带 KV Cache）
    2. Feed-forward Network (FFN)
    3. Layer Normalization
    4. Skip connections
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        n_experts: int = 8,
        topk: int = 2,
        use_mla: bool = False,
        latent_dim: int | None = None,
        rope_head_dim: int | None = None,
        use_attn_res: bool | str = False,
        n_shared_experts: int = 0,
    ):
        super().__init__()
        self.use_mla = use_mla
        self.use_attn_res = use_attn_res
        if use_mla:
            assert latent_dim is not None and rope_head_dim is not None
            self.self_attn = MultiheadLatentAttention(
                embedding_dim, num_heads, latent_dim, rope_head_dim, dropout
            )
        else:
            self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        if use_attn_res == "block":
            # [batch, seq_len, d_model]
            self.query_attn = nn.Parameter(torch.zeros(embedding_dim))
            self.query_ffn = nn.Parameter(torch.zeros(embedding_dim))
        elif use_attn_res == "full":
            self.query = nn.Parameter(torch.zeros(embedding_dim))

        self.norm_attn = nn.LayerNorm(embedding_dim)
        self.n_experts = n_experts
        if n_experts > 1:
            self.moe = MoELayer(
                n_experts, topk, hidden_dim, embedding_dim, dropout,
                n_shared_experts=n_shared_experts,
            )
        else:
            self.moe = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
                nn.Dropout(dropout),
            )
        self.norm_ffn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor | None = None,
        past_kv: tuple[Tensor, Tensor] | None = None,
        prev_layer_outputs: list[Tensor] | None = None,
        active_experts: list[int] | None = None,
        elastic_topk: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """
        Args:
            x: 输入张量
            freqs_cis: 预计算好的复数旋转因子
            mask: 注意力掩码
            past_kv: 来自上一轮的 KV 缓存
            active_experts: Elastic Width — 活跃专家索引列表 (None=全部)
            elastic_topk: Elastic Sparsity — 缩减后的 top-k (None=默认)

        Returns:
            x: 处理后的输出
            present_kv: 当前轮的 KV（含缓存），用于下一轮
            aux_loss: MoE 负载均衡 loss（如果不是 MoE 则为 0）
        """
        # Self-attention + 残差连接
        x_norm = self.norm_attn(x)
        attn_out, present_kv = self.self_attn(x_norm, freqs_cis, mask, past_kv)
        x = x + attn_out

        # Pre-Norm + FFN/MoE + 残差连接
        x_norm = self.norm_ffn(x)
        if self.n_experts > 1:
            moe_out, aux_loss = self.moe(x_norm, active_experts, elastic_topk)
            x = x + moe_out
        else:
            moe_out = self.moe(x_norm)
            x = x + moe_out
            # 这里写1是为了和上面MoE的loss对齐，=1的时候代表各专家已经负载均衡
            aux_loss = torch.tensor(1.0, device=x.device)
        return x, present_kv, aux_loss


class TransformerEncoder(nn.Module):
    """
    Multi-layer Transformer Encoder

    管理多个 TransformerLayer，并处理 KV Cache 在各层之间的传递
    """

    def __init__(
        self, layers: list[TransformerLayer], use_attn_res: bool | str = False
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.use_attn_res = use_attn_res
        if use_attn_res:
            assert use_attn_res in ["full", "block"]

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor | None = None,
        past_kv_list: list[tuple[Tensor, Tensor]] | None = None,
        active_layer_indices: list[int] | None = None,
        active_experts: list[int] | None = None,
        elastic_topk: int | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]], Tensor]:
        """
        Args:
            x: 输入张量
            freqs_cis: 预计算好的复数旋转因子
            mask: 注意力掩码
            past_kv_list: 各层的 KV 缓存列表
                         如果为 None，表示不使用缓存（首次前向传播）
            active_layer_indices: Elastic Depth — 活跃层索引 (None=全部)
            active_experts: Elastic Width — 活跃专家索引 (None=全部)
            elastic_topk: Elastic Sparsity — 缩减后的 top-k (None=默认)

        Returns:
            x: 处理后的输出
            present_kv_list: 各层当前轮的 KV 缓存，供下一轮使用
            total_aux_loss: 所有层的 MoE 负载均衡 loss 总和
        """
        present_kv_list = []
        total_aux_loss = torch.tensor(0.0, device=x.device)
        prev_layer_outputs = [x]  # used for attention residual

        num_active = 0
        for i, layer in enumerate(self.layers):
            # Elastic Depth: Bypassing 跳层（Stochastic Depth）
            # 被跳过的层执行恒等映射 X_{l+1} = X_l，即残差连接的本意。
            # 这比截断前缀更合理：梯度直接穿过被跳过的层（不更新其参数），
            # 且深层特征仍可流向输出层，不会强迫浅层去学深层特征。
            if active_layer_indices is not None and i not in active_layer_indices:
                past_kv = past_kv_list[i] if past_kv_list is not None else None
                present_kv_list.append(past_kv)
                if self.use_attn_res:
                    prev_layer_outputs.append(prev_layer_outputs[-1])
                continue

            # 获取当前层的历史缓存（如果有）
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            # attention residual
            if self.use_attn_res == "full":
                x = compute_full_attn_res(layer.query, prev_layer_outputs)
            # 前向传播，获取当前层的输出、新的 KV 和 aux_loss
            x, present_kv, aux_loss = layer(
                x, freqs_cis, mask, past_kv,
                active_experts=active_experts,
                elastic_topk=elastic_topk,
            )
            present_kv_list.append(present_kv)
            if self.use_attn_res == "full":
                prev_layer_outputs.append(x)
            total_aux_loss += aux_loss
            num_active += 1

        return x, present_kv_list, total_aux_loss / max(num_active, 1)


def compute_full_attn_res(query: Tensor, prev_layer_outputs: list[Tensor]) -> Tensor:
    # query: [embedding_dim]
    # prev_layer_outputs: [batch_size, seq_len, embedding_dim]
    # return: [batch_size, seq_len, embedding_dim]

    if len(prev_layer_outputs) == 1:
        return prev_layer_outputs[0]  # weights = 1

    # [batch_size, seq_len, num_prev_layers, embedding_dim]
    stacked = torch.stack(prev_layer_outputs, dim=2)
    # rms norm
    key = stacked * torch.rsqrt(torch.mean(stacked**2, dim=-1, keepdim=True) + 1e-8)

    # logits = query @ key.transpose(-1, -2)  # [batch_size, seq_len, num_prev_layers]
    logits = torch.einsum("d,bsnd->bsn", query, key)
    weights = F.softmax(logits, dim=-1)  # [batch_size, seq_len, num_prev_layers]

    # weights = weights.unsqueeze(-1)  # [batch_size, seq_len, num_prev_layers, 1]
    # return torch.sum(weights * stacked, dim=2)  # [batch_size, seq_len, embedding_dim]
    return torch.einsum("bsn,bsnd->bsd", weights, stacked)


class SongCiGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        # self.pos_emb = PositionalEmbedding(max_seq_len, embedding_dim)

        # MLA 使用 rope_head_dim，普通 Attention 使用 head_dim
        if getattr(config, "use_mla", False):
            head_dim = config.rope_head_dim
        else:
            head_dim = config.embedding_dim // config.num_heads
        freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.use_attn_res = getattr(config, "use_attn_res", False)

        decoder_layers = []
        n_shared = getattr(config, "n_shared_experts", 0)

        for i in range(config.num_layers):
            # 全 MoE 模式：所有层都是 MoE（当 n_shared_experts > 0 时）
            # 交替模式：奇数层 MoE，偶数层 Dense（向后兼容）
            if n_shared > 0:
                # 全 MoE：所有层使用相同的专家配置
                num_experts = config.n_experts
            elif i % 2 == 1 and i != config.num_layers - 1:
                num_experts = config.n_experts
            else:
                num_experts = 1  # Dense 层
            layer = TransformerLayer(
                config.embedding_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout,
                num_experts,
                config.topk,
                use_mla=getattr(config, "use_mla", False),
                latent_dim=getattr(config, "latent_dim", None),
                rope_head_dim=getattr(config, "rope_head_dim", None),
                use_attn_res=self.use_attn_res,
                n_shared_experts=n_shared,
            )
            decoder_layers.append(layer)

        self.transformer = TransformerEncoder(
            decoder_layers, use_attn_res=self.use_attn_res
        )

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

        self.ffn = nn.Linear(config.embedding_dim, config.vocab_size)
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
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_kv_list: list[tuple[Tensor, Tensor]] | None = None,
        active_layer_indices: list[int] | None = None,
        active_experts: list[int] | None = None,
        elastic_topk: int | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]], Tensor]:
        """
        前向传播

        Args:
            input_ids: token ID 序列 [batch_size, seq_len]
            attention_mask: padding 掩码 [batch_size, seq_len]
            past_kv_list: 历史 KV 缓存列表，用于 KV Cache 推理
            active_layer_indices: Elastic Depth — 活跃层索引 (None=全部)
            active_experts: Elastic Width — 活跃专家索引 (None=全部)
            elastic_topk: Elastic Sparsity — 缩减后的 top-k (None=默认)

        Returns:
            logits: 预测的 token 概率分布 [batch_size, seq_len, vocab_size]
            present_kv_list: 当前轮的 KV 缓存，供下一轮生成使用
            aux_loss: MoE 负载均衡 loss（如果不是 MoE 则为 0）
        """
        batch_size, seq_len = input_ids.size()

        # 1. Token Embedding
        x = self.emb(input_ids)

        # 2. Positional Embedding with KV Cache support
        # 计算已缓存的序列长度，用于正确生成新 token 的位置编码
        # past_length = 0
        # if past_kv_list is not None:
        #     # past_kv_list[0][0] 是第 0 层缓存的 K，shape 为 [batch, num_head, past_seq_len, head_dim]
        #     past_length = past_kv_list[0][0].size(2)
        # x = self.pos_emb(x, offset=past_length)

        # 3. 生成因果掩码（causal mask）
        # 确保每个位置只能看到之前的位置，实现自回归特性
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        if attention_mask is not None:
            # 合并 padding mask 和 causal mask
            # [batch, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # [1, 1, seq_len, seq_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            mask = attention_mask | causal_mask
        else:
            mask = causal_mask

        # 4. Transformer Encoder (带 KV Cache)
        x, present_kv_list, aux_loss = self.transformer(
            x, self.freqs_cis, mask, past_kv_list,
            active_layer_indices=active_layer_indices,
            active_experts=active_experts,
            elastic_topk=elastic_topk,
        )

        # 5. 输出层：映射到词表大小
        logits = self.ffn(x)
        return logits, present_kv_list, aux_loss

    @torch.no_grad()
    def generate(
        self,
        tokenizer: BPETokenizer,
        prompt_text: str,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        max_len: int = 256,
        active_layer_indices: list[int] | None = None,
        active_experts: list[int] | None = None,
        elastic_topk: int | None = None,
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
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        input_ids = input_ids.unsqueeze(0)

        # 2. 自回归生成
        for _ in range(self.max_seq_len - len(input_ids)):
            # KV Cache 优化：
            # - 首次：使用完整的 input_ids 进行前向
            # - 后续：只使用最后一个 token，大幅减少计算量
            if past_kv_list is None:
                x = input_ids
            else:
                x = input_ids[:, -1:]  # 只取最后一个 token

            logits, past_kv_list, _ = self(
                x, past_kv_list=past_kv_list,
                active_layer_indices=active_layer_indices,
                active_experts=active_experts,
                elastic_topk=elastic_topk,
            )
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

    def _top_k_logits(self, logits: Tensor, k: int) -> Tensor:
        if k == 0 or k > logits.size(-1):
            return logits
        # 将不要的 logits 设置为 float('-inf')
        value, idx = torch.topk(logits, k, dim=-1)
        probs = torch.full_like(logits, float("-inf"))
        probs.scatter_(-1, idx, value)
        return probs

    def _top_p_logits(self, logits: Tensor, p: float) -> Tensor:
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
    from config import ModelConfig

    # test positional embedding
    # emb = PositionalEmbedding(256, 512)

    # x = torch.rand(32, 256, 512)
    # print(emb(x).shape)

    # test SongCiGPT
    config = ModelConfig(
        vocab_size=10000,
        max_seq_len=256,
        embedding_dim=512,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6,
        n_experts=4,
        topk=1,
        dropout=0.1,
        use_attn_res="full",
    )
    model = SongCiGPT(config)
    input_ids = torch.randint(0, 1000, (4, 256))
    attention_mask = torch.ones(4, 256).bool()
    attention_mask = None
    logits, _, aux_loss = model(input_ids, attention_mask)
    print(f"logits shape: {logits.shape}, aux_loss: {aux_loss.item():.6f}")

    # test generate
    # model.load_state_dict(torch.load("scratch/ckpt/model.pt"))
    # model.to("cuda")
    # tokenizer = BPETokenizer()
    # tokenizer.load("scratch/ckpt/songci_tokenizer.json")
    # response = model.generate(tokenizer, "水调歌头", max_len=256, top_k=100, top_p=0.9)
    # print(response)
    # response = model.generate(tokenizer, "江城子", max_len=256, top_k=100, top_p=0.9)
    # print(response)
