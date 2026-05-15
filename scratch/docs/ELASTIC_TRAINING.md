# 弹性训练 (Elastic Training) 技术文档

## 1. 背景与动机

### 1.1 来源：ERNIE 5.0

弹性训练来自百度 ERNIE 5.0 技术报告（arxiv 2602.04705），核心思想：

> "Rather than compressing a pre-trained model post hoc, elastic training simultaneously optimizes a family of sub-networks during pre-training."

**解决的问题**：传统做法是先训练一个大模型，再通过剪枝/蒸馏压缩成小模型。弹性训练在预训练阶段就同时优化整个模型家族，一次训练产出多个可部署的子网络。

### 1.2 论文的关键设计

三个**正交维度**，每次训练迭代独立采样：

| 维度 | 全配置概率 | 子网络概率 | 论文原文 |
|------|-----------|-----------|---------|
| 深度 | 75% 全深度 | 25% 更浅 | "a reduced-depth sub-network is activated with a probability of 25%" |
| 宽度 | 80% 全专家 | 20% 子集 | "routing is restricted to a randomly sampled subset of experts" |
| 稀疏度 | 80% 默认 top-k | 20% 更小 k | "the routing top-k is randomly sampled from a predefined range" |

论文的关键短语是 **"bypassed"**（被绕过），不是 "truncated"（截断）或 "early exit"。

---

## 2. 弹性深度：Bypassing 跳层

### 2.1 为什么是 Bypassing 而不是截断

Transformer 的残差结构：$X_{l+1} = X_l + \text{Layer}_l(X_l)$

**Bypassing（跳层）**：跳过第 $l$ 层时，公式变为 $X_{l+1} = X_l$（恒等映射）。
- 这正是残差连接的本意——梯度可以直接无损穿过被跳过的层
- 被跳过的层不参与前向计算，也不接收梯度
- 深层特征仍可流向输出层
- 学术上叫 **Stochastic Depth**，是非常成熟的正则化技术

**截断前缀（错误做法）**：只取前 $N$ 层，第 $N$ 层直接接输出头。
- 强迫浅层去学深层特征，破坏特征空间
- 深层参数永远无梯度，无法训练
- 与论文的 "bypassed" 描述矛盾

### 2.2 实现方式

预定义深度配置库（`depth_levels`），每次从库中随机选一个深度，然后从全部层中均匀随机选取对应数量的层：

```
全部层: [0, 1, 2, 3, 4, 5, ..., 23]  (24层)
深度配置库: [24, 18, 12, 6]
选中 12 层 → 随机选: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
被跳过的层: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] → 执行恒等映射
```

**不是连续前缀**，是随机均匀选取，确保深层和浅层都有机会被激活。

### 2.3 代码路径

```
train_sft.py: _sample_elastic_config()
  → active_layer_indices = sorted(random.sample(range(24), 12))
  → model(input_ids, active_layer_indices=[0,2,4,...])

model.py: TransformerEncoder.forward()
  → for i, layer in enumerate(self.layers):
      if i not in active_layer_indices:
          continue  # 恒等映射，x 直接传递
```

被跳过的层：
- `x` 不变（恒等映射）
- `present_kv_list` 占位（保持索引对齐）
- `prev_layer_outputs` 复制上一层输出（用于 attention residual）
- 不参与 `total_aux_loss` 计算
- `num_active` 只计活跃层，aux_loss 除以活跃层数

---

## 3. 弹性宽度：Masking 路由专家

### 3.1 共享专家 vs 路由专家

参考 DeepSeek-V2 的设计，每层 MoE 包含两种专家：

| 类型 | 数量 | 激活方式 | 参与负载均衡 | 受弹性宽度影响 |
|------|------|---------|-------------|--------------|
| 共享专家 (Shared) | 1 | always-on，每个 token 必定经过 | 否 | 否 |
| 路由专家 (Routed) | 8 | Router 动态选择 top-k 个 | 是 | 是 |

**每个 token 的专家激活**：1 个共享专家 + 2 个路由专家（top-2）= 3 个专家

**输出计算**：
```
shared_output = sum(shared_expert(x) for each shared expert)
routed_output = sum(weight_i * expert_i(x) for i in top-k routed experts)
final_output = shared_output + routed_output
```

### 3.2 弹性宽度的实现

弹性宽度**只影响路由专家**，共享专家始终激活。

实现方式：将非活跃路由专家的 router logits 设为 $-\infty$，softmax 后概率为 0：

```python
# MoELayer.forward()
if active_experts is not None:
    mask_tensor = torch.full_like(logits, float("-inf"))
    for idx in active_experts:
        mask_tensor[:, idx] = 0.0
    logits = logits + mask_tensor
```

**核心专家（core_experts）**：在 `active_experts` 中指定的路由专家永不被屏蔽。这是弹性宽度的"安全网"，确保关键知识不会因随机屏蔽而丢失。

### 3.3 负载均衡 loss 的处理

当只有部分路由专家活跃时，负载均衡 loss 只考虑活跃专家：

```python
if active_experts is not None:
    active_mask = torch.zeros(self.n_experts, device=logits.device)
    for idx in active_experts:
        active_mask[idx] = 1.0
    P = P * active_mask      # 只看活跃专家的概率
    f = f * active_mask       # 只看活跃专家的频率
    P = P / P.sum()           # 重新归一化到 sum=1
# 使用固定的 self.n_experts 作为系数，保持正则化强度一致
# 不随活跃专家数量变化，避免弹性宽度时梯度信号减弱
load_balance_loss = self.n_experts * (f * P).sum()
```

### 3.4 代码路径

```
train_sft.py: _sample_elastic_config()
  → core = {0}  # 核心专家
  → remaining = [1,2,3,4,5,6,7]
  → random.sample(remaining, 1) → [3]
  → active_experts = [0, 3]  # 核心专家 + 随机选 1 个
  → model(input_ids, active_experts=[0, 3])

model.py: MoELayer.forward()
  → shared_experts: 始终执行，不受 active_experts 影响
  → router logits: 非活跃专家设为 -inf
  → topk: 只从活跃专家中选
  → load_balance_loss: 只考虑活跃专家
```

---

## 4. 弹性稀疏度：缩减 top-k

最简单的维度。将路由 top-k 从默认值缩减为更小的值：

```
默认 top-k = 2: 每个 token 激活 2 个路由专家
弹性 top-k = 1: 每个 token 只激活 1 个路由专家
```

共享专家不受影响，始终激活。所以 top-1 时每个 token 仍然激活 2 个专家（1 shared + 1 routed）。

### 代码路径

```
train_sft.py: _sample_elastic_config()
  → elastic_topk = random.choice([2, 1])
  → model(input_ids, elastic_topk=1)

model.py: MoELayer.forward()
  → effective_topk = elastic_topk if elastic_topk is not None else self.topk
  → topk_logits, topk_indices = torch.topk(logits, effective_topk, dim=-1)
```

---

## 5. 课程学习 (Curriculum Learning)

**问题**：模型初始化时权重是随机的，如果一开始就开启弹性训练（跳层、缺专家），模型很难收敛。

**解决**：`warmup_steps` 之前强制使用全量网络配置，等 Loss 稳定后再开启弹性。

```python
# train_sft.py: _sample_elastic_config()
if self.global_step < elastic_cfg.warmup_steps:
    return {}  # 空 dict = 不传弹性参数 = 全量网络
```

**global_step 递增位置**：在 `optimizer.step()` 之后，确保计数准确。

---

## 6. 训练循环完整流程

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 采样弹性配置
        elastic_params = self._sample_elastic_config()
        # warmup 阶段: {} (全量网络)
        # 弹性阶段: {"active_layer_indices": [...], "active_experts": [...], "elastic_topk": int}

        # 2. 前向传播（只走活跃路径）
        logits, _, aux_loss = model(input_ids, attention_mask, **elastic_params)

        # 3. 计算损失（标准交叉熵，不需要蒸馏 loss）
        loss = cross_entropy(logits, labels)
        total_loss = loss + dynamic_lambda * aux_loss

        # 4. 反向传播（梯度只更新活跃参数）
        total_loss.backward()

        # 5. 更新参数
        optimizer.step()
        optimizer.zero_grad()
        self.global_step += 1
```

**关键点**：
- 损失函数不变：无论用哪种子网络配置，都是标准的自回归交叉熵损失
- 只有一次前向-反向：子网络和全网络共享参数，不需要额外的蒸馏 loss
- 三个维度独立采样：一次迭代中可以同时缩减深度、宽度和稀疏度

---

## 7. 配置参考

### 24层全 MoE 模型 (mha_24l.yaml)

```yaml
model:
  vocab_size: 10000
  max_seq_len: 256
  embedding_dim: 512
  hidden_dim: 2048
  num_heads: 8
  num_layers: 24
  n_experts: 8           # 路由专家数
  topk: 2                # 每 token 激活 2 个路由专家
  n_shared_experts: 1    # 共享专家数 (always-on)
  dropout: 0.1

elastic:
  depth_prob: 0.25       # 25% 概率 Bypassing 跳层
  width_prob: 0.20       # 20% 概率缩减路由专家
  sparsity_prob: 0.20    # 20% 概率缩减路由 top-k
  depth_levels: [18, 12, 6]          # 预定义深度配置库（全量24层由 depth_prob 控制）
  width_levels: [4, 2]               # 预定义路由专家数配置库（全量8个由 width_prob 控制）
  sparsity_levels: [1]               # 预定义 top-k 配置库（全量 top-2 由 sparsity_prob 控制）
  warmup_steps: 1000     # 前 1000 步用全量网络
  core_experts: [0]      # 路由专家 0 永不被屏蔽
```

### 模型参数量

| 配置 | 总参数 | 激活参数 | 激活率 |
|------|--------|---------|--------|
| 6层 mha.yaml | 41.8M | 29.1M | 69.8% |
| 24层 mha_24l.yaml | 489.2M | 186.5M | 38.1% |

---

## 8. 弹性训练的效果（来自 ERNIE 5.0 论文）

- 将推理时 top-k 缩减到 25%，解码速度提升 **15%+**，精度损失很小
- 三维联合弹性：仅使用 **53.7% 激活参数**和 **35.8% 总参数**，平均分从 75.55 降到 75.17
- 弹性训练不仅是压缩技术，更是一种**原则性的训练范式**——模型学会在各种配置下重新分配表征能力
