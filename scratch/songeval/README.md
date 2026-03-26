# SongEval - 宋词格律评估系统

基于逆向工程从训练语料中提取格律标准，自动评估生成宋词的格律符合度。

## 评估指标

### 1. 结构匹配 (Structure Match)
- 检查生成宋词的句数和每句字数是否符合词牌标准
- 支持智能拆分：自动处理数据源中合并的段落（如浣溪沙的14字段落）

### 2. 平仄准确度 (Tonal Accuracy)
- 使用 `pypinyin` 提取每个字的声调
- 1,2声为平(P)，3,4声为仄(Z)
- 对比生成文本与标准模板的平仄符合度
- 只计算固定位置（`*`表示平仄不拘）

### 3. 押韵一致性 (Rhyme Consistency) ⭐ 已更新

**核心思想**：宋词可以一韵到底，也可以换韵。系统通过**韵脚组**来表示押韵规则。

**韵脚组格式**：`[[0,1,2,4,5]]` 或 `[[0,1], [2,3], [4,5]]`
- 单个韵脚组：表示这些句子押同一个韵（一韵到底）
- 多个韵脚组：表示换韵，每组使用不同韵部

**词牌示例**：
| 词牌 | 韵脚组 | 说明 |
|------|--------|------|
| 浣溪沙 | `[[0,1,2,4,5]]` | 句0,1,2,4,5押同韵，句3不押韵 |
| 菩萨蛮 | `[[0,1], [2,3], [6,7]]` | 换韵3次 |
| 虞美人 | `[[0,1], [2,3], [4,5], [6,7]]` | 换韵4次 |

**评估方法**：
1. 对每个韵脚组，检查组内所有句子的韵部是否一致
2. 计算各组的一致性，取平均值
3. 韵部识别使用 `pypinyin` 的 `Style.FINALS` + 韵部映射表

### 4. 综合格律分 (Form Score)
```
form_score = 0.4 * structure_match + 0.4 * tonal_accuracy + 0.2 * rhyme_consistency
```
若结构不匹配，分数大幅降低。

## 格律库 (standard.json)

包含 75 个常用词牌的格律标准：

```json
{
  "浣溪沙": {
    "structure": [7, 7, 7, 7, 7, 7],
    "tonal_template": ["*ZPP*ZP", "*P*ZZPP", "*P*ZZPP", "*Z*PPZZ", "*P*ZZPP", "*P*ZZPP"],
    "rhyme_groups": [[0, 1, 2, 4, 5]],
    "sample_size": 760
  },
  "虞美人": {
    "structure": [7, 5, 7, 9, 7, 5, 7, 9],
    "tonal_template": ["*P*ZPPZ", "*ZPPZ", "*P*ZZPP", "*Z*PPZZPP", "*P*ZPPZ", "*ZPPZ", "*P*ZZPP", "*Z*PPZZPP"],
    "rhyme_groups": [[0, 1], [2, 3], [4, 5], [6, 7]],
    "sample_size": 287
  }
}
```

**字段说明**：
- `structure`: 每句字数列表
- `tonal_template`: 平仄模板（P=平, Z=仄, *=不拘）
- `rhyme_groups`: 韵脚组列表（表示押韵规则）
- `sample_size`: 用于构建的样本数

## 使用方法

### 单条评估

```bash
uv run python eval.py --title "浣溪沙" --text "一曲新词酒一杯，去年天气旧亭台。夕阳西下几时回？无可奈何花落去，似曾相识燕归来。小园香径独徘徊。"
```

输出：
```
【浣溪沙】评估报告
✓ 结构匹配: [7, 7, 7, 7, 7, 7]
📊 平仄准确度: 86.67%
🎵 押韵一致性: 60.00%
⭐ 综合格律分: 86.67/100
```

### 模型评估

```bash
# 评估 MHA 模型
uv run python evaluate_model.py \
  --model_type scratch \
  --config_path ../configs/mha.yaml \
  --num_titles 50 \
  --output eval_mha.json

# 评估 MLA 模型
uv run python evaluate_model.py \
  --model_type scratch \
  --config_path ../configs/mla.yaml \
  --num_titles 50 \
  --output eval_mla.json
```

## 模型评估结果

### 最新评估结果（50词牌）

| 指标 | MHA | MLA | full_attn_res |
|------|-----|-----|---------------|
| 结构匹配率 | 78.00% | **82.00%** | 74.00% |
| 平均平仄准确度 | **85.88%** | 85.37% | 84.72% |
| 平均押韵一致性 | 66.24% | 55.90% | **69.22%** |
| 综合格律得分 | 75.06 | **75.34** | 72.88 |

**分析**：
- MLA 在结构匹配上表现最好
- MHA 在平仄准确度上略优
- full_attn_res 在押韵一致性上最高
- 综合来看，三个模型表现接近，各有优势

## 文件结构

```
scratch/songeval/
├── registry_builder.py   # 格律库构建器
├── evaluator.py          # 评估器
├── analyzer.py           # 格律库构建脚本
├── eval.py               # 单条评估脚本
├── evaluate_model.py     # 模型评估脚本
├── standard.json         # 格律库（75个词牌）
├── eval_mha.json         # MHA评估结果
├── eval_mla.json         # MLA评估结果
└── eval_full_attn_res.json
```

## 构建格律库

如需重新构建格律库：

```bash
uv run python analyzer.py --min_samples 50
```

参数：
- `--min_samples`: 最小样本量阈值（默认50）
- `--confidence`: 平仄判定阈值（默认0.8）
- `--top_n`: 保留Top N词牌
