# SongEval v2.0 - 宋词格律评估系统

基于 PRD 文档实现的宋词格律自动提取与多维度评估系统。

## 功能特性

### 1. RegistryBuilder - 格律先验库构建

- **Top N 词牌筛选**: 自动统计并保留样本量最大的词牌
- **众数过滤**: 基于 (句数, 总字数) 的众数过滤变体样本
- **平仄统计**: 使用 0.8 置信度阈值判定每个位置的平仄
- **韵部映射**: 支持完整的韵部映射表，识别押韵位置

### 2. Evaluator - 模型性能量化

- **PPL 计算**: 在验证集上计算困惑度
- **结构匹配**: 评估生成文本的句数和每句字数
- **平仄准确度**: 对比生成文本与标准模板的平仄符合度
- **押韵一致性**: 评估押韵位置韵部的一致性
- **综合格律分**: 加权计算综合得分

## 文件结构

```
scratch/songeval/
├── __init__.py           # 模块初始化
├── registry_builder.py   # 格律库构建器
├── evaluator.py          # 评估器
├── analyzer.py           # 格律库构建主脚本
├── eval.py               # 模型评估主脚本
├── standard.json         # 生成的格律库 (示例)
└── README.md             # 本文件
```

## 使用方法

### 1. 构建格律库

```bash
# 基本用法 (样本量 >= 50 的词牌，约75个，覆盖73.5%数据)
uv run python analyzer.py

# 使用 Top N 策略
uv run python analyzer.py --top_n 50

# 自定义样本量阈值 (更严格，约50个词牌)
uv run python analyzer.py --min_samples 100

# 自定义所有参数
uv run python analyzer.py \
  --data_path "../../dataset/宋词/*.json" \
  --output standard.json \
  --min_samples 50 \
  --confidence 0.8
```

**参数说明:**
- `--data_path`: 宋词数据文件路径，支持通配符
- `--output`: 输出文件路径 (默认: standard.json)
- `--top_n`: 保留 Top N 个词牌 (与 --min_samples 二选一，默认 None)
- `--min_samples`: 最小样本量阈值，只保留样本数 >= 此值的词牌 (默认: 50)
- `--confidence`: 平仄判定阈值 (默认: 0.8)

**数据统计:**
- 数据集共有 **1421** 个不同词牌
- 样本 >= 50 首的词牌: **75** 个，覆盖 **73.5%** 的样本
- 样本 >= 100 首的词牌: **50** 个，覆盖 **64.8%** 的样本

### 数据源处理说明

**重要**: 原始数据源（chinese-poetry）中部分词牌存在**段落合并**问题。例如浣溪沙的下阕前两句被合并为一个14字段落，而非两个7字句。系统已针对以下词牌进行智能拆分：

- **浣溪沙**: `[7,7,7,14,7]` → `[7,7,7,7,7,7]`
- **鹧鸪天**: `[7,7,14,6,7,14]` → `[7,7,7,7,6,7,7,7]`
- **玉楼春**: `[7,7,14,7,7,14]` → `[7,7,7,7,7,7,7,7]`
- **踏莎行**: `[8,7,14,8,7,14]` → `[8,7,7,7,8,7,7,7]`
- **减字木兰花**: 修复14字合并
- **瑞鹧鸪**: 修复14字合并
- **蝶恋花**: 修复14字合并

### 众数过滤的局限性

**当前策略**: 系统使用**众数过滤**（Mode Filtering），只保留出现频率最高的结构作为标准。

**问题**: 对于格律变体较多的词牌（如水调歌头、满江红、沁园春等），众数过滤会丢失大量 valid 的变体样本。

**示例**:
- **水调歌头**: 符合率仅 88.6%，意味着 11.4% 的样本因属于变体而被过滤
- **沁园春**: 符合率仅 61.5%，近 40% 的样本被过滤

### 训练集质量统计

基于当前格律库（min_samples=50）:

| 质量等级 | 符合率 | 词牌数量 | 样本覆盖 | 代表词牌 |
|---------|--------|---------|---------|---------|
| **高质量** | >=90% | 31个 | 7,887首 (50.3%) | 浣溪沙、鹧鸪天、玉楼春、蝶恋花 |
| **中质量** | 70-90% | 21个 | 3,863首 (24.6%) | 水调歌头 (88.6%)、念奴娇 (86.7%)、满江红 (82.9%) |
| **低质量** | <70% | 23个 | 3,940首 (25.1%) | 沁园春 (61.5%)、水龙吟 (55.3%)、满庭芳 (46.4%) |

**整体符合率**: 81.3% (12,750 / 15,690)

### 未来优化方向 (TODO)

#### 1. 多结构支持 (Top-K 变体)
对于变体多的词牌，不应只取众数，而应保留 Top 2-3 的主要变体：

```python
# 当前：只保留众数
structure = [7, 7, 7, 7, 7, 7]  # 浣溪沙唯一标准

# 优化：保留 Top 2 变体
structures = [
    [7, 7, 7, 7, 7, 7],      # 60% 样本
    [7, 7, 10, 7, 7, 10],    # 30% 样本 (鹧鸪天变体)
]
```

#### 2. 变体识别与标注
在 `standard.json` 中增加变体信息：

```json
{
  "水调歌头": {
    "primary_structure": [10, 11, 17, 10, 9, 11, 17, 10],
    "variants": [
      {"structure": [10, 11, 6, 6, 5, 10, 9, 11, 6, 6, 5, 10], "ratio": 0.02},
      {"structure": [10, 11, 17, 10, 9, 9, 17, 10], "ratio": 0.016}
    ],
    "sample_size": 744,
    "conformity_rate": 0.886
  }
}
```

#### 3. 训练数据筛选建议

**当前建议**:
1. **优先训练**: 31个高质量词牌（符合率>90%）
2. **次要训练**: 21个中质量词牌（符合率70-90%）
3. **剔除/降低权重**: 23个低质量词牌（符合率<70%）

**实现方式**:
```python
# 在 dataset.py 中增加过滤
high_quality_titles = ['浣溪沙', '鹧鸪天', '玉楼春', ...]  # 31个
medium_quality_titles = ['水调歌头', '念奴娇', '满江红', ...]  # 21个

# 只使用高质量词牌训练
train_data = [item for item in all_data 
              if item['rhythmic'] in high_quality_titles]
```

**输出格式 (standard.json):**
```json
{
  "浣溪沙": {
    "structure": [7, 7, 7, 14, 7],
    "tonal_template": ["*ZPP*ZP", "*P*ZZPP", "*P*ZZPP", "*Z*PPZZ*P*ZZPP", "*P*ZZPP"],
    "rhyme_indices": [0, 1, 2, 4],
    "sample_size": 754
  }
}
```

### 2. 评估生成文本

#### 单条评估

```bash
uv run python eval.py \
  --title "浣溪沙" \
  --text "一曲新词酒一杯，去年天气旧亭台。夕阳西下几时回。"
```

输出示例:
```
============================================================
【浣溪沙】评估报告
============================================================

✓ 结构匹配:
   标准: [7, 7, 7, 14, 7]
   实际: [7, 7, 7, 14, 7]

📊 平仄准确度: 85.71%

🎵 押韵一致性: 100.00%

⭐ 综合格律分: 87.43/100
============================================================
```

#### 批量评估

```bash
# 准备测试样本 (JSON 格式)
cat > test_samples.json << 'EOF'
[
  {"title": "浣溪沙", "text": "一曲新词酒一杯..."},
  {"title": "水调歌头", "text": "明月几时有..."}
]
EOF

# 批量评估
uv run python eval.py \
  --mode batch \
  --batch_file test_samples.json \
  --detailed \
  --output results.json
```

#### 生成并评估 (与模型集成)

```bash
uv run python eval.py \
  --mode generate \
  --config_path ../configs/mha.yaml \
  --num_samples 10 \
  --output eval_results.json
```

### 3. 在训练中使用 PPL 评估

```python
from songeval import Evaluator

# 创建评估器
evaluator = Evaluator(registry_path="standard.json")

# 计算 PPL
ppl = evaluator.compute_ppl(model, val_dataloader, device="cuda")
print(f"Validation PPL: {ppl:.2f}")
```

## 评分算法

### 平仄判定
- 使用 `pypinyin` 提取每个字的拼音和声调
- 1,2声为平 (P)，3,4声为仄 (Z)
- 某位置平声占比 > 0.8 → 标记为 `P`
- 某位置仄声占比 > 0.8 → 标记为 `Z`
- 否则标记为 `*` (平仄不拘)

### 韵部映射
完整的韵部映射表，包括:
- Group_AN: an, ian, uan, üan
- Group_ANG: ang, iang, uang
- Group_EN: en, in, un, ün
- Group_ENG: eng, ing, ong, iong
- ...等

### 综合格律分
```
form_score = 0.4 * structure_match + 0.4 * tonal_accuracy + 0.2 * rhyme_consistency
```
(若结构不匹配，分数大幅降低)

## 依赖

```bash
# 已添加到项目依赖
uv add pypinyin
```

## 注意事项

1. **先构建格律库**: 运行 `eval.py` 前必须先运行 `analyzer.py` 生成 `standard.json`
2. **数据质量**: 格律质量取决于训练数据的清洗程度
3. **变体处理**: 系统通过众数过滤自动处理词牌变体

## 扩展开发

### 添加新的韵部映射

编辑 `registry_builder.py` 和 `evaluator.py` 中的 `RHYME_MAPPING`:

```python
RHYME_MAPPING = {
    # 添加新的韵部映射
    'new_rhyme': 'GROUP_NAME',
}
```

### 自定义评分权重

编辑 `evaluator.py` 中的 `evaluate` 方法:

```python
form_score = (
    0.5 * structure_match +  # 提高结构权重
    0.3 * tonal_accuracy +
    0.2 * rhyme_consistency
)
```

## 4. 模型格律评估

使用 `evaluate_model.py` 对训练好的模型进行格律符合度评估。

### 评估 scratch 模型

```bash
# 评估 MHA 模型 (评估50个词牌，每个词牌生成1个样本)
python evaluate_model.py \
  --model_type scratch \
  --config_path ../configs/mha.yaml \
  --num_titles 50 \
  --output eval_mha.json

# 评估 MLA 模型 (评估20个词牌，每个词牌生成3个样本)
python evaluate_model.py \
  --model_type scratch \
  --config_path ../configs/mla.yaml \
  --num_titles 20 \
  --samples_per_title 3 \
  --output eval_mla.json
```

> **注意**: `evaluate_model.py` 主要设计用于 scratch 模型评估。unsloth 模型可使用 `eval.py` 进行单条评估。

### 评估结果示例

```
================================================================================
评估结果汇总
================================================================================
模型类型: scratch
评估样本数: 20

结构匹配率: 85.00% (17/20)
平均平仄准确度: 87.34%
平均押韵一致性: 100.00%
综合格律得分: 85.43/100
================================================================================
```

### 对比多个模型

```bash
# 对比 MHA 和 MLA
python3 << 'EOF'
import json
with open('eval_mha.json') as f: mha = json.load(f)
with open('eval_mla.json') as f: mla = json.load(f)

print(f"MHA: {mha['aggregate_scores']['avg_form_score']*100:.1f}/100")
print(f"MLA: {mla['aggregate_scores']['avg_form_score']*100:.1f}/100")
EOF
```

## 许可证

MIT License
