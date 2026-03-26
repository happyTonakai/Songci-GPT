#!/bin/bash
# train_and_eval.sh - 训练并评估模型
# 用法: ./train_and_eval.sh <config_path>
# 示例: ./train_and_eval.sh scratch/configs/mha.yaml

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供配置文件路径"
    echo "用法: $0 <config_path>"
    echo "示例: $0 scratch/configs/mha.yaml"
    exit 1
fi

CONFIG_PATH="$1"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 从配置文件提取模型名称（用于输出文件名）
MODEL_NAME=$(basename "$CONFIG_PATH" .yaml)
echo "=========================================="
echo "训练并评估模型: $MODEL_NAME"
echo "配置文件: $CONFIG_PATH"
echo "=========================================="

# 步骤1: 训练模型
echo ""
echo "步骤1: 开始训练模型..."
echo "=========================================="
cd /home/hanzerui/joyspace/gpt
uv run python scratch/train.py --config_path="$CONFIG_PATH"

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo "错误: 训练失败"
    exit 1
fi

echo ""
echo "✓ 训练完成"

# 步骤2: 评估模型
echo ""
echo "步骤2: 开始评估模型..."
echo "=========================================="

# 切换到 songeval 目录运行评估（因为 evaluate_model.py 使用相对路径）
cd /home/hanzerui/joyspace/gpt/scratch/songeval

# 生成输出文件名
EVAL_OUTPUT="eval_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).json"

uv run python evaluate_model.py \
    --model_type scratch \
    --config_path "../configs/${MODEL_NAME}.yaml" \
    --samples_per_title 10 \
    --output "$EVAL_OUTPUT"

# 检查评估是否成功
if [ $? -ne 0 ]; then
    echo "错误: 评估失败"
    exit 1
fi

echo ""
echo "✓ 评估完成"

# 步骤3: 显示结果摘要
echo ""
echo "=========================================="
echo "评估结果摘要"
echo "=========================================="

# 切回项目根目录显示结果
cd /home/hanzerui/joyspace/gpt

if [ -f "scratch/songeval/$EVAL_OUTPUT" ]; then
    python3 << EOF
import json
import sys

try:
    with open("scratch/songeval/$EVAL_OUTPUT", 'r') as f:
        results = json.load(f)
    
    scores = results['aggregate_scores']
    print(f"模型: $MODEL_NAME")
    print(f"评估样本数: {scores['total_samples']}")
    print(f"结构匹配率: {scores['structure_match_rate']*100:.2f}%")
    print(f"平均平仄准确度: {scores['avg_tonal_accuracy']*100:.2f}%")
    print(f"平均押韵一致性: {scores['avg_rhyme_consistency']*100:.2f}%")
    print(f"综合格律得分: {scores['avg_form_score']*100:.2f}/100")
    print("")
    print(f"详细结果已保存: scratch/songeval/$EVAL_OUTPUT")
except Exception as e:
    print(f"读取结果失败: {e}")
    sys.exit(1)
EOF
fi

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
