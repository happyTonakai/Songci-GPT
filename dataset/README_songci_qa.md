# Song Ci Q&A Generator

这个脚本可以根据宋词数据，调用 OpenAI API 生成多轮对话问答对。

## 安装依赖

```bash
pip install openai
```

## 使用方法

### 环境变量配置

首先设置环境变量：

```bash
# 必需：OpenAI API 密钥
export OPENAI_API_KEY='your-api-key-here'

# 可选：使用的模型（默认：gpt-3.5-turbo）
export OPENAI_MODEL='gpt-4'

# 可选：API 端点 URL（默认：https://api.openai.com/v1）
export OPENAI_BASE_URL='https://api.openai.com/v1'

# 可选：自定义模型提供商 ID（用于某些自定义 API 服务）
export X_MODEL_PROVIDER_ID='xiaomi'
```

### 基本用法

```bash
# 进入 dataset 目录
cd dataset

# 运行脚本（使用当前目录下的宋词文件）
python generate_songci_qa.py
```

这将使用默认设置：
- 输入目录：当前目录（`.`）
- 输出目录：`songci_qa`
- 模型：从 `OPENAI_MODEL` 环境变量获取，或默认 `gpt-3.5-turbo`
- API 端点：从 `OPENAI_BASE_URL` 环境变量获取，或默认 `https://api.openai.com/v1`

### 自定义参数

```bash
# 设置环境变量
export OPENAI_API_KEY='your-api-key-here'
export OPENAI_MODEL='gpt-4'
export OPENAI_BASE_URL='https://api.openai.com/v1'

# 运行脚本
python generate_songci_qa.py \
  --input-dir 宋词 \
  --output-dir songci_qa \
  --max-files 5 \
  --max-items 10 \
  --delay 2.0 \
  --save-interval 5
```

### 参数说明

- `--input-dir, -i`: 包含宋词 JSON 文件的目录（默认：`.`）
- `--output-dir, -o`: 保存生成的问答对的目录（默认：`songci_qa`）
- `--max-files`: 最大处理文件数（可选）
- `--max-items`: 每个文件最大处理条目数（可选）
- `--delay`: API 请求之间的延迟（秒，默认：1.0）
- `--save-interval`: 自动保存间隔（默认：10，表示每生成10个问答对自动保存一次）
- `--no-cleanup`: 保留临时文件（默认：False，处理完成后会自动清理临时文件）

### 环境变量说明

- `OPENAI_API_KEY`（必需）：OpenAI API 密钥
- `OPENAI_MODEL`（可选）：使用的模型（默认：`gpt-3.5-turbo`）
- `OPENAI_BASE_URL`（可选）：API 端点 URL（默认：`https://api.openai.com/v1`）
- `X_MODEL_PROVIDER_ID`（可选）：自定义模型提供商 ID（用于某些自定义 API 服务）

## 输出格式

生成的 JSON 文件格式如下：

```json
[
  {
    "conversation": [
      {
        "question": "问题1",
        "answer": "回答1"
      },
      {
        "question": "问题2",
        "answer": "回答2"
      }
    ],
    "metadata": {
      "author": "作者名",
      "rhythmic": "词牌名",
      "original_content": "原始内容摘要...",
      "original_author": "原始作者",
      "original_rhythmic": "原始词牌名",
      "original_paragraphs": ["段落1", "段落2", ...]
    }
  }
]
```

## 进度显示和自动保存

### 进度显示
脚本会显示实时进度条，包括：
- 当前处理的文件名
- 进度百分比（如：`[15/100] 15.0%`）
- 当前处理的宋词作者和词牌名

示例输出：
```
Processing file: dataset/宋词/ci.song.0.json
  Progress: [15/100] 15.0% - 苏轼 - 水调歌头
```

### 自动保存机制
- **默认设置**：每生成 10 个问答对自动保存一次
- **保存位置**：输出目录中的临时文件，文件名格式：`qa_ci.song.0.json.partial.10`
- **最终保存**：处理完成后，所有结果保存为最终文件：`qa_ci.song.0.json`

### 自定义保存间隔
```bash
# 每生成 5 个问答对保存一次
python generate_songci_qa.py --save-interval 5

# 每生成 20 个问答对保存一次
python generate_songci_qa.py --save-interval 20

# 保留临时文件（不自动清理，仅用于调试）
python generate_songci_qa.py --no-cleanup
```

### 临时文件说明
- 临时文件名格式：`qa_{原文件名}.partial.{已处理数量}`
- 例如：`qa_ci.song.0.json.partial.50` 表示已处理 50 个问答对
- 临时文件用于在处理过程中定期保存进度，防止数据丢失
- **处理完成后会自动清理所有临时文件**，只保留最终的完整文件
- 如果程序意外中断，临时文件可以用于恢复处理

## 生成的问答对特点

1. **多轮对话**：每个问答对包含 3-5 轮对话
2. **自然流畅**：模拟真实的学习或讨论场景
3. **回答简洁**：每个回答不超过 100 个字
4. **内容丰富**：围绕词的内容、意境、情感、写作技巧等方面
5. **准确深入**：回答准确、简洁、有深度

## 示例

假设输入宋词：
```
作者：和岘
词牌名：导引
内容：
气和玉烛，睿化著鸿明。
缇管一阳生。
...
```

可能生成的问答对：
```json
{
  "conversation": [
    {
      "question": "这首词描绘了什么样的场景？",
      "answer": "这首词描绘了庄严盛大的祭祀场景。'气和玉烛'形容天地和谐，'睿化著鸿明'赞美帝王的圣明教化。整首词展现了宋代宫廷祭祀的隆重氛围和对国家太平的颂扬。"
    },
    {
      "question": "词中'缇管一阳生'有什么含义？",
      "answer": "'缇管'指红色的律管，古代用来测定节气。'一阳生'指冬至时节阳气初生。这句词通过律管测气的典故，暗示冬至时节的到来，也象征着阳气回升、万物复苏的自然规律。"
    }
  ]
}
```

## 注意事项

1. **API 费用**：使用 OpenAI API 会产生费用，请注意控制处理数量
2. **速率限制**：建议设置适当的延迟（`--delay` 参数）避免触发速率限制
3. **错误处理**：脚本包含错误处理机制，失败的条目会被记录但不会中断整个处理过程
4. **数据备份**：建议在处理前备份原始数据

## 自定义提示词

如需修改生成问答对的提示词，可以编辑 `create_prompt` 方法中的提示模板。提示词包含：
- 任务描述
- 宋词信息（作者、词牌名、内容）
- 生成要求
- 输出格式说明

## 故障排除

### 常见问题

1. **API 密钥错误**
   - 确保正确设置了 `OPENAI_API_KEY` 环境变量
   - 检查账户是否有足够的余额
   - 使用 `echo $OPENAI_API_KEY` 验证环境变量是否设置正确

2. **输入目录不存在**
   - 确认宋词 JSON 文件在指定目录中
   - 使用 `--input-dir` 参数指定正确路径

3. **网络连接问题**
   - 检查网络连接
   - 如需使用代理，设置环境变量 `HTTP_PROXY` 和 `HTTPS_PROXY`

4. **API 端点错误**
   - 确认 `OPENAI_BASE_URL` 环境变量正确（例如：`https://api.openai.com/v1`）
   - 如果使用第三方 API 服务，确保 URL 格式正确
   - 使用 `echo $OPENAI_BASE_URL` 验证环境变量

5. **模型不支持**
   - 确认 `OPENAI_MODEL` 环境变量设置的模型可用
   - 检查 API 账户是否有访问该模型的权限

6. **内存不足**
   - 使用 `--max-files` 和 `--max-items` 参数限制处理数量
   - 分批处理大量数据

### 调试模式

如需查看详细处理过程，可以修改脚本中的日志级别或添加调试输出。

### 环境变量持久化

如果希望环境变量在每次登录后都生效，可以将其添加到 shell 配置文件中：

```bash
# 对于 bash 用户，添加到 ~/.bashrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export OPENAI_MODEL="gpt-3.5-turbo"' >> ~/.bashrc
echo 'export OPENAI_BASE_URL="https://api.openai.com/v1"' >> ~/.bashrc

# 对于 zsh 用户，添加到 ~/.zshrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
echo 'export OPENAI_MODEL="gpt-3.5-turbo"' >> ~/.zshrc
echo 'export OPENAI_BASE_URL="https://api.openai.com/v1"' >> ~/.zshrc

# 重新加载配置文件
source ~/.bashrc  # 或 source ~/.zshrc
```