# RAGtruth 数据集

本目录包含 RAGtruth 数据集的完整副本，用于幻觉检测评估。

## 文件说明

### 数据文件
- `response.jsonl` - 包含 17,790 个 LLM 响应及其幻觉标注
- `source_info.jsonl` - 包含 2,965 个源信息数据

### 文档文件  
- `RAGtruth_README.md` - RAGtruth 项目的原始说明文档
- `LICENSE` - RAGtruth 数据集的许可证文件

## 数据集统计

- **总响应数**: 17,790 条
- **总源信息数**: 2,965 条  
- **包含幻觉的响应**: 7,664 条 (43.1%)
- **无幻觉的响应**: 10,126 条 (56.9%)
- **训练集样本**: 15,090 条
- **测试集样本**: 2,700 条

## 模型分布

每个模型各有 2,965 个响应：
- gpt-3.5-turbo-0613
- gpt-4-0613  
- llama-2-13b-chat
- llama-2-70b-chat
- llama-2-7b-chat
- mistral-7B-instruct

## 任务类型

- **Data2txt**: 1,033 条 (结构化数据转文本)
- **QA**: 989 条 (问答)
- **Summary**: 943 条 (摘要)

## 幻觉类型

- **Evident Baseless Info**: 6,237 条 (明显的无根据信息)
- **Evident Conflict**: 5,324 条 (明显的冲突信息)
- **Subtle Baseless Info**: 2,527 条 (细微的无根据信息)  
- **Subtle Conflict**: 201 条 (细微的冲突信息)

## 使用方法

```python
from src.ragtruth_loader import RAGtruthLoader

# 加载数据（自动使用项目内的数据路径）
loader = RAGtruthLoader()

# 获取样本
samples = loader.get_samples(max_samples=10)

# 查看统计信息
loader.print_statistics()
```

## 数据格式

### response.jsonl 格式
```json
{
  "id": "响应ID",
  "source_id": "源信息ID", 
  "model": "模型名称",
  "temperature": 0.7,
  "labels": [
    {
      "start": 219,
      "end": 229, 
      "text": "幻觉文本",
      "meta": "标注说明",
      "label_type": "幻觉类型"
    }
  ],
  "split": "train/test",
  "quality": "good/incorrect_refusal/truncated",
  "response": "LLM的响应文本"
}
```

### source_info.jsonl 格式
```json
{
  "source_id": "源信息ID",
  "task_type": "任务类型", 
  "source": "数据来源",
  "source_info": "源信息内容",
  "prompt": "提示词"
}
```

## 引用

如果使用此数据集，请引用原始论文：
```
RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Generation
```

## 许可证

此数据集遵循原始 RAGtruth 项目的许可证条款。详见 LICENSE 文件。
