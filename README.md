# 集成幻觉评估器 (Integrated Hallucination Evaluator)

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/HEIcby/Integrated-Hallucination-Detection-System.svg)](https://github.com/HEIcby/Integrated-Hallucination-Detection-System/stargazers)

🎯 一个集成了HHEM和阿里云通义千问(Qwen)的AI幻觉检测系统，用于评估生成文本的准确性和事实一致性。

## 📖 简介

在AI生成内容日益普及的今天，确保生成文本的准确性至关重要。本项目通过结合两种先进的评估方法：
- **HHEM (Vectara)**: 专业的事实一致性评估模型
- **通义千问 (Qwen)**: 阿里云大语言模型的语义理解能力

提供更可靠、更全面的幻觉检测解决方案。

## ✨ 主要特性

### 🔄 双重评估系统
- **HHEM评估**: 使用Vectara的HHEM模型评估事实一致性
- **Qwen评估**: 使用阿里云通义千问模型检测幻觉内容  
- **集成评估**: 综合两种方法获得更可靠的评估结果

### 🚀 丰富的评估模式
- ✅ 单次评估 - 快速检测单个文本
- ✅ 批量评估 - 高效处理大量文本
- ✅ 方法对比 - 分析不同评估方法的差异
- ✅ 实时监控 - 持续评估内容质量

### 🎯 广泛的应用场景
- 📰 新闻事实核查
- 🤖 AI客服质量监控
- 📚 教育内容验证
- 🔍 用户内容审核

## � 快速开始

### 1️⃣ 安装依赖

```bash
# 克隆仓库
git clone https://github.com/HEIcby/Integrated-Hallucination-Detection-System.git
cd Integrated-Hallucination-Detection-System

# 安装依赖
pip install requests dashscope
```

### 2️⃣ 配置API密钥

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export VECTARA_API_KEY="your_vectara_api_key"  # 可选
```

> 📖 详细配置指南请参考 [`examples/pre_guidance/SETUP_GUIDE.md`](examples/pre_guidance/SETUP_GUIDE.md)

### 3️⃣ 运行示例

```bash
# 快速体验
python3 examples/quick_start.py

# 查看实际应用场景
python3 examples/practical_examples.py

# 运行完整测试
python3 tests/test_integrated_evaluator.py
```

## � 使用方法

### 基础用法

```python
import sys
import os
sys.path.append('path/to/Integrated-Hallucination-Detection-System')

from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod

# 初始化评估器
evaluator = IntegratedHallucinationEvaluator(
    vectara_api_key="your_vectara_key",      # 可选
    dashscope_api_key="your_dashscope_key"  # 或从环境变量获取
)

# 执行集成评估
result = evaluator.evaluate(
    generated_text="北京是中国的首都，人口约2100万。",
    source_texts=["北京是中华人民共和国的首都，常住人口约2100万。"],
    method=EvaluationMethod.ENSEMBLE
)

# 查看结果
if result.success:
    print(f"🎯 幻觉风险分数: {result.ensemble_score:.4f}")
    print(f"📝 评估解释: {result.ensemble_interpretation}")
    print(f"✅ 综合置信度: {result.ensemble_confidence:.4f}")
```

### 批量评估

```python
# 准备批量数据
batch_data = [
    {
        "generated_text": "太阳围绕地球转动",
        "source_texts": ["地球围绕太阳转动"]
    },
    {
        "generated_text": "Python是一种编程语言",
        "source_texts": ["Python是一种高级编程语言"]
    }
]

# 执行批量评估
results = evaluator.batch_evaluate(batch_data, method=EvaluationMethod.ENSEMBLE)

# 处理结果
for i, result in enumerate(results):
    print(f"文本 {i+1}: 风险分数 {result.ensemble_score:.3f}")
```

## 📊 评估方法与分数解释

### 评估方法对比

| 方法 | 说明 | 优势 | 适用场景 |
|------|------|------|----------|
| `HHEM_ONLY` | 仅使用HHEM评估 | 快速、专业的事实核查 | 快速事实验证 |
| `QWEN_ONLY` | 仅使用通义千问 | 强大的语义理解能力 | 复杂语义分析 |
| `BOTH` | 同时使用两种方法 | 获得双重验证结果 | 对比分析 |
| `ENSEMBLE` | 集成两种方法结果 | 最高的准确性和可靠性 | **推荐使用** |

### 分数含义

#### 🎯 HHEM一致性分数 (0-1，越高越好)
- `0.8-1.0`: 🟢 高度一致 - 生成文本与参考文档高度符合
- `0.6-0.8`: 🟡 较为一致 - 基本符合，存在轻微差异
- `0.4-0.6`: 🟠 部分一致 - 存在明显差异
- `0.2-0.4`: 🔴 不太一致 - 存在严重差异
- `0.0-0.2`: ❌ 严重不一致 - 事实严重冲突

#### 🎯 Qwen幻觉分数 (0-1，越低越好)
- `0.0-0.2`: 🟢 高度准确 - 与参考文档高度一致
- `0.2-0.4`: 🟡 基本准确 - 大部分准确，轻微问题
- `0.4-0.6`: 🟠 轻微幻觉 - 存在一些不准确之处
- `0.6-0.8`: 🔴 明显幻觉 - 存在明显事实错误
- `0.8-1.0`: ❌ 严重幻觉 - 包含大量虚假信息

## 📁 项目结构

```
Integrated-Hallucination-Detection-System/
├── 📖 README.md                           # 项目文档
├── 🚀 src/                                # 核心源代码
│   ├── __init__.py                        
│   ├── integrated_hallucination_evaluator.py  # 🎯 主要功能
│   ├── HHEM_API.py                        # Vectara HHEM接口
│   └── qwen_hallucination_evaluator.py   # 阿里云Qwen接口
├── 📚 examples/                           # 使用示例
│   ├── quick_start.py                     # ⚡ 快速开始
│   ├── practical_examples.py             # ⭐ 实际应用案例
│   └── pre_guidance/                      # 🔧 环境配置指南
│       ├── api_setup_guide.py             
│       └── SETUP_GUIDE.md                 # 详细配置文档
└── 🧪 tests/                             # 测试文件
    └── test_integrated_evaluator.py      # 功能测试
```

## 🎯 应用场景

### 1. 📰 新闻媒体
```python
# 验证AI生成新闻的准确性
result = evaluator.evaluate(
    generated_text="苹果公司将在2024年推出电动汽车",
    source_texts=["苹果公司的电动汽车项目仍在开发阶段，未宣布具体发布时间"],
    method=EvaluationMethod.ENSEMBLE
)
# 结果：高幻觉风险，需要人工核实
```

### 2. 🤖 客服系统
```python
# 监控AI客服回答质量
qa_pairs = [
    {
        "generated_text": "我们支持7天无理由退货",
        "source_texts": ["公司支持7天无理由退货政策，需保持商品原包装"]
    }
]
results = evaluator.batch_evaluate(qa_pairs)
# 实时监控回答准确性
```

### 3. 📚 教育平台
- ✅ 验证AI生成教学内容的科学性
- ✅ 确保知识点的准确传达  
- ✅ 提升在线教育质量

### 4. 🔍 内容审核
- ✅ 批量检测用户生成内容
- ✅ 识别和过滤虚假信息
- ✅ 维护平台内容质量

## � API密钥获取

### 🔑 HHEM (Vectara)
1. 访问 [Vectara官网](https://vectara.com/)
2. 注册账号并申请API密钥
3. 设置环境变量: `export VECTARA_API_KEY="your_key"`

### 🔑 通义千问 (DashScope)  
1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 开通灵积模型服务
3. 创建API密钥
4. 设置环境变量: `export DASHSCOPE_API_KEY="your_key"`

> 📖 详细配置步骤请参考 [`examples/pre_guidance/SETUP_GUIDE.md`](examples/pre_guidance/SETUP_GUIDE.md)

## ⚡ 性能特点

- 🎯 **准确性高**: 结合两种不同原理的评估方法
- 🛡️ **可靠性强**: 通过集成算法减少单一模型偏差  
- 📈 **扩展性好**: 支持批量评估和多种评估模式
- 🔌 **易于集成**: 提供简洁的Python API接口
- 🚀 **高效处理**: 支持异步调用和批量处理

## 🛠️ 高级配置

### 自定义模型参数

```python
from src.qwen_hallucination_evaluator import QwenModel

# 使用更强大的Qwen模型
result = evaluator.evaluate(
    generated_text="文本内容",
    source_texts=["参考文档"],
    method=EvaluationMethod.QWEN_ONLY,
    qwen_model=QwenModel.QWEN_MAX  # 使用Qwen-Max模型
)

# 调整评估参数
qwen_eval = QwenHallucinationEvaluator()
result = qwen_eval.evaluate_hallucination(
    generated_text="文本内容",
    source_texts=["参考文档"],
    temperature=0.1,  # 降低随机性
    max_tokens=1000   # 限制输出长度
)
```

### 集成到现有项目

```python
# 作为内容质量检查中间件
def content_quality_middleware(generated_content, references):
    evaluator = IntegratedHallucinationEvaluator()
    result = evaluator.evaluate(
        generated_text=generated_content,
        source_texts=references,
        method=EvaluationMethod.ENSEMBLE
    )
    
    # 根据风险分数决定是否发布
    if result.ensemble_score < 0.3:
        return {"status": "approved", "confidence": result.ensemble_confidence}
    else:
        return {"status": "needs_review", "reason": result.ensemble_interpretation}
```

## ⚠️ 重要提醒

### 💰 成本控制
- API调用需要费用，请注意使用量
- 建议先在小规模数据上测试
- 可以优先使用单一方法降低成本

### 🌐 网络要求  
- 需要稳定的网络连接访问API
- 建议配置重试机制处理网络异常
- 海外用户可能需要配置代理

### 🎯 准确性说明
- 评估结果仅供参考，重要决策需人工复核
- 不同领域的文本可能需要调整评估阈值
- 建议结合具体业务场景使用

## � 项目路线图

- [x] ✅ HHEM评估集成
- [x] ✅ 通义千问评估集成  
- [x] ✅ 集成评估算法
- [x] ✅ 批量评估支持
- [x] ✅ 完整示例和文档
- [ ] 🔄 支持更多评估模型 (GPT-4, Claude等)
- [ ] 🔄 添加缓存机制提升性能
- [ ] 🔄 实现异步评估批处理
- [ ] 🔄 增加评估结果可视化
- [ ] 🔄 支持自定义评估权重
- [ ] 🔄 提供Web API服务

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. 🍴 Fork 本仓库
2. 🌿 创建功能分支: `git checkout -b feature/amazing-feature`
3. 💾 提交更改: `git commit -m 'Add amazing feature'`
4. 📤 推送分支: `git push origin feature/amazing-feature`
5. 🔄 提交 Pull Request

### 开发环境设置
```bash
# 克隆你的Fork
git clone https://github.com/yourusername/Integrated-Hallucination-Detection-System.git
cd Integrated-Hallucination-Detection-System

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python3 -m pytest tests/
```

## � 支持与反馈

- 🐛 **Bug报告**: [提交Issue](https://github.com/HEIcby/Integrated-Hallucination-Detection-System/issues)
- 💡 **功能建议**: [讨论区](https://github.com/HEIcby/Integrated-Hallucination-Detection-System/discussions)  
- 📧 **邮件联系**: your.email@example.com
- 💬 **微信群**: 扫码加入技术交流群

## ⭐ 致谢

感谢以下开源项目和服务：
- [Vectara HHEM](https://vectara.com/) - 提供专业的事实一致性评估
- [阿里云DashScope](https://dashscope.console.aliyun.com/) - 提供通义千问模型服务
- 所有贡献者和用户的支持与反馈

## 📄 许可证

本项目采用 [MIT许可证](LICENSE)。详细信息请查看LICENSE文件。

---

<div align="center">

**如果这个项目对你有帮助，请给个⭐星标支持！**

[![GitHub stars](https://img.shields.io/github/stars/HEIcby/Integrated-Hallucination-Detection-System.svg?style=social&label=Star)](https://github.com/HEIcby/Integrated-Hallucination-Detection-System/stargazers)

</div>
