# 项目代码文件结构分析

## 📁 完整项目结构

```
Integrated-Hallucination-Detection-System/
├── 📄 项目配置文件
│   ├── README.md                                    # 主要项目文档
│   ├── README_EN.md                                 # 英文文档
│   ├── LICENSE                                      # 许可证文件
│   ├── requirements.txt                             # 生产依赖
│   ├── requirements-dev.txt                         # 开发依赖
│   ├── .gitignore                                   # Git忽略文件
│   ├── RAGtruth_Integration_Summary.md              # RAGtruth集成总结
│   └── RAGtruth_Evaluation_Scale_Guide.md          # 评估规模指南
│
├── 🧠 核心源代码 /src/ (1,427行)
│   ├── __init__.py                                  # 36行 - 包初始化
│   ├── integrated_hallucination_evaluator.py       # 427行 - 🎯 主评估器
│   ├── HHEM_API.py                                  # 248行 - HHEM API封装
│   ├── qwen_hallucination_evaluator.py             # 368行 - Qwen评估器
│   └── ragtruth_loader.py                          # 348行 - RAGtruth数据加载
│
├── 💡 使用示例 /examples/ (1,681行)
│   ├── __init__.py                                  # 3行
│   ├── quick_start.py                               # 91行 - ⚡ 快速开始示例
│   ├── practical_examples.py                       # 221行 - ⭐ 实际应用案例
│   ├── ragtruth_quick_test.py                      # 352行 - 🧪 RAGtruth快速测试
│   ├── ragtruth_evaluation.py                      # 314行 - 📊 RAGtruth完整评估
│   ├── ragtruth_large_scale_evaluation.py          # 326行 - 🔬 大规模评估
│   ├── hhem_vs_qwen_comparison.py                  # 218行 - ⚔️ 方法对比
│   └── pre_guidance/                               # 配置指南
│       ├── __init__.py                              # 4行
│       ├── api_setup_guide.py                      # 156行 - API配置指南
│       └── SETUP_GUIDE.md                          # 详细配置文档
│
├── 🧪 测试代码 /tests/ (312行)
│   ├── __init__.py                                  # 3行
│   └── test_integrated_evaluator.py                # 309行 - 集成测试
│
└── 📊 数据集 /data/
    └── ragtruth/                                    # RAGtruth数据集
        ├── README.md                                # 数据集说明
        ├── RAGtruth_README.md                      # 原始项目文档
        ├── LICENSE                                  # 数据集许可证
        ├── response.jsonl                           # 20MB - 17,790个响应
        └── source_info.jsonl                       # 14MB - 2,965个源信息
```

## 📊 代码统计

### 总体统计
- **总Python文件**: 16个
- **总代码行数**: 3,424行
- **核心代码**: 1,427行 (41.7%)
- **示例代码**: 1,681行 (49.1%)
- **测试代码**: 312行 (9.1%)

### 核心模块详解

#### 🎯 主评估器 (`integrated_hallucination_evaluator.py` - 427行)
- **功能**: 集成HHEM和Qwen两种评估方法
- **支持**: 单次评估、批量评估、集成评估
- **评估模式**: HHEM_ONLY, QWEN_ONLY, BOTH, ENSEMBLE

#### 🔧 API封装层
- **HHEM_API.py** (248行): Vectara HHEM API封装
- **qwen_hallucination_evaluator.py** (368行): 阿里云Qwen API封装

#### 📊 数据处理
- **ragtruth_loader.py** (348行): RAGtruth数据集加载和处理

### 示例和工具

#### 🧪 测试和评估工具
- **ragtruth_quick_test.py** (352行): 快速测试工具
- **ragtruth_large_scale_evaluation.py** (326行): 大规模评估工具
- **hhem_vs_qwen_comparison.py** (218行): 方法对比工具

#### 💡 应用示例
- **practical_examples.py** (221行): 实际应用场景演示
- **quick_start.py** (91行): 快速上手示例

### 数据集集成

#### 📊 RAGtruth数据集 (~35MB)
- **response.jsonl**: 17,790个LLM响应及幻觉标注
- **source_info.jsonl**: 2,965个源信息数据
- **覆盖模型**: GPT-4, GPT-3.5, LLaMA-2, Mistral
- **任务类型**: 摘要、问答、数据转文本

## 🎯 项目特点

### ✅ 优势
1. **模块化设计**: 核心功能、示例、测试分离清晰
2. **双重评估**: HHEM + Qwen 提供更可靠的结果
3. **大规模数据**: 集成标准RAGtruth数据集
4. **完整工具链**: 从快速测试到大规模评估
5. **详细文档**: 多层次的文档和示例

### 🔧 架构亮点
1. **API抽象**: 统一的评估接口支持不同后端
2. **数据适配**: 智能处理不同格式的源信息
3. **评估模式**: 支持多种评估策略和组合
4. **工具丰富**: 从简单测试到复杂对比分析

### 📈 代码质量
- **平均文件大小**: 214行/文件
- **功能聚焦**: 每个模块职责明确
- **示例丰富**: 示例代码占比49%，便于学习使用
- **测试覆盖**: 包含集成测试和单元测试

## 🚀 使用入口

### 快速开始
```bash
python examples/quick_start.py              # 基础使用
python examples/ragtruth_quick_test.py      # RAGtruth快速测试
```

### 大规模评估
```bash
python examples/ragtruth_large_scale_evaluation.py --samples 100
python examples/hhem_vs_qwen_comparison.py --samples 50
```

### 实际应用
```bash
python examples/practical_examples.py       # 应用场景演示
```

这是一个设计良好、功能完整的幻觉检测系统，具有清晰的模块化架构和丰富的工具集。
