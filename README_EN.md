# Integrated Hallucination Evaluator

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/HEIcby/Integrated-Hallucination-Detection-System.svg)](https://github.com/HEIcby/Integrated-Hallucination-Detection-System/stargazers)
[![RAGtruth Dataset](https://img.shields.io/badge/dataset-RAGtruth-orange.svg)](data/ragtruth/)

🎯 A comprehensive AI hallucination detection system that integrates HHEM (Vectara) and Qwen (Alibaba Cloud) for accurate evaluation of generated text factual consistency. Now integrated with **RAGtruth dataset** for large-scale standardized evaluation.

## 🚀 Features

### 🔄 Dual Evaluation System
- **HHEM Evaluation**: Professional fact consistency assessment using Vectara's model
- **Qwen Evaluation**: Semantic understanding and hallucination detection via Alibaba Cloud
- **Ensemble Method**: Combined approach for maximum reliability

### 📊 RAGtruth Dataset Integration
- **Standard Dataset**: Integrated with 17,790 responses for large-scale evaluation
- **Multi-model Coverage**: Supports GPT-4, GPT-3.5, LLaMA-2, Mistral evaluations
- **Multi-task Types**: Covers summarization, QA, data-to-text tasks
- **Professional Annotation**: 43.1% samples contain expert hallucination labels

### 🎯 Rich Evaluation Modes
- ✅ Single evaluation - Quick detection for individual texts
- ✅ Batch evaluation - Efficient processing of multiple texts
- ✅ Method comparison - Analyze differences between evaluation approaches
- ✅ Real-time monitoring - Continuous content quality assessment
- ✅ Benchmark testing - Standardized evaluation on RAGtruth dataset

## Quick Start

```bash
# Clone repository
git clone https://github.com/HEIcby/Integrated-Hallucination-Detection-System.git
cd Integrated-Hallucination-Detection-System

# Install dependencies
pip install requests dashscope

# Set API keys
export DASHSCOPE_API_KEY="your_dashscope_key"
export VECTARA_API_KEY="your_vectara_key"  # Optional

# Run quick demo
python3 examples/quick_start.py

# Test on RAGtruth dataset
python3 examples/ragtruth_quick_test.py
```

## Example Usage

### Basic Usage
```python
from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod

evaluator = IntegratedHallucinationEvaluator()
result = evaluator.evaluate(
    generated_text="Beijing is the capital of China with population of 21 million.",
    source_texts=["Beijing is the capital of China with about 21 million residents."],
    method=EvaluationMethod.ENSEMBLE
)

print(f"Hallucination risk score: {result.ensemble_score:.3f}")
print(f"Evaluation confidence: {result.ensemble_confidence:.3f}")
```

### RAGtruth Dataset Evaluation
```python
from src.ragtruth_loader import RAGtruthLoader, TaskType, SplitType

# Load RAGtruth dataset
loader = RAGtruthLoader()
samples = loader.get_samples(
    task_type=TaskType.SUMMARY,
    split=SplitType.TEST,
    max_samples=10
)

# Evaluate on dataset
from examples.ragtruth_large_scale_evaluation import LargeScaleEvaluator
evaluator = LargeScaleEvaluator()
metrics = evaluator.evaluate_system_performance(max_samples=50)

print(f"HHEM Accuracy: {metrics['hhem_accuracy']:.3f}")
print(f"Qwen Accuracy: {metrics['qwen_accuracy']:.3f}")
```

## Project Structure

```
Integrated-Hallucination-Detection-System/
├── 📖 README.md & README_EN.md              # Documentation
├── 🚀 src/                                  # Core source code (1,427 lines)
│   ├── integrated_hallucination_evaluator.py  # Main evaluator (427 lines)
│   ├── ragtruth_loader.py                   # RAGtruth data loader (348 lines)
│   ├── qwen_hallucination_evaluator.py     # Qwen API integration (368 lines)
│   └── HHEM_API.py                          # Vectara HHEM interface (248 lines)
├── 📚 examples/                             # Usage examples (1,681 lines)
│   ├── ragtruth_quick_test.py               # Quick RAGtruth testing (352 lines)
│   ├── ragtruth_large_scale_evaluation.py  # Large-scale evaluation (326 lines)
│   ├── hhem_vs_qwen_comparison.py          # Method comparison (218 lines)
│   ├── practical_examples.py               # Real-world examples (221 lines)
│   └── pre_guidance/                        # Setup guides
├── 🧪 tests/                               # Test files (312 lines)
└── 📊 data/ragtruth/                       # RAGtruth dataset (~35MB)
    ├── response.jsonl                       # 17,790 responses
    └── source_info.jsonl                   # 2,965 source info entries
```

## Key Features

### 🎯 Optimized Thresholds
- **HHEM Threshold**: 0.5 (higher scores = more consistent)
- **Qwen Threshold**: 0.2 (lower scores = less hallucination)
- **Accuracy**: 80%+ on RAGtruth benchmark after optimization

### 📊 Evaluation Metrics
- Single text evaluation with confidence scores
- Batch processing for large-scale analysis
- Comprehensive performance metrics (accuracy, F1, precision, recall)
- Cross-method comparison and ensemble results

### 🔧 Easy Integration
- Simple Python API with minimal dependencies
- Flexible evaluation modes (HHEM_ONLY, QWEN_ONLY, BOTH, ENSEMBLE)
- Support for both single and batch evaluations
- Comprehensive error handling and logging

## Documentation

- 📖 [Complete Chinese Documentation](README.md)
- 🔧 [Setup Guide](examples/pre_guidance/SETUP_GUIDE.md)  
- 💻 [API Reference](src/)
- 🧪 [Examples](examples/)
- 📊 [RAGtruth Integration](data/ragtruth/)

## Performance

### Benchmark Results on RAGtruth Dataset
- **Total Samples**: 17,790 responses across 6 LLM models
- **HHEM Accuracy**: ~75% (threshold: 0.5)
- **Qwen Accuracy**: ~80% (threshold: 0.2)  
- **Ensemble Method**: Best overall performance
- **Processing Speed**: ~2-3 samples/second

### Supported Models & Tasks
- **LLM Models**: GPT-4, GPT-3.5, LLaMA-2, Mistral, Alpaca, Vicuna
- **Task Types**: Summarization, Question Answering, Data-to-Text
- **Languages**: English (RAGtruth), Chinese (examples)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the following projects and services:
- [Vectara HHEM](https://vectara.com/) - Professional fact consistency evaluation
- [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/) - Qwen model service
- [RAGtruth Dataset](https://github.com/amazon-science/RAGtruth) - Standardized hallucination detection dataset
