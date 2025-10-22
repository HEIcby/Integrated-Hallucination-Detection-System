# Integrated Hallucination Evaluator

## Quick Demo

This repository provides a comprehensive hallucination detection system that combines:
- **HHEM (Vectara)**: Professional fact consistency evaluation
- **Qwen (Alibaba Cloud)**: Semantic understanding and hallucination detection

## Features
- 🎯 Dual evaluation system for higher accuracy
- 📊 Multiple evaluation methods (single, batch, comparison)
- 🚀 Easy integration with existing projects
- 📚 Comprehensive examples and documentation

## Quick Start

```bash
# Install dependencies
pip install requests dashscope

# Set API keys
export DASHSCOPE_API_KEY="your_dashscope_key"
export VECTARA_API_KEY="your_vectara_key"

# Run quick demo
python3 examples/quick_start.py
```

## Example Usage

```python
from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod

evaluator = IntegratedHallucinationEvaluator()
result = evaluator.evaluate(
    generated_text="Beijing is the capital of China with population of 21 million.",
    source_texts=["Beijing is the capital of China with about 21 million residents."],
    method=EvaluationMethod.ENSEMBLE
)

print(f"Hallucination risk score: {result.ensemble_score:.3f}")
```

## Documentation

- 📖 [Complete Documentation](README.md)
- 🔧 [Setup Guide](examples/pre_guidance/SETUP_GUIDE.md)  
- 💻 [API Reference](src/)
- 🧪 [Examples](examples/)

## License

MIT License - see [LICENSE](LICENSE) file for details.
