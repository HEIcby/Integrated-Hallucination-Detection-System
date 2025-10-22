"""
集成幻觉评估器包
支持HHEM和阿里云通义千问的幻觉检测系统
"""

from .integrated_hallucination_evaluator import (
    IntegratedHallucinationEvaluator,
    IntegratedEvaluationResult,
    EvaluationMethod
)

from .HHEM_API import (
    HHEMFactualConsistencyAPI,
    HHEMResponse
)

from .qwen_hallucination_evaluator import (
    QwenHallucinationEvaluator,
    QwenHallucinationResponse,
    QwenModel
)

__version__ = "1.0.0"
__author__ = "Ocean Chen"
__description__ = "集成HHEM和阿里云通义千问的幻觉检测系统"

__all__ = [
    'IntegratedHallucinationEvaluator',
    'IntegratedEvaluationResult', 
    'EvaluationMethod',
    'HHEMFactualConsistencyAPI',
    'HHEMResponse',
    'QwenHallucinationEvaluator',
    'QwenHallucinationResponse',
    'QwenModel'
]
