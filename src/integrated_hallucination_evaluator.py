"""
集成幻觉评估器
支持HHEM和通义千问两种评估方式的统一接口
"""

import asyncio
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from .HHEM_API import HHEMFactualConsistencyAPI, HHEMResponse
from .qwen_hallucination_evaluator import QwenHallucinationEvaluator, QwenHallucinationResponse, QwenModel


class EvaluationMethod(Enum):
    """评估方法枚举"""
    HHEM_ONLY = "hhem_only"  # 仅使用HHEM
    QWEN_ONLY = "qwen_only"  # 仅使用通义千问
    BOTH = "both"  # 同时使用两种方法
    ENSEMBLE = "ensemble"  # 集成评估（综合两种方法的结果）


@dataclass
class IntegratedEvaluationResult:
    """集成评估结果"""
    # 元信息（必需参数放在前面）
    method_used: EvaluationMethod
    success: bool = False
    
    # HHEM结果
    hhem_score: Optional[float] = None
    hhem_success: bool = False
    hhem_interpretation: Optional[str] = None
    
    # Qwen结果  
    qwen_hallucination_score: Optional[float] = None
    qwen_confidence: Optional[float] = None
    qwen_success: bool = False
    qwen_explanation: Optional[str] = None
    qwen_interpretation: Optional[str] = None
    
    # 集成结果
    ensemble_score: Optional[float] = None  # 综合评估分数 (0-1，越低越好)
    ensemble_confidence: Optional[float] = None  # 集成置信度
    ensemble_interpretation: Optional[str] = None
    
    # 错误信息
    error_messages: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class IntegratedHallucinationEvaluator:
    """
    集成幻觉评估器
    
    支持使用HHEM和通义千问两种方法进行幻觉评估，
    并提供集成评估功能以获得更可靠的结果
    """
    
    def __init__(
        self, 
        vectara_api_key: Optional[str] = None,
        dashscope_api_key: Optional[str] = None
    ):
        """
        初始化集成评估器
        
        Args:
            vectara_api_key: Vectara API密钥
            dashscope_api_key: 阿里云DashScope API密钥
        """
        self.hhem_evaluator = None
        self.qwen_evaluator = None
        
        # 初始化HHEM评估器
        if vectara_api_key:
            try:
                self.hhem_evaluator = HHEMFactualConsistencyAPI(api_key=vectara_api_key)
            except Exception as e:
                print(f"警告: HHEM评估器初始化失败: {e}")
        
        # 初始化Qwen评估器
        if dashscope_api_key:
            try:
                self.qwen_evaluator = QwenHallucinationEvaluator(api_key=dashscope_api_key)
            except Exception as e:
                print(f"警告: Qwen评估器初始化失败: {e}")
        
        if not self.hhem_evaluator and not self.qwen_evaluator:
            raise ValueError("至少需要提供一个有效的API密钥来初始化评估器")
    
    def evaluate(
        self,
        generated_text: str,
        source_texts: List[str],
        method: EvaluationMethod = EvaluationMethod.ENSEMBLE,
        qwen_model: QwenModel = QwenModel.QWEN_TURBO,
        hhem_model: str = "hhem_v2.3"
    ) -> IntegratedEvaluationResult:
        """
        执行集成幻觉评估
        
        Args:
            generated_text: 待评估的生成文本
            source_texts: 参考源文本列表
            method: 评估方法
            qwen_model: Qwen模型类型
            hhem_model: HHEM模型名称
            
        Returns:
            IntegratedEvaluationResult: 集成评估结果
        """
        result = IntegratedEvaluationResult(method_used=method)
        
        # 根据选择的方法执行评估
        if method in [EvaluationMethod.HHEM_ONLY, EvaluationMethod.BOTH, EvaluationMethod.ENSEMBLE]:
            if self.hhem_evaluator:
                hhem_result = self._evaluate_hhem(generated_text, source_texts, hhem_model)
                result.hhem_score = hhem_result.score if hhem_result.success else None
                result.hhem_success = hhem_result.success
                result.hhem_interpretation = (
                    self.hhem_evaluator.interpret_score(hhem_result.score) 
                    if hhem_result.success else None
                )
                if not hhem_result.success:
                    result.error_messages.append(f"HHEM评估失败: {hhem_result.error_message}")
            else:
                result.error_messages.append("HHEM评估器未初始化")
        
        if method in [EvaluationMethod.QWEN_ONLY, EvaluationMethod.BOTH, EvaluationMethod.ENSEMBLE]:
            if self.qwen_evaluator:
                qwen_result = self._evaluate_qwen(generated_text, source_texts, qwen_model)
                result.qwen_hallucination_score = qwen_result.hallucination_score if qwen_result.success else None
                result.qwen_confidence = qwen_result.confidence if qwen_result.success else None
                result.qwen_success = qwen_result.success
                result.qwen_explanation = qwen_result.explanation if qwen_result.success else None
                result.qwen_interpretation = (
                    self.qwen_evaluator.interpret_score(qwen_result.hallucination_score)
                    if qwen_result.success else None
                )
                if not qwen_result.success:
                    result.error_messages.append(f"Qwen评估失败: {qwen_result.error_message}")
            else:
                result.error_messages.append("Qwen评估器未初始化")
        
        # 计算集成结果
        if method == EvaluationMethod.ENSEMBLE:
            self._compute_ensemble_result(result)
        
        # 设置总体成功状态
        result.success = (
            (result.hhem_success and method in [EvaluationMethod.HHEM_ONLY]) or
            (result.qwen_success and method in [EvaluationMethod.QWEN_ONLY]) or
            (method in [EvaluationMethod.BOTH, EvaluationMethod.ENSEMBLE] and 
             (result.hhem_success or result.qwen_success))
        )
        
        return result
    
    def _evaluate_hhem(self, generated_text: str, source_texts: List[str], model: str) -> HHEMResponse:
        """执行HHEM评估"""
        return self.hhem_evaluator.evaluate_consistency(
            generated_text=generated_text,
            source_texts=source_texts,
            model_name=model
        )
    
    def _evaluate_qwen(
        self, 
        generated_text: str, 
        source_texts: List[str], 
        model: QwenModel
    ) -> QwenHallucinationResponse:
        """执行Qwen评估"""
        return self.qwen_evaluator.evaluate_hallucination(
            generated_text=generated_text,
            source_texts=source_texts,
            model=model
        )
    
    def _compute_ensemble_result(self, result: IntegratedEvaluationResult):
        """
        计算集成评估结果
        
        Args:
            result: 要更新的结果对象
        """
        scores = []
        confidences = []
        interpretations = []
        
        # 收集有效分数
        if result.hhem_success and result.hhem_score is not None:
            # HHEM分数越高表示越一致，转换为幻觉分数（越低越好）
            hallucination_score = 1.0 - result.hhem_score
            scores.append(hallucination_score)
            confidences.append(0.9)  # HHEM置信度假设为0.9
            interpretations.append(f"HHEM: {result.hhem_interpretation}")
        
        if result.qwen_success and result.qwen_hallucination_score is not None:
            scores.append(result.qwen_hallucination_score)
            confidences.append(result.qwen_confidence or 0.7)
            interpretations.append(f"Qwen: {result.qwen_interpretation}")
        
        # 计算集成分数
        if scores:
            # 使用加权平均，权重基于置信度
            if len(scores) > 1:
                weights = confidences
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                weight_sum = sum(weights)
                result.ensemble_score = weighted_sum / weight_sum if weight_sum > 0 else statistics.mean(scores)
                result.ensemble_confidence = statistics.mean(confidences)
            else:
                result.ensemble_score = scores[0]
                result.ensemble_confidence = confidences[0]
            
            result.ensemble_interpretation = self._interpret_ensemble_score(
                result.ensemble_score, 
                len(scores)
            )
    
    def _interpret_ensemble_score(self, score: float, num_evaluators: int) -> str:
        """解释集成评估分数"""
        base_interpretation = ""
        if score >= 0.8:
            base_interpretation = "严重幻觉风险 - 生成文本存在明显的事实错误或虚假信息"
        elif score >= 0.6:
            base_interpretation = "中等幻觉风险 - 生成文本存在一些不准确或可疑的内容"
        elif score >= 0.4:
            base_interpretation = "轻微幻觉风险 - 生成文本大部分准确，但存在细节问题"
        elif score >= 0.2:
            base_interpretation = "低幻觉风险 - 生成文本与参考文档基本一致"
        else:
            base_interpretation = "极低幻觉风险 - 生成文本高度准确和可信"
        
        evaluator_info = f"(基于{num_evaluators}个评估器的集成结果)"
        return f"{base_interpretation} {evaluator_info}"
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, Union[str, List[str]]]],
        method: EvaluationMethod = EvaluationMethod.ENSEMBLE,
        qwen_model: QwenModel = QwenModel.QWEN_TURBO,
        hhem_model: str = "hhem_v2.3"
    ) -> List[IntegratedEvaluationResult]:
        """
        批量评估
        
        Args:
            evaluations: 评估任务列表
            method: 评估方法
            qwen_model: Qwen模型类型
            hhem_model: HHEM模型名称
            
        Returns:
            List[IntegratedEvaluationResult]: 评估结果列表
        """
        results = []
        for i, eval_data in enumerate(evaluations):
            try:
                generated_text = eval_data.get('generated_text', '')
                source_texts = eval_data.get('source_texts', [])
                
                result = self.evaluate(
                    generated_text=generated_text,
                    source_texts=source_texts,
                    method=method,
                    qwen_model=qwen_model,
                    hhem_model=hhem_model
                )
                results.append(result)
                
            except Exception as e:
                error_result = IntegratedEvaluationResult(
                    method_used=method,
                    success=False,
                    error_messages=[f"批量评估第{i+1}项错误: {str(e)}"]
                )
                results.append(error_result)
        
        return results
    
    def compare_methods(
        self,
        generated_text: str,
        source_texts: List[str],
        qwen_model: QwenModel = QwenModel.QWEN_TURBO,
        hhem_model: str = "hhem_v2.3"
    ) -> Dict[str, IntegratedEvaluationResult]:
        """
        比较不同评估方法的结果
        
        Args:
            generated_text: 待评估文本
            source_texts: 参考文档
            qwen_model: Qwen模型
            hhem_model: HHEM模型
            
        Returns:
            Dict[str, IntegratedEvaluationResult]: 各种方法的评估结果
        """
        results = {}
        
        methods = [EvaluationMethod.HHEM_ONLY, EvaluationMethod.QWEN_ONLY, EvaluationMethod.ENSEMBLE]
        
        for method in methods:
            try:
                result = self.evaluate(
                    generated_text=generated_text,
                    source_texts=source_texts,
                    method=method,
                    qwen_model=qwen_model,
                    hhem_model=hhem_model
                )
                results[method.value] = result
            except Exception as e:
                results[method.value] = IntegratedEvaluationResult(
                    method_used=method,
                    success=False,
                    error_messages=[f"方法{method.value}评估失败: {str(e)}"]
                )
        
        return results


def demo_usage():
    """演示集成评估器的使用"""
    # 注意：需要设置相应的环境变量或提供API密钥
    try:
        # 初始化集成评估器（使用你的API密钥）
        evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g",  # 你的HHEM API密钥
            dashscope_api_key=None  # 需要设置你的DashScope API密钥
        )
        
        print("=== 集成幻觉评估器演示 ===\n")
        
        test_case = {
            "generated_text": "巴黎铁塔位于伦敦，是英国著名景点，高度为324米。",
            "source_texts": [
                "埃菲尔铁塔（Eiffel Tower）坐落于法国巴黎，是法国最著名的地标之一，高度为324米。",
                "伦敦是英国的首都，而巴黎是法国的首都。"
            ]
        }
        
        # 1. 单一方法评估
        print("1. 使用HHEM评估：")
        hhem_result = evaluator.evaluate(
            generated_text=test_case["generated_text"],
            source_texts=test_case["source_texts"],
            method=EvaluationMethod.HHEM_ONLY
        )
        
        if hhem_result.success:
            print(f"   ✅ HHEM分数: {hhem_result.hhem_score:.4f}")
            print(f"   📝 解释: {hhem_result.hhem_interpretation}")
        else:
            print(f"   ❌ 失败: {', '.join(hhem_result.error_messages)}")
        
        print("\n" + "-"*50 + "\n")
        
        # 2. 方法比较
        print("2. 方法比较：")
        comparison = evaluator.compare_methods(
            generated_text=test_case["generated_text"],
            source_texts=test_case["source_texts"]
        )
        
        for method_name, result in comparison.items():
            print(f"\n{method_name.upper()}:")
            if result.success:
                if result.hhem_score is not None:
                    print(f"   HHEM分数: {result.hhem_score:.4f}")
                if result.qwen_hallucination_score is not None:
                    print(f"   Qwen幻觉分数: {result.qwen_hallucination_score:.4f}")
                if result.ensemble_score is not None:
                    print(f"   集成分数: {result.ensemble_score:.4f}")
                    print(f"   集成置信度: {result.ensemble_confidence:.4f}")
                    print(f"   集成解释: {result.ensemble_interpretation}")
            else:
                print(f"   ❌ 失败: {', '.join(result.error_messages)}")
        
        print("\n" + "="*50 + "\n")
        
        # 3. 批量评估演示
        print("3. 批量评估演示：")
        batch_data = [
            {
                "generated_text": "苹果是一种蓝色的水果，富含维生素C",
                "source_texts": ["苹果通常是红色、绿色或黄色的水果，富含维生素C和纤维"]
            },
            {
                "generated_text": "北京是中国的首都，人口约2100万",
                "source_texts": ["中华人民共和国的首都是北京市，常住人口约2100万"]
            }
        ]
        
        batch_results = evaluator.batch_evaluate(
            evaluations=batch_data,
            method=EvaluationMethod.HHEM_ONLY  # 由于可能只有HHEM可用
        )
        
        for i, result in enumerate(batch_results, 1):
            print(f"\n批量评估任务 {i}:")
            if result.success:
                if result.hhem_score is not None:
                    print(f"   HHEM分数: {result.hhem_score:.4f} - {result.hhem_interpretation}")
                if result.ensemble_score is not None:
                    print(f"   集成分数: {result.ensemble_score:.4f} - {result.ensemble_interpretation}")
            else:
                print(f"   ❌ 失败: {', '.join(result.error_messages)}")
        
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        print("\n请确保：")
        print("1. 已安装必要的依赖包")
        print("2. 已设置正确的API密钥")
        print("3. 网络连接正常")


if __name__ == "__main__":
    demo_usage()
