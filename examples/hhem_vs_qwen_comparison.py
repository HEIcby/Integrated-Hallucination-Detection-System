"""
HHEM vs Qwen 对比评估脚本
直接对比两种评估方法在相同样本上的表现
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from dataclasses import dataclass
from typing import List, Dict

from src.ragtruth_loader import RAGtruthLoader, SplitType
from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod


@dataclass
class ComparisonResult:
    """对比结果"""
    sample_id: int
    actual_hallucination: bool
    
    # HHEM结果
    hhem_score: float = None
    hhem_predicted: bool = None
    hhem_correct: bool = None
    
    # Qwen结果
    qwen_score: float = None
    qwen_predicted: bool = None
    qwen_correct: bool = None
    qwen_confidence: float = None
    
    # 一致性
    predictions_agree: bool = None


def compare_methods(max_samples: int = 30):
    """对比HHEM和Qwen方法"""
    print("🔬 HHEM vs Qwen 对比评估")
    print("=" * 60)
    
    # 初始化
    loader = RAGtruthLoader()
    
    # 初始化两个评估器
    hhem_evaluator = IntegratedHallucinationEvaluator(
        vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g"
    )
    
    qwen_evaluator = IntegratedHallucinationEvaluator(
        dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
    )
    
    # 获取样本
    samples = loader.get_samples(
        split=SplitType.TEST,
        max_samples=max_samples,
        random_seed=42
    )
    
    print(f"📊 对比评估 {len(samples)} 个样本...")
    print("进度: ", end="", flush=True)
    
    results = []
    
    for i, sample in enumerate(samples):
        if i % max(1, len(samples) // 10) == 0:
            progress = (i + 1) / len(samples) * 100
            print(f"\r进度: {progress:.0f}%", end="", flush=True)
        
        # 跳过源信息过短的样本
        source_info_str = str(sample.source.source_info)
        if len(source_info_str.strip()) < 10:
            continue
        
        result = ComparisonResult(
            sample_id=i+1,
            actual_hallucination=sample.has_hallucination
        )
        
        try:
            # HHEM评估
            hhem_result = hhem_evaluator.evaluate(
                generated_text=sample.generated_text,
                source_texts=sample.source_texts,
                method=EvaluationMethod.HHEM_ONLY
            )
            
            if hhem_result.success and hhem_result.hhem_success:
                result.hhem_score = hhem_result.hhem_score
                result.hhem_predicted = hhem_result.hhem_score < 0.5  # HHEM阈值
                result.hhem_correct = result.hhem_predicted == sample.has_hallucination
            
        except Exception as e:
            print(f"\n⚠️ HHEM评估失败: {e}")
        
        try:
            # Qwen评估
            qwen_result = qwen_evaluator.evaluate(
                generated_text=sample.generated_text,
                source_texts=sample.source_texts,
                method=EvaluationMethod.QWEN_ONLY
            )
            
            if qwen_result.success and qwen_result.qwen_success:
                result.qwen_score = qwen_result.qwen_hallucination_score
                result.qwen_predicted = qwen_result.qwen_hallucination_score > 0.2  # Qwen阈值调整为0.2
                result.qwen_correct = result.qwen_predicted == sample.has_hallucination
                result.qwen_confidence = qwen_result.qwen_confidence
            
        except Exception as e:
            print(f"\n⚠️ Qwen评估失败: {e}")
        
        # 检查预测一致性
        if result.hhem_predicted is not None and result.qwen_predicted is not None:
            result.predictions_agree = result.hhem_predicted == result.qwen_predicted
        
        # 只保留两个方法都成功的结果
        if result.hhem_score is not None and result.qwen_score is not None:
            results.append(result)
    
    print()  # 换行
    
    # 分析结果
    analyze_comparison_results(results)


def analyze_comparison_results(results: List[ComparisonResult]):
    """分析对比结果"""
    if not results:
        print("❌ 没有有效的对比结果")
        return
    
    print(f"\n📊 对比分析结果 (基于 {len(results)} 个样本)")
    print("=" * 60)
    
    # 基础统计
    hhem_correct = sum(1 for r in results if r.hhem_correct)
    qwen_correct = sum(1 for r in results if r.qwen_correct)
    both_correct = sum(1 for r in results if r.hhem_correct and r.qwen_correct)
    predictions_agree = sum(1 for r in results if r.predictions_agree)
    
    hhem_accuracy = hhem_correct / len(results)
    qwen_accuracy = qwen_correct / len(results)
    agreement_rate = predictions_agree / len(results)
    
    print(f"🎯 准确率对比:")
    print(f"   HHEM准确率: {hhem_accuracy:.3f} ({hhem_correct}/{len(results)})")
    print(f"   Qwen准确率: {qwen_accuracy:.3f} ({qwen_correct}/{len(results)})")
    print(f"   两者都正确: {both_correct/len(results):.3f} ({both_correct}/{len(results)})")
    print(f"   预测一致率: {agreement_rate:.3f} ({predictions_agree}/{len(results)})")
    
    # 分数分析
    hhem_halluc_scores = [r.hhem_score for r in results if r.actual_hallucination]
    hhem_normal_scores = [r.hhem_score for r in results if not r.actual_hallucination]
    qwen_halluc_scores = [r.qwen_score for r in results if r.actual_hallucination]
    qwen_normal_scores = [r.qwen_score for r in results if not r.actual_hallucination]
    
    print(f"\n📈 分数分布:")
    if hhem_halluc_scores:
        print(f"   HHEM有幻觉样本平均分数: {sum(hhem_halluc_scores)/len(hhem_halluc_scores):.4f}")
    if hhem_normal_scores:
        print(f"   HHEM无幻觉样本平均分数: {sum(hhem_normal_scores)/len(hhem_normal_scores):.4f}")
    if qwen_halluc_scores:
        print(f"   Qwen有幻觉样本平均分数: {sum(qwen_halluc_scores)/len(qwen_halluc_scores):.4f}")
    if qwen_normal_scores:
        print(f"   Qwen无幻觉样本平均分数: {sum(qwen_normal_scores)/len(qwen_normal_scores):.4f}")
    
    # 不一致案例分析
    disagreements = [r for r in results if not r.predictions_agree]
    if disagreements:
        print(f"\n🔍 预测不一致案例分析 ({len(disagreements)} 个):")
        
        hhem_only_correct = [r for r in disagreements if r.hhem_correct and not r.qwen_correct]
        qwen_only_correct = [r for r in disagreements if r.qwen_correct and not r.hhem_correct]
        both_wrong = [r for r in disagreements if not r.hhem_correct and not r.qwen_correct]
        
        print(f"   仅HHEM正确: {len(hhem_only_correct)} 个")
        print(f"   仅Qwen正确: {len(qwen_only_correct)} 个")
        print(f"   两者都错: {len(both_wrong)} 个")
    
    # 详细案例展示
    print(f"\n📋 典型案例展示:")
    
    # 展示几个有代表性的案例
    show_cases = results[:5]  # 显示前5个案例
    
    for i, result in enumerate(show_cases, 1):
        actual_text = "有幻觉" if result.actual_hallucination else "无幻觉"
        hhem_text = "有幻觉" if result.hhem_predicted else "无幻觉"
        qwen_text = "有幻觉" if result.qwen_predicted else "无幻觉"
        agree_text = "✅" if result.predictions_agree else "❌"
        
        print(f"   案例{i}: 实际={actual_text} | HHEM={hhem_text}({result.hhem_score:.3f}) | Qwen={qwen_text}({result.qwen_score:.3f}) | 一致={agree_text}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HHEM vs Qwen 对比评估')
    parser.add_argument('--samples', type=int, default=30,
                        help='对比样本数量 (默认: 30)')
    
    args = parser.parse_args()
    
    # 检查环境变量
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    compare_methods(max_samples=args.samples)


if __name__ == "__main__":
    main()
