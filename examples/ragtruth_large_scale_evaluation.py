"""
RAGtruth 大规模评估脚本
在 RAGtruth 数据集上进行全面的系统性能评估
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass
import argparse

from src.ragtruth_loader import RAGtruthLoader, TaskType, SplitType, RAGtruthSample
from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod
from src.qwen_hallucination_evaluator import QwenModel


@dataclass
class EvaluationResults:
    """评估结果数据类"""
    total_samples: int = 0
    successful_evaluations: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # 混淆矩阵
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # 分数统计
    hallucination_scores: List[float] = None
    non_hallucination_scores: List[float] = None
    
    # 时间统计
    evaluation_time: float = 0.0
    
    def __post_init__(self):
        if self.hallucination_scores is None:
            self.hallucination_scores = []
        if self.non_hallucination_scores is None:
            self.non_hallucination_scores = []


class LargeScaleEvaluator:
    """大规模评估器"""
    
    def __init__(self, vectara_api_key: str = None, dashscope_api_key: str = None):
        """
        初始化评估器
        
        Args:
            vectara_api_key: Vectara API密钥
            dashscope_api_key: 阿里云DashScope API密钥
        """
        self.loader = RAGtruthLoader()
        self.evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key=vectara_api_key,
            dashscope_api_key=dashscope_api_key
        )
        self.hallucination_threshold = 0.5  # HHEM阈值
        self.qwen_threshold = 0.2  # Qwen阈值
    
    def evaluate_samples(
        self,
        samples: List[RAGtruthSample],
        method: EvaluationMethod = EvaluationMethod.HHEM_ONLY,
        show_progress: bool = True
    ) -> EvaluationResults:
        """
        评估样本列表
        
        Args:
            samples: 要评估的样本列表
            method: 评估方法
            show_progress: 是否显示进度
            
        Returns:
            EvaluationResults: 评估结果
        """
        results = EvaluationResults()
        results.total_samples = len(samples)
        
        start_time = time.time()
        
        if show_progress:
            print(f"🔍 开始评估 {len(samples)} 个样本...")
            print("进度: ", end="", flush=True)
        
        individual_results = []
        
        for i, sample in enumerate(samples):
            # 显示进度
            if show_progress and (i % max(1, len(samples) // 20) == 0 or i == len(samples) - 1):
                progress = (i + 1) / len(samples) * 100
                print(f"\r进度: {progress:.1f}% ({i+1}/{len(samples)})", end="", flush=True)
            
            # 跳过源信息过短的样本
            source_info_str = str(sample.source.source_info)
            if len(source_info_str.strip()) < 10:
                continue
            
            try:
                # 执行评估
                result = self.evaluator.evaluate(
                    generated_text=sample.generated_text,
                    source_texts=sample.source_texts,
                    method=method
                )
                
                if result.success:
                    results.successful_evaluations += 1
                    
                    # 获取相应的分数
                    score = None
                    if method == EvaluationMethod.HHEM_ONLY and result.hhem_score is not None:
                        score = result.hhem_score
                        predicted_hallucination = score < self.hallucination_threshold
                    elif method == EvaluationMethod.QWEN_ONLY and result.qwen_hallucination_score is not None:
                        score = result.qwen_hallucination_score
                        predicted_hallucination = score > self.qwen_threshold
                    elif method in [EvaluationMethod.BOTH, EvaluationMethod.ENSEMBLE] and result.ensemble_score is not None:
                        score = result.ensemble_score
                        predicted_hallucination = score > self.hallucination_threshold
                    else:
                        continue
                    
                    actual_hallucination = sample.has_hallucination
                    
                    # 更新混淆矩阵
                    if predicted_hallucination and actual_hallucination:
                        results.true_positives += 1
                    elif predicted_hallucination and not actual_hallucination:
                        results.false_positives += 1
                    elif not predicted_hallucination and not actual_hallucination:
                        results.true_negatives += 1
                    elif not predicted_hallucination and actual_hallucination:
                        results.false_negatives += 1
                    
                    # 收集分数
                    if actual_hallucination:
                        results.hallucination_scores.append(score)
                    else:
                        results.non_hallucination_scores.append(score)
                    
                    individual_results.append({
                        'predicted': predicted_hallucination,
                        'actual': actual_hallucination,
                        'score': score,
                        'correct': predicted_hallucination == actual_hallucination
                    })
                    
            except Exception as e:
                if show_progress and i < 5:  # 只显示前5个错误
                    print(f"\n⚠️ 样本 {i+1} 评估失败: {e}")
                continue
        
        results.evaluation_time = time.time() - start_time
        
        # 计算指标
        if results.successful_evaluations > 0:
            total_predictions = results.true_positives + results.false_positives + results.true_negatives + results.false_negatives
            
            if total_predictions > 0:
                results.accuracy = (results.true_positives + results.true_negatives) / total_predictions
            
            if results.true_positives + results.false_positives > 0:
                results.precision = results.true_positives / (results.true_positives + results.false_positives)
            
            if results.true_positives + results.false_negatives > 0:
                results.recall = results.true_positives / (results.true_positives + results.false_negatives)
            
            if results.precision + results.recall > 0:
                results.f1_score = 2 * (results.precision * results.recall) / (results.precision + results.recall)
        
        if show_progress:
            print()  # 换行
        
        return results
    
    def comprehensive_evaluation(
        self,
        max_samples: int = 100,
        task_types: List[TaskType] = None,
        methods: List[EvaluationMethod] = None
    ):
        """
        全面评估
        
        Args:
            max_samples: 最大样本数
            task_types: 要测试的任务类型列表
            methods: 要测试的评估方法列表
        """
        if task_types is None:
            task_types = [TaskType.ALL]
        
        if methods is None:
            methods = [EvaluationMethod.HHEM_ONLY]
        
        print("🚀 RAGtruth 大规模评估")
        print("=" * 60)
        
        all_results = {}
        
        for task_type in task_types:
            print(f"\n📋 任务类型: {task_type.value}")
            print("-" * 40)
            
            # 获取样本
            samples = self.loader.get_samples(
                task_type=task_type,
                split=SplitType.TEST,
                max_samples=max_samples,
                random_seed=42
            )
            
            print(f"📊 获取到 {len(samples)} 个样本")
            
            for method in methods:
                print(f"\n🔍 评估方法: {method.value}")
                
                results = self.evaluate_samples(samples, method)
                all_results[f"{task_type.value}_{method.value}"] = results
                
                self.print_results(results)
        
        # 总结对比
        if len(all_results) > 1:
            self.print_comparison(all_results)
    
    def print_results(self, results: EvaluationResults):
        """打印评估结果"""
        print(f"   ✅ 成功评估: {results.successful_evaluations}/{results.total_samples} 样本")
        print(f"   🎯 准确率: {results.accuracy:.3f}")
        print(f"   📊 精确率: {results.precision:.3f}")
        print(f"   📊 召回率: {results.recall:.3f}")
        print(f"   📊 F1分数: {results.f1_score:.3f}")
        print(f"   ⏱️ 评估时间: {results.evaluation_time:.1f}秒")
        
        print(f"   📈 混淆矩阵:")
        print(f"      TP: {results.true_positives}, FP: {results.false_positives}")
        print(f"      TN: {results.true_negatives}, FN: {results.false_negatives}")
        
        if results.hallucination_scores and results.non_hallucination_scores:
            halluc_mean = statistics.mean(results.hallucination_scores)
            non_halluc_mean = statistics.mean(results.non_hallucination_scores)
            print(f"   📊 有幻觉样本平均分数: {halluc_mean:.4f} (n={len(results.hallucination_scores)})")
            print(f"   📊 无幻觉样本平均分数: {non_halluc_mean:.4f} (n={len(results.non_hallucination_scores)})")
    
    def print_comparison(self, all_results: Dict[str, EvaluationResults]):
        """打印对比结果"""
        print(f"\n🏆 评估结果对比")
        print("=" * 60)
        
        print(f"{'配置':<20} {'准确率':<8} {'F1分数':<8} {'成功率':<8} {'时间(秒)':<10}")
        print("-" * 60)
        
        for config, results in all_results.items():
            success_rate = results.successful_evaluations / results.total_samples
            print(f"{config:<20} {results.accuracy:<8.3f} {results.f1_score:<8.3f} {success_rate:<8.3f} {results.evaluation_time:<10.1f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAGtruth 大规模评估')
    parser.add_argument('--samples', type=int, default=100,
                        help='最大样本数量 (默认: 100)')
    parser.add_argument('--task', choices=['ALL', 'Summary', 'QA', 'Data2txt'], 
                        default='ALL', help='任务类型 (默认: ALL)')
    parser.add_argument('--method', choices=['HHEM_ONLY', 'QWEN_ONLY', 'ENSEMBLE'], 
                        default='HHEM_ONLY', help='评估方法 (默认: HHEM_ONLY)')
    parser.add_argument('--vectara-key', type=str, 
                        default="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g",
                        help='Vectara API密钥')
    parser.add_argument('--dashscope-key', type=str, 
                        help='DashScope API密钥')
    parser.add_argument('--comprehensive', action='store_true',
                        help='运行全面对比评估')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = LargeScaleEvaluator(
        vectara_api_key=args.vectara_key,
        dashscope_api_key=args.dashscope_key
    )
    
    if args.comprehensive:
        # 全面评估
        task_types = [TaskType.ALL, TaskType.SUMMARY, TaskType.QA]
        methods = [EvaluationMethod.HHEM_ONLY]
        
        if args.dashscope_key:
            methods.extend([EvaluationMethod.QWEN_ONLY, EvaluationMethod.ENSEMBLE])
        
        evaluator.comprehensive_evaluation(
            max_samples=args.samples,
            task_types=task_types,
            methods=methods
        )
    else:
        # 单个配置评估
        task_type = getattr(TaskType, args.task)
        method = getattr(EvaluationMethod, args.method)
        
        samples = evaluator.loader.get_samples(
            task_type=task_type,
            split=SplitType.TEST,
            max_samples=args.samples,
            random_seed=42
        )
        
        print(f"🔍 评估 {len(samples)} 个样本 - 任务: {args.task}, 方法: {args.method}")
        results = evaluator.evaluate_samples(samples, method)
        evaluator.print_results(results)


if __name__ == "__main__":
    main()
