"""
RAGtruth 数据集集成评测示例
演示如何使用 RAGtruth 数据集对集成幻觉评估器进行测试和评估
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.ragtruth_loader import RAGtruthLoader, TaskType, SplitType, RAGtruthSample
from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod
from src.qwen_hallucination_evaluator import QwenModel


@dataclass
class EvaluationMetrics:
    """评估指标"""
    total_samples: int = 0
    successful_evaluations: int = 0
    
    # 准确性指标（与人工标注对比）
    true_positives: int = 0  # 正确识别的幻觉
    false_positives: int = 0  # 误报的幻觉
    true_negatives: int = 0  # 正确识别的非幻觉
    false_negatives: int = 0  # 漏报的幻觉
    
    # 评估器性能
    hhem_success_rate: float = 0.0
    qwen_success_rate: float = 0.0
    ensemble_success_rate: float = 0.0
    
    # 分数统计
    hallucination_scores: List[float] = None
    non_hallucination_scores: List[float] = None
    
    def __post_init__(self):
        if self.hallucination_scores is None:
            self.hallucination_scores = []
        if self.non_hallucination_scores is None:
            self.non_hallucination_scores = []
    
    @property
    def accuracy(self) -> float:
        """准确率"""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def precision(self) -> float:
        """精确率"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """召回率"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """F1分数"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


class RAGtruthEvaluator:
    """RAGtruth 数据集评测器"""
    
    def __init__(
        self,
        vectara_api_key: str = None,
        dashscope_api_key: str = None,
        dataset_path: str = None
    ):
        """
        初始化评测器
        
        Args:
            vectara_api_key: Vectara API密钥
            dashscope_api_key: 阿里云DashScope API密钥  
            dataset_path: RAGtruth 数据集路径，如果为None则使用项目内的默认路径
        """
        self.loader = RAGtruthLoader(dataset_path)
        self.evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key=vectara_api_key,
            dashscope_api_key=dashscope_api_key
        )
        
        # 阈值设置
        self.hallucination_threshold = 0.5  # 超过此阈值认为存在幻觉
    
    def evaluate_on_dataset(
        self,
        max_samples: int = 100,
        task_type: TaskType = TaskType.ALL,
        split: SplitType = SplitType.TEST,
        evaluation_method: EvaluationMethod = EvaluationMethod.ENSEMBLE,
        random_seed: int = 42
    ) -> EvaluationMetrics:
        """
        在 RAGtruth 数据集上进行评估
        
        Args:
            max_samples: 最大评估样本数
            task_type: 任务类型
            split: 数据集分割
            evaluation_method: 评估方法
            random_seed: 随机种子
            
        Returns:
            EvaluationMetrics: 评估指标
        """
        print(f"🔍 开始在 RAGtruth 数据集上进行评估")
        print(f"📊 参数设置:")
        print(f"   最大样本数: {max_samples}")
        print(f"   任务类型: {task_type.value}")
        print(f"   数据集分割: {split.value}")
        print(f"   评估方法: {evaluation_method.value}")
        print(f"   幻觉阈值: {self.hallucination_threshold}")
        print("=" * 60)
        
        # 获取样本
        samples = self.loader.get_samples(
            task_type=task_type,
            split=split,
            max_samples=max_samples,
            random_seed=random_seed
        )
        
        metrics = EvaluationMetrics()
        metrics.total_samples = len(samples)
        
        hhem_successes = 0
        qwen_successes = 0
        ensemble_successes = 0
        
        print(f"📝 开始评估 {len(samples)} 个样本...")
        
        for i, sample in enumerate(samples, 1):
            try:
                # 执行评估
                result = self.evaluator.evaluate(
                    generated_text=sample.generated_text,
                    source_texts=sample.source_texts,
                    method=evaluation_method
                )
                
                if result.success:
                    metrics.successful_evaluations += 1
                    
                    # 统计各评估器成功率
                    if result.hhem_success:
                        hhem_successes += 1
                    if result.qwen_success:
                        qwen_successes += 1
                    if result.ensemble_score is not None:
                        ensemble_successes += 1
                    
                    # 判断评估器预测结果
                    predicted_hallucination = False
                    score_to_use = None
                    
                    if evaluation_method == EvaluationMethod.HHEM_ONLY and result.hhem_score is not None:
                        # HHEM 分数越低表示越一致（无幻觉），所以高分数表示幻觉
                        predicted_hallucination = result.hhem_score > self.hallucination_threshold
                        score_to_use = result.hhem_score
                    elif evaluation_method == EvaluationMethod.QWEN_ONLY and result.qwen_hallucination_score is not None:
                        # Qwen 幻觉分数越高表示越可能是幻觉
                        predicted_hallucination = result.qwen_hallucination_score > self.hallucination_threshold
                        score_to_use = result.qwen_hallucination_score
                    elif evaluation_method in [EvaluationMethod.BOTH, EvaluationMethod.ENSEMBLE] and result.ensemble_score is not None:
                        # 集成分数
                        predicted_hallucination = result.ensemble_score > self.hallucination_threshold
                        score_to_use = result.ensemble_score
                    
                    # 实际标注结果
                    actual_hallucination = sample.has_hallucination
                    
                    # 计算混淆矩阵
                    if predicted_hallucination and actual_hallucination:
                        metrics.true_positives += 1
                    elif predicted_hallucination and not actual_hallucination:
                        metrics.false_positives += 1
                    elif not predicted_hallucination and not actual_hallucination:
                        metrics.true_negatives += 1
                    elif not predicted_hallucination and actual_hallucination:
                        metrics.false_negatives += 1
                    
                    # 收集分数统计
                    if score_to_use is not None:
                        if actual_hallucination:
                            metrics.hallucination_scores.append(score_to_use)
                        else:
                            metrics.non_hallucination_scores.append(score_to_use)
                
                # 显示进度
                if i % 10 == 0 or i == len(samples):
                    print(f"   进度: {i}/{len(samples)} ({i/len(samples)*100:.1f}%)")
                
            except Exception as e:
                print(f"   样本 {i} 评估失败: {e}")
                continue
        
        # 计算成功率
        if metrics.total_samples > 0:
            metrics.hhem_success_rate = hhem_successes / metrics.total_samples
            metrics.qwen_success_rate = qwen_successes / metrics.total_samples  
            metrics.ensemble_success_rate = ensemble_successes / metrics.total_samples
        
        return metrics
    
    def print_evaluation_results(self, metrics: EvaluationMetrics):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("📊 RAGtruth 数据集评估结果")
        print("=" * 60)
        
        print(f"📝 基础统计:")
        print(f"   总样本数: {metrics.total_samples}")
        print(f"   成功评估: {metrics.successful_evaluations} ({metrics.successful_evaluations/metrics.total_samples*100:.1f}%)")
        
        print(f"\n🎯 分类性能:")
        print(f"   准确率 (Accuracy): {metrics.accuracy:.4f}")
        print(f"   精确率 (Precision): {metrics.precision:.4f}")
        print(f"   召回率 (Recall): {metrics.recall:.4f}")
        print(f"   F1分数: {metrics.f1_score:.4f}")
        
        print(f"\n📈 混淆矩阵:")
        print(f"   真阳性 (TP): {metrics.true_positives}")
        print(f"   假阳性 (FP): {metrics.false_positives}")
        print(f"   真阴性 (TN): {metrics.true_negatives}")
        print(f"   假阴性 (FN): {metrics.false_negatives}")
        
        print(f"\n🔧 评估器成功率:")
        print(f"   HHEM成功率: {metrics.hhem_success_rate:.4f}")
        print(f"   Qwen成功率: {metrics.qwen_success_rate:.4f}")
        print(f"   集成成功率: {metrics.ensemble_success_rate:.4f}")
        
        if metrics.hallucination_scores and metrics.non_hallucination_scores:
            print(f"\n📊 分数分布:")
            halluc_mean = statistics.mean(metrics.hallucination_scores)
            halluc_std = statistics.stdev(metrics.hallucination_scores) if len(metrics.hallucination_scores) > 1 else 0
            non_halluc_mean = statistics.mean(metrics.non_hallucination_scores)
            non_halluc_std = statistics.stdev(metrics.non_hallucination_scores) if len(metrics.non_hallucination_scores) > 1 else 0
            
            print(f"   有幻觉样本分数: {halluc_mean:.4f} ± {halluc_std:.4f} (n={len(metrics.hallucination_scores)})")
            print(f"   无幻觉样本分数: {non_halluc_mean:.4f} ± {non_halluc_std:.4f} (n={len(metrics.non_hallucination_scores)})")
    
    def run_comprehensive_evaluation(self):
        """运行综合评估"""
        print("🚀 开始 RAGtruth 数据集综合评估")
        
        # 不同评估方法的对比
        methods = [
            EvaluationMethod.HHEM_ONLY,
            EvaluationMethod.QWEN_ONLY,
            EvaluationMethod.ENSEMBLE
        ]
        
        results = {}
        
        for method in methods:
            print(f"\n🔍 评估方法: {method.value}")
            try:
                metrics = self.evaluate_on_dataset(
                    max_samples=50,  # 为了快速测试，使用较小样本
                    evaluation_method=method
                )
                results[method.value] = metrics
                self.print_evaluation_results(metrics)
            except Exception as e:
                print(f"❌ 评估方法 {method.value} 失败: {e}")
        
        # 对比不同方法
        if len(results) > 1:
            print(f"\n🏆 方法对比总结:")
            print(f"{'方法':<15} {'准确率':<8} {'F1分数':<8} {'成功率':<8}")
            print("-" * 45)
            for method_name, metrics in results.items():
                print(f"{method_name:<15} {metrics.accuracy:<8.4f} {metrics.f1_score:<8.4f} {metrics.successful_evaluations/metrics.total_samples:<8.4f}")


def main():
    """主函数"""
    # 配置API密钥
    vectara_key = "zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g"  # 你的HHEM密钥
    dashscope_key = None  # 从环境变量获取或设置你的密钥
    
    # 创建评测器
    evaluator = RAGtruthEvaluator(
        vectara_api_key=vectara_key,
        dashscope_api_key=dashscope_key
    )
    
    # 首先查看数据集统计
    evaluator.loader.print_statistics()
    
    # 运行评估
    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
