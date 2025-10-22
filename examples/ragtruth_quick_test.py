"""
RAGtruth 数据集快速测试示例
使用少量样本测试评估系统在 RAGtruth 数据集上的表现
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ragtruth_loader import RAGtruthLoader, TaskType, SplitType
from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod


def quick_test_on_ragtruth(max_samples=5, use_qwen=False):
    """在 RAGtruth 数据集上进行快速测试"""
    method_name = "Qwen" if use_qwen else "HHEM"
    print(f"🚀 RAGtruth 数据集快速测试 - 使用 {method_name}")
    print("=" * 50)
    
    # 初始化数据加载器
    loader = RAGtruthLoader()
    
    # 根据选择初始化评估器
    if use_qwen:
        # 使用 Qwen 评估
        evaluator = IntegratedHallucinationEvaluator(
            dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
        )
        evaluation_method = EvaluationMethod.QWEN_ONLY
    else:
        # 使用 HHEM 评估
        evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g"
        )
        evaluation_method = EvaluationMethod.HHEM_ONLY
    
    # 获取测试样本
    print(f"📊 获取测试样本（最多 {max_samples} 个）...")
    samples = loader.get_samples(
        split=SplitType.TEST,
        max_samples=max_samples,
        random_seed=42
    )
    
    print(f"✅ 获取到 {len(samples)} 个测试样本")
    print("\n🔍 开始评估:")
    print("-" * 50)
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        print(f"\n📝 样本 {i}:")
        print(f"   任务类型: {sample.source.task_type}")
        print(f"   模型: {sample.response.model}")
        print(f"   实际包含幻觉: {'是' if sample.has_hallucination else '否'}")
        
        if sample.has_hallucination:
            print(f"   幻觉数量: {sample.response.hallucination_count}")
            print(f"   幻觉类型: {[label.label_type for label in sample.response.labels]}")
        
        # 显示部分源信息和生成文本
        source_info = sample.source.source_info
        if isinstance(source_info, dict):
            source_info_str = str(source_info)
        else:
            source_info_str = str(source_info)
        
        print(f"   源信息长度: {len(source_info_str)} 字符")
        print(f"   生成文本长度: {len(sample.generated_text)} 字符")
        print(f"   生成文本预览: {sample.generated_text[:100]}...")
        
        # 检查源信息是否有效
        if len(source_info_str.strip()) < 10:
            print(f"   ⚠️ 源信息过短，跳过评估: '{source_info_str[:50]}'")
            continue
        
        try:
            # 执行评估
            result = evaluator.evaluate(
                generated_text=sample.generated_text,
                source_texts=sample.source_texts,
                method=evaluation_method
            )
            
            if result.success:
                if use_qwen and result.qwen_success:
                    score = result.qwen_hallucination_score
                    print(f"   🎯 Qwen幻觉分数: {score:.4f}")
                    print(f"   🤖 Qwen置信度: {result.qwen_confidence:.4f}")
                    print(f"   📊 Qwen解释: {result.qwen_interpretation}")
                    
                    # Qwen幻觉分数越高表示越可能有幻觉
                    predicted_hallucination = score > 0.2
                    
                elif not use_qwen and result.hhem_success:
                    score = result.hhem_score
                    print(f"   🎯 HHEM评估分数: {score:.4f}")
                    print(f"   📊 HHEM解释: {result.hhem_interpretation}")
                    
                    # HHEM分数越低表示越可能有幻觉
                    predicted_hallucination = score < 0.5
                    
                else:
                    print(f"   ❌ {method_name}评估失败: {result.error_messages}")
                    continue
                
                actual_hallucination = sample.has_hallucination
                
                if predicted_hallucination == actual_hallucination:
                    print(f"   ✅ 预测正确: {'有幻觉' if predicted_hallucination else '无幻觉'}")
                else:
                    print(f"   ❌ 预测错误: 预测{'有幻觉' if predicted_hallucination else '无幻觉'}，实际{'有幻觉' if actual_hallucination else '无幻觉'}")
                
                results.append({
                    'sample_id': i,
                    'actual': actual_hallucination,
                    'predicted': predicted_hallucination,
                    'score': score,
                    'correct': predicted_hallucination == actual_hallucination
                })
            else:
                print(f"   ❌ {method_name}评估失败: {result.error_messages}")
                
        except Exception as e:
            print(f"   ⚠️ 评估异常: {e}")
    
    # 总结结果
    print(f"\n" + "=" * 50)
    print("📊 测试总结:")
    
    if results:
        correct_predictions = sum(1 for r in results if r['correct'])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions
        
        print(f"   成功评估样本: {total_predictions}/{len(samples)}")
        print(f"   预测准确率: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        
        # 分数分析
        halluc_scores = [r['score'] for r in results if r['actual']]
        non_halluc_scores = [r['score'] for r in results if not r['actual']]
        
        if halluc_scores:
            avg_halluc_score = sum(halluc_scores) / len(halluc_scores)
            print(f"   有幻觉样本平均分数: {avg_halluc_score:.4f} (n={len(halluc_scores)})")
        
        if non_halluc_scores:
            avg_non_halluc_score = sum(non_halluc_scores) / len(non_halluc_scores)
            print(f"   无幻觉样本平均分数: {avg_non_halluc_score:.4f} (n={len(non_halluc_scores)})")
    
    else:
        print("   ❌ 没有成功的评估结果")


def analyze_sample_details():
    """分析具体样本的详细信息"""
    print("\n\n🔍 详细样本分析")
    print("=" * 50)
    
    loader = RAGtruthLoader()
    
    # 获取一个有幻觉的样本
    samples_with_halluc = loader.get_samples(
        has_hallucination=True,
        max_samples=1,
        random_seed=123
    )
    
    # 获取一个无幻觉的样本
    samples_without_halluc = loader.get_samples(
        has_hallucination=False,
        max_samples=1,
        random_seed=456
    )
    
    for label, samples in [("有幻觉样本", samples_with_halluc), ("无幻觉样本", samples_without_halluc)]:
        if samples:
            sample = samples[0]
            print(f"\n📋 {label}分析:")
            
            # 安全地处理源信息
            source_info = sample.source.source_info
            if isinstance(source_info, dict):
                source_preview = str(source_info)[:200]
            else:
                source_preview = str(source_info)[:200]
            
            generated_preview = str(sample.generated_text)[:200]
            print(f"   源信息: {source_preview}...")
            print(f"   生成文本: {generated_preview}...")
            
            if sample.has_hallucination:
                print(f"   幻觉详情:")
                for j, label_info in enumerate(sample.response.labels, 1):
                    print(f"      {j}. 幻觉文本: \"{label_info.text}\"")
                    print(f"         类型: {label_info.label_type}")
                    print(f"         位置: {label_info.start}-{label_info.end}")
                    print(f"         说明: {label_info.meta[:100]}...")


def comprehensive_test():
    """进行更全面的测试"""
    print("\n" + "🔬 全面测试模式")
    print("=" * 60)
    
    # 不同规模的测试
    test_sizes = [10, 25, 50]
    
    for test_size in test_sizes:
        print(f"\n📊 测试 {test_size} 个样本:")
        print("-" * 40)
        
        try:
            loader = RAGtruthLoader()
            evaluator = IntegratedHallucinationEvaluator(
                vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g"
            )
            
            samples = loader.get_samples(
                split=SplitType.TEST,
                max_samples=test_size,
                random_seed=42
            )
            
            results = []
            successful_evaluations = 0
            
            for i, sample in enumerate(samples, 1):
                if i % 10 == 0 or i == len(samples):
                    print(f"   进度: {i}/{len(samples)} ({i/len(samples)*100:.1f}%)")
                
                # 跳过源信息过短的样本
                source_info_str = str(sample.source.source_info)
                if len(source_info_str.strip()) < 10:
                    continue
                
                try:
                    result = evaluator.evaluate(
                        generated_text=sample.generated_text,
                        source_texts=sample.source_texts,
                        method=EvaluationMethod.HHEM_ONLY
                    )
                    
                    if result.success and result.hhem_success:
                        successful_evaluations += 1
                        predicted_hallucination = result.hhem_score < 0.5
                        actual_hallucination = sample.has_hallucination
                        
                        results.append({
                            'actual': actual_hallucination,
                            'predicted': predicted_hallucination,
                            'score': result.hhem_score,
                            'correct': predicted_hallucination == actual_hallucination
                        })
                        
                except Exception as e:
                    continue
            
            # 计算统计结果
            if results:
                correct_predictions = sum(1 for r in results if r['correct'])
                accuracy = correct_predictions / len(results)
                
                # 计算混淆矩阵
                tp = sum(1 for r in results if r['actual'] and r['predicted'])  # 真阳性
                fp = sum(1 for r in results if not r['actual'] and r['predicted'])  # 假阳性
                tn = sum(1 for r in results if not r['actual'] and not r['predicted'])  # 真阴性
                fn = sum(1 for r in results if r['actual'] and not r['predicted'])  # 假阴性
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"   ✅ 成功评估: {successful_evaluations}/{len(samples)} 样本")
                print(f"   🎯 准确率: {accuracy:.3f} ({correct_predictions}/{len(results)})")
                print(f"   📊 精确率: {precision:.3f}")
                print(f"   📊 召回率: {recall:.3f}")
                print(f"   📊 F1分数: {f1:.3f}")
                
                # 分数分布
                halluc_scores = [r['score'] for r in results if r['actual']]
                non_halluc_scores = [r['score'] for r in results if not r['actual']]
                
                if halluc_scores and non_halluc_scores:
                    avg_halluc = sum(halluc_scores) / len(halluc_scores)
                    avg_non_halluc = sum(non_halluc_scores) / len(non_halluc_scores)
                    print(f"   📈 有幻觉样本平均分数: {avg_halluc:.4f} (n={len(halluc_scores)})")
                    print(f"   📈 无幻觉样本平均分数: {avg_non_halluc:.4f} (n={len(non_halluc_scores)})")
            else:
                print("   ❌ 没有成功的评估结果")
                
        except Exception as e:
            print(f"   ⚠️ 测试失败: {e}")


def dataset_overview():
    """数据集概览"""
    print("\n📊 RAGtruth 数据集概览")
    print("=" * 50)
    
    loader = RAGtruthLoader()
    loader.print_statistics()
    
    # 各任务类型的样本分布
    print(f"\n📋 测试集样本分布:")
    test_samples = loader.get_samples(split=SplitType.TEST, max_samples=None)
    
    task_stats = {}
    halluc_stats = {}
    
    for sample in test_samples:
        task_type = sample.source.task_type
        has_halluc = sample.has_hallucination
        
        # 任务类型统计
        if task_type not in task_stats:
            task_stats[task_type] = {'total': 0, 'with_halluc': 0}
        task_stats[task_type]['total'] += 1
        if has_halluc:
            task_stats[task_type]['with_halluc'] += 1
    
    for task_type, stats in task_stats.items():
        halluc_rate = stats['with_halluc'] / stats['total'] * 100
        print(f"   {task_type}: {stats['total']} 样本 (幻觉率: {halluc_rate:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGtruth 数据集测试')
    parser.add_argument('--samples', type=int, default=5, 
                        help='快速测试的样本数量 (默认: 5)')
    parser.add_argument('--comprehensive', action='store_true',
                        help='运行全面测试 (10, 25, 50 样本)')
    parser.add_argument('--overview', action='store_true',
                        help='显示数据集概览')
    parser.add_argument('--qwen', action='store_true',
                        help='使用 Qwen 评估器而不是 HHEM')
    
    args = parser.parse_args()
    
    if args.overview:
        dataset_overview()
    
    if args.comprehensive:
        comprehensive_test()
    else:
        # 运行快速测试
        quick_test_on_ragtruth(max_samples=args.samples, use_qwen=args.qwen)
    
    # 始终运行样本详情分析
    analyze_sample_details()
