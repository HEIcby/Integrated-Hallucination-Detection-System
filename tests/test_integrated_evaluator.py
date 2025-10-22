"""
集成幻觉评估测试脚本
演示如何使用HHEM和通义千问进行幻觉检测
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Dict
from src.integrated_hallucination_evaluator import (
    IntegratedHallucinationEvaluator, 
    EvaluationMethod
)
from src.qwen_hallucination_evaluator import QwenModel


def setup_api_keys():
    """设置API密钥"""
    # HHEM API密钥（Vectara）
    vectara_key = "zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g"  # 你现有的密钥
    
    # DashScope API密钥（需要你自己获取）
    dashscope_key = os.getenv('DASHSCOPE_API_KEY')  # 从环境变量获取
    if not dashscope_key:
        print("⚠️  警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("   只能使用HHEM评估功能")
        print("   要获取DashScope API密钥，请访问: https://dashscope.console.aliyun.com/")
        dashscope_key = None
    
    return vectara_key, dashscope_key


def test_single_evaluation():
    """测试单次评估"""
    print("=== 单次幻觉评估测试 ===\n")
    
    vectara_key, dashscope_key = setup_api_keys()
    
    try:
        # 初始化集成评估器
        evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key=vectara_key,
            dashscope_api_key=dashscope_key
        )
        
        # 测试用例：包含明显事实错误的文本
        test_cases = [
            {
                "name": "地理错误测试",
                "generated_text": "埃菲尔铁塔位于伦敦泰晤士河畔，是英国最著名的地标建筑。",
                "source_texts": [
                    "埃菲尔铁塔位于法国巴黎塞纳河畔，是法国最著名的地标建筑。",
                    "伦敦是英国的首都，巴黎是法国的首都。"
                ]
            },
            {
                "name": "数据错误测试", 
                "generated_text": "iPhone是由微软公司开发的智能手机产品。",
                "source_texts": [
                    "iPhone是苹果公司(Apple Inc.)开发和销售的智能手机产品系列。",
                    "微软公司主要产品包括Windows操作系统和Office办公软件。"
                ]
            },
            {
                "name": "准确信息测试",
                "generated_text": "北京是中华人民共和国的首都，也是政治和文化中心。",
                "source_texts": [
                    "中华人民共和国首都是北京市，是全国的政治中心、文化中心。"
                ]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"{i}. {test_case['name']}")
            print(f"   生成文本: {test_case['generated_text']}")
            print(f"   参考文档: {'; '.join(test_case['source_texts'][:1])}...")  # 只显示第一个文档
            
            # 使用HHEM评估
            if evaluator.hhem_evaluator:
                hhem_result = evaluator.evaluate(
                    generated_text=test_case["generated_text"],
                    source_texts=test_case["source_texts"],
                    method=EvaluationMethod.HHEM_ONLY
                )
                
                if hhem_result.success:
                    consistency_score = hhem_result.hhem_score
                    hallucination_score = 1.0 - consistency_score  # 转换为幻觉分数
                    print(f"   🔍 HHEM评估:")
                    print(f"      一致性分数: {consistency_score:.4f}")
                    print(f"      幻觉分数: {hallucination_score:.4f}")
                    print(f"      解释: {hhem_result.hhem_interpretation}")
                else:
                    print(f"   ❌ HHEM评估失败: {', '.join(hhem_result.error_messages)}")
            
            # 使用Qwen评估（如果可用）
            if evaluator.qwen_evaluator:
                qwen_result = evaluator.evaluate(
                    generated_text=test_case["generated_text"],
                    source_texts=test_case["source_texts"],
                    method=EvaluationMethod.QWEN_ONLY
                )
                
                if qwen_result.success:
                    print(f"   🤖 Qwen评估:")
                    print(f"      幻觉分数: {qwen_result.qwen_hallucination_score:.4f}")
                    print(f"      置信度: {qwen_result.qwen_confidence:.4f}")
                    print(f"      解释: {qwen_result.qwen_interpretation}")
                    print(f"      详细说明: {qwen_result.qwen_explanation[:100]}...")
                else:
                    print(f"   ❌ Qwen评估失败: {', '.join(qwen_result.error_messages)}")
            
            # 集成评估（如果两个评估器都可用）
            if evaluator.hhem_evaluator and evaluator.qwen_evaluator:
                ensemble_result = evaluator.evaluate(
                    generated_text=test_case["generated_text"],
                    source_texts=test_case["source_texts"],
                    method=EvaluationMethod.ENSEMBLE
                )
                
                if ensemble_result.success:
                    print(f"   🎯 集成评估:")
                    print(f"      综合幻觉分数: {ensemble_result.ensemble_score:.4f}")
                    print(f"      综合置信度: {ensemble_result.ensemble_confidence:.4f}")
                    print(f"      解释: {ensemble_result.ensemble_interpretation}")
            
            print("\n" + "-"*60 + "\n")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def test_batch_evaluation():
    """测试批量评估"""
    print("=== 批量幻觉评估测试 ===\n")
    
    vectara_key, dashscope_key = setup_api_keys()
    
    try:
        evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key=vectara_key,
            dashscope_api_key=dashscope_key
        )
        
        # 批量测试数据
        batch_data = [
            {
                "generated_text": "长江是中国最长的河流，全长约6300公里。",
                "source_texts": ["长江是中国最长的河流，全长约6397公里，发源于青藏高原。"]
            },
            {
                "generated_text": "太阳系有九大行星，包括冥王星。",
                "source_texts": ["太阳系有八大行星，冥王星在2006年被重新分类为矮行星。"]
            },
            {
                "generated_text": "人工智能是计算机科学的一个分支。",
                "source_texts": ["人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"]
            },
            {
                "generated_text": "比特币是由阿里巴巴公司发明的数字货币。",
                "source_texts": ["比特币是由中本聪(Satoshi Nakamoto)发明的去中心化数字货币。"]
            }
        ]
        
        print(f"正在批量评估 {len(batch_data)} 个文本...")
        
        # 执行批量评估
        results = evaluator.batch_evaluate(
            evaluations=batch_data,
            method=EvaluationMethod.HHEM_ONLY  # 使用HHEM评估
        )
        
        print("\n批量评估结果:")
        print("=" * 80)
        
        for i, (data, result) in enumerate(zip(batch_data, results), 1):
            print(f"\n{i}. 文本: {data['generated_text'][:50]}...")
            
            if result.success:
                if result.hhem_score is not None:
                    hallucination_score = 1.0 - result.hhem_score
                    risk_level = "🟢 低" if hallucination_score < 0.3 else "🟡 中" if hallucination_score < 0.6 else "🔴 高"
                    print(f"   风险等级: {risk_level}")
                    print(f"   一致性分数: {result.hhem_score:.4f}")
                    print(f"   幻觉分数: {hallucination_score:.4f}")
                    print(f"   评估: {result.hhem_interpretation}")
                
                if result.ensemble_score is not None:
                    print(f"   集成分数: {result.ensemble_score:.4f}")
                    print(f"   集成评估: {result.ensemble_interpretation}")
            else:
                print(f"   ❌ 评估失败: {', '.join(result.error_messages)}")
        
        # 统计摘要
        successful_results = [r for r in results if r.success]
        if successful_results:
            hhem_scores = [1.0 - r.hhem_score for r in successful_results if r.hhem_score is not None]
            if hhem_scores:
                avg_hallucination_score = sum(hhem_scores) / len(hhem_scores)
                high_risk_count = sum(1 for score in hhem_scores if score >= 0.6)
                
                print(f"\n📊 批量评估摘要:")
                print(f"   成功评估: {len(successful_results)}/{len(batch_data)}")
                print(f"   平均幻觉分数: {avg_hallucination_score:.4f}")
                print(f"   高风险文本数量: {high_risk_count}")
        
    except Exception as e:
        print(f"❌ 批量测试失败: {e}")


def test_method_comparison():
    """测试不同方法的比较"""
    print("=== 评估方法比较测试 ===\n")
    
    vectara_key, dashscope_key = setup_api_keys()
    
    try:
        evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key=vectara_key,
            dashscope_api_key=dashscope_key
        )
        
        test_text = "珠穆朗玛峰是世界上最高的山峰，位于中国西藏境内，海拔高度为8849米。"
        source_texts = [
            "珠穆朗玛峰是世界最高峰，位于中国西藏自治区与尼泊尔边境，海拔8848.86米。"
        ]
        
        print(f"测试文本: {test_text}")
        print(f"参考文档: {source_texts[0]}")
        print("\n比较结果:")
        print("=" * 60)
        
        # 比较不同评估方法
        comparison = evaluator.compare_methods(
            generated_text=test_text,
            source_texts=source_texts
        )
        
        for method_name, result in comparison.items():
            print(f"\n📋 {method_name.upper().replace('_', ' ')}:")
            
            if result.success:
                if result.hhem_score is not None:
                    print(f"   HHEM一致性分数: {result.hhem_score:.4f}")
                    print(f"   HHEM评估: {result.hhem_interpretation}")
                
                if result.qwen_hallucination_score is not None:
                    print(f"   Qwen幻觉分数: {result.qwen_hallucination_score:.4f}")
                    print(f"   Qwen置信度: {result.qwen_confidence:.4f}")
                    print(f"   Qwen评估: {result.qwen_interpretation}")
                
                if result.ensemble_score is not None:
                    print(f"   集成幻觉分数: {result.ensemble_score:.4f}")
                    print(f"   集成置信度: {result.ensemble_confidence:.4f}")
                    print(f"   集成评估: {result.ensemble_interpretation}")
            else:
                print(f"   ❌ 失败: {', '.join(result.error_messages)}")
        
    except Exception as e:
        print(f"❌ 比较测试失败: {e}")


def main():
    """主测试函数"""
    print("🚀 开始集成幻觉评估测试\n")
    print("=" * 80)
    
    # 检查依赖
    try:
        import requests
        print("✅ requests模块检查通过")
    except ImportError:
        print("❌ 缺少requests模块，请运行: pip install requests")
        return
    
    print()
    
    # 运行测试
    try:
        test_single_evaluation()
        print("\n" + "="*80 + "\n")
        
        test_batch_evaluation()
        print("\n" + "="*80 + "\n")
        
        test_method_comparison()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断了测试")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
    
    print("\n🏁 测试完成!")
    
    # 使用说明
    print("\n" + "="*80)
    print("📖 使用说明:")
    print("1. HHEM评估: 使用Vectara的HHEM模型评估事实一致性")
    print("2. Qwen评估: 使用阿里云通义千问评估幻觉程度")
    print("3. 集成评估: 综合两种方法获得更可靠的结果")
    print("\n🔑 API密钥设置:")
    print("- HHEM: 已使用你现有的Vectara API密钥")
    print("- Qwen: 需要设置环境变量DASHSCOPE_API_KEY")
    print("  获取地址: https://dashscope.console.aliyun.com/")


if __name__ == "__main__":
    main()
