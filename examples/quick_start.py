#!/usr/bin/env python3
"""
快速开始 - 集成幻觉评估器
这是一个简单的演示脚本，展示如何快速使用集成幻觉评估系统
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod


def quick_demo():
    """快速演示"""
    print("🚀 集成幻觉评估器 - 快速演示")
    print("=" * 50)
    
    # 初始化评估器
    try:
        evaluator = IntegratedHallucinationEvaluator(
            vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g",
            dashscope_api_key=None  # 从环境变量获取 DASHSCOPE_API_KEY
        )
        print("✅ 评估器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 测试用例
    test_cases = [
        {
            "name": "严重错误",
            "text": "太阳是围绕地球转动的，这是基本的天文常识。",
            "reference": ["地球围绕太阳转动，这是基本的天文学知识。"]
        },
        {
            "name": "基本正确", 
            "text": "北京是中国的首都，人口约2000多万。",
            "reference": ["北京是中华人民共和国的首都，常住人口约2100万。"]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   文本: {case['text']}")
        
        # 使用集成评估
        result = evaluator.evaluate(
            generated_text=case['text'],
            source_texts=case['reference'],
            method=EvaluationMethod.ENSEMBLE
        )
        
        if result.success:
            # 风险等级判断
            if result.ensemble_score < 0.3:
                risk = "🟢 低风险"
            elif result.ensemble_score < 0.6:
                risk = "🟡 中等风险" 
            else:
                risk = "🔴 高风险"
                
            print(f"   评估: {risk} (分数: {result.ensemble_score:.3f})")
            print(f"   说明: {result.ensemble_interpretation}")
        else:
            print(f"   ❌ 评估失败")


def setup_guide():
    """设置指南"""
    print("\n" + "=" * 50)
    print("📖 设置指南")
    print("=" * 50)
    
    print("\n🔑 需要的API密钥:")
    print("1. HHEM (Vectara): 已配置")
    print("2. DashScope (阿里云): 需要设置环境变量")
    
    print("\n⚡ 快速设置DashScope:")
    print("export DASHSCOPE_API_KEY=你的密钥")
    print("获取地址: https://dashscope.console.aliyun.com/")
    
    print("\n📚 更多示例:")
    print("python3 practical_examples.py    # 实际应用示例")
    print("python3 test_integrated_evaluator.py  # 完整功能测试")


if __name__ == "__main__":
    quick_demo()
    setup_guide()
