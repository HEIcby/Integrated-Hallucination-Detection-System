"""
实际使用示例：集成幻觉评估器的简单应用
演示如何在实际项目中使用HHEM和Qwen进行幻觉检测
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integrated_hallucination_evaluator import IntegratedHallucinationEvaluator, EvaluationMethod
from src.qwen_hallucination_evaluator import QwenModel


def example_news_fact_check():
    """新闻事实核查示例"""
    print("📰 新闻事实核查示例")
    print("=" * 50)
    
    # 初始化评估器
    evaluator = IntegratedHallucinationEvaluator(
        vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g",  # 你的HHEM密钥
        dashscope_api_key=None  # 从环境变量获取
    )
    
    # 模拟新闻生成场景
    news_cases = [
        {
            "headline": "AI生成新闻",
            "generated_content": "苹果公司今日宣布，将在2024年推出首款电动汽车Apple Car，售价预计为5万美元。",
            "reference_sources": [
                "苹果公司一直在研发自动驾驶技术，但尚未正式宣布推出电动汽车的具体时间和价格。",
                "多家媒体报道称苹果的电动汽车项目仍在开发阶段。"
            ]
        },
        {
            "headline": "科技新闻",
            "generated_content": "OpenAI发布的GPT-4模型在2023年3月正式推出，支持多模态输入。",
            "reference_sources": [
                "OpenAI在2023年3月14日正式发布了GPT-4，这是一个大型多模态模型。"
            ]
        }
    ]
    
    for i, case in enumerate(news_cases, 1):
        print(f"\n{i}. {case['headline']}")
        print(f"   内容: {case['generated_content']}")
        
        # 执行集成评估
        result = evaluator.evaluate(
            generated_text=case['generated_content'],
            source_texts=case['reference_sources'],
            method=EvaluationMethod.ENSEMBLE
        )
        
        if result.success:
            # 判断新闻可信度
            risk_level = "🟢 可信" if result.ensemble_score < 0.3 else "🟡 存疑" if result.ensemble_score < 0.6 else "🔴 不可信"
            
            print(f"   📊 评估结果: {risk_level}")
            print(f"   📈 综合幻觉分数: {result.ensemble_score:.4f}")
            print(f"   🎯 置信度: {result.ensemble_confidence:.4f}")
            print(f"   💬 建议: {result.ensemble_interpretation}")
            
            # 详细分析
            if result.hhem_success:
                print(f"   🔍 HHEM: 一致性 {result.hhem_score:.3f}")
            if result.qwen_success:
                print(f"   🤖 Qwen: 幻觉 {result.qwen_hallucination_score:.3f}")
        else:
            print(f"   ❌ 评估失败: {', '.join(result.error_messages)}")


def example_customer_service():
    """客服机器人回答准确性检查"""
    print("\n🤖 客服机器人回答质量检查")
    print("=" * 50)
    
    evaluator = IntegratedHallucinationEvaluator(
        vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g",
        dashscope_api_key=None
    )
    
    # 模拟客服场景
    customer_qa = [
        {
            "question": "你们公司的退货政策是什么？",
            "bot_answer": "我们支持7天无理由退货，但需要商品保持全新状态。",
            "official_policy": [
                "本公司支持7天无理由退货政策，商品需保持原包装和全新状态。",
                "退货时需要提供购买凭证和完整的商品包装。"
            ]
        },
        {
            "question": "支付方式有哪些？",
            "bot_answer": "我们支持微信支付、支付宝、银行卡和比特币支付。",
            "official_policy": [
                "支持的支付方式包括：微信支付、支付宝、银联卡支付。",
                "暂不支持虚拟货币支付。"
            ]
        }
    ]
    
    for i, qa in enumerate(customer_qa, 1):
        print(f"\n{i}. 客户问题: {qa['question']}")
        print(f"   机器人回答: {qa['bot_answer']}")
        
        # 评估回答准确性
        result = evaluator.evaluate(
            generated_text=qa['bot_answer'],
            source_texts=qa['official_policy'],
            method=EvaluationMethod.BOTH  # 同时使用两种方法
        )
        
        if result.success:
            if result.ensemble_score and result.ensemble_score < 0.2:
                quality = "✅ 优秀"
            elif result.ensemble_score and result.ensemble_score < 0.4:
                quality = "✅ 良好"  
            elif result.ensemble_score and result.ensemble_score < 0.6:
                quality = "⚠️ 需要改进"
            else:
                quality = "❌ 不准确"
                
            print(f"   📝 回答质量: {quality}")
            
            if result.hhem_success:
                print(f"   🔍 HHEM评估: 一致性 {result.hhem_score:.3f}")
            if result.qwen_success:
                print(f"   🤖 Qwen评估: 幻觉 {result.qwen_hallucination_score:.3f}")
        else:
            print(f"   ❌ 评估失败")


def example_education_content():
    """教育内容准确性验证"""
    print("\n📚 教育内容准确性验证")
    print("=" * 50)
    
    evaluator = IntegratedHallucinationEvaluator(
        vectara_api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g",
        dashscope_api_key=None
    )
    
    # 教育内容示例
    educational_content = [
        {
            "topic": "数学知识",
            "ai_explanation": "圆周率π约等于3.14，是圆的周长与直径的比值。",
            "textbook_reference": [
                "圆周率π是一个数学常数，约等于3.14159，表示圆的周长与直径的比值。"
            ]
        },
        {
            "topic": "历史事实",
            "ai_explanation": "中国的长城建于公元前7世纪，由秦始皇统一建造。",
            "textbook_reference": [
                "中国长城的建设始于春秋战国时期（公元前7世纪），后来秦始皇统一六国后，将各国长城连接并扩建。"
            ]
        }
    ]
    
    for content in educational_content:
        print(f"\n📖 主题: {content['topic']}")
        print(f"   AI解释: {content['ai_explanation']}")
        
        result = evaluator.evaluate(
            generated_text=content['ai_explanation'],
            source_texts=content['textbook_reference'],
            method=EvaluationMethod.ENSEMBLE
        )
        
        if result.success and result.ensemble_score is not None:
            if result.ensemble_score < 0.1:
                accuracy = "🎯 非常准确"
            elif result.ensemble_score < 0.3:
                accuracy = "✅ 基本准确"
            elif result.ensemble_score < 0.5:
                accuracy = "⚠️ 需要核实"
            else:
                accuracy = "❌ 存在错误"
            
            print(f"   📊 准确性: {accuracy}")
            print(f"   📈 幻觉风险: {result.ensemble_score:.3f}")
        else:
            print("   ❌ 无法评估")


def main():
    """主函数"""
    print("🚀 集成幻觉评估器实际应用示例")
    print("=" * 80)
    
    try:
        # 1. 新闻事实核查
        example_news_fact_check()
        
        # 2. 客服质量检查
        example_customer_service()
        
        # 3. 教育内容验证
        example_education_content()
        
        print("\n" + "=" * 80)
        print("📝 总结:")
        print("✅ HHEM评估: 专注于事实一致性，基于神经网络模型")
        print("✅ Qwen评估: 通过大语言模型进行语义理解和推理")
        print("✅ 集成评估: 结合两种方法，提供更可靠的判断")
        
        print("\n💡 应用建议:")
        print("• 新闻媒体: 使用集成评估验证AI生成内容的准确性")
        print("• 客服系统: 监控机器人回答质量，及时发现错误信息") 
        print("• 教育平台: 确保AI辅助教学内容的科学性和准确性")
        print("• 内容审核: 批量检测用户生成内容中的虚假信息")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        print("请确保已正确设置API密钥和网络连接")


if __name__ == "__main__":
    main()
