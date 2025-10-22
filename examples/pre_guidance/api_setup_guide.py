"""
阿里云DashScope API密钥设置指南和快速测试
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.qwen_hallucination_evaluator import QwenHallucinationEvaluator, QwenModel


def guide_dashscope_setup():
    """阿里云DashScope设置指南"""
    print("🔑 阿里云DashScope API密钥设置指南")
    print("=" * 60)
    
    print("\n📋 步骤1: 获取API密钥")
    print("1. 访问阿里云DashScope控制台: https://dashscope.console.aliyun.com/")
    print("2. 登录你的阿里云账号（如果没有账号需要先注册）")
    print("3. 开通灵积模型服务（DashScope）")
    print("4. 在控制台中创建API密钥")
    print("5. 复制生成的API密钥")
    
    print("\n📋 步骤2: 设置环境变量")
    print("方法1 - 临时设置（当前终端会话有效）:")
    print("export DASHSCOPE_API_KEY='你的API密钥'")
    
    print("\n方法2 - 永久设置（推荐）:")
    print("# macOS/Linux - 编辑 ~/.bashrc 或 ~/.zshrc")
    print("echo 'export DASHSCOPE_API_KEY=\"你的API密钥\"' >> ~/.zshrc")
    print("source ~/.zshrc")
    
    print("\n方法3 - 在代码中直接设置（不推荐，有安全风险）:")
    print("evaluator = QwenHallucinationEvaluator(api_key='你的API密钥')")
    
    print("\n📋 步骤3: 验证设置")
    print("运行本脚本来验证API密钥是否正确设置")


def test_dashscope_connection():
    """测试DashScope连接"""
    print("\n🔍 测试DashScope连接...")
    
    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 环境变量DASHSCOPE_API_KEY未设置")
        return False
    
    print(f"✅ 找到API密钥: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
    try:
        # 初始化评估器
        evaluator = QwenHallucinationEvaluator()
        print("✅ QwenHallucinationEvaluator初始化成功")
        
        # 简单测试
        print("\n🧪 执行简单幻觉检测测试...")
        result = evaluator.evaluate_hallucination(
            generated_text="北京是中国的首都，人口约2100万。",
            source_texts=["北京是中华人民共和国的首都，常住人口约2100万。"],
            model=QwenModel.QWEN_TURBO
        )
        
        if result.success:
            print("✅ DashScope API调用成功！")
            print(f"   幻觉分数: {result.hallucination_score:.4f}")
            print(f"   置信度: {result.confidence:.4f}")
            print(f"   评估说明: {result.explanation[:100]}...")
            return True
        else:
            print(f"❌ API调用失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def demo_qwen_evaluation():
    """演示Qwen幻觉评估功能"""
    print("\n🤖 Qwen幻觉评估功能演示")
    print("=" * 60)
    
    try:
        evaluator = QwenHallucinationEvaluator()
        
        test_cases = [
            {
                "name": "地理错误",
                "text": "长城位于印度北部，是印度古代的军事防御工程。",
                "source": ["中国的万里长城位于中国北部，是中国古代的军事防御工程。"]
            },
            {
                "name": "时间错误", 
                "text": "iPhone于1995年首次发布，彻底改变了手机行业。",
                "source": ["iPhone于2007年1月9日由苹果公司首次发布，彻底改变了智能手机行业。"]
            },
            {
                "name": "准确信息",
                "text": "Python是一种高级编程语言，由Guido van Rossum开发。",
                "source": ["Python是一种高级编程语言，最初由Guido van Rossum在1989年开始开发。"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}测试:")
            print(f"   生成文本: {test_case['text']}")
            print(f"   参考文档: {test_case['source'][0]}")
            
            result = evaluator.evaluate_hallucination(
                generated_text=test_case['text'],
                source_texts=test_case['source']
            )
            
            if result.success:
                risk_level = "🟢 低" if result.hallucination_score < 0.3 else "🟡 中" if result.hallucination_score < 0.7 else "🔴 高"
                print(f"   风险等级: {risk_level}")
                print(f"   幻觉分数: {result.hallucination_score:.4f}")
                print(f"   置信度: {result.confidence:.4f}")
                print(f"   解释: {evaluator.interpret_score(result.hallucination_score)}")
            else:
                print(f"   ❌ 评估失败: {result.error_message}")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")


def main():
    """主函数"""
    print("🚀 阿里云DashScope Qwen幻觉评估器设置与测试")
    print("=" * 80)
    
    # 显示设置指南
    guide_dashscope_setup()
    
    # 测试连接
    if test_dashscope_connection():
        # 如果连接成功，运行演示
        demo_qwen_evaluation()
    else:
        print("\n💡 提示:")
        print("1. 请先按照上面的指南获取和设置API密钥")
        print("2. 确保网络连接正常")
        print("3. 验证API密钥有效性")
        print("4. 重新运行此脚本进行测试")
    
    print("\n🎯 完成设置后，你可以:")
    print("1. 运行 python3 ../../tests/test_integrated_evaluator.py 进行完整测试")
    print("2. 运行 python3 ../practical_examples.py 查看实际应用示例")
    print("3. 在你的项目中使用集成幻觉评估器")
    print("4. 比较HHEM和Qwen两种评估方法的结果")


if __name__ == "__main__":
    main()
