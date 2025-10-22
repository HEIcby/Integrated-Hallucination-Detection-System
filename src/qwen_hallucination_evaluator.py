"""
阿里云通义千问幻觉评估器
通过DashScope平台调用Qwen模型来评估生成文本的幻觉程度
"""

import json
from typing import List, Dict, Optional, Union
import os
from dataclasses import dataclass
from enum import Enum

try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("⚠️ dashscope模块未安装，请运行: pip install dashscope")


class QwenModel(Enum):
    """支持的通义千问模型"""
    QWEN_TURBO = "qwen-turbo"
    QWEN_PLUS = "qwen-plus" 
    QWEN_MAX = "qwen-max"
    QWEN_MAX_LONGCONTEXT = "qwen-max-longcontext"


@dataclass
class QwenHallucinationResponse:
    """通义千问幻觉评估响应数据类"""
    hallucination_score: float  # 幻觉分数 0-1，越接近1表示幻觉越严重
    confidence: float  # 评估置信度 0-1
    explanation: str  # 评估说明
    success: bool
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None


class QwenHallucinationEvaluator:
    """
    通义千问幻觉评估器
    
    使用阿里云DashScope平台的Qwen模型来评估生成文本的幻觉程度
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            api_key: 阿里云DashScope API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY获取
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope模块未安装，请运行: pip install dashscope")
        
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        
        if not self.api_key:
            raise ValueError("API密钥未提供！请通过参数或环境变量DASHSCOPE_API_KEY提供")
        
        # 设置DashScope API密钥
        dashscope.api_key = self.api_key
    
    def _build_prompt(self, generated_text: str, source_texts: List[str]) -> str:
        """
        构建幻觉评估的提示词
        
        Args:
            generated_text: 待评估的生成文本
            source_texts: 参考源文本列表
            
        Returns:
            str: 构建好的提示词
        """
        source_content = "\n".join([f"参考文档{i+1}: {text}" for i, text in enumerate(source_texts)])
        
        prompt = f"""你是一个专业的事实核查和幻觉检测专家。请根据给定的参考文档，评估生成文本是否存在幻觉（虚假信息、不准确陈述或与参考文档不符的内容）。

参考文档：
{source_content}

生成文本：
{generated_text}

请从以下几个维度进行分析：
1. 事实准确性：生成文本中的事实是否与参考文档一致
2. 逻辑一致性：生成文本的逻辑是否与参考文档保持一致
3. 信息完整性：是否遗漏了重要信息或添加了不存在的信息
4. 细节准确性：具体的数字、日期、地点、人名等细节是否准确

请以如下JSON格式输出评估结果：
{{
    "hallucination_score": 0.0到1.0之间的浮点数（0表示无幻觉，1表示严重幻觉），
    "confidence": 0.0到1.0之间的浮点数表示评估置信度,
    "explanation": "详细的评估说明，包括发现的具体问题",
    "issues_found": ["发现的具体问题列表"]
}}

注意：请严格按照JSON格式输出，不要包含其他内容。"""
        
        return prompt
    
    def evaluate_hallucination(
        self,
        generated_text: str,
        source_texts: List[str],
        model: QwenModel = QwenModel.QWEN_TURBO,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> QwenHallucinationResponse:
        """
        评估生成文本的幻觉程度
        
        Args:
            generated_text: 待评估的生成文本
            source_texts: 参考源文本列表
            model: 使用的Qwen模型
            temperature: 生成温度，越低越确定性
            max_tokens: 最大生成token数
            
        Returns:
            QwenHallucinationResponse: 幻觉评估结果
        """
        if not generated_text.strip():
            return QwenHallucinationResponse(
                hallucination_score=1.0,
                confidence=1.0,
                explanation="生成文本为空，视为完全幻觉",
                success=False,
                error_message="生成文本不能为空"
            )
        
        if not source_texts or not any(text.strip() for text in source_texts):
            return QwenHallucinationResponse(
                hallucination_score=1.0,
                confidence=0.0,
                explanation="缺少参考文档，无法进行准确评估",
                success=False,
                error_message="参考文档不能为空"
            )
        
        # 构建请求提示词
        prompt = self._build_prompt(generated_text, source_texts)
        
        try:
            # 使用DashScope SDK调用模型
            messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            response = Generation.call(
                model=model.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format='message'
            )
            
            if response.status_code == 200:
                # 提取生成的文本
                if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                    generated_response = response.output.choices[0].message.content.strip()
                    
                    # 尝试解析JSON响应
                    try:
                        eval_result = json.loads(generated_response)
                        
                        return QwenHallucinationResponse(
                            hallucination_score=float(eval_result.get('hallucination_score', 0.5)),
                            confidence=float(eval_result.get('confidence', 0.5)),
                            explanation=eval_result.get('explanation', '评估完成'),
                            success=True,
                            raw_response=response.__dict__ if hasattr(response, '__dict__') else str(response)
                        )
                        
                    except json.JSONDecodeError:
                        # 如果无法解析JSON，尝试从文本中提取信息
                        return self._parse_text_response(generated_response, response.__dict__ if hasattr(response, '__dict__') else str(response))
                        
                else:
                    return QwenHallucinationResponse(
                        hallucination_score=0.5,
                        confidence=0.0,
                        explanation="API响应格式异常",
                        success=False,
                        error_message="响应中缺少choices字段",
                        raw_response=response.__dict__ if hasattr(response, '__dict__') else str(response)
                    )
            else:
                return QwenHallucinationResponse(
                    hallucination_score=0.5,
                    confidence=0.0,
                    explanation="API调用失败",
                    success=False,
                    error_message=f"状态码 {response.status_code}: {getattr(response, 'message', 'Unknown error')}",
                    raw_response=response.__dict__ if hasattr(response, '__dict__') else str(response)
                )
                
        except Exception as e:
            return QwenHallucinationResponse(
                hallucination_score=0.5,
                confidence=0.0,
                explanation="调用异常",
                success=False,
                error_message=f"调用异常: {str(e)}"
            )
    
    def _parse_text_response(self, text_response: str, raw_response: Dict) -> QwenHallucinationResponse:
        """
        解析非JSON格式的文本响应
        
        Args:
            text_response: 模型返回的文本
            raw_response: 原始API响应
            
        Returns:
            QwenHallucinationResponse: 解析后的评估结果
        """
        # 简单的文本解析逻辑
        text_lower = text_response.lower()
        
        # 尝试从文本中推断幻觉程度
        if any(keyword in text_lower for keyword in ['严重幻觉', '完全错误', '严重不符']):
            score = 0.9
        elif any(keyword in text_lower for keyword in ['存在幻觉', '部分错误', '不准确']):
            score = 0.6
        elif any(keyword in text_lower for keyword in ['轻微问题', '基本准确']):
            score = 0.3
        elif any(keyword in text_lower for keyword in ['无幻觉', '完全准确', '符合事实']):
            score = 0.1
        else:
            score = 0.5
        
        return QwenHallucinationResponse(
            hallucination_score=score,
            confidence=0.7,
            explanation=f"基于文本分析的评估: {text_response[:200]}...",
            success=True,
            raw_response=raw_response
        )
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, Union[str, List[str]]]],
        model: QwenModel = QwenModel.QWEN_TURBO,
        temperature: float = 0.1
    ) -> List[QwenHallucinationResponse]:
        """
        批量评估多个文本的幻觉程度
        
        Args:
            evaluations: 评估任务列表，每个元素包含generated_text和source_texts
            model: 使用的Qwen模型
            temperature: 生成温度
            
        Returns:
            List[QwenHallucinationResponse]: 评估结果列表
        """
        results = []
        for i, eval_data in enumerate(evaluations):
            try:
                generated_text = eval_data.get('generated_text', '')
                source_texts = eval_data.get('source_texts', [])
                
                result = self.evaluate_hallucination(
                    generated_text=generated_text,
                    source_texts=source_texts,
                    model=model,
                    temperature=temperature
                )
                results.append(result)
                
            except Exception as e:
                results.append(QwenHallucinationResponse(
                    hallucination_score=0.5,
                    confidence=0.0,
                    explanation=f"批量评估第{i+1}项错误",
                    success=False,
                    error_message=f"批量评估第{i+1}项错误: {str(e)}"
                ))
        
        return results
    
    def interpret_score(self, score: float) -> str:
        """
        解释幻觉分数的含义
        
        Args:
            score: 幻觉分数 (0-1)
            
        Returns:
            str: 分数解释
        """
        if score >= 0.8:
            return "严重幻觉 - 生成文本包含大量虚假或不准确信息"
        elif score >= 0.6:
            return "明显幻觉 - 生成文本存在明显的事实错误"
        elif score >= 0.4:
            return "轻微幻觉 - 生成文本存在一些不准确之处"
        elif score >= 0.2:
            return "基本准确 - 生成文本大部分准确，存在轻微问题"
        else:
            return "高度准确 - 生成文本与参考文档高度一致"


def demo_usage():
    """演示使用方法"""
    # 初始化评估器（注意：需要设置DASHSCOPE_API_KEY环境变量）
    try:
        evaluator = QwenHallucinationEvaluator()
        
        print("=== 通义千问幻觉评估器演示 ===\n")
        
        # 单次评估示例
        print("1. 单次评估示例：")
        result = evaluator.evaluate_hallucination(
            generated_text="巴黎铁塔位于伦敦，是英国著名景点，高度为324米。",
            source_texts=[
                "埃菲尔铁塔（Eiffel Tower）坐落于法国巴黎，是法国最著名的地标之一，高度为324米。",
                "伦敦是英国的首都，而巴黎是法国的首都。"
            ]
        )
        
        if result.success:
            print(f"✅ 评估成功")
            print(f"🎯 幻觉分数: {result.hallucination_score:.4f}")
            print(f"🎯 置信度: {result.confidence:.4f}")
            print(f"📝 分数解释: {evaluator.interpret_score(result.hallucination_score)}")
            print(f"💡 详细说明: {result.explanation}")
        else:
            print(f"❌ 评估失败: {result.error_message}")
        
        print("\n" + "="*50 + "\n")
        
        # 批量评估示例
        print("2. 批量评估示例：")
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
        
        batch_results = evaluator.batch_evaluate(batch_data)
        
        for i, result in enumerate(batch_results, 1):
            print(f"评估任务 {i}:")
            if result.success:
                print(f"  🎯 幻觉分数: {result.hallucination_score:.4f} - {evaluator.interpret_score(result.hallucination_score)}")
                print(f"  💡 说明: {result.explanation[:100]}...")
            else:
                print(f"  ❌ 错误: {result.error_message}")
        
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("请设置环境变量DASHSCOPE_API_KEY或在初始化时提供API密钥")


if __name__ == "__main__":
    demo_usage()
