"""
HHEM事实一致性评估API封装
使用Vectara的HHEM模型来评估生成文本与源文本之间的事实一致性
"""

import requests
import json
from typing import List, Dict, Optional, Union
import os
from dataclasses import dataclass


@dataclass
class HHEMResponse:
    """HHEM API响应数据类"""
    score: float
    success: bool
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None


class HHEMFactualConsistencyAPI:
    """
    HHEM事实一致性评估API封装类
    
    用于评估生成文本与参考文本之间的事实一致性
    分数范围: 0-1，越接近1表示事实越一致
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化API客户端
        
        Args:
            api_key: Vectara API密钥，如果不提供则从环境变量VECTARA_API_KEY获取
        """
        self.base_url = "https://api.vectara.io/v2/evaluate_factual_consistency"
        self.api_key = api_key or os.getenv('VECTARA_API_KEY')
        
        if not self.api_key:
            raise ValueError("API密钥未提供！请通过参数或环境变量VECTARA_API_KEY提供")
        
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-api-key': self.api_key
        }
    
    def evaluate_consistency(
        self, 
        generated_text: str, 
        source_texts: List[str],
        model_name: str = "hhem_v2.3"
    ) -> HHEMResponse:
        """
        评估生成文本与源文本的事实一致性
        
        Args:
            generated_text: 需要评估的生成文本
            source_texts: 参考源文本列表
            model_name: 使用的模型名称，默认为hhem_v2.3
            
        Returns:
            HHEMResponse: 包含评估分数和相关信息的响应对象
        """
        if not generated_text.strip():
            return HHEMResponse(
                score=0.0,
                success=False,
                error_message="生成文本不能为空"
            )
        
        if not source_texts or not any(text.strip() for text in source_texts):
            return HHEMResponse(
                score=0.0,
                success=False,
                error_message="源文本不能为空"
            )
        
        # 准备请求负载
        payload = {
            "model_parameters": {
                "model_name": model_name
            },
            "generated_text": generated_text.strip(),
            "source_texts": [text.strip() for text in source_texts if text.strip()]
        }
        
        try:
            # 发送请求
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            # 检查HTTP状态码
            if response.status_code == 200:
                result = response.json()
                return HHEMResponse(
                    score=result.get('score', 0.0),
                    success=True,
                    raw_response=result
                )
            else:
                return HHEMResponse(
                    score=0.0,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                    raw_response={"status_code": response.status_code, "response": response.text}
                )
                
        except requests.exceptions.Timeout:
            return HHEMResponse(
                score=0.0,
                success=False,
                error_message="请求超时"
            )
        except requests.exceptions.RequestException as e:
            return HHEMResponse(
                score=0.0,
                success=False,
                error_message=f"请求异常: {str(e)}"
            )
        except json.JSONDecodeError as e:
            return HHEMResponse(
                score=0.0,
                success=False,
                error_message=f"响应JSON解析错误: {str(e)}"
            )
        except Exception as e:
            return HHEMResponse(
                score=0.0,
                success=False,
                error_message=f"未知错误: {str(e)}"
            )
    
    def batch_evaluate(
        self, 
        evaluations: List[Dict[str, Union[str, List[str]]]],
        model_name: str = "hhem_v2.3"
    ) -> List[HHEMResponse]:
        """
        批量评估多个文本对的事实一致性
        
        Args:
            evaluations: 评估任务列表，每个元素包含generated_text和source_texts
            model_name: 使用的模型名称
            
        Returns:
            List[HHEMResponse]: 评估结果列表
        """
        results = []
        for i, eval_data in enumerate(evaluations):
            try:
                generated_text = eval_data.get('generated_text', '')
                source_texts = eval_data.get('source_texts', [])
                
                result = self.evaluate_consistency(
                    generated_text=generated_text,
                    source_texts=source_texts,
                    model_name=model_name
                )
                results.append(result)
                
            except Exception as e:
                results.append(HHEMResponse(
                    score=0.0,
                    success=False,
                    error_message=f"批量评估第{i+1}项错误: {str(e)}"
                ))
        
        return results
    
    def interpret_score(self, score: float) -> str:
        """
        解释评估分数的含义
        
        Args:
            score: 事实一致性分数 (0-1)
            
        Returns:
            str: 分数解释
        """
        if score >= 0.8:
            return "高度一致 - 生成文本与源文本在事实上高度符合"
        elif score >= 0.6:
            return "较为一致 - 生成文本与源文本基本符合，存在轻微差异"
        elif score >= 0.4:
            return "部分一致 - 生成文本与源文本存在明显差异"
        elif score >= 0.2:
            return "不太一致 - 生成文本与源文本存在严重差异"
        else:
            return "严重不一致 - 生成文本与源文本在事实上严重冲突"


def demo_usage():
    """演示API使用方法"""
    # 初始化API客户端（注意：实际使用时应使用环境变量）
    api = HHEMFactualConsistencyAPI(api_key="zut_aE6JAgOK4H3CaDxnaz4rEfq8fI9FrYOJr_Es2g")
    
    print("=== HHEM事实一致性评估API演示 ===\n")
    
    # 单次评估示例
    print("1. 单次评估示例：")
    result = api.evaluate_consistency(
        generated_text="巴黎铁塔位于伦敦，是英国著名景点。",
        source_texts=[
            "埃菲尔铁塔（Eiffel Tower）坐落于法国巴黎，是法国最著名的地标之一。",
            "伦敦是英国的首都，而巴黎是法国的首都。"
        ]
    )
    
    if result.success:
        print(f"✅ 评估成功")
        print(f"📊 一致性分数: {result.score:.4f}")
        print(f"📝 分数解释: {api.interpret_score(result.score)}")
    else:
        print(f"❌ 评估失败: {result.error_message}")
    
    print("\n" + "="*50 + "\n")
    
    # 批量评估示例
    print("2. 批量评估示例：")
    batch_data = [
        {
            "generated_text": "苹果是一种蓝色的水果",
            "source_texts": ["苹果通常是红色、绿色或黄色的水果"]
        },
        {
            "generated_text": "北京是中国的首都",
            "source_texts": ["中华人民共和国的首都是北京市"]
        }
    ]
    
    batch_results = api.batch_evaluate(batch_data)
    
    for i, result in enumerate(batch_results, 1):
        print(f"评估任务 {i}:")
        if result.success:
            print(f"  ✅ 分数: {result.score:.4f} - {api.interpret_score(result.score)}")
        else:
            print(f"  ❌ 错误: {result.error_message}")


if __name__ == "__main__":
    demo_usage()
