"""
RAGtruth 数据集加载器
用于加载和处理 RAGtruth 数据集，支持集成到幻觉评估系统中
"""

import json
import os
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import random


class TaskType(Enum):
    """任务类型枚举"""
    SUMMARY = "Summary"
    QA = "QA"
    ALL = "ALL"


class SplitType(Enum):
    """数据集分割类型"""
    TRAIN = "train"
    TEST = "test"
    ALL = "all"


@dataclass
class HallucinationLabel:
    """幻觉标注信息"""
    start: int  # 幻觉开始位置
    end: int    # 幻觉结束位置
    text: str   # 幻觉文本
    label_type: str  # 幻觉类型
    meta: str   # 标注者评论
    due_to_null: Optional[bool] = None  # 是否由于空值导致
    implicit_true: Optional[bool] = None  # 是否隐含正确


@dataclass
class RAGtruthResponse:
    """RAGtruth 响应数据"""
    id: str
    source_id: str
    model: str
    temperature: float
    labels: List[HallucinationLabel]
    split: str
    quality: str
    response: str
    
    @property
    def has_hallucination(self) -> bool:
        """是否包含幻觉"""
        return len(self.labels) > 0
    
    @property
    def hallucination_count(self) -> int:
        """幻觉数量"""
        return len(self.labels)


@dataclass
class RAGtruthSource:
    """RAGtruth 源信息数据"""
    source_id: str
    task_type: str
    source: str
    source_info: str
    prompt: str


@dataclass
class RAGtruthSample:
    """RAGtruth 完整样本（包含源信息和响应）"""
    source: RAGtruthSource
    response: RAGtruthResponse
    
    @property
    def generated_text(self) -> str:
        """获取生成的文本"""
        return self.response.response
    
    @property
    def source_texts(self) -> List[str]:
        """获取源文本列表"""
        source_info = self.source.source_info
        
        # 处理不同的源信息格式
        if isinstance(source_info, str):
            return [source_info]
        elif isinstance(source_info, dict):
            # 如果是字典格式，提取所有文本内容
            texts = []
            if 'passages' in source_info:
                texts.append(source_info['passages'])
            if 'question' in source_info:
                texts.append(f"Question: {source_info['question']}")
            # 如果没有找到预期的键，将整个字典转换为字符串
            if not texts:
                texts.append(str(source_info))
            return texts
        else:
            # 其他情况转换为字符串
            return [str(source_info)]
    
    @property
    def has_hallucination(self) -> bool:
        """是否包含幻觉"""
        return self.response.has_hallucination


class RAGtruthLoader:
    """RAGtruth 数据集加载器"""
    
    def __init__(self, dataset_path: str = None):
        """
        初始化数据加载器
        
        Args:
            dataset_path: RAGtruth 数据集路径，如果为None则使用项目内的默认路径
        """
        if dataset_path is None:
            # 默认使用项目内的数据路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            dataset_path = os.path.join(project_root, "data", "ragtruth")
        
        self.dataset_path = dataset_path
        self.dataset_path = dataset_path
        self.response_file = os.path.join(dataset_path, "response.jsonl")
        self.source_file = os.path.join(dataset_path, "source_info.jsonl")
        
        # 验证文件存在
        if not os.path.exists(self.response_file):
            raise FileNotFoundError(f"响应文件不存在: {self.response_file}")
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"源信息文件不存在: {self.source_file}")
        
        # 缓存数据
        self._responses_cache = None
        self._sources_cache = None
    
    def load_responses(self) -> List[RAGtruthResponse]:
        """加载所有响应数据"""
        if self._responses_cache is None:
            responses = []
            with open(self.response_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    # 解析幻觉标注
                    labels = []
                    for label_data in data.get('labels', []):
                        label = HallucinationLabel(
                            start=label_data['start'],
                            end=label_data['end'],
                            text=label_data['text'],
                            label_type=label_data['label_type'],
                            meta=label_data['meta'],
                            due_to_null=label_data.get('due_to_null'),
                            implicit_true=label_data.get('implicit_true')
                        )
                        labels.append(label)
                    
                    # 创建响应对象
                    response = RAGtruthResponse(
                        id=data['id'],
                        source_id=data['source_id'],
                        model=data['model'],
                        temperature=data['temperature'],
                        labels=labels,
                        split=data['split'],
                        quality=data['quality'],
                        response=data['response']
                    )
                    responses.append(response)
            
            self._responses_cache = responses
        
        return self._responses_cache
    
    def load_sources(self) -> List[RAGtruthSource]:
        """加载所有源信息数据"""
        if self._sources_cache is None:
            sources = []
            with open(self.source_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    source = RAGtruthSource(
                        source_id=data['source_id'],
                        task_type=data['task_type'],
                        source=data['source'],
                        source_info=data['source_info'],
                        prompt=data['prompt']
                    )
                    sources.append(source)
            
            self._sources_cache = sources
        
        return self._sources_cache
    
    def get_samples(
        self,
        task_type: TaskType = TaskType.ALL,
        split: SplitType = SplitType.ALL,
        has_hallucination: Optional[bool] = None,
        max_samples: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> List[RAGtruthSample]:
        """
        获取样本数据
        
        Args:
            task_type: 任务类型筛选
            split: 数据集分割类型筛选
            has_hallucination: 是否包含幻觉的筛选（None表示不筛选）
            max_samples: 最大样本数量
            random_seed: 随机种子
            
        Returns:
            List[RAGtruthSample]: 样本列表
        """
        responses = self.load_responses()
        sources = self.load_sources()
        
        # 创建源信息映射
        source_map = {source.source_id: source for source in sources}
        
        # 筛选响应
        filtered_responses = []
        for response in responses:
            # 任务类型筛选
            if task_type != TaskType.ALL:
                if response.source_id not in source_map:
                    continue
                source = source_map[response.source_id]
                if source.task_type != task_type.value:
                    continue
            
            # 分割类型筛选
            if split != SplitType.ALL and response.split != split.value:
                continue
            
            # 幻觉筛选
            if has_hallucination is not None:
                if response.has_hallucination != has_hallucination:
                    continue
            
            filtered_responses.append(response)
        
        # 创建样本
        samples = []
        for response in filtered_responses:
            if response.source_id in source_map:
                source = source_map[response.source_id]
                sample = RAGtruthSample(source=source, response=response)
                samples.append(sample)
        
        # 随机采样
        if max_samples and len(samples) > max_samples:
            if random_seed is not None:
                random.seed(random_seed)
            samples = random.sample(samples, max_samples)
        
        return samples
    
    def get_statistics(self) -> Dict[str, any]:
        """获取数据集统计信息"""
        responses = self.load_responses()
        sources = self.load_sources()
        
        stats = {
            "total_responses": len(responses),
            "total_sources": len(sources),
            "responses_with_hallucination": sum(1 for r in responses if r.has_hallucination),
            "responses_without_hallucination": sum(1 for r in responses if not r.has_hallucination),
            "train_samples": sum(1 for r in responses if r.split == "train"),
            "test_samples": sum(1 for r in responses if r.split == "test"),
            "models": list(set(r.model for r in responses)),
            "task_types": list(set(s.task_type for s in sources)),
            "quality_distribution": {},
            "hallucination_types": {}
        }
        
        # 质量分布统计
        quality_counts = {}
        for response in responses:
            quality_counts[response.quality] = quality_counts.get(response.quality, 0) + 1
        stats["quality_distribution"] = quality_counts
        
        # 幻觉类型统计
        hallucination_type_counts = {}
        for response in responses:
            for label in response.labels:
                hallucination_type_counts[label.label_type] = hallucination_type_counts.get(label.label_type, 0) + 1
        stats["hallucination_types"] = hallucination_type_counts
        
        return stats
    
    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_statistics()
        
        print("🔍 RAGtruth 数据集统计信息")
        print("=" * 50)
        print(f"📊 总响应数: {stats['total_responses']:,}")
        print(f"📚 总源信息数: {stats['total_sources']:,}")
        print(f"🚨 包含幻觉的响应: {stats['responses_with_hallucination']:,} ({stats['responses_with_hallucination']/stats['total_responses']*100:.1f}%)")
        print(f"✅ 无幻觉的响应: {stats['responses_without_hallucination']:,} ({stats['responses_without_hallucination']/stats['total_responses']*100:.1f}%)")
        print(f"🏋️ 训练集样本: {stats['train_samples']:,}")
        print(f"🧪 测试集样本: {stats['test_samples']:,}")
        
        print(f"\n🤖 模型分布:")
        for model in sorted(stats['models']):
            count = sum(1 for r in self.load_responses() if r.model == model)
            print(f"   {model}: {count:,}")
        
        print(f"\n📋 任务类型:")
        for task_type in sorted(stats['task_types']):
            count = sum(1 for s in self.load_sources() if s.task_type == task_type)
            print(f"   {task_type}: {count:,}")
        
        print(f"\n⚡ 质量分布:")
        for quality, count in sorted(stats['quality_distribution'].items()):
            print(f"   {quality}: {count:,}")
        
        print(f"\n🔍 幻觉类型分布:")
        for halluc_type, count in sorted(stats['hallucination_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {halluc_type}: {count:,}")


if __name__ == "__main__":
    # 测试数据加载器
    loader = RAGtruthLoader()
    loader.print_statistics()
    
    # 获取一些样本进行测试
    print(f"\n🧪 获取测试样本:")
    samples = loader.get_samples(max_samples=5, random_seed=42)
    for i, sample in enumerate(samples, 1):
        print(f"\n样本 {i}:")
        print(f"  任务类型: {sample.source.task_type}")
        print(f"  模型: {sample.response.model}")
        print(f"  包含幻觉: {sample.has_hallucination}")
        if sample.has_hallucination:
            print(f"  幻觉数量: {sample.response.hallucination_count}")
        print(f"  响应长度: {len(sample.generated_text)} 字符")
