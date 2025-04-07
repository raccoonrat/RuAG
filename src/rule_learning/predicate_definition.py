"""
谓词定义模块 - 负责自动定义目标谓词和体谓词
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional


class PredicateDefiner:
    """
    谓词定义器 - 自动定义目标谓词和体谓词
    """
    
    def __init__(self, llm_provider: Optional[Callable] = None):
        """
        初始化谓词定义器
        
        Args:
            llm_provider: 语言模型提供者，用于利用LLM的常识推理能力
        """
        self.llm_provider = llm_provider
    
    def define_predicates(self, data: pd.DataFrame, task_type: str) -> Tuple[str, List[str]]:
        """
        定义目标谓词和体谓词
        
        Args:
            data: 训练数据集
            task_type: 任务类型，如"relation_extraction"或"log_anomaly_detection"
            
        Returns:
            Tuple[str, List[str]]: (目标谓词, 体谓词列表)
        """
        if task_type == "relation_extraction":
            return self._define_predicates_for_relation_extraction(data)
        elif task_type == "log_anomaly_detection":
            return self._define_predicates_for_log_anomaly_detection(data)
        else:
            return self._define_predicates_generic(data)
    
    def _define_predicates_for_relation_extraction(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        为关系抽取任务定义谓词
        
        Args:
            data: 训练数据集
            
        Returns:
            Tuple[str, List[str]]: (目标谓词, 体谓词列表)
        """
        # 如果有标签列，使用它作为目标谓词
        if "label" in data.columns:
            target = "label"
        else:
            # 尝试找到可能的目标谓词
            possible_targets = [col for col in data.columns if "relation" in col.lower()]
            target = possible_targets[0] if possible_targets else data.columns[-1]
        
        # 体谓词是除目标谓词外的所有列
        body_predicates = [col for col in data.columns if col != target]
        
        # 如果有LLM提供者，使用LLM进一步优化谓词定义
        if self.llm_provider:
            prompt = self._build_relation_extraction_prompt(data)
            response = self.llm_provider(prompt)
            
            # 解析LLM响应，提取目标谓词和体谓词
            # 这里是简化处理，实际应用中需要更复杂的解析逻辑
            if "目标谓词" in response and "体谓词" in response:
                try:
                    target_part = response.split("目标谓词")[1].split("体谓词")[0]
                    body_part = response.split("体谓词")[1]
                    
                    target_candidates = [t.strip() for t in target_part.split(",")]
                    body_candidates = [b.strip() for b in body_part.split(",")]
                    
                    # 验证谓词是否在数据集中
                    if target_candidates and target_candidates[0] in data.columns:
                        target = target_candidates[0]
                    
                    valid_body_predicates = [b for b in body_candidates if b in data.columns]
                    if valid_body_predicates:
                        body_predicates = valid_body_predicates
                except:
                    pass
        
        return target, body_predicates
    
    def _define_predicates_for_log_anomaly_detection(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        为日志异常检测任务定义谓词
        
        Args:
            data: 训练数据集
            
        Returns:
            Tuple[str, List[str]]: (目标谓词, 体谓词列表)
        """
        # 对于日志异常检测，目标谓词通常是"异常"或"abnormal"
        target_candidates = ["异常", "abnormal", "anomaly", "label"]
        target = next((col for col in target_candidates if col in data.columns), data.columns[-1])
        
        # 体谓词是日志事件ID或特征
        body_predicates = [col for col in data.columns if col != target and (col.startswith("E") or col.startswith("event"))]
        
        # 如果没有找到合适的体谓词，使用除目标谓词外的所有列
        if not body_predicates:
            body_predicates = [col for col in data.columns if col != target]
        
        # 如果有LLM提供者，使用LLM进一步优化谓词定义
        if self.llm_provider:
            prompt = self._build_log_anomaly_detection_prompt(data)
            response = self.llm_provider(prompt)
            
            # 解析LLM响应
            # 简化处理，实际应用中需要更复杂的解析逻辑
            if "目标谓词" in response and "体谓词" in response:
                try:
                    target_part = response.split("目标谓词")[1].split("体谓词")[0]
                    body_part = response.split("体谓词")[1]
                    
                    target_candidates = [t.strip() for t in target_part.split(",")]
                    body_candidates = [b.strip() for b in body_part.split(",")]
                    
                    # 验证谓词是否在数据集中
                    if target_candidates and target_candidates[0] in data.columns:
                        target = target_candidates[0]
                    
                    valid_body_predicates = [b for b in body_candidates if b in data.columns]
                    if valid_body_predicates:
                        body_predicates = valid_body_predicates
                except:
                    pass
        
        return target, body_predicates
    
    def _define_predicates_generic(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        通用谓词定义方法
        
        Args:
            data: 训练数据集
            
        Returns:
            Tuple[str, List[str]]: (目标谓词, 体谓词列表)
        """
        # 默认使用最后一列作为目标谓词
        target = data.columns[-1]
        
        # 体谓词是除目标谓词外的所有列
        body_predicates = [col for col in data.columns if col != target]
        
        return target, body_predicates
    
    def _build_relation_extraction_prompt(self, data: pd.DataFrame) -> str:
        """
        构建关系抽取任务的提示
        
        Args:
            data: 训练数据集
            
        Returns:
            str: 提示文本
        """
        # 获取数据集的基本信息
        columns = data.columns.tolist()
        sample_data = data.head(5).to_string()
        
        prompt = f"""
        我正在处理一个关系抽取任务，需要定义目标谓词和体谓词。
        
        数据集包含以下列：
        {', '.join(columns)}
        
        以下是数据集的前几行：
        {sample_data}
        
        请帮我确定：
        1. 目标谓词（head predicate）：应该是表示实体间关系的列，通常是任务的标签或目标变量。
        2. 体谓词（body predicates）：应该是可以用来推断目标谓词的特征或属性列。
        
        请按以下格式回答：
        目标谓词：[列名]
        体谓词：[列名1], [列名2], ...
        """
        
        return prompt
    
    def _build_log_anomaly_detection_prompt(self, data: pd.DataFrame) -> str:
        """
        构建日志异常检测任务的提示
        
        Args:
            data: 训练数据集
            
        Returns:
            str: 提示文本
        """
        # 获取数据集的基本信息
        columns = data.columns.tolist()
        sample_data = data.head(5).to_string()
        
        prompt = f"""
        我正在处理一个日志异常检测任务，需要定义目标谓词和体谓词。
        
        数据集包含以下列：
        {', '.join(columns)}
        
        以下是数据集的前几行：
        {sample_data}
        
        请帮我确定：
        1. 目标谓词（head predicate）：应该是表示日志是否异常的列，通常是"异常"、"abnormal"或类似的标签。
        2. 体谓词（body predicates）：应该是可以用来推断异常的日志事件ID或特征列，通常以"E"或"event"开头。
        
        请按以下格式回答：
        目标谓词：[列名]
        体谓词：[列名1], [列名2], ...
        """
        
        return prompt