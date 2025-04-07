"""
谓词筛选模块 - 负责移除不相关的体谓词
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
from sklearn.feature_selection import mutual_info_classif


class PredicateFilter:
    """
    谓词筛选器 - 移除不相关的体谓词
    """
    
    def __init__(self, llm_provider: Optional[Callable] = None):
        """
        初始化谓词筛选器
        
        Args:
            llm_provider: 语言模型提供者，用于利用LLM的常识推理能力
        """
        self.llm_provider = llm_provider
    
    def filter_predicates(
        self, 
        data: pd.DataFrame, 
        target: str, 
        body_predicates: List[str],
        task_type: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        筛选相关的体谓词
        
        Args:
            data: 训练数据集
            target: 目标谓词
            body_predicates: 初始体谓词列表
            task_type: 任务类型
            top_k: 保留的体谓词数量，如果为None则根据相关性阈值筛选
            
        Returns:
            List[str]: 筛选后的体谓词列表
        """
        # 使用统计方法筛选
        filtered_predicates = self._filter_by_statistics(data, target, body_predicates)
        
        # 如果有LLM提供者，使用LLM进一步筛选
        if self.llm_provider:
            filtered_predicates = self._filter_by_llm(data, target, filtered_predicates, task_type)
        
        # 如果指定了top_k，只保留前k个体谓词
        if top_k and len(filtered_predicates) > top_k:
            # 计算互信息
            X = data[filtered_predicates]
            y = data[target]
            mi_scores = mutual_info_classif(X, y)
            
            # 按互信息排序
            predicate_scores = list(zip(filtered_predicates, mi_scores))
            predicate_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 只保留前k个
            filtered_predicates = [p[0] for p in predicate_scores[:top_k]]
        
        return filtered_predicates
    
    def _filter_by_statistics(self, data: pd.DataFrame, target: str, body_predicates: List[str]) -> List[str]:
        """
        使用统计方法筛选体谓词
        
        Args:
            data: 训练数据集
            target: 目标谓词
            body_predicates: 初始体谓词列表
            
        Returns:
            List[str]: 筛选后的体谓词列表
        """
        # 计算每个体谓词与目标谓词的互信息
        X = data[body_predicates]
        y = data[target]
        mi_scores = mutual_info_classif(X, y)
        
        # 筛选互信息大于阈值的体谓词
        threshold = np.mean(mi_scores) * 0.5  # 使用平均互信息的一半作为阈值
        filtered_predicates = [pred for pred, score in zip(body_predicates, mi_scores) if score > threshold]
        
        # 如果筛选后的体谓词太少，保留原始体谓词
        if len(filtered_predicates) < 3:
            return body_predicates
        
        return filtered_predicates
    
    def _filter_by_llm(
        self, 
        data: pd.DataFrame, 
        target: str, 
        body_predicates: List[str],
        task_type: str
    ) -> List[str]:
        """
        使用LLM筛选体谓词
        
        Args:
            data: 训练数据集
            target: 目标谓词
            body_predicates: 初始体谓词列表
            task_type: 任务类型
            
        Returns:
            List[str]: 筛选后的体谓词列表
        """
        if task_type == "relation_extraction":
            prompt = self._build_relation_filtering_prompt(data, target, body_predicates)
        elif task_type == "log_anomaly_detection":
            prompt = self._build_log_filtering_prompt(data, target, body_predicates)
        else:
            prompt = self._build_generic_filtering_prompt(data, target, body_predicates)
        
        response = self.llm_provider(prompt)
        
        # 解析LLM响应，提取相关体谓词
        # 简化处理，实际应用中需要更复杂的解析逻辑
        if "相关体谓词" in response:
            try:
                relevant_part = response.split("相关体谓词")[1]
                relevant_predicates = [p.strip() for p in relevant_part.split(",")]
                
                # 验证谓词是否在原始列表中
                valid_predicates = [p for p in relevant_predicates if p in body_predicates]
                
                # 如果筛选后的体谓词太少，保留原始体谓词
                if len(valid_predicates) >= 3:
                    return valid_predicates
            except:
                pass
        
        return body_predicates
    
    def _build_relation_filtering_prompt(
        self, 
        data: pd.DataFrame, 
        target: str, 
        body_predicates: List[str]
    ) -> str:
        """
        构建关系抽取任务的筛选提示
        
        Args:
            data: 训练数据集
            target: 目标谓词
            body_predicates: 初始体谓词列表
            
        Returns:
            str: 提示文本
        """
        # 获取数据集的基本信息
        sample_data = data.head(5).to_string()
        
        prompt = f"""
        我正在处理一个关系抽取任务，需要筛选与目标谓词相关的体谓词。
        
        目标谓词（head predicate）是：{target}
        
        候选体谓词（body predicates）有：
        {', '.join(body_predicates)}
        
        以下是数据集的前几行：
        {sample_data}
        
        请帮我筛选出与目标谓词"{target}"最相关的体谓词，这些体谓词应该能够有效地推断目标谓词。
        
        请按以下格式回答：
        相关体谓词：[列名1], [列名2], ...
        """
        
        return prompt
    
    def _build_log_filtering_prompt(
        self, 
        data: pd.DataFrame, 
        target: str, 
        body_predicates: List[str]
    ) -> str:
        """
        构建日志异常检测任务的筛选提示
        
        Args:
            data: 训练数据集
            target: 目标谓词
            body_predicates: 初始体谓词列表
            
        Returns:
            str: 提示文本
        """
        # 获取数据集的基本信息
        sample_data = data.head(5).to_string()
        
        prompt = f"""
        我正在处理一个日志异常检测任务，需要筛选与异常检测相关的体谓词。
        
        目标谓词（head predicate）是：{target}
        
        候选体谓词（body predicates）有：
        {', '.join(body_predicates)}
        
        以下是数据集的前几行：
        {sample_data}
        
        请帮我筛选出与目标谓词"{target}"最相关的体谓词，这些体谓词应该能够有效地推断日志是否异常。
        
        请按以下格式回答：
        相关体谓词：[列名1], [列名2], ...
        """
        
        return prompt
    
    def _build_generic_filtering_prompt(
        self, 
        data: pd.DataFrame, 
        target: str, 
        body_predicates: List[str]
    ) -> str:
        """
        构建通用任务的筛选提示
        
        Args:
            data: 训练数据集
            target: 目标谓词
            body_predicates: 初始体谓词列表
            
        Returns:
            str: 提示文本
        """
        # 获取数据集的基本信息
        sample_data = data.head(5).to_string()
        
        prompt = f"""
        我正在处理一个机器学习任务，需要筛选与目标变量相关的特征。
        
        目标变量是：{target}
        
        候选特征有：
        {', '.join(body_predicates)}
        
        以下是数据集的前几行：
        {sample_data}
        
        请帮我筛选出与目标变量"{target}"最相关的特征，这些特征应该能够有效地预测目标变量。
        
        请按以下格式回答：
        相关体谓词：[列名1], [列名2], ...
        """
        
        return prompt