"""
规则提取器 - 负责从训练数据中提取逻辑规则
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
import os
import json

from .predicate_definition import PredicateDefiner
from .predicate_filtering import PredicateFilter
from .mcts import mcts_search, extract_rules_batch


class RuleExtractor:
    """
    规则提取器 - 从训练数据中提取逻辑规则
    """
    
    def __init__(self, llm_provider: Optional[Callable] = None):
        """
        初始化规则提取器
        
        Args:
            llm_provider: 语言模型提供者，用于谓词定义和筛选
        """
        self.llm_provider = llm_provider
        self.predicate_definer = PredicateDefiner(llm_provider)
        self.predicate_filter = PredicateFilter(llm_provider)
    
    def extract_rules(
        self, 
        data: pd.DataFrame, 
        task_type: str,
        num_rules: int = 10,
        min_precision: float = 0.5,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        从训练数据中提取规则
        
        Args:
            data: 训练数据集
            task_type: 任务类型，如"relation_extraction"或"log_anomaly_detection"
            num_rules: 提取的规则数量
            min_precision: 最小精度阈值
            save_path: 保存规则的路径，如果为None则不保存
            
        Returns:
            List[Dict[str, Any]]: 规则列表，每条规则包含body、head和accuracy
        """
        # 1. 谓词定义
        target, body_predicates = self.predicate_definer.define_predicates(data, task_type)
        
        # 2. 谓词筛选
        filtered_predicates = self.predicate_filter.filter_predicates(
            data, target, body_predicates, task_type
        )
        
        # 3. 规则搜索
        rules = extract_rules_batch(
            data, target, filtered_predicates, num_rules, min_precision
        )
        
        # 4. 添加任务类型和关键词
        for rule in rules:
            rule["task_type"] = task_type
            rule["keywords"] = self._extract_keywords(rule, task_type)
        
        # 5. 保存规则
        if save_path:
            self._save_rules(rules, save_path)
        
        return rules
    
    def _extract_keywords(self, rule: Dict[str, Any], task_type: str) -> List[str]:
        """
        从规则中提取关键词，用于后续检索
        
        Args:
            rule: 规则
            task_type: 任务类型
            
        Returns:
            List[str]: 关键词列表
        """
        keywords = []
        
        # 添加头部关键词
        if isinstance(rule["head"], str):
            keywords.extend(rule["head"].split())
        
        # 添加体部关键词
        for body_item in rule["body"]:
            if isinstance(body_item, str):
                keywords.extend(body_item.split())
        
        # 添加任务类型关键词
        if task_type == "relation_extraction":
            keywords.extend(["关系", "实体", "抽取"])
        elif task_type == "log_anomaly_detection":
            keywords.extend(["日志", "异常", "检测"])
        
        # 去重
        keywords = list(set(keywords))
        
        return keywords
    
    def _save_rules(self, rules: List[Dict[str, Any]], save_path: str) -> None:
        """
        保存规则到文件
        
        Args:
            rules: 规则列表
            save_path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存为JSON格式
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
    
    def load_rules(self, load_path: str) -> List[Dict[str, Any]]:
        """
        从文件加载规则
        
        Args:
            load_path: 加载路径
            
        Returns:
            List[Dict[str, Any]]: 规则列表
        """
        if not os.path.exists(load_path):
            return []
        
        # 从JSON文件加载规则
        with open(load_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        return rules
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据，将连续特征离散化
        
        Args:
            data: 原始数据集
            
        Returns:
            pd.DataFrame: 预处理后的数据集
        """
        processed_data = data.copy()
        
        # 处理连续特征
        for col in processed_data.columns:
            if processed_data[col].dtype in [np.float64, np.int64]:
                # 使用Gini指数离散化连续特征
                processed_data = self._discretize_by_gini(processed_data, col)
        
        return processed_data
    
    def _discretize_by_gini(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        使用Gini指数离散化连续特征
        
        Args:
            data: 数据集
            column: 要离散化的列名
            
        Returns:
            pd.DataFrame: 离散化后的数据集
        """
        # 如果列已经是布尔类型，直接返回
        if data[column].dtype == bool:
            return data
        
        # 获取唯一值并排序
        unique_values = sorted(data[column].unique())
        
        # 如果唯一值太少，直接转换为布尔特征
        if len(unique_values) <= 2:
            data[column] = data[column] == unique_values[-1]
            return data
        
        # 计算每个可能的分割点的Gini指数
        best_gini = float('inf')
        best_threshold = None
        
        for i in range(1, len(unique_values)):
            threshold = (unique_values[i-1] + unique_values[i]) / 2
            
            # 分割数据
            left = data[data[column] <= threshold]
            right = data[data[column] > threshold]
            
            # 计算Gini指数
            gini = (len(left) / len(data)) * self._calculate_gini(left) + \
                   (len(right) / len(data)) * self._calculate_gini(right)
            
            # 更新最佳分割点
            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold
        
        # 使用最佳分割点创建新的布尔特征
        new_column = f"{column}_geq_{best_threshold:.2f}"
        data[new_column] = data[column] > best_threshold
        
        # 删除原始列
        data = data.drop(column, axis=1)
        
        return data
    
    def _calculate_gini(self, data: pd.DataFrame) -> float:
        """
        计算数据集的Gini指数
        
        Args:
            data: 数据集
            
        Returns:
            float: Gini指数
        """
        # 获取最后一列作为目标变量
        target = data.iloc[:, -1]
        
        # 计算每个类别的比例
        proportions = target.value_counts(normalize=True)
        
        # 计算Gini指数
        gini = 1 - sum(p * p for p in proportions)
        
        return gini