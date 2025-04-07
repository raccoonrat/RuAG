"""
数据增强工具
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union


class DataAugmenter:
    """
    数据增强器
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化数据增强器
        
        Args:
            seed: 随机种子，用于确保结果可复现
        """
        if seed is not None:
            np.random.seed(seed)
    
    def augment(self, data: pd.DataFrame, task_type: str, n_samples: int) -> pd.DataFrame:
        """
        增强数据集
        
        Args:
            data: 原始数据集
            task_type: 任务类型，如"relation_extraction"或"log_anomaly_detection"
            n_samples: 增强后的样本数量
            
        Returns:
            pd.DataFrame: 增强后的数据集
        """
        if task_type == "relation_extraction":
            return self._augment_relation_data(data, n_samples)
        elif task_type == "log_anomaly_detection":
            return self._augment_log_data(data, n_samples)
        else:
            return self._augment_generic_data(data, n_samples)
    
    def _augment_relation_data(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        增强关系抽取数据
        
        Args:
            data: 原始数据集
            n_samples: 增强后的样本数量
            
        Returns:
            pd.DataFrame: 增强后的数据集
        """
        # 如果原始数据集已经足够大，直接返回
        if len(data) >= n_samples:
            return data
        
        # 获取实体和关系
        entity_a_list = data["entity_a"].unique().tolist()
        entity_b_list = data["entity_b"].unique().tolist()
        
        # 创建新样本
        new_samples = []
        while len(new_samples) < n_samples - len(data):
            # 随机选择实体
            entity_a = np.random.choice(entity_a_list)
            entity_b = np.random.choice(entity_b_list)
            
            # 获取实体类型
            entity_a_type = data[data["entity_a"] == entity_a]["entity_a_type"].iloc[0]
            entity_b_type = data[data["entity_b"] == entity_b]["entity_b_type"].iloc[0]
            
            # 设置关系
            relation = "效力于"  # 简化处理，实际应用中可能需要更复杂的逻辑
            
            # 添加到新样本
            new_samples.append({
                "entity_a": entity_a,
                "entity_b": entity_b,
                "entity_a_type": entity_a_type,
                "entity_b_type": entity_b_type,
                "relation": relation
            })
        
        # 创建新的DataFrame
        new_df = pd.DataFrame(new_samples)
        
        # 合并原始数据集和新样本
        augmented_data = pd.concat([data, new_df], ignore_index=True)
        
        return augmented_data
    
    def _augment_log_data(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        增强日志异常检测数据
        
        Args:
            data: 原始数据集
            n_samples: 增强后的样本数量
            
        Returns:
            pd.DataFrame: 增强后的数据集
        """
        # 如果原始数据集已经足够大，直接返回
        if len(data) >= n_samples:
            return data
        
        # 获取事件列
        event_columns = [col for col in data.columns if col.startswith("E")]
        
        # 创建新样本
        new_samples = []
        while len(new_samples) < n_samples - len(data):
            # 随机选择3-5个事件
            n_events = np.random.randint(3, 6)
            selected_events = np.random.choice(event_columns, n_events, replace=False)
            
            # 创建新样本
            new_sample = {col: False for col in event_columns}
            for event in selected_events:
                new_sample[event] = True
            
            # 根据规则设置异常标记
            # 规则1：如果日志包含E11和E7，则标记为异常
            # 规则2：如果日志包含E3、E5和E9，则标记为异常
            if (new_sample.get("E11", False) and new_sample.get("E7", False)) or \
               (new_sample.get("E3", False) and new_sample.get("E5", False) and new_sample.get("E9", False)):
                new_sample["abnormal"] = True
            else:
                new_sample["abnormal"] = False
            
            # 添加到新样本
            new_samples.append(new_sample)
        
        # 创建新的DataFrame
        new_df = pd.DataFrame(new_samples)
        
        # 合并原始数据集和新样本
        augmented_data = pd.concat([data, new_df], ignore_index=True)
        
        return augmented_data
    
    def _augment_generic_data(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        通用数据增强
        
        Args:
            data: 原始数据集
            n_samples: 增强后的样本数量
            
        Returns:
            pd.DataFrame: 增强后的数据集
        """
        # 如果原始数据集已经足够大，直接返回
        if len(data) >= n_samples:
            return data
        
        # 简单的过采样
        augmented_data = data.sample(n_samples, replace=True)
        
        return augmented_data