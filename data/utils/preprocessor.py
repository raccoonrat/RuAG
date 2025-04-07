"""
数据预处理工具
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    数据预处理器
    """
    
    def __init__(self):
        """
        初始化数据预处理器
        """
        self.label_encoders = {}
        self.scalers = {}
    
    def preprocess(self, data: pd.DataFrame, task_type: str) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            data: 原始数据集
            task_type: 任务类型，如"relation_extraction"或"log_anomaly_detection"
            
        Returns:
            pd.DataFrame: 预处理后的数据集
        """
        if task_type == "relation_extraction":
            return self._preprocess_relation_data(data)
        elif task_type == "log_anomaly_detection":
            return self._preprocess_log_data(data)
        else:
            return self._preprocess_generic_data(data)
    
    def _preprocess_relation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理关系抽取数据
        
        Args:
            data: 原始数据集
            
        Returns:
            pd.DataFrame: 预处理后的数据集
        """
        processed_data = data.copy()
        
        # 编码实体类型
        for col in ["entity_a_type", "entity_b_type"]:
            if col in processed_data.columns:
                processed_data = self._encode_categorical(processed_data, col)
        
        # 编码关系
        if "relation" in processed_data.columns:
            processed_data = self._encode_categorical(processed_data, "relation")
        
        # 编码实体
        for col in ["entity_a", "entity_b"]:
            if col in processed_data.columns:
                processed_data = self._encode_categorical(processed_data, col)
        
        return processed_data
    
    def _preprocess_log_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理日志异常检测数据
        
        Args:
            data: 原始数据集
            
        Returns:
            pd.DataFrame: 预处理后的数据集
        """
        processed_data = data.copy()
        
        # 如果有events列，将其拆分为独立的事件列
        if "events" in processed_data.columns:
            # 获取所有可能的事件ID
            all_events = set()
            for events_str in processed_data["events"]:
                events = events_str.split(", ")
                all_events.update(events)
            
            # 为每个事件创建一个布尔列
            for event in sorted(all_events):
                processed_data[event] = processed_data["events"].apply(lambda x: event in x)
            
            # 删除原始events列
            processed_data = processed_data.drop("events", axis=1)
        
        # 确保异常标记为布尔值
        if "abnormal" in processed_data.columns and processed_data["abnormal"].dtype != bool:
            processed_data["abnormal"] = processed_data["abnormal"].astype(bool)
        
        return processed_data
    
    def _preprocess_generic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        通用数据预处理
        
        Args:
            data: 原始数据集
            
        Returns:
            pd.DataFrame: 预处理后的数据集
        """
        processed_data = data.copy()
        
        # 处理分类特征
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                processed_data = self._encode_categorical(processed_data, col)
        
        # 处理数值特征
        for col in processed_data.columns:
            if processed_data[col].dtype in [np.float64, np.int64] and col != processed_data.columns[-1]:
                processed_data = self._normalize_numerical(processed_data, col)
        
        return processed_data
    
    def _encode_categorical(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            data: 数据集
            column: 要编码的列名
            
        Returns:
            pd.DataFrame: 编码后的数据集
        """
        # 如果列的唯一值只有两个，直接转换为布尔值
        if len(data[column].unique()) == 2:
            unique_values = sorted(data[column].unique())
            data[column] = data[column] == unique_values[1]
            return data
        
        # 否则，使用独热编码
        # 创建标签编码器
        le = LabelEncoder()
        le.fit(data[column])
        self.label_encoders[column] = le
        
        # 编码数据
        encoded = le.transform(data[column])
        
        # 为每个类别创建一个布尔特征
        for i, category in enumerate(le.classes_):
            new_column = f"{column}_{category}"
            data[new_column] = encoded == i
        
        # 删除原始列
        data = data.drop(column, axis=1)
        
        return data
    
    def _normalize_numerical(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        归一化数值特征
        
        Args:
            data: 数据集
            column: 要归一化的列名
            
        Returns:
            pd.DataFrame: 归一化后的数据集
        """
        # 创建标准化器
        scaler = StandardScaler()
        
        # 归一化数据
        data[column] = scaler.fit_transform(data[[column]])
        self.scalers[column] = scaler
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        将预处理后的数据转换回原始形式
        
        Args:
            data: 预处理后的数据集
            
        Returns:
            pd.DataFrame: 转换后的数据集
        """
        # 这个方法在实际应用中可能需要更复杂的逻辑
        # 这里只是一个简单的示例
        return data