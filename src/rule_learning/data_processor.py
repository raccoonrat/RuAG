"""
数据处理模块 - 负责预处理训练数据
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """
    数据处理器 - 预处理训练数据
    """
    
    def __init__(self):
        """
        初始化数据处理器
        """
        self.label_encoders = {}
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
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
        
        # 处理连续特征
        for col in processed_data.columns:
            if processed_data[col].dtype in [np.float64, np.int64]:
                processed_data = self._discretize_continuous(processed_data, col)
        
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
    
    def _discretize_continuous(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        离散化连续特征
        
        Args:
            data: 数据集
            column: 要离散化的列名
            
        Returns:
            pd.DataFrame: 离散化后的数据集
        """
        # 如果列已经是布尔类型，直接返回
        if data[column].dtype == bool:
            return data
        
        # 计算分位数
        q1 = data[column].quantile(0.25)
        q2 = data[column].quantile(0.5)
        q3 = data[column].quantile(0.75)
        
        # 创建布尔特征
        data[f"{column}_lt_{q1:.2f}"] = data[column] < q1
        data[f"{column}_lt_{q2:.2f}"] = data[column] < q2
        data[f"{column}_lt_{q3:.2f}"] = data[column] < q3
        data[f"{column}_geq_{q3:.2f}"] = data[column] >= q3
        
        # 删除原始列
        data = data.drop(column, axis=1)
        
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