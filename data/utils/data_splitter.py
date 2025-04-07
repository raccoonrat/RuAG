"""
数据分割工具
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    数据分割器
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化数据分割器
        
        Args:
            seed: 随机种子，用于确保结果可复现
        """
        self.seed = seed
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        test_size: float = 0.2, 
        val_size: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            data: 数据集
            test_size: 测试集比例
            val_size: 验证集比例，如果为None则不创建验证集
            
        Returns:
            Dict[str, pd.DataFrame]: 分割后的数据集，包含"train"、"val"和"test"
        """
        # 如果验证集比例为None，则不创建验证集
        if val_size is None:
            # 分割为训练集和测试集
            train_data, test_data = train_test_split(
                data, test_size=test_size, random_state=self.seed
            )
            
            return {
                "train": train_data,
                "test": test_data
            }
        else:
            # 计算训练集比例
            train_size = 1.0 - test_size - val_size
            
            # 分割为训练集、验证集和测试集
            train_data, temp_data = train_test_split(
                data, test_size=test_size + val_size, random_state=self.seed
            )
            
            val_data, test_data = train_test_split(
                temp_data, test_size=test_size / (test_size + val_size), random_state=self.seed
            )
            
            return {
                "train": train_data,
                "val": val_data,
                "test": test_data
            }
    
    def k_fold_split(
        self, 
        data: pd.DataFrame, 
        n_folds: int = 5
    ) -> List[Dict[str, pd.DataFrame]]:
        """
        K折交叉验证分割
        
        Args:
            data: 数据集
            n_folds: 折数
            
        Returns:
            List[Dict[str, pd.DataFrame]]: 分割后的数据集列表，每个元素包含"train"和"test"
        """
        # 创建索引数组
        indices = np.arange(len(data))
        
        # 设置随机种子
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # 打乱索引
        np.random.shuffle(indices)
        
        # 计算每折的大小
        fold_size = len(data) // n_folds
        
        # 分割数据
        folds = []
        for i in range(n_folds):
            # 计算测试集索引
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else len(data)
            test_indices = indices[start:end]
            
            # 计算训练集索引
            train_indices = np.setdiff1d(indices, test_indices)
            
            # 创建训练集和测试集
            train_data = data.iloc[train_indices].reset_index(drop=True)
            test_data = data.iloc[test_indices].reset_index(drop=True)
            
            # 添加到结果列表
            folds.append({
                "train": train_data,
                "test": test_data
            })
        
        return folds