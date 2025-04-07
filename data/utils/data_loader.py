"""
数据加载工具
"""
import os
import pandas as pd
from typing import Optional, Union, Dict, Any


class DataLoader:
    """
    数据加载器
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录，如果为None则使用默认目录
        """
        if data_dir is None:
            # 使用默认数据目录
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        else:
            self.data_dir = data_dir
    
    def load_data(
        self, 
        task_type: str, 
        file_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载数据集
        
        Args:
            task_type: 任务类型，如"relation"或"log"
            file_name: 文件名，如果为None则使用默认文件名
            
        Returns:
            pd.DataFrame: 加载的数据集
        """
        if file_name is None:
            # 使用默认文件名
            if task_type == "relation" or task_type == "relation_extraction":
                file_name = "relation_extraction_sample.csv"
            elif task_type == "log" or task_type == "log_anomaly_detection":
                file_name = "log_anomaly_detection_sample.csv"
            else:
                raise ValueError(f"未知任务类型：{task_type}")
        
        # 构建文件路径
        file_path = os.path.join(self.data_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 尝试生成示例数据
            if task_type == "relation" or task_type == "relation_extraction":
                from ..datasets.relation_extraction import generate_sample_data
                data = generate_sample_data(save_path=file_path)
                return data
            elif task_type == "log" or task_type == "log_anomaly_detection":
                from ..datasets.log_anomaly_detection import generate_sample_data
                data = generate_sample_data(save_path=file_path)
                return data
            else:
                raise FileNotFoundError(f"文件不存在：{file_path}")
        
        # 加载数据
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{file_path}")
    
    def save_data(
        self, 
        data: pd.DataFrame, 
        task_type: str, 
        file_name: Optional[str] = None
    ) -> str:
        """
        保存数据集
        
        Args:
            data: 数据集
            task_type: 任务类型，如"relation"或"log"
            file_name: 文件名，如果为None则使用默认文件名
            
        Returns:
            str: 保存的文件路径
        """
        if file_name is None:
            # 使用默认文件名
            if task_type == "relation" or task_type == "relation_extraction":
                file_name = "relation_extraction_sample.csv"
            elif task_type == "log" or task_type == "log_anomaly_detection":
                file_name = "log_anomaly_detection_sample.csv"
            else:
                file_name = f"{task_type}_sample.csv"
        
        # 构建文件路径
        file_path = os.path.join(self.data_dir, file_name)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存数据
        if file_path.endswith(".csv"):
            data.to_csv(file_path, index=False)
        elif file_path.endswith(".json"):
            data.to_json(file_path, orient="records", force_ascii=False)
        else:
            # 默认保存为CSV
            data.to_csv(file_path, index=False)
        
        return file_path