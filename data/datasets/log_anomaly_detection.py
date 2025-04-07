"""
日志异常检测示例数据集
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional


def generate_sample_data(n_samples: int = 100, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    生成日志异常检测示例数据集
    
    Args:
        n_samples: 样本数量
        save_path: 保存路径，如果为None则不保存
        
    Returns:
        pd.DataFrame: 生成的数据集
    """
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    
    # 创建事件ID列
    events = []
    for _ in range(n_samples):
        # 随机选择3-5个事件
        n_events = np.random.randint(3, 6)
        sample_events = np.random.choice(
            ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"], 
            n_events, 
            replace=False
        )
        events.append(", ".join(sample_events))
    
    # 创建异常标记
    # 规则1：如果日志包含E11和E7，则标记为异常
    # 规则2：如果日志包含E3、E5和E9，则标记为异常
    abnormal = []
    for event_str in events:
        if ("E11" in event_str and "E7" in event_str) or ("E3" in event_str and "E5" in event_str and "E9" in event_str):
            abnormal.append(True)
        else:
            abnormal.append(False)
    
    # 创建DataFrame
    data = pd.DataFrame({
        "events": events,
        "abnormal": abnormal
    })
    
    # 将事件拆分为独立的列
    for event in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"]:
        data[event] = data["events"].apply(lambda x: event in x)
    
    # 保存数据集
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".csv"):
            data.to_csv(save_path, index=False)
        elif save_path.endswith(".json"):
            data.to_json(save_path, orient="records", force_ascii=False)
        else:
            data.to_csv(save_path + ".csv", index=False)
    
    return data


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载日志异常检测数据集
    
    Args:
        file_path: 文件路径
        
    Returns:
        pd.DataFrame: 加载的数据集
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{file_path}")


if __name__ == "__main__":
    # 生成示例数据集并保存
    data_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(data_dir, "log_anomaly_detection_sample.csv")
    
    df = generate_sample_data(n_samples=100, save_path=save_path)
    print(f"生成了 {len(df)} 条日志异常检测示例数据，保存到 {save_path}")