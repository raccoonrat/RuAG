"""
关系抽取示例数据集
"""
import pandas as pd
import os
import json
from typing import Dict, List, Any, Optional


def generate_sample_data(n_samples: int = 100, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    生成关系抽取示例数据集
    
    Args:
        n_samples: 样本数量
        save_path: 保存路径，如果为None则不保存
        
    Returns:
        pd.DataFrame: 生成的数据集
    """
    # 定义实体和关系
    person_entities = [
        "梅西", "C罗", "内马尔", "姆巴佩", "哈兰德", "贝克汉姆", "齐达内", "罗纳尔多", "马拉多纳", "贝利",
        "莱万多夫斯基", "苏亚雷斯", "本泽马", "莫德里奇", "克罗斯", "德布劳内", "萨拉赫", "孙兴慜", "凯恩", "斯特林"
    ]
    
    team_entities = [
        "巴塞罗那", "曼联", "巴黎圣日耳曼", "皇马", "曼城", "拜仁慕尼黑", "利物浦", "切尔西", "尤文图斯", "AC米兰",
        "国际米兰", "阿森纳", "多特蒙德", "马德里竞技", "那不勒斯", "桑托斯", "托特纳姆热刺", "阿贾克斯", "波尔图", "本菲卡"
    ]
    
    # 生成数据
    data = []
    for _ in range(n_samples):
        # 随机选择实体
        import random
        person = random.choice(person_entities)
        team = random.choice(team_entities)
        
        # 设置关系
        relation = "效力于"
        
        # 添加到数据集
        data.append({
            "entity_a": person,
            "entity_b": team,
            "entity_a_type": "球员",
            "entity_b_type": "球队",
            "relation": relation
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据集
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".csv"):
            df.to_csv(save_path, index=False)
        elif save_path.endswith(".json"):
            df.to_json(save_path, orient="records", force_ascii=False)
        else:
            df.to_csv(save_path + ".csv", index=False)
    
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载关系抽取数据集
    
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
    save_path = os.path.join(data_dir, "relation_extraction_sample.csv")
    
    df = generate_sample_data(n_samples=100, save_path=save_path)
    print(f"生成了 {len(df)} 条关系抽取示例数据，保存到 {save_path}")