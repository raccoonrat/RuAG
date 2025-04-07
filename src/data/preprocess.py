# src/data/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path, task_type="classification"):
    """
    预处理数据：加载、清洗、转换。
    :param input_path: 原始数据路径
    :param output_path: 处理后数据路径
    :param task_type: 任务类型（"classification" 或 "regression"）
    :return: 处理后的 DataFrame
    """
    # 加载数据
    df = pd.read_csv(input_path)

    # 处理缺失值（示例：删除含 NaN 的行）
    df.dropna(inplace=True)

    # 标签编码（仅分类任务）
    if task_type == "classification":
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])

    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    return df
