"""
RuAG 示例数据集模块
"""
from .relation_extraction import generate_sample_data as generate_relation_data, load_data as load_relation_data
from .log_anomaly_detection import generate_sample_data as generate_log_data, load_data as load_log_data

__all__ = [
    'generate_relation_data',
    'load_relation_data',
    'generate_log_data',
    'load_log_data'
]