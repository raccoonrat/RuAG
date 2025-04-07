"""
RuAG 规则存储模块
"""
import os
import json
from typing import List, Dict, Any, Optional


def load_rules(file_path: str) -> List[Dict[str, Any]]:
    """
    从文件加载规则
    
    Args:
        file_path: 规则文件路径
        
    Returns:
        List[Dict[str, Any]]: 规则列表
    """
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    
    return rules


def save_rules(rules: List[Dict[str, Any]], file_path: str) -> None:
    """
    保存规则到文件
    
    Args:
        rules: 规则列表
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def get_rule_file_path(task_type: str) -> str:
    """
    获取规则文件路径
    
    Args:
        task_type: 任务类型，如"relation"或"log"
        
    Returns:
        str: 规则文件路径
    """
    rules_dir = os.path.dirname(os.path.abspath(__file__))
    
    if task_type == "relation" or task_type == "relation_extraction":
        return os.path.join(rules_dir, "relation_rules.json")
    elif task_type == "log" or task_type == "log_anomaly_detection":
        return os.path.join(rules_dir, "log_rules.json")
    else:
        return os.path.join(rules_dir, f"{task_type}_rules.json")