"""
数据评估工具
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DataEvaluator:
    """
    数据评估器
    """
    
    def __init__(self):
        """
        初始化数据评估器
        """
        pass
    
    def evaluate_rule(
        self, 
        data: pd.DataFrame, 
        rule: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        评估规则在数据集上的性能
        
        Args:
            data: 数据集
            rule: 规则，包含body和head
            
        Returns:
            Dict[str, float]: 评估结果，包含精度、召回率、F1值等
        """
        # 计算规则覆盖的样本
        covered_indices = np.ones(len(data), dtype=bool)
        for predicate in rule["body"]:
            if isinstance(predicate, str):
                # 处理字符串形式的谓词
                try:
                    covered_indices = covered_indices & eval(f"data[{predicate}]")
                except:
                    # 如果谓词是列名，直接使用
                    if predicate in data.columns:
                        covered_indices = covered_indices & data[predicate].values
            else:
                # 处理列名形式的谓词
                covered_indices = covered_indices & data[predicate].values
        
        # 获取覆盖的样本
        covered = data[covered_indices]
        
        # 如果没有覆盖任何样本，返回零值
        if len(covered) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "support": 0.0
            }
        
        # 计算头部谓词的真实值
        head = rule["head"]
        if isinstance(head, str):
            # 处理字符串形式的头部谓词
            try:
                y_true = eval(f"covered[{head}]")
            except:
                # 如果头部谓词是列名，直接使用
                if head in covered.columns:
                    y_true = covered[head].values
                else:
                    # 处理否定形式，如"not abnormal"
                    if head.startswith("not ") and head[4:] in covered.columns:
                        y_true = ~covered[head[4:]].values
                    else:
                        raise ValueError(f"无法解析头部谓词：{head}")
        else:
            # 处理列名形式的头部谓词
            y_true = covered[head].values
        
        # 计算预测值（规则预测的结果总是True）
        y_pred = np.ones_like(y_true, dtype=bool)
        
        # 计算评估指标
        try:
            precision = precision_score(y_true, y_pred)
        except:
            precision = np.mean(y_true == y_pred)
        
        try:
            recall = recall_score(y_true, y_pred)
        except:
            recall = np.sum(y_true & y_pred) / np.sum(y_true)
        
        try:
            f1 = f1_score(y_true, y_pred)
        except:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        try:
            accuracy = accuracy_score(y_true, y_pred)
        except:
            accuracy = np.mean(y_true == y_pred)
        
        # 计算支持度（覆盖的样本比例）
        support = len(covered) / len(data)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "support": float(support)
        }
    
    def evaluate_rules(
        self, 
        data: pd.DataFrame, 
        rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        评估多条规则在数据集上的性能
        
        Args:
            data: 数据集
            rules: 规则列表
            
        Returns:
            List[Dict[str, Any]]: 评估结果列表
        """
        results = []
        
        for rule in rules:
            # 评估规则
            metrics = self.evaluate_rule(data, rule)
            
            # 添加规则信息
            result = {
                "rule": rule,
                "metrics": metrics
            }
            
            results.append(result)
        
        return results
    
    def evaluate_rule_set(
        self, 
        data: pd.DataFrame, 
        rules: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        评估规则集在数据集上的整体性能
        
        Args:
            data: 数据集
            rules: 规则列表
            
        Returns:
            Dict[str, float]: 整体评估结果
        """
        # 如果没有规则，返回零值
        if not rules:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "coverage": 0.0
            }
        
        # 获取所有规则覆盖的样本
        covered_indices = np.zeros(len(data), dtype=bool)
        
        for rule in rules:
            # 计算规则覆盖的样本
            rule_covered = np.ones(len(data), dtype=bool)
            for predicate in rule["body"]:
                if isinstance(predicate, str):
                    # 处理字符串形式的谓词
                    try:
                        rule_covered = rule_covered & eval(f"data[{predicate}]")
                    except:
                        # 如果谓词是列名，直接使用
                        if predicate in data.columns:
                            rule_covered = rule_covered & data[predicate].values
                else:
                    # 处理列名形式的谓词
                    rule_covered = rule_covered & data[predicate].values
            
            # 更新覆盖的样本
            covered_indices = covered_indices | rule_covered
        
        # 获取覆盖的样本
        covered = data[covered_indices]
        
        # 如果没有覆盖任何样本，返回零值
        if len(covered) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "coverage": 0.0
            }
        
        # 计算整体评估指标
        # 这里简化处理，实际应用中可能需要更复杂的逻辑
        rule_metrics = [self.evaluate_rule(data, rule) for rule in rules]
        
        precision = np.mean([m["precision"] for m in rule_metrics])
        recall = np.mean([m["recall"] for m in rule_metrics])
        f1 = np.mean([m["f1"] for m in rule_metrics])
        accuracy = np.mean([m["accuracy"] for m in rule_metrics])
        
        # 计算覆盖率（被规则集覆盖的样本比例）
        coverage = len(covered) / len(data)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "coverage": float(coverage)
        }