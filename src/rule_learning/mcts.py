# src/rule_learning/mcts.py
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd

class Config:
    """MCTS配置参数"""
    MCTS_ROUNDS = 500
    MAX_RULE_LENGTH = 5
    PRECISION_THRESHOLD = 0.85
    UCT_C = 2.0  # UCT公式中的常数C

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # 当前规则（体谓词列表）
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

def select_node(node):
    """使用 UCT 策略选择节点"""
    if not node.children:
        return node
    
    # 使用UCT公式选择最有潜力的子节点
    return max(node.children, key=lambda n: n.reward / max(n.visits, 1) + 
               Config.UCT_C * np.sqrt(2 * np.log(max(node.visits, 1)) / max(n.visits, 1)))

def expand_node(node, body_predicates):
    """扩展节点，添加新的体谓词"""
    for predicate in body_predicates:
        if predicate not in node.state:
            new_state = node.state + [predicate]
            new_node = MCTSNode(new_state, parent=node)
            node.children.append(new_node)
    
    # 如果没有可添加的谓词，返回原节点
    return node.children[0] if node.children else node

def simulate(node, data, target):
    """模拟并计算奖励（规则精度）"""
    rule = node.state
    if len(rule) == 0:
        return 0
    
    # 计算规则覆盖的数据子集
    covered_indices = np.ones(len(data), dtype=bool)
    for predicate in rule:
        covered_indices = covered_indices & data[predicate].values
    
    covered = data[covered_indices]
    if len(covered) == 0:
        return 0
    
    # 计算精度（目标谓词为真的比例）
    precision = np.mean(covered[target].values)
    
    # 计算支持度（覆盖的样本比例）
    support = len(covered) / len(data)
    
    # 如果支持度太低，降低奖励
    if support < 0.01:
        precision *= support * 100
    
    return precision

def backpropagate(node, reward):
    """回溯更新节点的统计信息"""
    while node:
        node.visits += 1
        node.reward += reward
        node = node.parent

def mcts_search(data, target, body_predicates, rounds=Config.MCTS_ROUNDS):
    """
    使用 MCTS 搜索最佳规则。
    
    Args:
        data: DataFrame，训练数据集
        target: 目标谓词
        body_predicates: 体谓词列表
        rounds: MCTS 迭代次数
        
    Returns:
        Tuple[List[str], float]: (最佳规则（体谓词列表）, 规则精度)
    """
    root = MCTSNode([])
    best_rule = []
    best_precision = 0
    
    for _ in range(rounds):
        # 选择
        node = select_node(root)
        
        # 扩展
        if len(node.state) < Config.MAX_RULE_LENGTH and node.visits > 0:
            node = expand_node(node, body_predicates)
        
        # 模拟
        reward = simulate(node, data, target)
        
        # 回溯
        backpropagate(node, reward)
        
        # 更新最佳规则
        if node.state and reward > best_precision:
            best_rule = node.state
            best_precision = reward
            
            # 如果精度超过阈值，提前结束搜索
            if best_precision >= Config.PRECISION_THRESHOLD:
                break
    
    # 如果没有找到规则，返回空规则
    if not best_rule:
        return [], 0
    
    return best_rule, best_precision

def extract_rules_batch(data, target, body_predicates, num_rules=10, min_precision=0.5):
    """
    批量提取多条规则
    
    Args:
        data: DataFrame，训练数据集
        target: 目标谓词
        body_predicates: 体谓词列表
        num_rules: 提取的规则数量
        min_precision: 最小精度阈值
        
    Returns:
        List[Dict[str, Any]]: 规则列表，每条规则包含body、head和accuracy
    """
    rules = []
    remaining_data = data.copy()
    
    for _ in range(num_rules):
        if len(remaining_data) < len(data) * 0.01:  # 如果剩余数据太少，停止提取
            break
            
        rule, precision = mcts_search(remaining_data, target, body_predicates)
        
        if not rule or precision < min_precision:
            break
            
        # 将规则添加到结果列表
        rules.append({
            "body": rule,
            "head": target,
            "accuracy": float(precision)
        })
        
        # 从数据集中移除被当前规则覆盖的样本
        covered_indices = np.ones(len(remaining_data), dtype=bool)
        for predicate in rule:
            covered_indices = covered_indices & remaining_data[predicate].values
        
        remaining_data = remaining_data[~covered_indices]
    
    return rules
