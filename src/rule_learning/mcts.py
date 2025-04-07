# src/rule_learning/mcts.py
import numpy as np
from config import Config  # 假设 Config 包含 MCTS 参数，如 rounds 和 MAX_RULE_LENGTH

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # 当前规则（体谓词列表）
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

def select_node(node):
    """使用 UCT 策略选择节点"""
    while node.children:
        node = max(node.children, key=lambda n: n.reward / n.visits + np.sqrt(2 * np.log(node.visits) / n.visits))
    return node

def expand_node(node, body_predicates):
    """扩展节点，添加新的体谓词"""
    for predicate in body_predicates:
        if predicate not in node.state:
            new_state = node.state + [predicate]
            new_node = MCTSNode(new_state, parent=node)
            node.children.append(new_node)

def simulate(node, data, target):
    """模拟并计算奖励（规则精度）"""
    rule = node.state
    if len(rule) == 0:
        return 0
    # 计算规则覆盖的数据子集
    covered = data[np.all(data[rule], axis=1)]
    if len(covered) == 0:
        return 0
    # 计算精度（目标谓词为真的比例）
    precision = np.mean(covered[target])
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
    :param data: DataFrame，训练数据集
    :param target: 目标谓词
    :param body_predicates: 体谓词列表
    :param rounds: MCTS 迭代次数
    :return: 最佳规则（体谓词列表）
    """
    root = MCTSNode([])
    for _ in range(rounds):
        node = select_node(root)
        if len(node.state) < Config.MAX_RULE_LENGTH:
            expand_node(node, body_predicates)
            node = node.children[0]  # 选择第一个子节点
        reward = simulate(node, data, target)
        backpropagate(node, reward)
    # 选择访问价值最高的规则
    best_node = max(root.children, key=lambda n: n.reward / n.visits)
    return best_node.state
