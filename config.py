"""
RuAG 全局配置文件
"""

class Config:
    """全局配置参数"""
    
    # MCTS 参数
    MCTS_ROUNDS = 500  # MCTS 搜索轮数
    MAX_RULE_LENGTH = 5  # 规则最大长度
    PRECISION_THRESHOLD = 0.85  # 规则精度阈值
    UCT_C = 2.0  # UCT公式中的常数C
    
    # 规则学习参数
    MIN_RULE_PRECISION = 0.5  # 最小规则精度
    MAX_RULES_PER_TASK = 20  # 每个任务最大规则数量
    
    # 规则整合参数
    MAX_RULES_IN_PROMPT = 5  # 提示中最大规则数量
    
    # 生成参数
    DEFAULT_TEMPERATURE = 0.7  # 默认生成温度
    MAX_TOKENS = 1000  # 最大生成token数
    
    # 后处理参数
    MAX_ADJUSTMENT_ATTEMPTS = 3  # 最大调整尝试次数
    
    # 路径配置
    RULES_DIR = "data/rules"  # 规则存储目录
    EXAMPLES_DIR = "examples"  # 示例目录
    
    # LLM配置
    DEFAULT_MODEL = "gpt-3.5-turbo"  # 默认模型
    ADVANCED_MODEL = "gpt-4"  # 高级模型