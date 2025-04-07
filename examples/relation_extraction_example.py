"""
关系抽取示例 - 展示如何使用RuAG框架进行关系抽取
"""
import os
import sys
import pandas as pd
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.rule_learning import RuleExtractor, RuleTranslator, DataProcessor
from src.integration import RuleIntegrator
from src.generation import RuleAugmentedGenerator
from src.generation.llm_provider import MockLLMProvider, OpenAIProvider


def load_sample_data():
    """加载示例数据"""
    # 这里使用模拟数据，实际应用中应该加载真实数据集
    data = {
        "entity_a": ["梅西", "C罗", "内马尔", "姆巴佩", "哈兰德", "贝克汉姆", "齐达内", "罗纳尔多", "马拉多纳", "贝利"],
        "entity_b": ["巴塞罗那", "曼联", "巴黎圣日耳曼", "巴黎圣日耳曼", "曼城", "皇马", "皇马", "皇马", "那不勒斯", "桑托斯"],
        "entity_a_type": ["球员", "球员", "球员", "球员", "球员", "球员", "球员", "球员", "球员", "球员"],
        "entity_b_type": ["球队", "球队", "球队", "球队", "球队", "球队", "球队", "球队", "球队", "球队"],
        "relation": ["效力于", "效力于", "效力于", "效力于", "效力于", "效力于", "效力于", "效力于", "效力于", "效力于"]
    }
    
    return pd.DataFrame(data)


def main():
    """主函数"""
    print("=== 关系抽取示例 ===")
    
    # 1. 加载数据
    print("1. 加载数据...")
    data = load_sample_data()
    print(f"加载了 {len(data)} 条数据")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    processor = DataProcessor()
    processed_data = processor.preprocess(data)
    print("预处理完成")
    
    # 3. 规则学习
    print("\n3. 规则学习...")
    # 使用模拟LLM提供者
    llm_provider = MockLLMProvider()
    rule_extractor = RuleExtractor(llm_provider)
    
    # 提取规则
    rules = rule_extractor.extract_rules(
        processed_data,
        task_type="relation_extraction",
        num_rules=Config.MAX_RULES_PER_TASK,
        min_precision=Config.MIN_RULE_PRECISION,
        save_path=os.path.join(Config.RULES_DIR, "relation_rules.json")
    )
    
    print(f"提取了 {len(rules)} 条规则")
    
    # 4. 规则翻译
    print("\n4. 规则翻译...")
    rule_translator = RuleTranslator(llm_provider)
    translated_rules = rule_translator.translate_rules_batch(rules)
    
    print("规则示例:")
    for i, rule_text in enumerate(translated_rules[:3]):
        print(f"  规则 {i+1}: {rule_text}")
    
    # 5. 规则整合和生成
    print("\n5. 规则整合和生成...")
    rule_integrator = RuleIntegrator(rules=rules)
    generator = RuleAugmentedGenerator(llm_provider, rules=rules)
    
    # 示例查询
    query = "分析以下文本中的实体关系：'姆巴佩是巴黎圣日耳曼的球员，他在2017年加入球队。'"
    context = "姆巴佩 球员 巴黎圣日耳曼"
    
    # 生成回答
    print("\n示例查询:")
    print(query)
    
    print("\n生成回答:")
    response = generator.generate(query, context)
    print(response)
    
    # 6. 后处理（可选）
    print("\n6. 使用后处理生成回答:")
    response_with_post = generator.generate_with_post_processing(query, context)
    print(response_with_post)
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    # 确保规则目录存在
    os.makedirs(Config.RULES_DIR, exist_ok=True)
    main()