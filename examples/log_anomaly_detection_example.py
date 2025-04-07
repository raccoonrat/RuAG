"""
日志异常检测示例 - 展示如何使用RuAG框架进行日志异常检测
"""
import os
import sys
import pandas as pd
import numpy as np
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.rule_learning import RuleExtractor, RuleTranslator, DataProcessor
from src.integration import RuleIntegrator
from src.generation import RuleAugmentedGenerator
from src.generation.llm_provider import MockLLMProvider, OpenAIProvider


def load_sample_log_data():
    """加载示例日志数据"""
    # 这里使用模拟数据，实际应用中应该加载真实日志数据集
    # 创建一个简单的日志数据集，包含事件ID和异常标记
    np.random.seed(42)
    n_samples = 100
    
    # 创建事件ID列
    events = []
    for _ in range(n_samples):
        # 随机选择3-5个事件
        n_events = np.random.randint(3, 6)
        sample_events = np.random.choice(["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"], n_events, replace=False)
        events.append(", ".join(sample_events))
    
    # 创建异常标记
    # 规则：如果日志包含E11和E7，则标记为异常
    abnormal = []
    for event_str in events:
        if "E11" in event_str and "E7" in event_str:
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
    
    return data


def main():
    """主函数"""
    print("=== 日志异常检测示例 ===")
    
    # 1. 加载数据
    print("1. 加载数据...")
    data = load_sample_log_data()
    print(f"加载了 {len(data)} 条日志数据")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    processor = DataProcessor()
    # 删除events列，只保留独立的事件列和异常标记
    processed_data = data.drop("events", axis=1)
    print("预处理完成")
    
    # 3. 规则学习
    print("\n3. 规则学习...")
    # 使用模拟LLM提供者
    llm_provider = MockLLMProvider()
    rule_extractor = RuleExtractor(llm_provider)
    
    # 提取规则
    rules = rule_extractor.extract_rules(
        processed_data,
        task_type="log_anomaly_detection",
        num_rules=Config.MAX_RULES_PER_TASK,
        min_precision=Config.MIN_RULE_PRECISION,
        save_path=os.path.join(Config.RULES_DIR, "log_rules.json")
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
    query = "分析以下日志序列是否存在异常：'E1, E5, E7, E11, E12'"
    context = "日志 E1 E5 E7 E11 E12 异常检测"
    
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