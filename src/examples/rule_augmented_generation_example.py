"""
规则增强生成示例 - 展示如何使用RuAG框架
"""
import os
import json
from typing import List, Dict, Any

# 导入RuAG模块
from src.integration.rule_integrator import RuleIntegrator
from src.generation.generator import RuleAugmentedGenerator
from src.generation.llm_provider import MockLLMProvider, OpenAIProvider


def create_sample_rules() -> List[Dict[str, Any]]:
    """
    创建示例规则
    
    Returns:
        List[Dict[str, Any]]: 示例规则列表
    """
    return [
        {
            "body": ["实体A是球员", "实体B是球队"],
            "head": "实体A是实体B的成员",
            "accuracy": 0.95,
            "task_type": "relation_extraction",
            "keywords": ["球员", "球队", "成员"]
        },
        {
            "body": ["实体A是教授", "实体B是大学"],
            "head": "实体A在实体B工作",
            "accuracy": 0.90,
            "task_type": "relation_extraction",
            "keywords": ["教授", "大学", "工作"]
        },
        {
            "body": ["E11", "E28"],
            "head": "异常",
            "accuracy": 1.0,
            "task_type": "log_anomaly_detection",
            "keywords": ["日志", "异常", "E11", "E28"]
        }
    ]


def main():
    """主函数"""
    # 创建示例规则
    rules = create_sample_rules()
    
    # 保存规则到文件
    os.makedirs("data", exist_ok=True)
    with open("data/sample_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    
    # 创建模拟LLM提供者
    llm_provider = MockLLMProvider()
    
    # 创建规则增强生成器
    generator = RuleAugmentedGenerator(
        llm_provider=llm_provider,
        rules=rules,
        max_rules=3
    )
    
    # 示例1：关系抽取
    prompt1 = "请分析以下文本中的实体关系：'梅西是巴塞罗那足球俱乐部的球员。'"
    context1 = "梅西 球员 巴塞罗那 足球俱乐部"
    
    print("=== 示例1：关系抽取 ===")
    result1 = generator.generate(prompt1, context1)
    print(result1)
    print("\n")
    
    # 示例2：日志异常检测
    prompt2 = "请分析以下日志序列是否存在异常：'E1, E5, E11, E28, E30'"
    context2 = "日志 E11 E28 异常检测"
    
    print("=== 示例2：日志异常检测 ===")
    result2 = generator.generate_with_post_processing(prompt2, context2)
    print(result2)
    print("\n")
    
    # 示例3：使用OpenAI API（如果有API密钥）
    try:
        openai_provider = OpenAIProvider(model="gpt-3.5-turbo")
        openai_generator = RuleAugmentedGenerator(
            llm_provider=openai_provider,
            rules=rules,
            max_rules=3
        )
        
        print("=== 示例3：使用OpenAI API ===")
        result3 = openai_generator.generate(prompt1, context1)
        print(result3)
    except Exception as e:
        print(f"OpenAI示例失败：{e}")


if __name__ == "__main__":
    main()