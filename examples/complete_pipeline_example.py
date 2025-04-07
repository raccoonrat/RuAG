"""
完整流程示例 - 展示RuAG框架的完整工作流程
"""
import os
import sys
import pandas as pd
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.rule_learning import RuleExtractor, RuleTranslator, DataProcessor
from src.integration import RuleIntegrator
from src.generation import RuleAugmentedGenerator
from src.post_processing import RuleValidator, TextAdjuster
from src.generation.llm_provider import MockLLMProvider, OpenAIProvider


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RuAG完整流程示例")
    parser.add_argument("--task", type=str, default="relation", choices=["relation", "log"],
                        help="任务类型：relation（关系抽取）或log（日志异常检测）")
    parser.add_argument("--model", type=str, default="mock", choices=["mock", "gpt-3.5-turbo", "gpt-4"],
                        help="使用的语言模型")
    parser.add_argument("--data", type=str, default=None,
                        help="数据文件路径，如果不提供则使用示例数据")
    parser.add_argument("--rules", type=str, default=None,
                        help="规则文件路径，如果不提供则从数据中学习规则")
    parser.add_argument("--query", type=str, default=None,
                        help="查询文本，如果不提供则使用示例查询")
    
    return parser.parse_args()


def get_llm_provider(model_name):
    """获取语言模型提供者"""
    if model_name == "mock":
        return MockLLMProvider()
    elif model_name in ["gpt-3.5-turbo", "gpt-4"]:
        try:
            return OpenAIProvider(model=model_name)
        except Exception as e:
            print(f"OpenAI API初始化失败：{e}")
            print("使用模拟LLM提供者代替")
            return MockLLMProvider()
    else:
        print(f"未知模型：{model_name}，使用模拟LLM提供者")
        return MockLLMProvider()


def load_data(task_type, data_path=None):
    """加载数据"""
    if data_path and os.path.exists(data_path):
        # 从文件加载数据
        if data_path.endswith(".csv"):
            return pd.read_csv(data_path)
        elif data_path.endswith(".json"):
            return pd.read_json(data_path)
        else:
            print(f"不支持的文件格式：{data_path}")
            return None
    
    # 使用示例数据
    if task_type == "relation":
        from relation_extraction_example import load_sample_data
        return load_sample_data()
    elif task_type == "log":
        from log_anomaly_detection_example import load_sample_log_data
        data = load_sample_log_data()
        return data.drop("events", axis=1)
    else:
        print(f"未知任务类型：{task_type}")
        return None


def load_rules(rules_path=None):
    """加载规则"""
    if rules_path and os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_example_query(task_type):
    """获取示例查询"""
    if task_type == "relation":
        return "分析以下文本中的实体关系：'梅西是巴塞罗那足球俱乐部的球员。'", "梅西 球员 巴塞罗那 足球俱乐部"
    elif task_type == "log":
        return "分析以下日志序列是否存在异常：'E1, E5, E7, E11, E12'", "日志 E1 E5 E7 E11 E12 异常检测"
    else:
        return "请生成回答", ""


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    print(f"=== RuAG完整流程示例：{args.task} ===")
    
    # 1. 初始化LLM提供者
    print("1. 初始化LLM提供者...")
    llm_provider = get_llm_provider(args.model)
    print(f"使用模型：{args.model}")
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    data = load_data(args.task, args.data)
    if data is None:
        print("数据加载失败，退出")
        return
    print(f"加载了 {len(data)} 条数据")
    
    # 3. 数据预处理
    print("\n3. 数据预处理...")
    processor = DataProcessor()
    processed_data = processor.preprocess(data)
    print("预处理完成")
    
    # 4. 规则学习或加载
    print("\n4. 规则学习或加载...")
    rules = load_rules(args.rules)
    
    if rules:
        print(f"从文件加载了 {len(rules)} 条规则")
    else:
        # 学习规则
        rule_extractor = RuleExtractor(llm_provider)
        task_type = "relation_extraction" if args.task == "relation" else "log_anomaly_detection"
        
        rules = rule_extractor.extract_rules(
            processed_data,
            task_type=task_type,
            num_rules=Config.MAX_RULES_PER_TASK,
            min_precision=Config.MIN_RULE_PRECISION,
            save_path=os.path.join(Config.RULES_DIR, f"{args.task}_rules.json")
        )
        
        print(f"学习了 {len(rules)} 条规则")
    
    # 5. 规则翻译
    print("\n5. 规则翻译...")
    rule_translator = RuleTranslator(llm_provider)
    translated_rules = rule_translator.translate_rules_batch(rules)
    
    print("规则示例:")
    for i, rule_text in enumerate(translated_rules[:3]):
        print(f"  规则 {i+1}: {rule_text}")
    
    # 6. 规则整合
    print("\n6. 规则整合...")
    rule_integrator = RuleIntegrator(rules=rules)
    
    # 7. 生成
    print("\n7. 生成...")
    generator = RuleAugmentedGenerator(llm_provider, rules=rules)
    
    # 获取查询
    query = args.query
    if not query:
        query, context = get_example_query(args.task)
    else:
        context = query  # 简单地使用查询作为上下文
    
    print("\n查询:")
    print(query)
    
    # 生成回答
    print("\n生成回答:")
    response = generator.generate(query, context)
    print(response)
    
    # 8. 后处理
    print("\n8. 后处理...")
    validator = RuleValidator(rules)
    adjuster = TextAdjuster(llm_provider, rules, validator)
    
    # 验证生成的文本
    is_valid, violations = validator.validate(response, context, rules)
    
    if is_valid:
        print("生成的文本符合所有规则约束")
    else:
        print(f"生成的文本违反了 {len(violations)} 条规则约束")
        violation_details = validator.get_violation_details(violations)
        for i, detail in enumerate(violation_details):
            print(f"  违规 {i+1}: {detail}")
        
        # 调整文本
        print("\n调整后的回答:")
        adjusted_response = adjuster.adjust(response, context, query)
        print(adjusted_response)
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    # 确保规则目录存在
    os.makedirs(Config.RULES_DIR, exist_ok=True)
    main()