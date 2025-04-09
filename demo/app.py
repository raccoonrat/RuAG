from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.llm_provider import MockLLMProvider, OpenAIProvider, VolcArkDeepSeekProvider
from src.rule_learning.rule_extractor import RuleExtractor
from src.rule_learning.rule_translator import RuleTranslator
from src.integration.rule_integrator import RuleIntegrator
from src.generation.rule_augmented_generator import RuleAugmentedGenerator
from src.post_processing.rule_validator import RuleValidator
from src.post_processing.text_adjuster import TextAdjuster

app = Flask(__name__, static_folder='.')

# 模型提供者映射
MODEL_PROVIDERS = {
    'mock': MockLLMProvider(),
    'gpt-3.5-turbo': lambda: OpenAIProvider(model="gpt-3.5-turbo"),
    'gpt-4': lambda: OpenAIProvider(model="gpt-4"),
    'volc-ark-deepseek': lambda: VolcArkDeepSeekProvider()
}

# 示例数据
EXAMPLE_DATA = {
    'relation': [
        {"text": "哈兰德是曼城足球俱乐部的前锋。", "entities": ["哈兰德", "曼城足球俱乐部"], "relation": "效力于"},
        {"text": "姆巴佩是巴黎圣日耳曼足球俱乐部的球员。", "entities": ["姆巴佩", "巴黎圣日耳曼足球俱乐部"], "relation": "效力于"},
        {"text": "梅西曾经效力于巴塞罗那足球俱乐部。", "entities": ["梅西", "巴塞罗那足球俱乐部"], "relation": "效力于"}
    ],
    'log': [
        {"sequence": ["系统启动", "数据库连接失败", "重试连接", "连接成功"], "anomaly": True},
        {"sequence": ["系统启动", "用户登录", "查询数据", "用户登出"], "anomaly": False},
        {"sequence": ["系统启动", "数据库连接失败", "数据库连接失败", "数据库连接失败", "系统崩溃"], "anomaly": True}
    ]
}

# 示例规则
EXAMPLE_RULES = {
    'relation': [
        {"condition": "entity_a_哈兰德", "prediction": "relation_效力于", "accuracy": 1.0},
        {"condition": "entity_a_姆巴佩", "prediction": "relation_效力于", "accuracy": 1.0},
        {"condition": "entity_a_梅西", "prediction": "relation_效力于", "accuracy": 1.0}
    ],
    'log': [
        {"condition": "contains_ERROR & next_contains_retry", "prediction": "anomaly_minor", "accuracy": 0.95},
        {"condition": "ERROR_count >= 3", "prediction": "anomaly_major", "accuracy": 0.98}
    ]
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/run', methods=['POST'])
def run_demo():
    data = request.json
    task = data.get('task', 'relation')
    model_name = data.get('model', 'mock')
    query = data.get('query', '')
    
    try:
        # 1. 初始化LLM提供者
        if model_name in MODEL_PROVIDERS:
            llm_provider = MODEL_PROVIDERS[model_name]() if callable(MODEL_PROVIDERS[model_name]) else MODEL_PROVIDERS[model_name]
        else:
            return jsonify({"error": f"不支持的模型: {model_name}"}), 400
        
        # 2. 获取示例数据
        data = EXAMPLE_DATA.get(task, [])
        
        # 3. 获取示例规则
        rules = EXAMPLE_RULES.get(task, [])
        
        # 4. 规则翻译
        translator = RuleTranslator(llm_provider)
        translated_rules = []
        for rule in rules:
            translated = translator.translate(rule)
            translated_rules.append(translated)
        
        # 5. 规则整合
        integrator = RuleIntegrator(rules)
        context = integrator.integrate(query)
        
        # 6. 生成
        generator = RuleAugmentedGenerator(llm_provider)
        original_response = llm_provider(query)
        enhanced_response = generator.generate(query, context)
        
        # 7. 规则验证
        validator = RuleValidator(rules)
        is_valid, violations = validator.validate(enhanced_response, context)
        
        # 8. 返回结果
        return jsonify({
            "task": task,
            "model": model_name,
            "query": query,
            "rules": translated_rules,
            "original_response": original_response,
            "enhanced_response": enhanced_response,
            "is_valid": is_valid,
            "violations": violations
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)