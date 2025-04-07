"""
规则整合机制模块 - 负责将学习到的逻辑规则与语言模型的生成过程结合
"""
import os
import json
from typing import List, Dict, Any, Optional


class RuleIntegrator:
    """
    规则整合器 - 将逻辑规则整合到语言模型的提示中
    """
    
    def __init__(self, rules_path: Optional[str] = None, rules: Optional[List[Dict[str, Any]]] = None):
        """
        初始化规则整合器
        
        Args:
            rules_path: 规则文件路径，JSON格式
            rules: 直接提供的规则列表
        """
        self.rules = []
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
        elif rules:
            self.rules = rules
    
    def translate_rule_to_natural_language(self, rule: Dict[str, Any]) -> str:
        """
        将逻辑规则翻译为自然语言
        
        Args:
            rule: 逻辑规则，格式为 {"body": [...], "head": "...", "accuracy": float}
            
        Returns:
            str: 自然语言形式的规则
        """
        # 根据规则类型选择不同的翻译模板
        if rule.get("task_type") == "relation_extraction":
            # 关系抽取任务的规则翻译
            body = " 且 ".join(rule["body"])
            return f"如果 {body}，那么 {rule['head']}（精确度：{rule['accuracy']:.2f}）"
        
        elif rule.get("task_type") == "log_anomaly_detection":
            # 日志异常检测任务的规则翻译
            body = " 和 ".join(rule["body"])
            return f"当日志中出现 {body} 时，表明系统可能存在异常（精确度：{rule['accuracy']:.2f}）"
        
        else:
            # 通用规则翻译
            body = " AND ".join(rule["body"])
            return f"IF {body} THEN {rule['head']} (accuracy: {rule['accuracy']:.2f})"
    
    def retrieve_relevant_rules(self, context: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        根据上下文检索最相关的规则
        
        Args:
            context: 当前上下文或查询
            top_k: 返回的最相关规则数量
            
        Returns:
            List[Dict[str, Any]]: 最相关的规则列表
        """
        # 简单实现：基于关键词匹配的相关性评分
        # 实际应用中可以使用更复杂的语义相似度计算
        scored_rules = []
        
        for rule in self.rules:
            score = 0
            # 检查规则头部和体部中的关键词是否出现在上下文中
            for keyword in rule.get("keywords", []):
                if keyword.lower() in context.lower():
                    score += 1
            
            # 也可以考虑规则的精确度
            score *= rule.get("accuracy", 0.5)
            
            scored_rules.append((rule, score))
        
        # 按相关性得分排序并返回top_k个规则
        sorted_rules = [r[0] for r in sorted(scored_rules, key=lambda x: x[1], reverse=True)]
        return sorted_rules[:top_k]
    
    def inject_rules_into_prompt(self, prompt: str, context: str = "", top_k: int = 5) -> str:
        """
        将规则注入到提示中
        
        Args:
            prompt: 原始提示
            context: 当前上下文或查询，用于检索相关规则
            top_k: 注入的规则数量
            
        Returns:
            str: 注入规则后的提示
        """
        # 检索相关规则
        relevant_rules = self.retrieve_relevant_rules(context, top_k)
        
        # 将规则翻译为自然语言
        rule_texts = [self.translate_rule_to_natural_language(rule) for rule in relevant_rules]
        
        # 构建规则部分
        if rule_texts:
            rules_section = "请根据以下规则生成回答：\n" + "\n".join([f"- {rule}" for rule in rule_texts])
            # 将规则部分插入到提示中
            enhanced_prompt = f"{prompt}\n\n{rules_section}"
        else:
            enhanced_prompt = prompt
        
        return enhanced_prompt