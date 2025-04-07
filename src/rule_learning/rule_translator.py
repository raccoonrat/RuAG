"""
规则翻译器 - 负责将逻辑规则翻译为自然语言
"""
from typing import List, Dict, Any, Optional, Callable


class RuleTranslator:
    """
    规则翻译器 - 将逻辑规则翻译为自然语言
    """
    
    def __init__(self, llm_provider: Optional[Callable] = None):
        """
        初始化规则翻译器
        
        Args:
            llm_provider: 语言模型提供者，用于更自然的翻译
        """
        self.llm_provider = llm_provider
    
    def translate_rule(self, rule: Dict[str, Any]) -> str:
        """
        将规则翻译为自然语言
        
        Args:
            rule: 规则，包含body、head和accuracy等字段
            
        Returns:
            str: 自然语言形式的规则
        """
        task_type = rule.get("task_type", "generic")
        
        if task_type == "relation_extraction":
            return self._translate_relation_rule(rule)
        elif task_type == "log_anomaly_detection":
            return self._translate_log_rule(rule)
        else:
            return self._translate_generic_rule(rule)
    
    def _translate_relation_rule(self, rule: Dict[str, Any]) -> str:
        """
        翻译关系抽取规则
        
        Args:
            rule: 规则
            
        Returns:
            str: 自然语言形式的规则
        """
        body = " 且 ".join(rule["body"])
        head = rule["head"]
        accuracy = rule.get("accuracy", 0)
        
        # 使用LLM进行更自然的翻译
        if self.llm_provider:
            prompt = f"""
            请将以下关系抽取规则翻译为自然、流畅的中文句子：
            
            规则：如果 {body}，那么 {head}（精确度：{accuracy:.2f}）
            
            翻译：
            """
            
            translation = self.llm_provider(prompt)
            if translation:
                return translation.strip()
        
        # 默认翻译
        return f"如果 {body}，那么 {head}（精确度：{accuracy:.2f}）"
    
    def _translate_log_rule(self, rule: Dict[str, Any]) -> str:
        """
        翻译日志异常检测规则
        
        Args:
            rule: 规则
            
        Returns:
            str: 自然语言形式的规则
        """
        body = " 和 ".join(rule["body"])
        head = rule["head"]
        accuracy = rule.get("accuracy", 0)
        
        # 使用LLM进行更自然的翻译
        if self.llm_provider:
            prompt = f"""
            请将以下日志异常检测规则翻译为自然、流畅的中文句子：
            
            规则：当日志中出现 {body} 时，表明系统可能存在{head}（精确度：{accuracy:.2f}）
            
            翻译：
            """
            
            translation = self.llm_provider(prompt)
            if translation:
                return translation.strip()
        
        # 默认翻译
        return f"当日志中出现 {body} 时，表明系统可能存在{head}（精确度：{accuracy:.2f}）"
    
    def _translate_generic_rule(self, rule: Dict[str, Any]) -> str:
        """
        翻译通用规则
        
        Args:
            rule: 规则
            
        Returns:
            str: 自然语言形式的规则
        """
        body = " AND ".join(rule["body"])
        head = rule["head"]
        accuracy = rule.get("accuracy", 0)
        
        # 使用LLM进行更自然的翻译
        if self.llm_provider:
            prompt = f"""
            请将以下逻辑规则翻译为自然、流畅的中文句子：
            
            规则：IF {body} THEN {head} (accuracy: {accuracy:.2f})
            
            翻译：
            """
            
            translation = self.llm_provider(prompt)
            if translation:
                return translation.strip()
        
        # 默认翻译
        return f"如果满足条件 {body}，则 {head} 成立（精确度：{accuracy:.2f}）"
    
    def translate_rules_batch(self, rules: List[Dict[str, Any]]) -> List[str]:
        """
        批量翻译规则
        
        Args:
            rules: 规则列表
            
        Returns:
            List[str]: 翻译后的规则列表
        """
        return [self.translate_rule(rule) for rule in rules]