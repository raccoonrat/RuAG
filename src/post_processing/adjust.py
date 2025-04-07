"""
调整模块 - 负责调整不符合规则的生成文本
"""
from typing import List, Dict, Any, Optional, Callable
from .validate import RuleValidator


class TextAdjuster:
    """
    文本调整器 - 调整不符合规则的生成文本
    """
    
    def __init__(
        self, 
        llm_provider: Callable,
        rules: Optional[List[Dict[str, Any]]] = None,
        validator: Optional[RuleValidator] = None
    ):
        """
        初始化文本调整器
        
        Args:
            llm_provider: 语言模型提供者，用于重新生成文本
            rules: 规则列表
            validator: 规则验证器，如果为None则创建新的验证器
        """
        self.llm_provider = llm_provider
        self.rules = rules or []
        self.validator = validator or RuleValidator(rules)
    
    def adjust(
        self, 
        text: str, 
        context: str, 
        original_prompt: str,
        max_attempts: int = 3,
        temperature: float = 0.7
    ) -> str:
        """
        调整不符合规则的文本
        
        Args:
            text: 待调整的文本
            context: 上下文
            original_prompt: 原始提示
            max_attempts: 最大尝试次数
            temperature: 生成的随机性
            
        Returns:
            str: 调整后的文本
        """
        # 验证文本是否符合规则
        is_valid, violations = self.validator.validate(text, context, self.rules)
        
        # 如果文本符合规则，直接返回
        if is_valid:
            return text
        
        # 获取违规详情
        violation_details = self.validator.get_violation_details(violations)
        
        # 尝试调整文本
        adjusted_text = text
        attempt = 0
        
        while not is_valid and attempt < max_attempts:
            # 构建调整提示
            adjustment_prompt = self._build_adjustment_prompt(
                original_prompt, adjusted_text, violation_details
            )
            
            # 使用语言模型重新生成
            adjusted_text = self.llm_provider(adjustment_prompt, temperature=temperature)
            
            # 再次验证
            is_valid, violations = self.validator.validate(adjusted_text, context, self.rules)
            violation_details = self.validator.get_violation_details(violations)
            
            attempt += 1
        
        return adjusted_text
    
    def _build_adjustment_prompt(
        self, 
        original_prompt: str, 
        text: str, 
        violation_details: List[str]
    ) -> str:
        """
        构建调整提示
        
        Args:
            original_prompt: 原始提示
            text: 当前文本
            violation_details: 违规详情列表
            
        Returns:
            str: 调整提示
        """
        prompt = f"{original_prompt}\n\n"
        prompt += "您之前的回答存在以下问题：\n"
        
        for detail in violation_details:
            prompt += f"- {detail}\n"
        
        prompt += "\n您的回答是：\n"
        prompt += f"{text}\n\n"
        prompt += "请根据上述规则调整您的回答，确保符合所有规则约束。"
        
        return prompt
    
    def adjust_with_specific_rules(
        self, 
        text: str, 
        context: str, 
        original_prompt: str,
        specific_rules: List[Dict[str, Any]],
        max_attempts: int = 3,
        temperature: float = 0.7
    ) -> str:
        """
        使用特定规则调整文本
        
        Args:
            text: 待调整的文本
            context: 上下文
            original_prompt: 原始提示
            specific_rules: 特定规则列表
            max_attempts: 最大尝试次数
            temperature: 生成的随机性
            
        Returns:
            str: 调整后的文本
        """
        # 验证文本是否符合特定规则
        is_valid, violations = self.validator.validate(text, context, specific_rules)
        
        # 如果文本符合规则，直接返回
        if is_valid:
            return text
        
        # 获取违规详情
        violation_details = self.validator.get_violation_details(violations)
        
        # 尝试调整文本
        adjusted_text = text
        attempt = 0
        
        while not is_valid and attempt < max_attempts:
            # 构建调整提示
            adjustment_prompt = self._build_adjustment_prompt(
                original_prompt, adjusted_text, violation_details
            )
            
            # 使用语言模型重新生成
            adjusted_text = self.llm_provider(adjustment_prompt, temperature=temperature)
            
            # 再次验证
            is_valid, violations = self.validator.validate(adjusted_text, context, specific_rules)
            violation_details = self.validator.get_violation_details(violations)
            
            attempt += 1
        
        return adjusted_text