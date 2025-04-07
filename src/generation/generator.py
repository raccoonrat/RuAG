"""
生成模块 - 负责生成符合规则约束的高质量文本
"""
from typing import List, Dict, Any, Optional, Callable
import json
import os

# 导入规则整合器
from src.integration.rule_integrator import RuleIntegrator


class RuleAugmentedGenerator:
    """
    规则增强生成器 - 生成符合规则约束的文本
    """
    
    def __init__(
        self, 
        llm_provider: Callable,
        rules_path: Optional[str] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        max_rules: int = 5
    ):
        """
        初始化规则增强生成器
        
        Args:
            llm_provider: 语言模型提供者，一个可调用对象，接收提示并返回生成的文本
            rules_path: 规则文件路径
            rules: 直接提供的规则列表
            max_rules: 每次生成使用的最大规则数量
        """
        self.llm_provider = llm_provider
        self.rule_integrator = RuleIntegrator(rules_path, rules)
        self.max_rules = max_rules
    
    def generate(self, prompt: str, context: str = "", temperature: float = 0.7) -> str:
        """
        生成符合规则约束的文本
        
        Args:
            prompt: 原始提示
            context: 上下文，用于检索相关规则
            temperature: 生成的随机性，值越高随机性越大
            
        Returns:
            str: 生成的文本
        """
        # 1. 规则整合：将相关规则注入到提示中
        enhanced_prompt = self.rule_integrator.inject_rules_into_prompt(
            prompt, context, self.max_rules
        )
        
        # 2. 初始生成：使用增强后的提示生成文本
        generated_text = self.llm_provider(enhanced_prompt, temperature=temperature)
        
        # 3. 返回生成的文本
        return generated_text
    
    def generate_with_post_processing(
        self, 
        prompt: str, 
        context: str = "", 
        temperature: float = 0.7,
        max_attempts: int = 3
    ) -> str:
        """
        生成文本并进行后处理，确保符合规则约束
        
        Args:
            prompt: 原始提示
            context: 上下文，用于检索相关规则
            temperature: 生成的随机性
            max_attempts: 最大尝试次数，用于重新生成不符合规则的文本
            
        Returns:
            str: 经过后处理的生成文本
        """
        # 1. 初始生成
        generated_text = self.generate(prompt, context, temperature)
        
        # 2. 规则验证和后处理
        attempt = 1
        while attempt < max_attempts:
            # 检查生成的文本是否符合规则
            is_valid, violations = self._validate_against_rules(generated_text, context)
            
            if is_valid:
                # 文本符合规则，返回
                return generated_text
            
            # 文本不符合规则，添加违规信息并重新生成
            correction_prompt = f"{prompt}\n\n您之前的回答存在以下问题：\n"
            for violation in violations:
                correction_prompt += f"- {violation}\n"
            correction_prompt += "\n请重新生成符合所有规则的回答。"
            
            # 重新生成
            generated_text = self.generate(correction_prompt, context, temperature)
            attempt += 1
        
        # 达到最大尝试次数，返回最后一次生成的文本
        return generated_text
    
    def _validate_against_rules(self, text: str, context: str) -> tuple[bool, List[str]]:
        """
        验证生成的文本是否符合规则
        
        Args:
            text: 生成的文本
            context: 上下文
            
        Returns:
            tuple: (是否有效, 违规列表)
        """
        # 获取相关规则
        relevant_rules = self.rule_integrator.retrieve_relevant_rules(context, self.max_rules)
        
        # 检查每条规则
        violations = []
        for rule in relevant_rules:
            # 这里需要根据规则类型实现具体的验证逻辑
            # 简单示例：检查规则的头部是否在文本中体现
            if rule.get("head") and rule.get("head") not in text:
                # 检查是否满足规则的条件
                conditions_met = all(body_item in context for body_item in rule.get("body", []))
                if conditions_met:
                    violations.append(
                        f"违反规则：当满足 {' 且 '.join(rule.get('body', []))} 时，应该包含 {rule.get('head')}"
                    )
        
        return len(violations) == 0, violations