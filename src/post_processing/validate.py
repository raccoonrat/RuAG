"""
验证模块 - 负责验证生成文本是否符合规则约束
"""
from typing import List, Dict, Any, Tuple, Optional
import re


class RuleValidator:
    """
    规则验证器 - 验证生成文本是否符合规则约束
    """
    
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        """
        初始化规则验证器
        
        Args:
            rules: 规则列表，每条规则包含body、head和accuracy等字段
        """
        self.rules = rules or []
    
    def validate(self, text: str, context: str, rules: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        验证文本是否符合规则约束
        
        Args:
            text: 待验证的文本
            context: 上下文信息
            rules: 用于验证的规则列表，如果为None则使用初始化时提供的规则
            
        Returns:
            Tuple[bool, List[Dict[str, Any]]]: (是否有效, 违规规则列表)
        """
        rules_to_check = rules or self.rules
        violations = []
        
        for rule in rules_to_check:
            # 检查规则是否适用于当前上下文
            if not self._is_rule_applicable(rule, context):
                continue
                
            # 验证规则
            is_valid = self._validate_rule(rule, text, context)
            
            if not is_valid:
                violations.append(rule)
        
        return len(violations) == 0, violations
    
    def _is_rule_applicable(self, rule: Dict[str, Any], context: str) -> bool:
        """
        检查规则是否适用于当前上下文
        
        Args:
            rule: 规则
            context: 上下文
            
        Returns:
            bool: 规则是否适用
        """
        # 检查规则的体部条件是否在上下文中满足
        # 对于不同类型的任务，可能需要不同的适用性检查逻辑
        
        if rule.get("task_type") == "relation_extraction":
            # 关系抽取任务：检查实体和关系是否在上下文中
            for body_item in rule.get("body", []):
                # 简化处理：检查体部条件的关键词是否在上下文中
                if not any(keyword in context for keyword in body_item.split()):
                    return False
            return True
            
        elif rule.get("task_type") == "log_anomaly_detection":
            # 日志异常检测任务：检查特定日志事件是否在上下文中
            for body_item in rule.get("body", []):
                if body_item not in context:
                    return False
            return True
            
        else:
            # 通用规则：检查体部条件的关键词是否在上下文中
            for body_item in rule.get("body", []):
                if isinstance(body_item, str) and body_item not in context:
                    return False
            return True
    
    def _validate_rule(self, rule: Dict[str, Any], text: str, context: str) -> bool:
        """
        验证文本是否符合特定规则
        
        Args:
            rule: 规则
            text: 待验证的文本
            context: 上下文
            
        Returns:
            bool: 文本是否符合规则
        """
        # 根据规则类型使用不同的验证逻辑
        
        if rule.get("task_type") == "relation_extraction":
            # 关系抽取任务：检查头部关系是否在文本中体现
            head = rule.get("head", "")
            
            # 检查所有体部条件是否满足
            body_satisfied = all(self._check_condition_in_context(body, context) 
                                for body in rule.get("body", []))
            
            # 如果体部条件满足，则头部关系应该在文本中体现
            if body_satisfied:
                return self._check_relation_in_text(head, text)
            return True  # 体部条件不满足，规则不适用
            
        elif rule.get("task_type") == "log_anomaly_detection":
            # 日志异常检测任务：检查异常标记是否在文本中
            head = rule.get("head", "")
            
            # 检查所有体部条件（日志事件）是否在上下文中
            body_satisfied = all(body in context for body in rule.get("body", []))
            
            # 如果体部条件满足，则应该标记为异常
            if body_satisfied and head == "异常":
                return "异常" in text or "不正常" in text
            return True  # 体部条件不满足，规则不适用
            
        else:
            # 通用规则：简单检查头部是否在文本中
            head = rule.get("head", "")
            
            # 检查所有体部条件是否满足
            body_satisfied = all(body in context for body in rule.get("body", []))
            
            # 如果体部条件满足，则头部应该在文本中
            if body_satisfied and head:
                return head in text
            return True  # 体部条件不满足或没有头部，规则不适用
    
    def _check_condition_in_context(self, condition: str, context: str) -> bool:
        """
        检查条件是否在上下文中满足
        
        Args:
            condition: 条件
            context: 上下文
            
        Returns:
            bool: 条件是否满足
        """
        # 简化处理：检查条件的关键词是否在上下文中
        keywords = condition.split()
        return any(keyword in context for keyword in keywords)
    
    def _check_relation_in_text(self, relation: str, text: str) -> bool:
        """
        检查关系是否在文本中体现
        
        Args:
            relation: 关系
            text: 文本
            
        Returns:
            bool: 关系是否在文本中体现
        """
        # 提取关系的主要部分
        match = re.search(r"实体A(.*?)实体B", relation)
        if match:
            relation_core = match.group(1).strip()
            # 检查关系核心是否在文本中
            return relation_core in text
        
        # 如果无法提取核心关系，直接检查整个关系
        return relation in text
    
    def get_violation_details(self, violations: List[Dict[str, Any]]) -> List[str]:
        """
        获取违规详情
        
        Args:
            violations: 违规规则列表
            
        Returns:
            List[str]: 违规详情列表
        """
        details = []
        
        for rule in violations:
            task_type = rule.get("task_type", "通用")
            body = " 且 ".join(rule.get("body", []))
            head = rule.get("head", "")
            accuracy = rule.get("accuracy", 0)
            
            if task_type == "relation_extraction":
                detail = f"违反关系抽取规则：当 {body} 时，应有关系 {head}（精确度：{accuracy:.2f}）"
            elif task_type == "log_anomaly_detection":
                detail = f"违反日志异常检测规则：当出现 {body} 时，应标记为 {head}（精确度：{accuracy:.2f}）"
            else:
                detail = f"违反规则：当 {body} 时，应有 {head}（精确度：{accuracy:.2f}）"
            
            details.append(detail)
        
        return details