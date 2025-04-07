"""
RuAG: Learned-Rule-Augmented Generation for Large Language Models

RuAG是一种通过学习到的逻辑规则增强大型语言模型生成能力的框架，
旨在提高生成文本的质量、一致性和可控性。
"""

from .config import Config
from src.rule_learning import RuleExtractor, RuleTranslator
from src.integration import RuleIntegrator
from src.generation import RuleAugmentedGenerator
from src.post_processing import RuleValidator, TextAdjuster

__version__ = "0.1.0"
__author__ = "RuAG Team"

__all__ = [
    'Config',
    'RuleExtractor',
    'RuleTranslator',
    'RuleIntegrator',
    'RuleAugmentedGenerator',
    'RuleValidator',
    'TextAdjuster'
]