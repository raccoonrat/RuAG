"""
语言模型提供者 - 提供与不同语言模型交互的接口
"""
import os
import requests
from typing import Dict, Any, Optional


class LLMProvider:
    """
    语言模型提供者基类
    """
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        调用语言模型生成文本
        
        Args:
            prompt: 提示文本
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        raise NotImplementedError("子类必须实现此方法")


class OpenAIProvider(LLMProvider):
    """
    OpenAI API提供者
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        初始化OpenAI提供者
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的模型名称
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供，请设置OPENAI_API_KEY环境变量或直接传入")
        
        self.model = model
    
    def __call__(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        调用OpenAI API生成文本
        
        Args:
            prompt: 提示文本
            temperature: 生成的随机性
            max_tokens: 最大生成token数
            
        Returns:
            str: 生成的文本
        """
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("请安装openai包：pip install openai")
        except Exception as e:
            print(f"OpenAI API调用失败：{e}")
            return f"生成失败：{e}"


class MockLLMProvider(LLMProvider):
    """
    模拟语言模型提供者，用于测试
    """
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        模拟生成文本
        
        Args:
            prompt: 提示文本
            **kwargs: 其他参数
            
        Returns:
            str: 模拟生成的文本
        """
        # 简单的模拟响应，实际应用中可以根据prompt内容生成更复杂的响应
        return f"这是对提示的模拟响应：\n{prompt[:100]}...\n\n这是一个模拟的语言模型生成的文本，用于测试RuAG框架。"