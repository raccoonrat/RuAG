# src/generation/llm.py
from transformers import pipeline
from config import Config

def generate_text(prompt, rules):
    """生成增强文本"""
    generator = pipeline("text-generation", model=Config.LLM_MODEL)
    augmented_prompt = f"{prompt}\nRules: {rules}"
    return generator(augmented_prompt, max_length=100)[0]["generated_text"]
