#### 3.4 `src/config.py`
配置全局参数和 API 密钥：
```python
# src/config.py
import os

class Config:
    DATA_DIR = "data/"
    LLM_MODEL = "gpt-4"  # 或 "bert-base-uncased" 等
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    MCTS_ROUNDS = 500
    MAX_RULE_LENGTH = 5
    PRECISION_THRESHOLD = 0.85
