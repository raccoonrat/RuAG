# RuAG: Learned-Rule-Augmented Generation

RuAG 是一个结合逻辑规则学习和大型语言模型（LLM）的框架，旨在提升生成文本的质量和可控性。

## 安装

1. 克隆仓库：

   ```bash
   git clone https://github.com/raccoonrat/RuAG.git
   cd RuAG

2. 创建虚拟环境并安装依赖：
  
   ```bash
      python -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
  

## 使用

   ```bash
    python src/main.py --input "example query" --output "generated_text.txt"

## 贡献

欢迎提交 PR 和 Issues！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

3. 核心文件内容

4. 开发与测试

1. **编写模块**
  
  * 根据需求逐步实现 `src/` 下各模块功能。
  * 确保每个模块有独立的 `__init__.py`，支持导入。
2. **测试**
  
  * 在 `tests/` 下编写单元测试，例如：
    
        # tests/test_data.py
        import pytest
        from src.data.preprocess import preprocess_data
        
        def test_preprocess():
            assert preprocess_data("data/raw/test.csv", "data/processed/test.csv") is not None
    
  * 运行测试：
    
        pytest
    

* * *

### 5. 提交到 GitHub

1. **提交代码**：
  
      git add .
      git commit -m "Initial project structure and core files"
      git push origin main
  
2. **后续开发**：
  
  * 使用 feature 分支开发新功能，例如：
    
        git checkout -b feature/data-processing
    

* * *

### 6. 文档完善

* 在 `docs/api.md` 中记录模块 API。
* 在 `docs/usage.md` 中提供详细使用示例。
