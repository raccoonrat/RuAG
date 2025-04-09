# RuAG: Learned-Rule-Augmented Generation for Large Language Models

[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python 38](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 论文出处：ICLR 2025  

> 论文链接：[https://arxiv.org/abs/2411.03349](https://arxiv.org/abs/2411.03349)

## 简介

RuAG (Learned-Rule-Augmented Generation) 是一种通过学习到的逻辑规则增强大型语言模型生成能力的框架，旨在提高生成文本的质量、一致性和可控性。本框架通过从训练数据中自动提取高质量的逻辑规则，并将这些规则整合到生成过程中，使大型语言模型能够在生成时遵循这些规则，从而产生更加可靠和符合预期的输出。

RuAG 框架的主要特点：

1. **自动规则学习**：使用蒙特卡洛树搜索 (MCTS) 算法从训练数据中自动提取高质量的逻辑规则
  
2. **规则增强生成**：将学习到的规则整合到大型语言模型的生成过程中
  
3. **规则验证与调整**：对生成的文本进行规则验证，并在必要时进行调整
  
4. **多任务支持**：支持关系抽取、日志异常检测等多种任务
  

## 安装

### 环境要求

* Python 3.8+
  
* PyTorch 1.10+
  
* pandas, numpy, scikit-learn
  
* OpenAI API (可选，用于使用 GPT 模型)
  

### 安装步骤

    
    # 克隆仓库
    
    git clone https://github.com/raccoonrat/RuAG.git
    
    cd RuAG
    
    
    
    # 安装依赖
    
    pip install -r requirements.txt
    

## 项目结构

    
    RuAG/
    
    ├── config.py                # 全局配置文件
    
    ├── data/                    # 数据模块
    
    │   ├── datasets/            # 示例数据集
    
    │   ├── rules/               # 规则存储
    
    │   └── utils/               # 数据处理工具
    
    ├── examples/                # 示例代码
    
    │   ├── relation_extraction_example.py
    
    │   ├── log_anomaly_detection_example.py
    
    │   └── complete_pipeline_example.py
    
    ├── src/                     # 源代码
    
    │   ├── rule_learning/       # 规则学习模块
    
    │   ├── integration/         # 规则整合模块
    
    │   ├── generation/          # 规则增强生成模块
    
    │   └── post_processing/     # 后处理模块
    
    └── tests/                   # 测试代码
    

## 快速开始

### 关系抽取示例

    
    from config import Config
    
    from src.rule_learning import RuleExtractor
    
    from src.integration import RuleIntegrator
    
    from src.generation import RuleAugmentedGenerator
    
    from data.utils import DataLoader
    
    
    
    # 1. 加载数据
    
    loader = DataLoader()
    
    data = loader.load_data(task_type="relation")
    
    
    
    # 2. 提取规则
    
    rule_extractor = RuleExtractor()
    
    rules = rule_extractor.extract_rules(
    
        data,
    
        task_type="relation_extraction",
    
        num_rules=10,
    
        min_precision=0.5
    
    )
    
    
    
    # 3. 规则增强生成
    
    generator = RuleAugmentedGenerator(rules=rules)
    
    query = "分析以下文本中的实体关系：'梅西是巴塞罗那足球俱乐部的球员。'"
    
    response = generator.generate(query, context="梅西 球员 巴塞罗那")
    
    print(response)
    

### 日志异常检测示例

    
    from config import Config
    
    from src.rule_learning import RuleExtractor
    
    from src.integration import RuleIntegrator
    
    from src.generation import RuleAugmentedGenerator
    
    from data.utils import DataLoader
    
    
    
    # 1. 加载数据
    
    loader = DataLoader()
    
    data = loader.load_data(task_type="log")
    
    
    
    # 2. 提取规则
    
    rule_extractor = RuleExtractor()
    
    rules = rule_extractor.extract_rules(
    
        data,
    
        task_type="log_anomaly_detection",
    
        num_rules=10,
    
        min_precision=0.5
    
    )
    
    
    
    # 3. 规则增强生成
    
    generator = RuleAugmentedGenerator(rules=rules)
    
    query = "分析以下日志序列是否存在异常：'E1, E5, E7, E11, E12'"
    
    response = generator.generate(query, context="日志 E1 E5 E7 E11 E12")
    
    print(response)
    

### 使用命令行运行完整示例

    
    # 运行关系抽取示例
    
    python examples/complete_pipeline_example.py --task relation --model gpt-3.5-turbo
    
    
    
    # 运行日志异常检测示例
    
    python examples/complete_pipeline_example.py --task log --model gpt-3.5-turbo
    

## 核心模块

### 规则学习模块 (rule_learning)

规则学习模块负责从训练数据中自动提取高质量的逻辑规则。主要组件包括：

* 谓词定义 (predicate_definition.py) ：定义规则中使用的谓词
  
* 谓词筛选 (predicate_filtering.py) ：筛选有用的谓词
  
* MCTS 搜索 (mcts.py) ：使用蒙特卡洛树搜索算法搜索规则
  
* 规则提取器 (rule_extractor.py) ：整合上述组件，提取规则
  

### 规则整合模块 (integration)

规则整合模块负责将学习到的规则整合到生成过程中。主要组件包括：

* 规则整合器 (rule_integrator.py) ：将规则整合到提示中
  
* 规则检索器 (rule_retriever.py) ：根据上下文检索相关规则
  

### 规则增强生成模块 (generation)

规则增强生成模块负责使用整合了规则的提示生成文本。主要组件包括：

* LLM 提供者 (llm_provider.py) ：提供语言模型接口
  
* 规则增强生成器 (rule_augmented_generator.py) ：使用规则增强生成文本
  

### 后处理模块 (post_processing)

后处理模块负责验证生成的文本是否符合规则，并在必要时进行调整。主要组件包括：

* 规则验证器 (rule_validator.py) ：验证文本是否符合规则
  
* 文本调整器 (text_adjuster.py) ：调整不符合规则的文本
  

## 数据模块

数据模块提供了示例数据集、规则存储和数据处理工具。详细信息请参阅 data/README.md 。

## 示例代码

示例目录包含了几个完整的示例，展示了 RuAG 框架的使用方法：

* 关系抽取示例 (relation_extraction_example.py) ：展示如何使用 RuAG 框架进行关系抽取
  
* 日志异常检测示例 (log_anomaly_detection_example.py) ：展示如何使用 RuAG 框架进行日志异常检测
  
* 完整流程示例 (complete_pipeline_example.py) ：展示 RuAG 框架的完整工作流程
  

## 示例运行结果

在项目根目录下运行以下命令，查看示例运行结果：

    python examples/complete_pipeline_example.py --task relation --model volc-ark-deepseek

输出结果如下：

    === RuAG完整流程示例：relation ===
    1. 初始化LLM提供者...
    使用模型：volc-ark-deepseek
    
    2. 加载数据...
    加载了 10 条数据
    
    3. 数据预处理...
    预处理完成
    
    4. 规则学习或加载...
    学习了 10 条规则
    
    5. 规则翻译...
    规则示例:
      规则 1: 当实体A为哈兰德时，存在效力于关系（准确度：1.00）。
    
    注：翻译在保持原规则逻辑的基础上，通过以下优化使表达更自然：
    1. 将条件状语"如果...那么"转换为更符合中文习惯的"当...时"
    2. 采用"存在...关系"替代机械直译，使语义更完整
    3. 保留数值精度标记格式（1.00），符合技术文档规范
    4. 使用中文全角括号包裹附加参数，保持视觉一致性
    5. 省略冗余的"relation_"前缀，通过语境自然传达关系属性
      规则 2: 翻译：当实体为姆巴佩时，可以确定其效力于关系，精确度为1.00。
    
    解析说明：
    1. 保留原规则的条件结构，将"entity_a_姆巴佩"自然转化为"实体为姆巴佩"的完整中文表达
    2. 关系"relation_效力于"转化为动词短语"效力于关系"，既保留了关系抽取任务的专业性，又符合中文语序
    3. 精确度数值1.00采用中文科技文本通用的"精确度为1.00"表述方式，符合学术规范
    4. 整句采用"当...时，可以确定..."的条件逻辑句式，既准确传达规则判断逻辑，又保持中文行文的流畅性
    5. 专业术语（如"实体"、"精确度"）的保留确保了在自然语言转换过程中不丢失关键信息要素
      规则 3: 当检测到实体A为"梅西"时，可以确定其存在"效力于"的关联关系，该判断的置信度为100%。
    
    （说明：在保持技术规范性的同时采用了更符合中文表达习惯的句式结构。将"精确度"转换为更符合中文技术文档习惯的"置信度"表述，并用百分比形式强化了1.00的准确程度，整体表述既保留了原始规则的逻辑严谨性，又增强了自然语言的流畅度。）
    
    6. 规则整合...
    
    7. 生成...
    
    查询:
    分析以下文本中的实体关系：'梅西是巴塞罗那足球俱乐部的球员。'
    
    生成回答:
    
    
    在句子“梅西是巴塞罗那足球俱乐部的球员”中，可以提取以下实体关系：
    
    1. **实体**
       - **梅西**（类型：`人物`）
       - **巴塞罗那足球俱乐部**（类型：`组织`）
    
    2. **关系**
       - **关系类型**：`效力于`（或`属于`）
       - **关系描述**：梅西（人物）作为球员效力于巴塞罗那足球俱乐部（组织）。
    
    **结构化表示**：

    {
      "entities": [
        {"name": "梅西", "type": "人物"},
        {"name": "巴塞罗那足球俱乐部", "type": "组织"}
      ],
      "relations": [
        {
          "subject": "梅西",
          "predicate": "效力于",
          "object": "巴塞罗那足球俱乐部",
          "context": "球员身份"
        }
      ]
    }

    8. 后处理...
    生成的文本符合所有规则约束
    
    === 示例完成 ===
    
    

## 论文引用

如果您在研究中使用了 RuAG，请引用我们的论文：

    @inproceedings{
    
        ruag2025,
    
        title={RuAG: Learned-Rule-Augmented Generation for Large Language Models},
    
        author={Author Names},
    
        booktitle={International Conference on Learning Representations},
    
        year={2025},
    
        url={https://arxiv.org/abs/2411.03349}
    
    }

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 贡献

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork 本仓库
  
2. 创建您的特性分支 ( git checkout -b feature/amazing-feature )
  
3. 提交您的更改 ( git commit -m 'Add some amazing feature' )
  
4. 推送到分支 ( git push origin feature/amazing-feature )
  
5. 打开一个 Pull Request
  

## 联系方式

如有任何问题，请通过以下方式联系我们：

* 电子邮件： example@example.com
  
* GitHub Issues： https://github.com/raccoonrat/RuAG/issues