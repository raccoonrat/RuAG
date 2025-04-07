# RuAG 数据模块

本目录包含 RuAG 框架的数据模块，包括示例数据集、规则存储和数据处理工具。

## 目录结构

- `datasets/`: 示例数据集
  - `relation_extraction.py`: 关系抽取示例数据集
  - `log_anomaly_detection.py`: 日志异常检测示例数据集
  - `relation_extraction_sample.csv`: 关系抽取示例数据文件
  - `log_anomaly_detection_sample.csv`: 日志异常检测示例数据文件

- `rules/`: 规则存储
  - `relation_rules.json`: 关系抽取规则
  - `log_rules.json`: 日志异常检测规则

- `utils/`: 数据处理工具
  - `preprocessor.py`: 数据预处理工具
  - `augmenter.py`: 数据增强工具
  - `evaluator.py`: 数据评估工具
  - `data_loader.py`: 数据加载工具
  - `data_splitter.py`: 数据分割工具

## 使用方法

### 加载示例数据集

```python
from data.utils import DataLoader

# 初始化数据加载器
loader = DataLoader()

# 加载关系抽取数据集
relation_data = loader.load_data(task_type="relation")

# 加载日志异常检测数据集
log_data = loader.load_data(task_type="log")
```

### 数据预处理
```python
from data.utils import DataPreprocessor

# 初始化数据预处理器
preprocessor = DataPreprocessor()

# 预处理关系抽取数据
processed_relation_data = preprocessor.preprocess(relation_data, task_type="relation_extraction")

# 预处理日志异常检测数据
processed_log_data = preprocessor.preprocess(log_data, task_type="log_anomaly_detection")
```

### 数据分割
```python
from data.utils import DataSplitter

# 初始化数据分割器
splitter = DataSplitter(seed=42)

# 分割数据集为训练集和测试集
split_data = splitter.split_data(processed_relation_data, test_size=0.2)
train_data = split_data["train"]
test_data = split_data["test"]

# K折交叉验证分割
folds = splitter.k_fold_split(processed_relation_data, n_folds=5)
```
### 加载规则
```python
from data.rules import load_rules

# 加载关系抽取规则
relation_rules = load_rules("relation_rules.json")

# 加载日志异常检测规则
log_rules = load_rules("log_rules.json")
```


这样，我们就完成了 `data` 目录下的所有核心模块，包括：

1. 示例数据集模块
2. 规则存储模块
3. 数据处理工具模块
   - 数据预处理器
   - 数据增强器
   - 数据评估器
   - 数据加载器
   - 数据分割器

这些模块共同构成了 RuAG 框架的数据基础设施，为规则学习和规则增强生成提供数据支持。