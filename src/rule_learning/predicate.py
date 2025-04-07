# src/rule_learning/predicate.py
import openai
from config import Config  # 假设 Config 包含 LLM 模型和 API 配置

def define_predicates(df, task_description):
    """
    使用 LLM 定义目标谓词和体谓词。
    :param df: Pandas DataFrame，包含特征和标签
    :param task_description: 任务描述字符串
    :return: tuple (目标谓词, 体谓词列表)
    """
    # 设计提示
    prompt = f"""
    给定以下数据集和任务描述，请定义目标谓词和体谓词。
    数据集特征：{df.columns.tolist()}
    任务描述：{task_description}
    目标谓词通常是任务的标签，体谓词是与目标相关的特征。
    请以 JSON 格式返回：{{"target": "目标谓词", "body": ["体谓词1", "体谓词2", ...]}}
    """

    # 调用 LLM
    response = openai.Completion.create(
        model=Config.LLM_MODEL,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )

    # 提取谓词
    result = eval(response.choices[0].text.strip())  # 注意：生产环境中应使用 json.loads
    return result["target"], result["body"]

def filter_predicates(target, body_predicates, task_description):
    """
    使用 LLM 筛选体谓词。
    :param target: 目标谓词
    :param body_predicates: 初始体谓词列表
    :param task_description: 任务描述
    :return: 精简后的体谓词列表
    """
    # 设计提示
    prompt = f"""
    给定目标谓词 '{target}' 和任务描述 '{task_description}'，请评估以下体谓词的相关性：
    {body_predicates}
    返回一个 JSON 列表，包含与目标谓词相关的体谓词。
    """

    # 调用 LLM
    response = openai.Completion.create(
        model=Config.LLM_MODEL,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )

    # 提取相关谓词
    related_predicates = eval(response.choices[0].text.strip())  # 注意：生产环境中应使用 json.loads
    return related_predicates
