# src/data/discretize.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def discretize_features(df, feature, target, n_bins=10):
    """
    使用 Gini 指数离散化连续特征。
    :param df: DataFrame
    :param feature: 连续特征列名
    :param target: 目标列名
    :param n_bins: 离散化后的区间数
    :return: 离散化后的 DataFrame
    """
    # 使用决策树寻找最佳分割点
    tree = DecisionTreeClassifier(max_leaf_nodes=n_bins)
    tree.fit(df[[feature]], df[target])

    # 获取分割点
    thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
    thresholds = sorted(set(thresholds))

    # 离散化
    bins = [-float('inf')] + thresholds + [float('inf')]
    labels = [f"{feature}<{thresholds[0]}"] + \
             [f"{thresholds[i-1]}<={feature}<{thresholds[i]}" for i in range(1, len(thresholds))] + \
             [f"{feature}>={thresholds[-1]}"]
    df[f"{feature}_discrete"] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

    return df
