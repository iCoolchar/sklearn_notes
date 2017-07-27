#!/usr/bin/env python
# -*- coding: utf-8 -*-

from minepy import MINE
from numpy import array
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

iris = load_iris()

print iris.data
print iris.target

# 方差选择法
print VarianceThreshold(threshold=3).fit_transform(iris.data)

# 相关系数, 用协方差和方差的比值衡量两者是不是相关
print SelectKBest(lambda X, Y: map(tuple, array(map(lambda x:pearsonr(x, Y), X.T)).T), k=2).fit_transform(iris.data, iris.target)

# 卡方检验, 用自变量等于i且因变量等于j的样本频数的观察值与期望的差距来衡量两者是不是相关
print SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)


# mic返回最大互信息系数,返回二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5

# 选择K个最好的特征，返回特征选择后的数据
print SelectKBest(lambda X, Y: map(tuple, array(map(lambda x: mic(x, Y), X.T)).T), k=2).fit_transform(iris.data, iris.target)
# SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

# 递归特征消除法,每轮训练后,消除若干权值系数的特征,再基于新的特征集进行下一轮训练
print RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

# 带L1惩罚项的逻辑回归作为基模型的特征选择
print SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

# GBDT作为基模型的特征选择
print SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)




