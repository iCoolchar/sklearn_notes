#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import vstack, array, nan, log1p
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder, Imputer, PolynomialFeatures, FunctionTransformer

iris = load_iris()
print iris.data
print iris.target

# 标准化, 用求z-score的方法
print StandardScaler().fit_transform(iris.data) # 转换后的X
a = StandardScaler().fit(iris.data) # 带mean_和std_
print a.mean_
print a.std_
print a.transform(iris.data)

# 区间缩放法
print MinMaxScaler().fit_transform(iris.data)

# 归一化, 依照特征矩阵的行处理数据, 其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”
print Normalizer().fit_transform(iris.data)

# 二值化
print Binarizer(threshold=3).fit_transform(iris.data)

# 哑编码, 就是把定性数据(如label)用只有0或1的向量表示
print OneHotEncoder().fit_transform(iris.target.reshape((-1,1))) # reshape先把数据变成一列

# 缺失值填充, 默认用均值
print Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))

# 多项式转换
# 例如 [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
print PolynomialFeatures().fit_transform(iris.data)

# 单变元函数的数据变换, log1p = log e (1+x)
print FunctionTransformer(log1p).fit_transform(iris.data)