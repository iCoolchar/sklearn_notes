#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.lda import LDA

iris = load_iris()

print iris.data
print iris.target

# 主成分分析法
print PCA(n_components=2).fit_transform(iris.data)

# 线性判别分析法
print LDA(n_components=2).fit_transform(iris.data, iris.target)
