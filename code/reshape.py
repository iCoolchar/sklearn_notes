#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder

a = np.array([1,2,3,4,5,6,7,8])
print a
b = a.reshape((-1,1))
print b
c = a.reshape((2,2,2))
print c
d = a.reshape((2,4))
print d

