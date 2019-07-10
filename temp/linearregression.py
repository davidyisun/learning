#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 线性回归
Created on 2019-07-10
@author: David Yisun
@group: data
@e-mail: david_yisun@163.com
@describe:
"""

# coding:utf-8
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# 使用以后的数据集进行线性回归（这里是波士顿房价数据）
##1 数据的加载
# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_y = loaded_data.target

data_X = np.array([2017, 2018, 2019])
data_y = np.array([3.35, 3.95, 4.54])

data_X = np.reshape(data_X, [data_X.shape[0], 1])
data_X = np.reshape(data_y, [data_y.shape[0], 1])

##2 模型的加载
model = LinearRegression()

##3 模型的训练
model.fit(data_X, data_y)

# print(model.predict(data_X[:4, :]))
# print(data_y[:4])

##4.查看模型拟合效果


# 使用生成线性回归的数据集，最后的数据集结果用散点图表示
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
# n_samples表示样本数目，n_features特征的数目  n_tragets  noise噪音

plt.scatter(X, y)
plt.show()

# 预测
test_x = np.array([[2022]])
test_y = model.predict(test_x)
# print('the result of test_x:\n{x}\n{y}'.find(x=test_x, y=test_y))
print(test_y)
pass