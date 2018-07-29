#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: tf 基本概念
Created on 2018-07-29
@author:David Yisun
@group:data
"""
import tensorflow as tf


# 1.1 tensor
a = tf.zeros(shape=[1, 2])
print('-'*20)
print(a)
        # --- 只有在训练过程开始后，才能获得a的实际值 sess.run()
sess = tf.InteractiveSession()
print('-'*20)
print(sess.run(a))


# 1.2 variable
tensor = tf.zeros(shape=[1,2])
variable = tf.Variable(tensor)
sess = tf.InteractiveSession()
# print(sess.run(variable))  # 会报错
        # --- Variable必须初始化以后才有具体的值
sess.run(tf.initialize_all_variables()) # 对variable进行初始化
print(sess.run(variable))


# 1.3 placeholder
"""
又叫占位符，同样是一个抽象的概念。用于表示输入输出数据的格式。
告诉系统：这里有一个值/向量/矩阵，现在我没法给你具体数值，
不过我正式运行的时候会补上的！
例如上式中的x和y。因为没有具体数值，所以只要指定尺寸即可
"""
x = tf.placeholder(tf.float32, [1, 5], name='input')
y = tf.placeholder(tf.float32, [None, 5], name='input')


# 1.4 session
"""
    session是抽象模型的实现者。
"""

# example ---- 官方tutorial中的mnist数据集的分类
    # 训练模型
    # --- 建立抽象模型
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])     # 输入占位符
y = tf.placeholder(dtype=tf.float32, shape=[None, 784])     # 输出占位符（预期输出）
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
a = tf.nn.softmax(tf.matmul(x, W)+b)        # a表示模型的实际输出

    # 定义损失函数和训练方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)  # 梯度下降法 学习率为0.5
train = optimizer.minimize(cross_entropy)  # 训练目标： 最小化损失函数

    # 测试模型
correct_prediction = tf.equal(tf.argmax(a, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    # 实际训练
sess = tf.InteractiveSession()  # 建立交互式会话
tf.initialize_all_tables().run()  # 所有变量初始化
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train.run({x: batch_xs, y: batch_ys})
print(sess.run(accuracy, feed_dict={x: mnist.test, y: mnist.test.labels}))