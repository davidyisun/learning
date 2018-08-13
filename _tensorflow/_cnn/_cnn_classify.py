#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 多层CNN代码分析 a very simple MNIST classifier
Created on 2018-08-13
@author:David Yisun
@group:data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# Import data
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data/MNIST_data', 'Directory for storing data')  # 把数据放在../data/文件夹中

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

# 权重命名
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正太分布初始化变量
    return tf.Variable(initial)

# 偏移函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 用0.1初始化
    return tf.Variable(initial)

# 卷积核
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')