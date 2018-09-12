#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
@contact:davidhu@wezhuiyi.com
"""

import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

test = tf.Variable(output)
t1 = tf.get_variable(name="input2", shape=[1])

writer = tf.train.SummaryWriter("./test/log")


