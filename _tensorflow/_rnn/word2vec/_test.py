#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""
import tensorflow as tf

e = tf.Variable(tf.random_uniform([4, 6], 10, 20))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    l = sess.run(e)
    l2 = sess.run(tf.square(l))
    l3 = sess.run(tf.sqrt(tf.reduce_sum(l2, 1, keepdims=True)))
    l4 = l/l3
    print(l)
    # print(tf.shape(l))
    # print('-'*20)
    # print(l2)
    # print(tf.shape(l2))
    print('-'*20)
    print(l3)
    # print(tf.shape(l3))
    print('-'*20)
    print(l4)