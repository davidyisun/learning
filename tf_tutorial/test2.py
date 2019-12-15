#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: name scope test2
Created on 2018-03-09
@author:David Yisun
@group:data
"""


d = tf.Variable(name="d", )
with tf.name_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var4 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var3))
    print(var4.name, sess.run(var4))
