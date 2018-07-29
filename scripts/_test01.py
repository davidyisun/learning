#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""


import tensorflow as tf

x = tf.constant([[1., 2.], [5., 2.], [1., 2.], [5., 2.]])

sess1 = tf.InteractiveSession()
sess1.run(x)
sess1.run(tf.shape(x))


xShape = tf.shape(x)
z1 = tf.reduce_mean(x, axis=0)  # 沿axis=0操作
z2 = tf.reduce_mean(x, axis=1)  # 沿axis=1操作


with tf.Session() as sess:
    xShapeValue, d1, d2 = sess.run([xShape, z1, z2])
    print('shape= %s' % (xShapeValue))
    print(d1)
    print(d2)
