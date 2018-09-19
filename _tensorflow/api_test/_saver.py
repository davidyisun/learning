#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 模型的保存与加载
Created on 2018-09-14
@author:David Yisun
@group:data
"""
import tensorflow as tf
import numpy as np
def save():
    with tf.variable_scope('test'):
        w1 = tf.Variable(tf.random_normal(shape=[2, 3]), name='w1')
        w2 = tf.Variable(tf.random_normal(shape=[3, 5]), name='w2')
        x = tf.placeholder('float', shape=[2, 2], name='x')
        res = tf.matmul(tf.matmul(x, w1), w2, name='res')
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print(res.name)
        saver.save(sess, './save_model/model.ckpt', global_step=2000)


def read():
    # 加载网络图
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./save_model/model.ckpt-1000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./save_model/'))
        gragh = tf.get_default_graph()
        res = gragh.get_tensor_by_name('test/res:0')
        print(sess.run(res, feed_dict={x: np.array([10, 20])}))
        # w = sess.run('w1:0')
        # print(tf.shape(w))

def get_w():
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./save_model/model.ckpt-1000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./save_model/'))
        with tf.get_variable_scope():
            w = tf.get_variable('w1')
        print(sess.run(w))

if __name__ == '__main__':
    save()
    # read()
    # get_w()