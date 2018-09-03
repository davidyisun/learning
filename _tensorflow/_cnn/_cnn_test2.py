
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

import tqdm
import tensorflow as tf
# Import data
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data/MNIST_data', 'Directory for storing data')  # 把数据放在../data/文件夹中

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

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
    """
        convolution layer
    :param x: input tensor [batch, in_height, in_width, in_channels] 当前输入
    :param W: filter tensor [filter_height, filter_width, in_channels, out_channels]
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1 表示剩下的维度


# 第一层 conv 输入: 28*28*1 输出:28*28*32 32个channel filter: 5*5
# with tf.variable_scope('layer01_con01'):
W_conv1 = weight_variable(shape=[5, 5, 1, 32])
b_conv1 = bias_variable(shape=[32])
h_conv1 = tf.nn.elu(features=conv2d(x=x_image, W=W_conv1)+b_conv1)
# h_conv1 = tf.nn.elu(features=tf.nn.bias_add(conv2d(x=x_image, W=W_conv1), b_conv1))

# 第二层 pooling 输入: 28*28*32 输出: 14*14*32
# with tf.name_scope('layer02_pool01'):
h_pool1 = max_pool_2x2(x=h_conv1)

# 第三层 卷积层  输入:14*14*32 filter:5*5 64个 输出:14*14*64
# with tf.variable_scope('layer03_conv02'):
W_conv2 = weight_variable(shape=[5, 5, 32, 64])
b_conv2 = bias_variable(shape=[64])
h_conv2 = tf.nn.elu(conv2d(x=h_pool1, W=W_conv2)+b_conv2)
# h_conv2 = tf.nn.elu(tf.nn.bias_add(h_conv2, b_conv2))

# 第四层 池化层 输入:h_conv2 14*14*64 输出 7*7*64
# with tf.name_scope('layer04_pool02'):
h_pool2 = max_pool_2x2(x=h_conv2)

# 第五层 全连接层 输入: h_pool2 7*7*64 需要拉直成一个向量 输出: 1024*1
# with tf.variable_scope('layer05-fc01'):
h_pool2_shape = h_pool2.get_shape()
nodes_fc1 = h_pool2_shape[1]*h_pool2_shape[2]*h_pool2_shape[3]
h_pool2_flat = tf.reshape(h_pool2, shape=[-1, nodes_fc1])
# h_pool2_flat = tf.reshape(h_pool2, shape=[h_pool2_shape[0], nodes_fc1])
W_fc1 = weight_variable(shape=[nodes_fc1.value,1024])
b_fc1 = bias_variable(shape=[1024])
h_fc1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))
# 添加dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(x=h_fc1, keep_prob=keep_prob)

# 第六层 全连接层 输入: h_fc1_drop 1024 输出: 10  具体分为0~9类别
# with tf.variable_scope('layer06_fc02'):
w_fc2 = weight_variable(shape=[1024, 10])
b_fc2 = bias_variable(shape=[10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

y_ = tf.placeholder(tf.float32, [None, 10])

# 定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  #准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast() 转型函数

# 每个批次大小
batch_size = 100
# 一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
# 训练轮数
epoches = 20

pbar = tqdm.tqdm(total=epoches)

# 训练
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epoches):
        pbar.set_description('Processing epoch {0}'.format(epoch))
        pbar.update(1)
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 每100个batch算一下准确率
            if batch%100==0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
                print("step %d, training accuracy %g"%(batch, train_accuracy))
            train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
        total_acc = accuracy.run(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('> > > > > > > epoch %d test accuracy %g'%(epoch, total_acc))