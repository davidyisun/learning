#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 加载 ptb 模型 做出预测
Created on 2018-09-19
@author:David Yisun
@group:data
"""
import tensorflow as tf
import numpy as np


# 1.设置参数
TRAIN_DATA = "ptb.train"          # 训练数据路径。
EVAL_DATA = "ptb.valid"           # 验证数据路径。
TEST_DATA = "ptb.test"            # 测试数据路径。
HIDDEN_SIZE = 300                 # 隐藏层规模。
NUM_LAYERS = 2                    # 深层循环神经网络中LSTM结构的层数。
VOCAB_SIZE = 10000                # 词典规模。
TRAIN_BATCH_SIZE = 20             # 训练数据batch的大小。
TRAIN_NUM_STEP = 35               # 训练数据截断长度。

EVAL_BATCH_SIZE = 1               # 测试数据batch的大小。
EVAL_NUM_STEP = 1                 # 测试数据截断长度。
NUM_EPOCH = 5                     # 使用训练数据的轮数。
LSTM_KEEP_PROB = 0.9              # LSTM节点不被dropout的概率。
EMBEDDING_KEEP_PROB = 0.9         # 词向量不被dropout的概率。
MAX_GRAD_NORM = 5                 # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True      # 在Softmax层和词向量层之间共享参数。


# 通过一个PTBModel类来描述模型，这样方便维护循环神经网络中的状态。
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度。
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义每一步的输入和预期输出。两者的维度都是[batch_size, num_steps]。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络。
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # 初始化最初的状态，即全零的向量。这个量只在每个epoch初始化第一个batch
        # 时使用。
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 定义单词的词向量矩阵。
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转化为词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只在训练时使用dropout。
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再一起提供给
        # softmax层。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                # 把输出队列展开成[batch, hidden_size*num_steps]的形状，然后再
        # reshape成[batch*numsteps, hidden_size]的形状。
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # Softmax层：将RNN在每个位置上的输出转化为各个单词的logits。
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias    # logits 是输出

        # 定义交叉熵损失函数和平均损失。
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size  # 平均损失
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training: return

        trainable_variables = tf.trainable_variables()  # 获取需要训练的变量列表
        # 控制梯度大小，定义优化方法和训练步骤。  tf.clip_by_global_norm()  梯度剪裁
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables),
                                          MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))  #  自由计算梯度


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义模型
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        model = PTBModel()
    # 定义输入数据



