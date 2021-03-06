#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: PTB 语言模型
Created on 2018-09-12
@author:David Yisun
@group:data
"""
import numpy as np
import tensorflow as tf

# 1.设置参数
TRAIN_DATA = "ptb.train"          # 训练数据路径。
EVAL_DATA = "ptb.valid"           # 验证数据路径。
TEST_DATA = "ptb.test"            # 测试数据路径。
HIDDEN_SIZE = 300                 # 隐藏层规模。
NUM_LAYERS = 3                    # 深层循环神经网络中LSTM结构的层数。
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

# 2.定义模型

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
        self.initial_state = cell.zero_state(batch_size, tf.float32)  # state 由lstm层个数的 [batch_size, hidden] 组成的tuple,  其中 state 的最后一层  表示各batch的最后一个step的output

        # 定义单词的词向量矩阵。
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转化为词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)   # 经过embedding_lookup之后 有 20*35 变为 20*35*300  将每一个词映射成一个词向量

        # 只在训练时使用dropout。
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再一起提供给softmax层。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)      # time_step 表示一句话中第time_step个词语 cell_output 由 [batch, hidden] 组成 表示 各个batch下个step的输出
                outputs.append(cell_output)       # 实际上 outputs和states得到的是 step个 [batch_size,  hideden] 组成的list
        # 把输出队列展开成[batch, hidden_size*num_steps]的形状，然后再
        # reshape成[batch*numsteps, hidden_size]的形状。
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])  # outputs是list

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


# 3.定义数据和训练过程。
# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值。
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量。
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)   # 初始化 state
    # 训练一个epoch。
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单
        # 词为给定单词的概率。
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                     {model.input_data: x,
                                      model.targets: y,
                                      model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        # 只有在训练时输出日志。
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                  step, np.exp(total_costs / iters)))
        step += 1

    # 返回给定模型在给定数据上的perplexity值。
    return step, np.exp(total_costs / iters)


# 从文件中读取数据，并返回包含单词编号的数组。
def read_data(file_path):
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串。
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    return id_list


def make_batches(id_list, batch_size, num_step):
    # 计算总的batch数量。每个batch包含的单词数量是batch_size * num_step。
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # 如9-4图所示，将数据整理成一个维度为[batch_size, num_batches * num_step]
    # 的二维数组。
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成num_batches个batch，存入一个数组。
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位。这里得到的是RNN每一步输出所需要预测的
    # 下一个单词。
    label = np.array(id_list[1 : num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)
    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵。
    return list(zip(data_batches, label_batches))    #  输出为 [(data, labele)] 的形式

    """
    [(array([[ 0,  1,  2,  3,  4],
            [15, 16, 17, 18, 19],
            [30, 31, 32, 33, 34],
            [45, 46, 47, 48, 49],
            [60, 61, 62, 63, 64]]), 
      array([[ 1,  2,  3,  4,  5],
            [16, 17, 18, 19, 20],
            [31, 32, 33, 34, 35],
            [46, 47, 48, 49, 50],
            [61, 62, 63, 64, 65]]))]
    """


# 4.主函数
def main():
    # 数据读入
    train_batches = make_batches(
        read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    eval_batches = make_batches(
        read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    test_batches = make_batches(
        read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model",
                           reuse=None,
                           initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试用的循环神经网络模型。它与train_model共用参数，但是没有dropout。
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 保存模型
    saver = tf.train.Saver()

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session,
                                         train_model,
                                         train_batches,
                                         train_model.train_op,
                                         True,
                                         step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)
            print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))

        _, test_pplx = run_epoch(session, eval_model, test_batches,
                                 tf.no_op(), False, 0)
        print("Test Perplexity: %.3f" % test_pplx)

        # 保存模型
        saver.save(session, '../../models/PTB.train')


if __name__ == "__main__":
    main()

