#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: Attention 注意力机制 机器翻译
Created on 2018-10-10
@author:David Yisun
@group:data
"""
import tensorflow as tf
import parameters
import data_preprocess


def get_para(type='en_to_zh', process='train'):
    if type == 'en_to_zh':
        para_train = parameters.para_train()
        para_predict = parameters.para_predict()
    else:
        para_train = parameters.para_train2()
        para_predict = parameters.para_predict2()

    if process == 'train':
        p = para_train
    else:
        p = para_predict
    return p

para = get_para()


class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器所使用的lstm结构
        # --- 编码器 前向
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size)
        # --- 编码器 反向
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size)
        # --- 解码器
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size) for i in range(para.decoder_layers)])

        # 定义词向量
        self.src_embedding = tf.get_variable('src_emb', [para.src_vocab_size, para.hidden_size])
        self.trg_embedding = tf.get_variable('trg_emb', [para.trg_vocab_size, para.hidden_size])

        # 定义softmax层
        if para.share_emb_and_softmax:  # 此处共享 是指与 embedding时的 weight 共享
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [para.hidden_size, para.trg_vocab_size])
        self.softmax_bias = tf.get_variable('softmax_bias', [para.trg_vocab_size])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):

        # 将输入的单词编码转换为embedding
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量层上进行dropout
        src_emb = tf.nn.dropout(src_emb, keep_prob=para.keep_prob)
        trg_emb = tf.nn.dropout(trg_emb, keep_prob=para.keep_prob)

        # 构建模型
        # --- 编码 ---
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw,
                                                                     self.enc_cell_bw,
                                                                     src_emb,
                                                                     src_size,
                                                                     dtype=tf.float32)
            # 将前向和后向lstm连接起来
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], axis=-1)  # -1 表示将连个tensor最里层的数拼在一起

        # --- 解码 --- 选择注意力权重模型
        with tf.variable_scope('decoder'):
            # 1.BahdanauAttention是使用一个隐藏层的前馈神经网络。BahdanauAttention是使用一个隐藏层的前馈神经网络。
            # 2.memory_sequence_length是一个维度为[batch_size]的张量，代表batch中每个句子的长度，Attention需要根据这个信息把填充位置的注意力权重设置为0。
            attention_machanism = tf.contrib.seq2seq.BahdanauAttention(num_units=para.hidden_size,
                                                                       memory=enc_outputs,
                                                                       memory_sequence_length=src_size)
            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,
                                                                 attention_machanism,
                                                                 attention_layer_size=para.hidden_size)
            # 使用attention_cell和dynamic_rnn构造编码器，这里没有指定init_state,也就是没有使用编码器的输出来初始化
            # 输入，而完全依赖注意力作为信息来源。
            dec_outputs, _ = tf.nn.dynamic_rnn(cell=attention_cell,
                                               inputs=trg_emb,
                                               sequence_length=trg_size,
                                               dtype=tf.float32)

            # 计算解码器每一步的log perplexity
            output = tf.reshape(dec_outputs, [-1, para.hidden_size])
            logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]),
                                                                  logits=logits)

            # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练
            label_weights = tf.sequence_mask(trg_size,
                                             maxlen=tf.shape(trg_label)[1],
                                             dtype=tf.float32)
            label_weights = tf.reshape(label_weights, [-1])
            cost = tf.reduce_sum(loss * label_weights)
            cost_per_token = cost / tf.reduce_sum(label_weights)

            # 定义反向传播操作。
            trainable_variables = tf.trainable_variables()
            # 控制梯度大小，定义优化方法和训练步骤
            grads = tf.gradients(ys=cost/tf.to_float(para.batch_size),
                                 xs=trainable_variables)
            grads, _ = tf.clip_by_global_norm(t_list=grads,
                                              clip_norm=para.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
            return cost_per_token, train_op



def run_epoch(session, cost_op, train_op, saver, step):
    while True:  # 符合dataset的方式运行
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            if step % 200 == 0:
                saver.save(sess=session, save_path=para.checkpoint_path, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main_train():
    initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)  # 初始化函数
    # --- 数据 ---
    dataset = data_preprocess.MakeSrcTrgDataset(para=para)
    iterator = dataset.make_initializable_iterator()
    (src_input, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()
    # --- 模型 ---
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        model = NMTModel()
    cost_op, train_op = model.forward(src_input=src_input,
                                      src_size=src_size,
                                      trg_input=trg_input,
                                      trg_label=trg_label,
                                      trg_size=trg_size)
    # --- 存储 ---
    saver = tf.train.Saver()
    step = 0
    # --- 训练过程 ---
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(para.num_epoch):
            print("In iteration: %d" % (epoch + 1))
            sess.run(iterator.initializer)
            step = run_epoch(session=sess,
                             cost_op=cost_op,
                             train_op=train_op,
                             saver=saver,
                             step=step)


def main_predict():
    pass


if __name__ == '__main__':
    main_train()