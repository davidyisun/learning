#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: sequence to sequence 英文翻译
Created on 2018-09-20
@author:David Yisun
@group:data
"""
import tensorflow as tf
import codecs
import tqdm


# 1.参数设置。
class parameters():
    def __init__(self):
        self.src_train_data = './data/train.en'  # 英文源语言输入文件
        self.trg_train_data = './data/train.zh'  # 中文目标语言输入文件
        self.checkpoint_path = './model/seq2seq_ckpt'  # checkpoint保存路径
        # 模型参数
        self.hidden_size = 1024           # lstm 隐藏层规模
        self.num_layers = 2               # lstm 深层lstm层数
        self.src_vocab_size = 10000       # 英文词表大小
        self.trg_vocab_size = 4000        # 中文词表大小
        self.batch_size = 100             # 训练数据batch的大小
        self.num_epoch = 5                # 训练数据的轮数
        self.keep_prob = 0.8              # dropout的保留程度
        self.max_grad_norm = 5            # 控制梯度膨胀的梯度大小上限
        self.share_emb_and_softmax = True # 在softmax层和隐藏层之间共享参数
        # 文本内容参数
        self.max_len = 50                 # 限定句子的最大单词数量
        self.sos_id = 1                   # 目标语言词表中<sos>的

para = parameters()


class vocab():
    def __init__(self, path):
        pass
    def query_id(self):
        pass
    def query_word(self):
        pass


# 获取中英文词表
def get_vocab(path):
    with codecs.open(path, 'r', 'utf-8') as f:
        data = f.read()
    data = data.splitlines()
    return data


# 2.读取训练数据并创建Dataset。¶
def MakeDataset(file_path):
    """
        使用dataset从一个文件中读取一个语言的数据
        数据的格式为每行一句话， 单词已经转化为单词编号
    :param file_path: 
    :return: 
    """
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言和目标语言数据
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个dataset合并成一个dataset，构建一个由4个张量组成的数据项
    dataset = tf.data.Dataset.zip((src_data, trg_data))
    # 删除内容为空（只包含<eos>）和 内容过多的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, para.max_len))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, para.max_len))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(FilterLength)

    #   1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
    #   2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
    # 形式并加入到Dataset中。
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[para.sos_id], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据。
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),      # 源句子是长度未知的向量
         tf.TensorShape([])),         # 源句子长度是单个数字
        (tf.TensorShape([None]),      # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),      # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))         # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


# 3.定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的多层lstm结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size)]*para.num_layers)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size)]*para.num_layers)
        
        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('src_emb', [para.src_vocab_size, para.hidden_size])
        self.trg_embedding = tf.get_variable('trg_emb', [para.trg_vocab_size, para.hidden_size])
        
        # 定义softmax层的变量 (判断是否共享参数向量)
        if para.share_emb_and_softmax:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('softmax_weight', [para.hidden_size, para.trg_vocab_size])
        
        self.softmax_bias = tf.get_variable('softmax_bias', [para.trg_vocab_size])

        # 在forward函数中定义模型的前向计算图。
        # src_input, src_size, trg_input, trg_label, trg_size分别是上面
        # MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, keep_prob=para.keep_prob)
        trg_emb = tf.nn.dropout(trg_emb, keep_prob=para.keep_prob)

        # 使用dynamic_rnn构造编码器。
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类
        # 张量的tuple，每个LSTMStateTuple对应编码器中的一层。
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size,
        # max_time, HIDDEN_SIZE]。Seq2Seq模型中不需要用到enc_outputs，而
        # 后面介绍的attention模型会用到它。

        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                       inputs=src_emb,
                                                       sequence_length=src_size,
                                                       dtype=tf.float32)
        # 使用dyanmic_rnn构造解码器。
        # 解码器读取目标句子每个位置的词向量，输出的dec_outputs为每一步
        # 顶层LSTM的输出。dec_outputs的维度是 [batch_size, max_time,
        # HIDDEN_SIZE]。
        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        with tf.variable_scope('decoder'):
            dec_outputs, _ = tf.nn.dynamic_rnn(cell=self.dec_cell,
                                               inputs=trg_emb,
                                               sequence_length=trg_size,
                                               initial_state=enc_state)    # 将enc_state(由编码器的最后一个time_step组成的LSTMStateTuple('c', 'h'))作为解码器的初始状态
        # 计算解码器每一步的log perplexity。  计算的是解码器的loss
        output = tf.reshape(dec_outputs, [-1, para.hidden_size])
        logits = tf.matmul(output, self.softmax_weight)+self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练。
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss*label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。反向操作的实现与语言模型代码相同。
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(ys=cost/tf.to_float(batch_size),
                             xs=trainable_variables)

        grads, _ = tf.clip_by_global_norm(t_list=grads,
                                          clip_norm=para.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


# 4.训练过程和主函数。
def run_epoch(session, cost_op, train_op, saver, step):
    """
          使用给定的模型model上训练一个epoch，并返回全局步数。
          每训练200步便保存一个checkpoint。
    :param session: 
    :param cost_op: 
    :param train_op: 
    :param saver: 
    :param step: 
    :return: 
    """
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供。
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            # 每200步保存一个checkpoint。
            if step % 200 == 0:
                saver.save(session, para.checkpoint_path, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NMTModel()

    # 定义输入数据。
    data = MakeSrcTrgDataset(src_path=para.src_train_data,
                             trg_path=para.trg_train_data,
                             batch_size=para.batch_size)
    # 创建dataset的iterator
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    # 训练模型。
    saver = tf.train.Saver()
    step = 0

    pbar = tqdm.tqdm(list(range(para.num_epoch)))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(para.num_epoch):
            pbar.set_description('epoch {0}'.format(i+1))
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)
            pbar.update(1)


if __name__ == '__main__':
    # get_vocab(path='./data/zh.vocab')
    # MakeDataset(file_path=para.trg_train_data)
    main()