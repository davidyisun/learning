#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 机器翻译 完全模型
Created on 2018-10-08
@author:David Yisun
@group:data
"""

import tensorflow as tf
import codecs
import tqdm
import sys

# 0.设置test输入
class input():
    flags = tf.flags
    flags.DEFINE_string('en_sentence', 'this is a book.', 'str: the sentence needed to be translated.')
    FLags = flags.FLAGS


# 1.参数设置。
class parameters():
    def __init__(self):
        self.src_train_data = './data/train.en'  # 英文源语言输入文件
        self.trg_train_data = './data/train.zh'  # 中文目标语言输入文件
        self.checkpoint_path = '../../../../model/seq2seq/seq2seq_ckpt'  # checkpoint保存路径
        # 模型参数
        self.hidden_size = 1024  # lstm 隐藏层规模
        self.num_layers = 2  # lstm 深层lstm层数
        self.src_vocab_size = 10000  # 英文词表大小
        self.trg_vocab_size = 4000  # 中文词表大小
        self.batch_size = 100  # 训练数据batch的大小
        self.num_epoch = 5  # 训练数据的轮数
        self.keep_prob = 0.8  # dropout的保留程度
        self.max_grad_norm = 5  # 控制梯度膨胀的梯度大小上限
        self.share_emb_and_softmax = True  # 在softmax层和隐藏层之间共享参数
        # 文本内容参数
        self.max_len = 50  # 限定句子的最大单词数量
        self.sos_id = 1  # 目标语言词表中<sos>的
        pass


class predict_parameters():
    checkpoint_path = '../../../../model/seq2seq/seq2seq_ckpt-9000'  # 读取checkpoint的路径。9000表示是训练程序在第9000步保存的checkpoint
    # 模型参数 必须 和训练模型参数保持一致
    hidden_size = 1024  # lstm的隐藏层规模
    num_layers = 2  # 深层循环神经网络中lstm结构的层数
    src_vocab_size = 10000  # 源语言词汇表大小
    trg_vocab_size = 4000  # 目标语言词汇表大小
    share_emb_and_softmax = True  # 在softmax层和词向量层之间共享参数
    # 词汇表文件
    src_vocab = './data/en.vocab'
    trg_vocab = './data/zh.vocab'
    # 词汇表中<sos>和<eos>的ID，在解码过程中需要用<sos>作为第一步的输入, 并将检查是否是<eos>，因此需要知道这两个符号的ID
    sos_id = 1
    eos_id = 2


para = parameters()
para_predict = predict_parameters()


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

    dataset = dataset.filter(FilterLength)   # 根据条件过滤需要的数据

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
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))  # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


# 3.定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的多层lstm结构
        # self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size)] * para.num_layers)
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size) for _ in range(para.num_layers)])
        # self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size)] * para.num_layers)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(para.hidden_size) for _ in range(para.num_layers) ])

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
                                               initial_state=enc_state)  # 将enc_state(由编码器的最后一个time_step组成的LSTMStateTuple('c', 'h'))作为解码器的初始状态
        # 计算解码器每一步的log perplexity。  计算的是解码器的loss
        output = tf.reshape(dec_outputs, [-1, para.hidden_size])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练。
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。反向操作的实现与语言模型代码相同。
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(ys=cost / tf.to_float(batch_size),
                             xs=trainable_variables)

        grads, _ = tf.clip_by_global_norm(t_list=grads,
                                          clip_norm=para.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


    def inference(self, src_input):  # 推理
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里将输入句子整理为大小为1的batch
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器. 这一步与训练时相同。
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                       inputs=src_emb,
                                                       sequence_length=src_size,
                                                       dtype=tf.float32)
        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题
        max_dec_len=100

        with tf.variable_scope('decoder/rnn/multi_rnn_cell'):
            # 使用一个变长的tensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入
            init_array = init_array.write(0, para_predict.sos_id)
            # 构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的TensorArray,以及记录解码步数的一个整数step
            init_loop_var = (enc_state, init_array, 0)
            # tf.while_loop的循环条件: 循环直到解码器输出<eos>，或者达到最大步数为止
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step),
                                                                 para_predict.eos_id),
                                                    tf.less(step, max_dec_len)

                ))
            def loop_body(state, trg_ids, step):  # trg_ids 为上一个词的id
                # 读取最后一步输出的单词， 并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # 这里不适用dynamic_rnn, 而是直接调用dec_cell向前计算一步。
                dec_outputs, next_state = self.dec_cell.call(inputs=trg_emb,
                                                             state=state)
                # 计算每个可能的输出单词对应的logit， 并选取logit值最大的单词作为这一步的id而输出。
                output = tf.reshape(dec_outputs, [-1, para_predict.hidden_size])
                logits = (tf.matmul(output, self.softmax_weight)+self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # 执行tf.while_loop, 返回最终状态
            state, trg_ids, step = tf.while_loop(cond=continue_loop_condition,
                                                 body=loop_body,
                                                 loop_vars=init_loop_var)
            return trg_ids.stack()


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


# 训练主函数
def train_main():
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
            pbar.set_description('epoch {0}'.format(i + 1))
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)
            pbar.update(1)


def predict_main():
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('nmt_model', reuse=None):
        model = NMTModel()
    # 定义个测试句子
    flags = input()
    test_en_text = flags.FLags.en_sentence
    # 根据英文词汇表, 将句子转为单词id
    with codecs.open(predict_parameters.src_vocab, 'r', 'utf-8') as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>']) for token in test_en_text.split()]
    if '<eos>' not in test_en_text:
        test_en_ids.append(src_id_dict['<eos>'])

    # print(test_en_ids)

    # 建立解码所需的计算图
    output_op = model.inference(test_en_ids)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, predict_parameters.checkpoint_path)  # 加载模型

    # 读取翻译结果
        output_ids = sess.run(output_op)

    # 根据中文词汇表, 将翻译结果转换为中文文字
    with codecs.open(para_predict.trg_vocab, 'r', 'utf-8') as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = [trg_vocab[x] for x in output_ids]

    # 输出翻译结果
    # print(output_text.encode('utf8').decode(sys.stdout.encoding))
    return test_en_text, test_en_ids, output_text


if __name__ == '__main__':
    # get_vocab(path='./data/zh.vocab')
    # MakeDataset(file_path=para.trg_train_data)
    # --- 预测 ---
    # test_en_text, test_en_ids, output_text = predict_main()
    # print('---'*20)
    # print(test_en_text)
    # print(test_en_ids)
    # output_text = ''.join(output_text)
    # print(output_text.encode('ascii').decode(sys.getdefaultencoding()))
    # print(output_text)
    # --- 训练 ---
    train_main()