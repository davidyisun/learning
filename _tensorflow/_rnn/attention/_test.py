#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018-10-10
@author:David Yisun
@group:data
"""
def t1():
    import tensorflow as tf

    # dataset = tf.data.TextLineDataset('./data/text_test.txt')
    dataset = tf.data.TextLineDataset('./data/train.raw.en')
    # dataset = dataset.map(lambda string:string+'123')
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    iterator = dataset.make_initializable_iterator()
    t = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        # print(str(sess.run(t), encoding='utf-8'))
        for i in range(1):
            print('----'*10)
            k = sess.run(t)
            # print(str(k, encoding='utf-8'))
            # print(type(k))
            print([str(j, encoding='utf-8') for j in k])
            print(sess.run(tf.size(k)))

def t2():
    import tensorflow as tf
    MAX_LEN = 50
    SOS_ID = 1
    # 使用Dataset从一个文件中读取一个语言的数据。
    # 数据的格式为每行一句话，单词已经转化为单词编号。
    def MakeDataset(file_path):
        dataset = tf.data.TextLineDataset(file_path)
        # 根据空格将单词编号切分开并放入一个一维向量。
        dataset = dataset.map(lambda string: tf.string_split([string]).values)
        # 将字符串形式的单词编号转化为整数。
        dataset = dataset.map(
            lambda string: tf.string_to_number(string, tf.int32))
        # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
        dataset = dataset.map(lambda x: (x, tf.size(x)))
        return dataset

    # 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和
    # batching操作。
    def MakeSrcTrgDataset(src_path, trg_path, batch_size):
        # 首先分别读取源语言数据和目标语言数据。
        src_data = MakeDataset(src_path)
        trg_data = MakeDataset(trg_path)
        # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds
        # 由4个张量组成：
        #   ds[0][0]是源句子
        #   ds[0][1]是源句子长度
        #   ds[1][0]是目标句子
        #   ds[1][1]是目标句子长度
        dataset = tf.data.Dataset.zip((src_data, trg_data))

        # 删除内容为空（只包含<EOS>）的句子和长度过长的句子。
        def FilterLength(src_tuple, trg_tuple):
            ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
            src_len_ok = tf.logical_and(
                tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
            trg_len_ok = tf.logical_and(
                tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
            return tf.logical_and(src_len_ok, trg_len_ok)

        dataset = dataset.filter(FilterLength)

        # 从图9-5可知，解码器需要两种格式的目标句子：
        #   1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
        #   2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
        # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"
        # 形式并加入到Dataset中。
        def MakeTrgInput(src_tuple, trg_tuple):
            ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
            trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
            return ((src_input, src_len), (trg_input, trg_label, trg_len))

        dataset = dataset.map(MakeTrgInput)

        # 随机打乱训练数据。
        dataset = dataset.shuffle(10000)

        # 规定填充后输出的数据维度。 经过padded_batch 将单条数据以batch格式输出
        padded_shapes = (
            (tf.TensorShape([None]),  # 源句子是长度未知的向量
             tf.TensorShape([])),  # 源句子长度是单个数字
            (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
             tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
             tf.TensorShape([])))  # 目标句子长度是单个数字
        # 调用padded_batch方法进行batching操作。
        batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
        return batched_dataset


    # 定义输入数据。
    data = MakeSrcTrgDataset(src_path='./data/train.en',
                             trg_path='./data/train.zh',
                             batch_size=2)
    iterator = data.make_initializable_iterator()
    t = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        # print(str(sess.run(t), encoding='utf-8'))
        for i in range(3):
            print('----'*10)
            k = sess.run(t)
            # print(str(k, encoding='utf-8'))
            # print(type(k))
            # print([str(j, encoding='utf-8') for j in k])
            # print(sess.run(tf.size(k)))
            pass


def t3():
    import tensorflow as tf
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess=sess,
                      save_path='../../../../model/attention/en_to_zh/attention_ckpt-8800')
        model_variables = tf.contrib.slim.get_variables()
        restore_variables = [var for var in model_variables]
        for var in restore_variables:
            print(var.name)
    pass


if __name__ == '__main__':
    # t2()
    t3()