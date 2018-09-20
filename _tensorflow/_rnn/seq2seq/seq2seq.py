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


class parameters():
    def __init__(self):
        self.src_train_data = './data/train.en'  # 英文源语言输入文件
        self.trg_train_data = './data/train.zh'  # 中文目标语言输入文件
        self.checkpoint_path = './seq2seq_ckpt'  # checkpoint保存路径
        # 模型参数
        self.hidden_size = 1024           # lstm 隐藏层规模
        self.num_layers = 2               # lstm 深层lstm层数
        self.src_vocab_size = 10000       # 英文词表大小
        self.trg_vocal_size = 4000        # 中文词表大小
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


def MakeSrcTrgDaset(src_path, trg_path, batch_size):
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



if __name__ == '__main__':
    # get_vocab(path='./data/zh.vocab')
    MakeDataset(file_path=para.trg_train_data)
