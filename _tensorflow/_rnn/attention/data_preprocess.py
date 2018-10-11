#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 数据预处理
Created on 2018-10-10
@author:David Yisun
@group:data
"""
import tensorflow as tf


# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每行一句话，单词已经转化为单词编号。
# 格式转换: '90 13 1799 0 4 11 86 6012 0 4 2' ---> [[90, 13, 1799, 0, 4, 11, 86, 6012, 0, 4, 2], [11]] 的形式 int32
def MakeDateset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(para):
    src_data = MakeDateset(para.src_path)
    trg_data = MakeDateset(para.trg_path)
    dataset = tf.data.Dataset.zip((src_data, trg_data))
    # 删除内容为空（只包含<eos>）和长度过长的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_size), (trg_input, trg_size)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(x=tf.greater(src_size, 1), y=tf.less(src_size, para.max_len))
        trg_len_ok = tf.logical_and(x=tf.greater(trg_size, 1), y=tf.less(trg_size, para.max_len))
        res = tf.logical_and(x=src_len_ok, y=trg_len_ok)
        return res

    dataset = dataset.filter(predicate=FilterLength)

    # 构建解码器需要的两种格式的目标句子:
    # 1.解码器输入 input '<sos> x y z'
    # 2.解码器目标输出 label 'x y z <eos>'
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_size), (trg_input, trg_size)) = (src_tuple, trg_tuple)
        trg_lable = tf.concat([para.sos_id], trg_input[:-1], axis=0)
        res = ((src_input, src_size), (trg_input, trg_lable, trg_size))
        return res

    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据
    dataset = dataset.shuffle(buffer_size=10000)

    # 规定填充后输出的数据维度
    padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])),
                     (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))

    # 用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size=para.batch_size,
                                           padded_shapes=padded_shapes)
    return batched_dataset




