#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: sequence to sequence 模型实现 中文翻译
Created on 2018-09-19
@author:David Yisun
@group:data
"""
import tensorflow as tf
import codecs
import collections
from operator import itemgetter


# ---- 文本预处理参数 ----
RAW_DATA = "./data/train.txt.zh"  # 训练集数据文件
VOCAB_OUTPUT = "zh.vocab"  # 输出的词汇表文件
VOCAB_SIZE = 4000  # 中文词表词汇长度
OUTPUT_DATA = "train.zh"  # 转换为单词id编号的输出文件

# ---- seq2seq 模型训练参数 ----
SRC_TRAIN_DATA = ''


# 生成词表
def generated_vocab():
    counter = collections.Counter()
    # 统计词频
    with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1
    # 按词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [i[0] for i in sorted_word_to_cnt]
    # 插入特殊符号  中文中 需要 加 句子结束符<eos> 无法识别（低频词汇）<unk> 句子起始符<sos>
    sorted_words = ['<unk>', '<sos>', '<eos>']+sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]  # 截断
    # 保存词表文件
    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word+'\n')
    word_to_id = {j: i for i, j in enumerate(sorted_words)}
    return word_to_id


# 获取id, 没在词表里的命名为<unk>
def get_id(word, vocab):
    return vocab[word] if word in vocab else vocab['<unk>']


# 生成训练文件
def generated_train_dataset():
    word_to_id = generated_vocab()
    words = []
    fout = codecs.open()
    with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
         for line in f:
            words = line.strip().split()+['<eos>']
            out_line = ' '.join([str(get_id(word=i, vocab=word_to_id)) for i in words])
            words.append(out_line)
            fout.write(out_line+'\n')
    fout.close()
    return words




if __name__ == '__main__':
    pass

