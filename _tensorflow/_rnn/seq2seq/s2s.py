#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: sequence to sequence 模型实现
Created on 2018-09-19
@author:David Yisun
@group:data
"""
import tensorflow as tf
import codecs
import collections
from operator import itemgetter

RAW_DATA = "./data/ptb.train.txt"  # 训练集数据文件
VOCAB_OUTPUT = "ptb.vocab"  # 输出的词汇表文件
VOCAB_SIZE = 4000  # 中文词表词汇长度



def read(filepath):
    counter = collections.Counter()
    # 统计词频
    with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
        for line in f:
            pass
