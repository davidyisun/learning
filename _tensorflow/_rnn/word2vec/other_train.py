#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: word2vector 实现 https://blog.csdn.net/u014595019/article/details/54093161
Created on 2018-11-01
@author:David Yisun
@group:data
"""

import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
from pprint import pprint
from pymongo import MongoClient
import re
import jieba
import os.path as path
import os
import codecs


def get_stop_word(path):
    """
        读取停用词
    :param path: 停用词表path
    :return:
    """
    stop_words = []
    with codecs.open(path, 'r', 'utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个单词'.format(n=len(stop_words)))
    return stop_words


def data_prepocess(path, stop_words):
    """
        过滤 停用词 读取文本，预处理，分词，得到词典
    :param path: 
    :param stop_words: 
    :return: 
    """
    raw_word_list = []
    sentence_list = []
    with codecs.open(path, 'r', encoding='gbk') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line)>0: # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words and word not in ['qingkan520','www','com','http']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    word_count = collections.Counter(raw_word_list)
    print('文本中总共有{n1}个单词,不重复单词数{n2},选取前30000个单词进入词典'.format(n1=len(raw_word_list),n2=len(word_count)))
    word_count = word_count.most_common(30000)
    word_list = [x[0] for x in word_count]
    return word_list

if __name__ == '__main__':
    stop_words = get_stop_word(path='./data/stop_words.txt')
    word_list = data_prepocess(path='./data/280.txt', stop_words=stop_words)
        