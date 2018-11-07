#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 模块测试脚本
Created on 2018-11-07
@author:David Yisun
@group:data
"""
from word2vector_self import *

def model_test():
    print('build model ......')
    model = word2vec(vocabulary_size=10,
                     dictionary={},
                     reverse_dictionary={},
                     log_dir=para['log_dir'],
                     model_dir=para['model_dir'])
    print('build graph ......')
    model.build_graph()
    print('model initialize ......')
    model.init_op()

if __name__ == '__main__':
    model_test()