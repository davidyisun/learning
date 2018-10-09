#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""
import codecs
import chardet

# 根据中文词汇表, 将翻译结果转换为中文文字
with codecs.open('./data/zh.vocab', 'r', 'utf-8') as f_vocab:
    trg_vocab = [w.strip() for w in f_vocab.readlines()]
print(trg_vocab)
output_ids = [19, 13, 9, 0]
output_text = ''.join([trg_vocab[x] for x in output_ids])
# print(output_text)
fenconding = chardet.detect(output_text)
print(fenconding)