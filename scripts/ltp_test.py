#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: ltp测试
Created on 2018-07-16
@author:David Yisun
@group:data
"""
import sys
sys.path.append('../')
from _ltp.basic import sentence_splitter, segmentor, posttagger, ner, parse, role_label
import codecs


def fenju(s=''):
    return sentence_splitter(s)


def fenci(s=''):
    return segmentor(s)

if __name__ == '__main__':
    s = '深圳美丽生态股份有限公司（以下简称“公司”）董事会于 2018 年 2 月 1 日收到公司总经理郑方先生的书面辞职报告，郑方先生因个人原因，辞去公司总经理职务。根据相关规定，辞职报告自送达公司董事会时生效。郑方先生辞去总经理职务不会影响公司相关工作的正常运行，辞职后继续担任公司董事。公司董事会将按照相关规定尽快聘请新任总经理。截止本公告日，郑方先生间接持有公司股份 39,366,373 股，占公司总股本的 4.80%。郑方先生将继续履行本人做出的承诺及《公司法》、《深圳证券交易所主板上市公司规范运作指引》和《深圳证券交易所上市公司股东及董事、监事、高级管理人员减持股份实施细则》等相关法律法规、规范性文件的规定。郑方先生在任职期间积极履行总经理职责，公司董事会对郑方先生在任职期间为公司发展所做出的贡献表示衷心的感谢！'
    s = '上市公司拥有的辽宁民族54%的股权、香港民族54%的股权、国贸公司100%的股权、大和公司45%的股权、锦冠公司45%的股权和信添公司45%的股权。'
    # with codecs.open('scripts/data/test_data', 'r', 'utf8') as f:
    #     d = f.read()
    d1 = list(segmentor(s))
    d2 = list(posttagger(segmentor(s)))
    d3 = list(zip(d1, d2))
    d4 = list(ner(d1, d2))
    d5 = list(zip(d1, d2, d4))
    d6 = list(parse(d1, d2))
    d7 = list(zip(d1, d6))


    # 结合jieba的效果
    s1 = '上市公司拥有的辽宁民族54%的股权、香港民族54%的股权、国贸公司100%的股权、大和公司45%的股权、锦冠公司45%的股权和信添公司45%的股权。'

    pass