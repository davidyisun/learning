#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: jieba_fenci test
Created on 2018--
@author:David Yisun
@group:data
"""
import jieba
s = '上市公司拥有的辽宁民族54%的股权、香港民族54%的股权、国贸公司100%的股权、大和公司45%的股权、锦冠公司45%的股权和信添公司45%的股权。'
l = list(jieba.cut(s))
jieba.suggest_freq(('辽宁民族'), True)