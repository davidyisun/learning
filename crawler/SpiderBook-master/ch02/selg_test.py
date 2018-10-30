#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 自学test
Created on 2018-10-30
@author:David Yisun
@group:data
"""
import codecs
from bs4 import BeautifulSoup

with codecs.open('./self_test', 'r', 'utf-8') as f:
    text = f.read()
soup = BeautifulSoup(text, 'lxml')
for tag in soup:
    print('---'*10)
    print(tag)
l = [i for i in soup]
pass
