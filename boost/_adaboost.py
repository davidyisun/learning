#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""
a = range(10)

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def g1(x):
    if x < 2.5:
        return 1
    else:
        return -1


a1 = [sign(i*0.4236) for i in list(map(g1, a))]

def g2(x):
    if x < 8.5:
        return 1
    else:
        return -1

