#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 美团美食评论数据
Created on 2018-11-09
@author:David Yisun
@group:data
"""
import scrapy
from scrapy import Selector

class MeituanMeishiSpider(scrapy.Spider):
    name = 'meituanmeishi'
    start_urls = ['http://bj.meituan.com/meishi/pn3/']
    def parse(self, response):
        pass