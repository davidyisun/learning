#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 千千音乐数据爬取
Created on 2018-11-09
@author:David Yisun
@group:data
"""
import scrapy

class QianqianyinyueSpider(scrapy.Spider):
    name = 'qianqianmusic'
    start_urls = ['http://music.taihe.com/artist/']
    def parse(self, response):
        pass