#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: scrapy meituan spider
Created on 2018-11-02
@author:David Yisun
@group:data
"""
import scrapy
from scrapy import Selector

from meituan.items import CnblogspiderItem

class CnblogsSpider(scrapy.Spider):
    name = "cnblogs"  # 爬虫的名称
    allowed_domains = ["cnblogs.com"]  # 允许的域名
    start_urls = [
      "https://bj.meituan.com/meishi/pn01/"
    ]
    
    def parse(self, response):
        pass
    
    def parse_body(self, response):
        pass