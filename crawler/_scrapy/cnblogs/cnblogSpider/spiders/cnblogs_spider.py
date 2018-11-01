#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""
import scrapy

from cnblogSpider.items import CnblogspiderItem

class CnblogsSpider(scrapy.Spider):
    name = "cnblogs"  # 爬虫的名称
    allowed_domains = ["cnblogs.com"]  # 允许的域名
    start_urls = [
      "http://www.cnblogs.com/qiyeboy/default.html?page=1"
    ]
    def parse(self, response):
        #里面实现网页的而解析
        # 里面实现网页的而解析
        # 首先抽取所有的文章
        papers = response.xpath(".//*[@class='day']")
        # 从每篇文章中抽取数据
        for paper in papers:
            url = paper.xpath(".//*[@class='postTitle']/a/@href").extract()[0]
            title = paper.xpath(".//*[@class='postTitle']/a/text()").extract()[0]
            time = paper.xpath(".//*[@class='dayTitle']/a/text()").extract()[0]
            content = paper.xpath(".//*[@class='postCon']/div/text()").extract()[0]
            item = CnblogspiderItem(url=url, title=title, time=time, content=content)
            # print(item.keys())
            # print(item.items())
            yield item