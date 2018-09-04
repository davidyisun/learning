#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: selenium爬虫实践 01
Created on 2018-08-30
@author:David Yisun
@group:data
"""
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome()
driver.get('http://www.baidu.com')
# 页名

# # 通过元素名称wd获取输入框
# elem = driver.find_element_by_name('wd')
# elem.clear()
# elem.send_keys(u'黄嘉恩')
# elem.send_keys(Keys.RETURN)  # 回车
# time.sleep(3)  # 延时3秒 等待加载
# driver.close() # 关闭

# 定位搜索框
e = driver.find_element_by_xpath('//*[@id="kw"]')
e.clear()
e.send_keys(u'黄嘉恩')
res = e.send_keys(Keys.RETURN)
# login_button = driver.find_element_by_xpath('//*[@id="su"]')
pass