#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 千千音乐
Created on 2018-09-29
@author:David Yisun
@group:data
"""
from selenium import webdriver
url = 'http://music.taihe.com/artist'
# 驱动chrome
driver = webdriver.Chrome()
driver.set_page_load_timeout(50)

# 打开网页
driver.get(url)
driver.maximize_window()  # 将浏览器最大化显示
driver.implicitly_wait(10)  # 控制间隔时间，等待浏览器反映