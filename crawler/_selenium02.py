#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: selenium爬虫实践 02
Created on 2018-08-30
@author:David Yisun
@group:data
"""
import sys
sys.path.append('../')
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# 初始化chrome驱动 打开html
driver = webdriver.Chrome()
driver.get("D:/workspace_for_python/to_github/learning2/crawler/9.4.4.login.html")
# 获取用户名和密码的输入框和登录按钮
username = driver.find_element_by_name('username')
password = driver.find_element_by_xpath(".//*[@id='loginForm']/input[2]")
login_button = driver.find_element_by_xpath("//input[@type='submit']")
#