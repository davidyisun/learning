#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 去哪儿网 数据爬取
Created on 2018-09-04
@author:David Yisun
@group:data
"""
import codecs
import datetime
from bs4 import BeautifulSoup

from selenium import webdriver
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time

url = 'http://hotel.qunar.com/'

# 获取日期 今明两天
today = datetime.date.today().strftime('%Y-%m-%d')
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
tomorrow = tomorrow.strftime('%Y-%m-%d')

# 驱动chrome
driver = webdriver.Chrome()
driver.set_page_load_timeout(50)

# 打开网页
driver.get(url)
driver.maximize_window()  # 将浏览器最大化显示
driver.implicitly_wait(10)  # 控制间隔时间，等待浏览器反映

# 设置起始日期
fromdate = today
todate = tomorrow

# 获取数据操作
ele_toCity = driver.find_element_by_name('toCity')   # 目的地
ele_fromDate = driver.find_element_by_id('fromDate')   # 入住日期
ele_toDate = driver.find_element_by_id('toDate')   # 离店日期
ele_search = driver.find_element_by_class_name('search-btn')   # 搜索按钮

ele_toCity.clear()   # 清除已有目的地
ele_toCity.send_keys(u'泸州')
ele_toCity.click()   # 点击

# 填写【入住日期】和【离店日期】 然后点击【搜索】
ele_fromDate.clear()
ele_fromDate.send_keys(fromdate)
ele_toDate.clear()
ele_toDate.send_keys(todate)
ele_search.click()

page_num = 0

while True:
    try:
        WebDriverWait(driver, 10).until(
            EC.title_contains(u'泸州'))
    except Exception as e:
        print(e)
        break
    time.sleep(5)

    js = "window.scrollTo(0, document.body.scrollHeight);"   # 滑动窗口到底部
    driver.execute_script(js)

    htm_const = driver.page_source   # 获取html







