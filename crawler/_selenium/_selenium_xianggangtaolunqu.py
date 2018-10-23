#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 香港讨论区selenium的demo
Created on 2018-10-19
@author:David Yisun
@group:data
"""
import codecs
import datetime
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time

url = 'https://finance.discuss.com.hk/forumdisplay.php?fid=57'

# 打开网页
driver = webdriver.Chrome()
driver.set_page_load_timeout(100)
driver.get(url=url)
driver.maximize_window()
driver.implicitly_wait(10)

while True:
    try:
        shut_1 = driver.find_element_by_xpath('/html/body/div[9]/div/div/a')
        shut_1.click()
    except:
        pass
    print('one')
    try:
        WebDriverWait(driver, 10).until(EC.visibility_of(driver.find_element_by_css_selector('next')))
    except Exception as e:
        print(e)
        break
    print('here')
    next = driver.find_element_by_css_selector('next')
    next.click()




