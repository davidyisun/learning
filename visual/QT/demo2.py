#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""
from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QWidget()             # 创建窗口
window.setWindowTitle("窗口标题")  # 设置窗口标题
window.resize(300, 50)                   # 设置窗口大小
window.show()                            # 显示窗口
sys.exit(app.exec_())