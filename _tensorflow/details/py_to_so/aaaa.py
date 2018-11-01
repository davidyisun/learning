#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名:
Created on 2018--
@author:David Yisun
@group:data
"""

import ctypes
import os
so = ctypes.cdll.LoadLibrary(os.getcwd()+'/t1.cpython-36m-x86_64-linux-gnu.so')