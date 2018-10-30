#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: so 的 setup 文件
Created on 2018-10-29
@author:David Yisun
@group:data
"""
from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize(['tt.py']))

# import py_compile
# py_compile.compile(r'D:\workspace_for_python\to_github\learning2\_tensorflow\details\py_to_so\tt.py')