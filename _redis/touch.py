#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: redis 初探
Created on 2018-11-21
@author:David Yisun
@group:data
"""
import redis

# 建立连接
r = redis.Redis(host='127.0.0.1', port=6379)

# 建立连接池管理redis
pool = redis.ConnectionPool(host='127.0.0.1', port=6379)
r = redis.Redis(connection_pool=pool)

r.lpush("digit", 11,22,33)
print(r.lrange('digit', ))
